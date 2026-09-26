"""The Qwen3.8-27B CUDA engine behind ``tensorfold.cuda.server``: one GPU, or two ranks over NCCL.

Rank 0 serves HTTP and sends each request's header and prompt to rank 1, which runs the same calls
(``follow``). Prefix reuse: the committed state after the last request's prompt and after its reply are
kept, and a prompt that extends either resumes from it; rows never depend on their chunk-mates, so a resumed
prompt ends in the same state as a fresh one.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any, Callable


def _words(value: int) -> list[int]:
    """A 64-bit unsigned value as four 16-bit words (the rank share carries int32 values)."""

    return [(value >> (16 * i)) & 0xFFFF for i in range(4)]


def _value(words) -> int:
    return sum(int(w) << (16 * i) for i, w in enumerate(words))


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _float(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


class Qwen27Engine:
    """Qwen3.8-27B on one GPU or two ranks (rank 0 here), DFlash2 drafting, prefix reuse."""

    def __init__(self, model_dir: Path, draft_dir: Path | None, *, max_rows: int = 12, tp: int = 1,
                 rank: int = 0, master: str = "", port: int = 29551, split_head: bool = False,
                 tp_draft: bool = False, allow_copy: bool = True):
        import torch

        from .weights import load

        self.torch = torch
        self.tp, self.rank, self.max_rows, self.allow_copy = tp, rank, max_rows, allow_copy
        torch.cuda.set_device(0)
        if tp == 2:
            import torch.distributed as dist

            from .distributed import split_weights

            dist.init_process_group("nccl", init_method=f"tcp://{master}:{port}", rank=rank, world_size=2)
            # both ranks must run the same calls: refuse to start when they were given different settings
            flags = torch.tensor([int(draft_dir is not None and tp_draft), max_rows, int(split_head), int(allow_copy)],
                                 dtype=torch.int64, device="cuda")
            both = torch.empty((2, flags.numel()), dtype=torch.int64, device="cuda")
            dist.all_gather_into_tensor(both, flags)
            if not torch.equal(both[0], both[1]):
                raise RuntimeError("the two ranks were started with different settings (two-rank drafter, rows, "
                                   f"head split, copies): rank 0 {both[0].tolist()}, rank 1 {both[1].tolist()}; "
                                   "pull the draft model on both machines, or pass --no-drafts to both")
            full = load(model_dir)
            self.w = split_weights(full, rank, tiled=True, split_head=split_head)
        else:
            full = load(model_dir, tiled=True)
            self.w = full
        self.draft = None
        if draft_dir is not None and (rank == 0 or (tp == 2 and tp_draft)):
            from .dflash2 import DFlash2

            # tp_draft: both ranks hold half the drafter and draft together (rank 1 needs --draft too)
            self.draft = DFlash2(draft_dir, full, rank=rank, world=2) if tp == 2 and tp_draft else DFlash2(draft_dir, full)
        del full
        torch.cuda.empty_cache()
        self.eos = tuple(self.w.config.eos)
        self.cache: list[tuple[list[int], Any, Any]] = []   # (committed ids, state, drafter snapshot)

    def _resume(self, prompt: list[int]):
        best = None
        for ids, st, snap in self.cache:
            if len(ids) < len(prompt) and prompt[:len(ids)] == ids and (best is None or len(ids) > len(best[0])):
                best = (ids, st, snap)
        if best is not None:
            self._drop_extensions(best[0])
        return best

    def _drop_extensions(self, ids: list[int]) -> None:
        """Forget cached entries that extend ``ids``, before resuming from it.

        A decode clones its start state and writes KV rows past the clone's length into the
        buffers they share, so entries that share buffers always form one chain of prefixes. The
        entries longer than the one resumed from are exactly its extensions, and the resumed decode
        overwrites their rows. A fresh prefill allocates new buffers and drops nothing.
        """

        n = len(ids)
        self.cache = [c for c in self.cache if len(c[0]) <= n or c[0][:n] != ids]

    def _remember(self, ids: list[int], st, snap) -> None:
        self.cache = [c for c in self.cache if c[0] != ids][-1:] + [(ids, st, snap)]

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True):
        """``draft=False``: one token a round with no draft model and no copies, the serial reference. It runs
        from a fresh prefill and leaves the prefix cache alone."""

        from .decode import draft_decode, prefill

        t0 = time.perf_counter()
        hit = self._resume(prompt) if draft else None
        if self.tp == 2:
            return self._generate_tp(prompt, max_tokens, sampling, on_tokens, hit, t0, draft)
        drafter = self.draft if draft else None
        if hit is not None and drafter is not None:
            drafter.restore(hit[2])
        elif drafter is not None:
            drafter.restore(([None] * drafter.layers, [None] * drafter.layers, 0, 0))
        st, pending = prefill(self.w, prompt, sampling, drafter, state=hit[1] if hit else None)
        if draft:
            self._remember(list(prompt), st, drafter.snapshot() if drafter else None)
        prefill_s = time.perf_counter() - t0
        if on_tokens([pending]):
            return {"prefill_s": prefill_s, "cached": hit[1].pos if hit else 0}
        result = draft_decode(self.w, st, prompt, pending, max_tokens, sampling, drafter,
                              max_rows=self.max_rows, allow_copy=self.allow_copy and draft, on_tokens=on_tokens)
        if draft:
            self._remember(list(prompt) + result.tokens[:-1], result.state, drafter.snapshot() if drafter else None)
        return {"prefill_s": prefill_s, "decode_s": result.seconds, "rounds": result.rounds,
                "cached": hit[1].pos if hit else 0, "drafts": draft}

    # two ranks: rank 0 sends each request's header and prompt to rank 1, both run the same calls
    def _generate_tp(self, prompt, max_tokens, sampling, on_tokens, hit, t0, draft):
        from .decode_tp import _share, decode_tp, prefill_tp

        dev = self.w.norm.device
        cached = hit[1].pos if hit else 0
        # the seed and the float settings cross as their exact bits, so both ranks hold the same Sampling
        header = [1, max_tokens, cached, int(draft), int(sampling is not None), sampling.top_k if sampling else 0,
                  *_words(sampling.seed if sampling else 0), *_words(_bits(sampling.temperature if sampling else 0)),
                  *_words(_bits(sampling.top_p if sampling else 1))]
        _share(header, 0, dev)
        _share(prompt, 0, dev)
        drafter = self.draft if draft else None
        if hit is not None and drafter is not None:
            drafter.restore(hit[2])
        elif drafter is not None:
            drafter.restore(([None] * drafter.layers, [None] * drafter.layers, 0, 0))
        st, pending = prefill_tp(self.w, prompt, sampling, 0, drafter, state=hit[1] if hit else None)
        if draft:
            self._remember(list(prompt), st, drafter.snapshot() if drafter else None)
        prefill_s = time.perf_counter() - t0
        stop_now = bool(on_tokens([pending]))
        result = decode_tp(self.w, st, prompt, pending, 1 if stop_now else max_tokens, sampling, 0, drafter,
                           max_rows=self.max_rows, allow_copy=self.allow_copy and draft, on_tokens=on_tokens)
        if draft:
            self._remember(list(prompt) + result.tokens[:-1], result.state, drafter.snapshot() if drafter else None)
        return {"prefill_s": prefill_s, "decode_s": result.seconds, "rounds": result.rounds, "cached": cached,
                "drafts": draft}

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling
        from .decode_tp import _share, decode_tp, prefill_tp

        dev = self.w.norm.device
        while True:
            header = _share(None, 1, dev)
            _, max_tokens, cached, draft, sampled, top_k = header[:6]
            seed, temp, top_p = _value(header[6:10]), _float(_value(header[10:14])), _float(_value(header[14:18]))
            prompt = _share(None, 1, dev)
            drafter = self.draft if draft else None
            sampling = Sampling(seed, temp, top_k, top_p) if sampled else None
            hit = next(((ids, st, snap) for ids, st, snap in self.cache if len(ids) == cached and prompt[:cached] == ids),
                       None) if cached else None
            if cached and hit is None:
                raise RuntimeError(f"rank 1 has no cached state for the {cached} tokens rank 0 resumes from")
            if hit is not None:
                self._drop_extensions(hit[0])
            if drafter is not None:             # a two-rank drafter: mirror rank 0's drafter state
                drafter.restore(hit[2] if hit is not None else
                                ([None] * drafter.layers, [None] * drafter.layers, 0, 0))
            st, pending = prefill_tp(self.w, prompt, sampling, 1, drafter, state=hit[1] if hit else None)
            if draft:
                self._remember(list(prompt), st, drafter.snapshot() if drafter else None)
            result = decode_tp(self.w, st, prompt, pending, max_tokens, sampling, 1, drafter, max_rows=self.max_rows)
            if draft:
                self._remember(list(prompt) + result.tokens[:-1], result.state, drafter.snapshot() if drafter else None)
