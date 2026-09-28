"""The Flash Next CUDA engine: MTP chains verified exactly on one GPU or two ranks in lockstep."""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from . import CONFIDENCE, DEPTH

MAX_DEPTH = 15           # a verify window of at most 16 rows
KEEP = 8                 # prompt ends a concurrent decoder keeps to resume from


class FlashNextEngine:
    """``eos``, ``generate`` (rank 0 or one GPU) and ``follow`` (rank 1), as ``tensorfold.cuda.server`` expects."""

    def __init__(self, model_dir: Path, *, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 draft_vocab: str | int | None = "default", max_len: int | None = None,
                 context_explicit: bool | None = None, tp: int = 1, rank: int = 0, master: str = "", port: int = 29551,
                 prefetch: bool = True, graphs: bool = True, streams: int = 1, ple_on_ssd: bool = False) -> None:
        import torch

        from .decode import Engine
        from .weights import draft_token_ids, load
        from tensorfold.cuda.capacity import admit, gather_ints
        from tensorfold.cuda.geometry import gdn_geometry, indexed_stream_geometry, indexed_weights

        if tp not in (1, 2) or rank not in range(tp):
            raise ValueError(f"rank {rank} of {tp}: Flash Next runs on one GPU or two")
        if streams > 1 and tp > 1:
            raise ValueError("--parallel decodes several Flash Next requests together on one GPU; with --tp 2 it "
                             "serves one request at a time for now, so drop --parallel")
        if not 0 <= int(depth) <= MAX_DEPTH:
            raise ValueError(f"MTP drafts a round: 0 to {MAX_DEPTH}, not {depth}")
        torch.cuda.set_device(0)
        self.tp, self.rank, self.depth, self.confidence = tp, rank, int(depth), float(confidence)
        self.comm = None
        ids = draft_token_ids(draft_vocab) if self.depth > 0 else None
        if tp == 2:
            from tensorfold.cuda.comm import NCCL

            if not master:
                raise ValueError("two ranks need rank 0's address (master)")
            self.comm = NCCL(rank, 2, master, port)
            self.comm.barrier()
        gather = (lambda values: gather_ints(torch, self.comm.all_gather, values)) if tp == 2 else None
        each, mtp = self.depth + 1, self.depth > 0
        # one admission for one stream or many (every slot, the shared rows and kept snapshots), before any load
        geometry = ((lambda text: indexed_stream_geometry(text, streams, each, KEEP, mtp=mtp)) if streams > 1 else
                    (lambda text: gdn_geometry(text, tp, each, indexed=True, mtp=mtp)))
        self.capacity_plan = admit(model_dir, max_len, context_explicit, torch, geometry,
                                   indexed_weights(tp, mtp, mapped_tables=not ple_on_ssd), rank=rank, world=tp,
                                   gather=gather)
        self.max_len = self.capacity_plan["cache_slots"]
        if tp == 2:
            self._same_settings(torch, ids)
        w = load(model_dir, mtp=self.depth > 0, tp=(rank, 2) if tp == 2 else None,
                 draft_vocab=draft_vocab if self.depth > 0 else None, ple_on_ssd=ple_on_ssd)
        w.comm = self.comm
        if self.depth > 0 and w.mtp is None:
            raise ValueError("this checkpoint has no MTP head, which Flash Next's CUDA engine drafts with: use one "
                             "that has it, or --no-drafts for the serial reference (one token a round)")
        self.w = w
        # ``streams`` > 1: up to that many requests decoded together, every stream's chain in one forward
        self.concurrent = streams > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder

            self.e = None
            self.multi = MultiDecoder(w, slots=streams, capacity=self.max_len, depth=self.depth,
                                      confidence=self.confidence, keep=KEEP)
            self.scheduler = Scheduler(self.multi, max_streams=streams)
        else:
            self.e = Engine(w, capacity=self.max_len, max_rows=max(8, self.depth + 1), graphs=graphs)
        started = time.perf_counter()
        locked = False
        if prefetch and not ple_on_ssd:               # the n-gram tables' pages, read now rather than by requests
            tables = {id(layer.ple.table): layer.ple.table for layer in w.layers if layer.ple is not None}
            size = sum(a.nbytes for t in tables.values() for a in t.words + t.scales + t.biases)
            # pinned pages are no longer reclaimable: lock only what the startup budget leaves room for
            room = self.capacity_plan["budget_bytes"] - self.capacity_plan["total_bytes_estimate"]
            for table in tables.values():
                locked = room >= size and table.lock()
                if not locked:
                    table.prefetch()
        read_s = time.perf_counter() - started
        captured = self.e.graphs.warm(self.depth + 1) if self.e is not None and self.e.graphs is not None else 0
        self.eos = tuple(w.cfg.eos)
        self.served = 0
        self.cache: list[tuple[list[int], dict]] = []    # (committed ids, what resuming from them needs)
        self.serial = None                                # the serial requests' engine, made on first use
        rule = (f"1 to {self.depth} MTP drafts a round, a chain stops before a later draft under "
                f"{self.confidence:.0%}" if self.depth else "no drafts: the serial reference, one token a round")
        where = (f"{streams} streams of {self.context_window} prompt/reply tokens "
                 f"({self.multi.slot_bytes / 2**20:.0f} MiB a stream), eager" if self.concurrent else
                 f"{self.context_window}-token prompt/reply window; {self.max_len}-token cache")
        how = ("read from SSD at each lookup" if ple_on_ssd else
               f"{'locked in memory' if locked else 'read'} in {read_s:.1f}s")
        print(f"[tensorfold] Flash Next on CUDA: {rule}; {where}; n-gram tables {how}; {captured} "
              "decode graphs captured", flush=True)

    def _same_settings(self, torch, ids) -> None:
        """Both ranks must decode with the same rule, context and draft vocabulary, or they would fall out of step: refuse to start otherwise."""

        total = int(ids.sum()) if ids is not None else -1
        mine = torch.tensor([self.depth, round(self.confidence * 1e6), self.max_len,
                             len(ids) if ids is not None else -1, total], dtype=torch.int64, device="cuda")
        both = torch.empty((2 * mine.numel(),), dtype=torch.int64, device="cuda")
        self.comm.all_gather(mine, both)
        both = both.view(2, -1).cpu()
        if not torch.equal(both[0], both[1]):
            raise RuntimeError(f"the two ranks were started with different settings (drafts, confidence, context, "
                               f"draft vocabulary): rank 0 {both[0].tolist()}, rank 1 {both[1].tolist()}")

    def _key(self, n: int) -> str:
        return f"tensorfold/flashnext/request/{n}"

    def shutdown(self) -> None:
        """Rank 0: tell rank 1 to leave ``follow``."""

        if self.tp == 2 and self.rank == 0:
            self.comm.store.set(self._key(self.served), json.dumps({"stop": True}))

    def _share(self, prompt: list[int], max_tokens: int, sampling, draft: bool, cached: int) -> tuple:
        body = {"prompt": prompt, "max_tokens": max_tokens, "draft": bool(draft), "cached": int(cached),
                "sampling": None if sampling is None else [int(sampling.seed), float(sampling.temperature),
                                                           int(sampling.top_k), float(sampling.top_p)]}
        text = json.dumps(body)
        self.comm.store.set(self._key(self.served), text)
        return self._unpack(text)

    def _receive(self) -> tuple | None:
        from torch.distributed import DistNetworkError

        key = self._key(self.served)
        while True:
            try:
                self.comm.store.wait([key], timedelta(hours=1))
                break
            except DistNetworkError:                        # rank 0 is gone: leave ``follow``
                print("[tensorfold] rank 0 closed the connection; rank 1 stops", flush=True)
                return None
            except Exception:                               # noqa: BLE001  (no request within the hour: wait on)
                continue
        text = self.comm.store.get(key).decode()
        self.comm.store.delete_key(key)
        return self._unpack(text)

    @staticmethod
    def _unpack(text: str) -> tuple | None:
        from tensorfold.engine.exact_sampling import Sampling

        body = json.loads(text)
        if body.get("stop"):
            return None
        s = body["sampling"]
        return (body["prompt"], body["max_tokens"], None if s is None else Sampling(s[0], s[1], s[2], s[3]),
                body["draft"], body["cached"])

    @property
    def context_window(self) -> int:
        """Prompt and reply capacity after reserving speculative scratch positions."""

        return max(0, self.max_len - self.depth - 1)

    def _limit(self, prompt: list[int], max_tokens: int) -> int:
        room = self.max_len - len(prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.max_len}-token context")
        return max(1, min(max_tokens, room))

    def _resume(self, prompt: list[int]):
        """The longest kept state the prompt extends (with at least one new token), or None."""

        best = None
        for ids, snap in self.cache:
            if len(ids) < len(prompt) and prompt[:len(ids)] == ids and (best is None or len(ids) > len(best[0])):
                best = (ids, snap)
        return best

    def _start_from(self, hit) -> None:
        """Before a prefill: resuming overwrites the cache rows past the kept prefix, so the states that extend it go; a fresh prompt overwrites them all."""

        if hit is None:
            self.cache = []
        else:
            n = len(hit[0])
            self.cache = [c for c in self.cache if len(c[0]) <= n or c[0][:n] != hit[0]]

    def _remember(self, ids: list[int], snap: dict) -> None:
        self.cache = [c for c in self.cache if c[0] != ids][-1:] + [(ids, snap)]

    def _serial(self, prompt: list[int], max_tokens: int, sampling, on_tokens) -> dict[str, Any]:
        """One token a round from a fresh prefill in the serial engine's own state (no drafts, no kept states)."""

        import torch

        from .decode import prefill, serial_decode

        if self.serial is None:
            self.serial = self.e.twin()
        t0 = time.perf_counter()
        first = prefill(self.serial, prompt, sampling, mtp=False)
        torch.cuda.synchronize()
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": 0, "drafts": False}
        if (on_tokens is not None and on_tokens([first])) or first in self.eos or max_tokens <= 1:
            return stats
        res = serial_decode(self.serial, first, max_tokens, sampling, stop_eos=True, on_tokens=on_tokens)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def _decode(self, prompt: list[int], max_tokens: int, sampling, on_tokens, hit) -> dict[str, Any]:
        import torch

        from .decode import mtp_decode, prefill, serial_decode

        t0 = time.perf_counter()
        self._start_from(hit)
        first = prefill(self.e, prompt, sampling, resume=hit[1] if hit else None)
        # the prompt's state: the MTP head has absorbed every position but the last, whose streams resume needs
        self._remember(list(prompt), {"state": self.e.st.snapshot(),
                                      "tail": self.e.last_streams.clone() if self.e.mbuf is not None else None})
        torch.cuda.synchronize()
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": len(hit[0]) if hit else 0,
                                 "drafts": True}
        if (on_tokens is not None and on_tokens([first])) or first in self.eos or max_tokens <= 1:
            return stats
        if self.depth > 0:
            res = mtp_decode(self.e, first, max_tokens, sampling, depth=self.depth, confidence=self.confidence,
                             stop_eos=True, on_tokens=on_tokens)
            stats.update(drafted=res.drafted, accepted=res.accepted, min_rows=min(res.widths, default=0))
        else:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=True, on_tokens=on_tokens)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def generate(self, prompt: list[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None], draft: bool = True) -> dict[str, Any]:
        """``draft=False``: one token a round with no MTP drafts, from a fresh prefill that leaves the kept states alone: the serial reference."""

        max_tokens = self._limit(prompt, max_tokens)
        if self.scheduler is not None:
            return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens)
        hit = self._resume(prompt) if draft else None
        if self.tp == 2:                     # rank 0 decodes exactly what it hands rank 1
            prompt, max_tokens, sampling, draft, _ = self._share(prompt, max_tokens, sampling, draft,
                                                                 len(hit[0]) if hit else 0)
            self.served += 1
            emit = on_tokens
            on_tokens = lambda new: (emit(new), False)[1]       # noqa: E731  both ranks decode to the end
        if not draft:
            return self._serial(prompt, max_tokens, sampling, on_tokens)
        return self._decode(prompt, max_tokens, sampling, on_tokens, hit)

    def follow(self) -> None:
        """Rank 1: decode every request rank 0 serves, until rank 0 stops."""

        while True:
            request = self._receive()
            if request is None:
                return
            prompt, max_tokens, sampling, draft, cached = request
            self.served += 1
            hit = None
            if draft and cached:
                hit = next(((ids, snap) for ids, snap in self.cache if len(ids) == cached and prompt[:cached] == ids),
                           None)
                if hit is None:
                    raise RuntimeError(f"rank 1 has no kept state for the {cached} tokens rank 0 resumes from")
            try:
                if draft:
                    self._decode(prompt, max_tokens, sampling, None, hit)
                else:
                    self._serial(prompt, max_tokens, sampling, None)
            except ValueError as exc:                       # rank 0 raised at the same point on the same input
                print(f"[tensorfold] request {self.served} failed on both ranks: {exc}", flush=True)
