"""The Nemotron CUDA engine: MTP chains verified exactly; ``draft=False`` runs a serial twin engine."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from tensorfold.cuda import prompt_precision
from . import CONFIDENCE, DRAFTS


class NemotronEngine:
    """``eos``, ``generate`` (rank 0 or one GPU), ``follow`` (rank 1) and ``shutdown``, as the server expects."""

    def __init__(self, model_dir: Path, *, drafts: int = DRAFTS, confidence: float = CONFIDENCE,
                 draft_ids: str | list[int] | None = "default", context: int | None = None,
                 context_explicit: bool | None = None, tp: int = 1, rank: int = 0, master: str = "",
                 port: int = 29571) -> None:
        """``draft_ids``: "default" (the family's ``draft_ids.txt``, the Mac engine's list), a list, or None (all)."""
        import torch

        from tensorfold.cuda.capacity import admit, gather_ints
        from tensorfold.cuda.geometry import hybrid_geometry, hybrid_weights

        from .attention import CHUNK
        from .engine import ROWS, Engine
        from .mtp import MTPHead
        from .weights import MTP_FILE, load

        if tp not in (1, 2) or rank not in range(tp):
            raise ValueError(f"rank {rank} of {tp}: Nemotron runs on one GPU or two")
        if not 0 <= int(drafts) <= 8:
            raise ValueError(f"MTP drafts a round: 0 to 8, not {drafts}")
        torch.cuda.set_device(0)
        if draft_ids == "default":
            draft_ids = [int(v) for v in (Path(__file__).parent.parent / "draft_ids.txt").read_text().split()]
        vocab = int(json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"])
        if draft_ids is not None and (len(set(draft_ids)) != len(draft_ids) or
                                      not all(0 <= int(t) < vocab for t in draft_ids)):
            raise ValueError(f"draft ids must be distinct token ids below the vocabulary's {vocab}")
        self.tp, self.rank, self.drafts, self.confidence = tp, rank, int(drafts), float(confidence)
        self.comm = None
        if tp == 2:
            from tensorfold.cuda.comm import NCCL

            if not master:
                raise ValueError("two ranks need rank 0's address (master)")
            self.comm = NCCL(rank, 2, master, port)
            self.comm.barrier()
        gather = (lambda values: gather_ints(torch, self.comm.all_gather, values)) if tp == 2 else None
        head = Path(model_dir) / MTP_FILE
        # the engine, its serial twin, the MTP head and the kept prompt ends, on every rank before any weight loads
        self.capacity_plan = admit(model_dir, context, context_explicit, torch,
                                   lambda text: hybrid_geometry(text, tp, ROWS, rows=ROWS, chunk=CHUNK,
                                                                drafts=self.drafts > 0, draft=len(draft_ids or [])),
                                   hybrid_weights(tp), rank=rank, world=tp, gather=gather,
                                   startup_copies=2 if tp == 2 else 0,     # the whole model loads before its split
                                   files=sorted(Path(model_dir).glob("model*.safetensors")),      # as ``load`` reads
                                   extra_files=(head,) if self.drafts and head.is_file() else ())
        self.max_len = -(-self.capacity_plan["cache_slots"] // CHUNK) * CHUNK
        if tp == 2:
            self._same_settings(torch, draft_ids)
        w = load(model_dir, mtp=self.drafts > 0)
        if self.comm is not None:
            self.comm.ready("loading")               # a peer stuck loading is named, not waited on in NCCL
        if self.drafts and w.mtp is None:
            raise ValueError("this checkpoint has no MTP head (mtp-4bit.safetensors), which Nemotron's CUDA engine "
                             "drafts with: use one that has it, or --no-drafts for the serial reference")
        self._make = lambda: Engine(w, max_len=self.max_len)                     # noqa: E731
        if tp == 2:
            from .tp import TPEngine, split_weights

            w = split_weights(w, rank)
            torch.cuda.empty_cache()
            self._make = lambda: TPEngine(w, self._gather, max_len=self.max_len)  # noqa: E731
        self.e = self._make()
        self.mtp = MTPHead(self.e, draft_ids=draft_ids, split=tp == 2) if self.drafts else None
        started = time.perf_counter()
        for mode in (_default_sampling(), None):             # the common modes get CUDA graphs
            self.e.capture(range(1, self.e.max_rows + 1), mode)
            if self.mtp is not None:
                self.mtp.capture(range(1, self.e.max_rows + 1), (0, self.drafts))
        self.eos = tuple(self.e.c.eos)
        self.model_dir = Path(model_dir)
        self.served = 0
        self.cache: list[tuple[list[int], dict]] = []       # (committed ids, what resuming from them needs)
        self.serial = None                                    # the serial requests' engine, made on first use
        rule = (f"up to {self.drafts} MTP drafts a round, verified while their running confidence stays at or above "
                f"{self.confidence:.0%}" if self.drafts else "no drafts: the serial reference, one token a round")
        print(f"[tensorfold] Nemotron on CUDA: {rule}; {self.max_len}-token context; graphs captured in "
              f"{time.perf_counter() - started:.1f}s", flush=True)

    def _gather(self, local):
        import torch

        local = local.contiguous()
        out = torch.empty((2 * local.shape[0], *local.shape[1:]), dtype=local.dtype, device=local.device)
        self.comm.all_gather(local.view(-1), out.view(-1))
        return out

    def _same_settings(self, torch, draft_ids) -> None:
        """Both ranks must decode with the same rule, context and draft ids, or they fall out of step."""

        ids = list(draft_ids) if draft_ids is not None else []
        digest = int.from_bytes(hashlib.sha256(" ".join(map(str, ids)).encode()).digest()[:7], "big")   # order too
        mine = torch.tensor([self.drafts, round(self.confidence * 1e6), self.max_len, len(ids), digest,
                             int(prompt_precision.fp8())], dtype=torch.int64, device="cuda")
        both = torch.empty((2 * mine.numel(),), dtype=torch.int64, device="cuda")
        self.comm.all_gather(mine, both)
        both = both.view(2, -1).cpu()
        prompt_precision.same_on_ranks(int(both[0, -1]), int(both[1, -1]))
        if not torch.equal(both[0], both[1]):
            raise RuntimeError(f"the two ranks were started with different settings (drafts, confidence, context, "
                               f"draft ids): rank 0 {both[0].tolist()}, rank 1 {both[1].tolist()}")

    # -- two ranks: rank 0 hands each request to rank 1 ------------------------------------------------------
    def _key(self, n: int) -> str:
        return f"tensorfold/nemotron/request/{n}"

    def shutdown(self) -> None:
        """Rank 0: tell rank 1 to leave ``follow``."""

        if self.tp == 2 and self.rank == 0:
            self.comm.store.set(self._key(self.served), json.dumps({"stop": True}))

    def _share(self, prompt: list[int], max_tokens: int, sampling, draft: bool, cached: int, constraint=None,
               stop_eos: bool = True) -> tuple:
        from tensorfold.engine.grammar import pack

        body = {"prompt": prompt, "max_tokens": max_tokens, "draft": bool(draft), "cached": int(cached),
                "stop_eos": bool(stop_eos),
                "sampling": None if sampling is None else [int(sampling.seed), float(sampling.temperature),
                                                           int(sampling.top_k), float(sampling.top_p),
                                                           float(sampling.min_p)],
                "grammar": pack(constraint)}                 # rank 1 walks and masks the same rows
        text = json.dumps(body)
        self.comm.store.set(self._key(self.served), text)
        return _unpack(text)

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
        return _unpack(text)

    # -- prefix reuse ----------------------------------------------------------------------------------------------
    def _resume(self, prompt: list[int]):
        """The longest kept state the prompt extends (with at least one new token), or None."""

        best = None
        for ids, snap in self.cache:
            if len(ids) < len(prompt) and prompt[:len(ids)] == ids and (best is None or len(ids) > len(best[0])):
                best = (ids, snap)
        return best

    def _remember(self, ids: list[int], snap: dict) -> None:
        self.cache = [c for c in self.cache if c[0] != ids][-1:] + [(ids, snap)]

    # -- decoding ------------------------------------------------------------------------------------------------
    @property
    def context_window(self) -> int:
        """Prompt and reply capacity: the cache rows less one verify window."""

        return max(0, self.max_len - self.e.max_rows)

    def _limit(self, prompt: list[int], max_tokens: int) -> int:
        room = self.max_len - len(prompt) - self.e.max_rows
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.max_len}-token context")
        return max(1, min(max_tokens, room))

    def _serial(self, prompt: list[int], max_tokens: int, sampling, on_tokens, constraint=None,
                stop_eos: bool = True) -> dict[str, Any]:
        """One token a round from a fresh prefill in the serial engine's own state (no drafts, no kept states)."""

        from .decode import prefill, serial_decode

        if self.serial is None:
            self.serial = self._make()
            for mode in (_default_sampling(), None):
                self.serial.capture([1], mode)
        t0 = time.perf_counter()
        pre = prefill(self.serial, None, prompt, sampling, constraint=constraint)
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": 0, "drafts": False}
        if (on_tokens is not None and on_tokens([pre.pending])) or (stop_eos and pre.pending in self.eos) or \
                max_tokens <= 1:
            return stats
        res = serial_decode(self.serial, pre, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                            constraint=constraint)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def _decode(self, prompt: list[int], max_tokens: int, sampling, on_tokens, hit, constraint=None,
                stop_eos: bool = True) -> dict[str, Any]:
        from .decode import draft_decode, prefill, serial_decode

        t0 = time.perf_counter()
        if hit is None:
            self.cache = []
        else:                               # resuming overwrites the cache rows past the kept prefix
            n = len(hit[0])
            self.cache = [c for c in self.cache if len(c[0]) <= n or c[0][:n] != hit[0]]
        resume = None if hit is None else (hit[1]["engine"], hit[1]["mtp"], len(hit[0]), hit[1]["tail"])
        end = max(1, len(prompt) - 1)
        pre = prefill(self.e, self.mtp, prompt, sampling, resume=resume, constraint=constraint, keep_at=end)
        if pre.kept is None:
            raise RuntimeError(f"prefill did not retain the required {end}-token prefix of the {len(prompt)}-token prompt")
        self._remember(list(prompt[:end]), pre.kept)
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": len(hit[0]) if hit else 0,
                                 "drafts": True}
        if (on_tokens is not None and on_tokens([pre.pending])) or (stop_eos and pre.pending in self.eos) or \
                max_tokens <= 1:
            return stats
        if self.mtp is not None:
            res = draft_decode(self.e, self.mtp, pre, max_tokens, sampling, drafts=self.drafts,
                               confidence=self.confidence, stop_eos=stop_eos, on_tokens=on_tokens,
                               constraint=constraint)
            stats.update(drafted=res.drafted, accepted=res.accepted, min_rows=min(res.widths, default=0))
        else:
            res = serial_decode(self.e, pre, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                                constraint=constraint)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def generate(self, prompt: list[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None], draft: bool = True, constraint=None,
                 stop_eos: bool = True) -> dict[str, Any]:
        """``draft=False``: serial one-token rounds from a fresh prefill; ``stop_eos=False``: past end tokens."""

        max_tokens = self._limit(prompt, max_tokens)
        hit = self._resume(prompt) if draft else None
        if self.tp == 2:                     # rank 0 decodes exactly what it hands rank 1
            prompt, max_tokens, sampling, draft, _, _, stop_eos = self._share(
                prompt, max_tokens, sampling, draft, len(hit[0]) if hit else 0, constraint, stop_eos)
            self.served += 1
            emit = on_tokens
            on_tokens = lambda new: (emit(new), False)[1]       # noqa: E731  both ranks decode to the end
        if not draft:
            return self._serial(prompt, max_tokens, sampling, on_tokens, constraint, stop_eos)
        return self._decode(prompt, max_tokens, sampling, on_tokens, hit, constraint, stop_eos)

    def follow(self) -> None:
        """Rank 1: decode every request rank 0 serves, until rank 0 stops."""

        while True:
            request = self._receive()
            if request is None:
                return
            prompt, max_tokens, sampling, draft, cached, packed, stop_eos = request
            constraint = None
            if packed:                                      # the request's grammar, compiled here as on rank 0
                from tensorfold.engine import grammar

                constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
            self.served += 1
            hit = None
            if draft and cached:
                hit = next(((ids, snap) for ids, snap in self.cache if len(ids) == cached and prompt[:cached] == ids),
                           None)
                if hit is None:
                    raise RuntimeError(f"rank 1 has no kept state for the {cached} tokens rank 0 resumes from")
            try:
                if draft:
                    self._decode(prompt, max_tokens, sampling, None, hit, constraint, stop_eos)
                else:
                    self._serial(prompt, max_tokens, sampling, None, constraint, stop_eos)
            except ValueError as exc:                       # rank 0 raised at the same point on the same input
                print(f"[tensorfold] request {self.served} failed on both ranks: {exc}", flush=True)


def _default_sampling():
    from tensorfold.engine.exact_sampling import Sampling

    return Sampling(0, 1.0, 20, 0.95)


def _unpack(text: str) -> tuple | None:
    from tensorfold.engine.exact_sampling import Sampling

    body = json.loads(text)
    if body.get("stop"):
        return None
    s = body["sampling"]
    return (body["prompt"], body["max_tokens"], None if s is None else Sampling(*s), body["draft"],
            body["cached"], body.get("grammar") or [], body.get("stop_eos", True))
