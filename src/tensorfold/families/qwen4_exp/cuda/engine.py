"""The Flash Next CUDA engine: MTP chains verified exactly on one GPU or two ranks in lockstep."""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from tensorfold.cuda import prompt_precision
from . import CONFIDENCE, DEPTH

MAX_DEPTH = 15           # a verify window of at most 16 rows
KEEP_SERIAL = 4          # prompt states the serial engine keeps (they share its attention rows)
KEEP = 8                 # prompt states (one token before each end) a concurrent decoder keeps to resume from
# Reserve bounded tower workspace separately from its weights; override for measured deployments.
VISION_WORKSPACE = 4 * 2**30


def vision_workspace() -> int:
    import os

    value = os.environ.get("TENSORFOLD_VISION_WORKSPACE_MIB")
    if value is None or value == "":
        return VISION_WORKSPACE
    if not value.isdecimal() or int(value) > 16384:
        raise ValueError(f"TENSORFOLD_VISION_WORKSPACE_MIB: 0 to 16,384 MiB, not {value!r}")
    return int(value) * 2**20


class FlashNextEngine:
    """``eos``, ``generate`` (rank 0 or one GPU) and ``follow`` (rank 1), as ``tensorfold.cuda.server`` expects."""

    def __init__(self, model_dir: Path, *, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 draft_vocab: str | int | None = "default", max_len: int | None = None,
                 context_explicit: bool | None = None, tp: int = 1, rank: int = 0, master: str = "", port: int = 29551,
                 prefetch: bool = True, graphs: bool = True, streams: int = 1, ple_on_ssd: bool = False,
                 kv_dtype: str = "bf16", share: float = 0.0, vision: bool = False, vision_urls: bool = False) -> None:
        import torch

        from .exl3_pack import admission, extra_files, is_exl3

        from tensorfold.families import quant_method, read_config

        exl3 = is_exl3(model_dir)
        if (exl3 or quant_method(read_config(model_dir)) == "modelopt") and tp != 1:
            raise ValueError(f"{'EXL3 packs' if exl3 else 'NVFP4 checkpoints'} of Flash Next run on one GPU: drop --tp "
                             "2, or serve the MLX checkpoint (Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) on two")
        if vision and (streams < 2 or tp != 1):
            raise ValueError("image input on Flash Next runs on one GPU with --parallel 2 or more")
        if exl3 and ple_on_ssd:
            raise ValueError("--ple-on-ssd reads the MLX checkpoint's n-gram tables; an EXL3 pack maps its own table "
                             "from its file, so drop --ple-on-ssd")
        from .decode import Engine
        from .kvcache import BITS_OF, check as check_kv
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
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"MTP draft confidence: a probability from 0 to 1, not {confidence}")
        torch.cuda.set_device(0)
        self.tp, self.rank, self.depth, self.confidence = tp, rank, int(depth), float(confidence)
        self.kv_dtype = check_kv(kv_dtype)
        self.comm = None
        self.vision = None                   # the image tower (``QwenCudaVision``) with --vision
        ids = draft_token_ids(draft_vocab) if self.depth > 0 else None
        if tp == 2:
            from tensorfold.cuda.comm import NCCL

            if not master:
                raise ValueError("two ranks need rank 0's address (master)")
            self.comm = NCCL(rank, 2, master, port)
            self.comm.barrier()
        gather = (lambda values: gather_ints(torch, self.comm.all_gather, values)) if tp == 2 else None
        each, mtp, bits = self.depth + 1, self.depth > 0, BITS_OF[self.kv_dtype]
        # one admission for one stream or many (every slot, the shared rows and kept snapshots), before any load
        geometry = ((lambda text: indexed_stream_geometry(text, streams, each, KEEP, mtp=mtp, kv_bits=bits))
                    if streams > 1 else
                    (lambda text: gdn_geometry(text, tp, each, indexed=True, mtp=mtp, kv_bits=bits,
                                               kept=KEEP_SERIAL + 1)))
        if exl3:
            geometry = admission(geometry)
        from tensorfold.vision.qwen_cuda import capacity_geometry, weight_transform as vision_weights

        workspace = vision_workspace() if vision else 0
        self.capacity_plan = admit(model_dir, max_len, context_explicit, torch,
                                   capacity_geometry(geometry, model_dir, vision, rank, workspace),
                                   vision_weights(indexed_weights(tp, mtp, mapped_tables=not ple_on_ssd), vision, rank),
                                   rank=rank, world=tp,
                                   gather=gather, extra_files=extra_files(model_dir) if exl3 else ())
        self.max_len = self.capacity_plan["cache_slots"]
        if tp == 2:
            self._same_settings(torch, ids)
        from concurrent.futures import wait

        from tensorfold.cuda.direct_read import wait_all

        reads: list = []                              # the n-gram tables' pages, read while the weights load
        try:
            w = load(model_dir, mtp=self.depth > 0, tp=(rank, 2) if tp == 2 else None,
                     draft_vocab=draft_vocab if self.depth > 0 else None, ple_on_ssd=ple_on_ssd,
                     table_reads=reads if prefetch and not ple_on_ssd else None)
        except BaseException:
            wait(reads)                               # a failed load leaves no table read behind it
            raise
        tables_read = bool(reads)
        waited = time.perf_counter()
        wait_all(reads)                               # raises a table read's error
        waited = time.perf_counter() - waited
        w.comm = self.comm
        if self.comm is not None:
            self.comm.ready("loading")               # a peer stuck loading is named, not waited on in NCCL
        if self.depth > 0 and w.mtp is None:
            raise ValueError("this checkpoint has no MTP head, which Flash Next's CUDA engine drafts with: use one "
                             "that has it, or --no-drafts for the serial reference (one token a round)")
        self.w = w
        from tensorfold.cuda.markers import resume_points

        self.points = resume_points(model_dir)          # a prompt's message starts to keep states at, or None
        if vision:
            from tensorfold.vision.qwen_cuda import QwenCudaVision

            self.vision = QwenCudaVision(model_dir, torch.device("cuda", 0),
                                         allow_urls=vision_urls)
            torch.cuda.empty_cache()
            print(f"[tensorfold] vision: image input, a "
                  f"{self.vision.weight_bytes / 2**30:.2f} GiB tower with {vision_workspace() / 2**30:.2f} GiB of "
                  f"workspace reserved{'; https URLs allowed' if vision_urls else ''}", flush=True)
        # ``streams`` > 1: up to that many requests decoded together, every stream's chain in one forward
        self.concurrent = streams > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder

            self.e = None
            self.multi = MultiDecoder(w, slots=streams, capacity=self.max_len, depth=self.depth,
                                      confidence=self.confidence, keep=KEEP, points=self.points,
                                      kv_dtype=self.kv_dtype, share=share, vision=self.vision)
            self.scheduler = Scheduler(self.multi, max_streams=streams)
        else:
            self.e = Engine(w, capacity=self.max_len, max_rows=max(8, self.depth + 1), graphs=graphs,
                            kv_dtype=self.kv_dtype)
        started = time.perf_counter()
        locked = False
        if prefetch and not ple_on_ssd:               # the n-gram tables' pages, read now rather than by requests
            tables = {id(layer.ple.table): layer.ple.table for layer in w.layers if layer.ple is not None}
            size = sum(t.nbytes for t in tables.values())
            # pinned pages are no longer reclaimable: lock only what the startup budget leaves room for
            room = self.capacity_plan["budget_bytes"] - self.capacity_plan["total_bytes_estimate"]
            for table in tables.values():
                if not tables_read:
                    table.prefetch()                  # eight readers first: mlock alone faults the pages in one by one
                locked = room >= size and table.lock()
        read_s = time.perf_counter() - started
        captured = self.e.graphs.warm(self.depth + 1) if self.e is not None and self.e.graphs is not None else 0
        started = time.perf_counter()
        if self.concurrent:
            self.multi.warm()
        else:
            from .decode import warm

            warm(self.e)
        if self.vision is not None:
            self.vision.warm()
            torch.cuda.empty_cache()
        warm_s = time.perf_counter() - started
        self.eos = tuple(w.cfg.eos)
        self.model_dir = Path(model_dir)
        self.served = 0
        self.cache: list[tuple[list[int], dict]] = []    # (committed ids, what resuming from them needs)
        self.serial = None                                # the serial requests' engine, made on first use
        rule = (f"1 to {self.depth} MTP drafts a round, a chain stops before a later draft under "
                f"{self.confidence:.0%}" if self.depth else "no drafts: the serial reference, one token a round")
        where = (f"up to {streams} streams, each growing to {self.context_window} prompt/reply tokens while memory "
                 f"lasts ({self.multi.memory_gate.room / 2**30:.1f} GiB free for their caches, "
                 f"{self.multi.window_bytes / 2**30:.2f} GiB for one at the full window), eager" if self.concurrent else
                 f"{self.context_window}-token prompt/reply window; {self.max_len}-token cache")
        if ple_on_ssd:
            how = "read from SSD at each lookup"
        elif tables_read:                             # read during the load: the wait after it, then any lock
            how = f"read alongside the weights ({waited:.1f}s after them)" + (
                f", locked in memory in {read_s:.1f}s" if locked else "")
        else:
            how = f"{'locked in memory' if locked else 'read'} in {read_s:.1f}s"
        kv = "" if self.kv_dtype == "bf16" else f"; {self.kv_dtype} KV cache (fp16 scale per 32 values)"
        print(f"[tensorfold] Flash Next on CUDA: {rule}; {where}{kv}; n-gram tables {how}; {captured} "
              f"decode graphs captured; prompt kernels warmed in {warm_s:.1f}s", flush=True)

    def _same_settings(self, torch, ids) -> None:
        """Both ranks must decode with the same rule, context, draft vocabulary and KV cache, or they would fall out of step: refuse to start otherwise."""

        from .kvcache import BITS_OF

        total = int(ids.sum()) if ids is not None else -1
        mine = torch.tensor([self.depth, round(self.confidence * 1e6), self.max_len,
                             len(ids) if ids is not None else -1, total, BITS_OF[self.kv_dtype],
                             int(prompt_precision.fp8())], dtype=torch.int64, device="cuda")
        both = torch.empty((2 * mine.numel(),), dtype=torch.int64, device="cuda")
        self.comm.all_gather(mine, both)
        both = both.view(2, -1).cpu()
        prompt_precision.same_on_ranks(int(both[0, -1]), int(both[1, -1]))
        if not torch.equal(both[0], both[1]):
            raise RuntimeError(f"the two ranks were started with different settings (drafts, confidence, context, "
                               f"draft vocabulary, KV cache): rank 0 {both[0].tolist()}, rank 1 {both[1].tolist()}")

    def _key(self, n: int) -> str:
        return f"tensorfold/flashnext/request/{n}"

    def shutdown(self) -> None:
        """Rank 0: tell rank 1 to leave ``follow``."""

        if self.tp == 2 and self.rank == 0:
            self.comm.store.set(self._key(self.served), json.dumps({"stop": True}))

    def _share(self, prompt: list[int], max_tokens: int, sampling, draft: bool, cached: int, constraint=None,
               stop_eos: bool = True) -> tuple:
        from tensorfold.engine.grammar import pack

        points = getattr(self, "points", None)
        body = {"prompt": prompt, "max_tokens": max_tokens, "draft": bool(draft), "cached": int(cached),
                "points": list(points(prompt)) if draft and points is not None else [],
                "stop_eos": bool(stop_eos),
                "sampling": None if sampling is None else [int(sampling.seed), float(sampling.temperature),
                                                           int(sampling.top_k), float(sampling.top_p),
                                                           float(sampling.min_p)],
                "grammar": pack(constraint)}                 # rank 1 walks and masks the same rows
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
        return (body["prompt"], body["max_tokens"], None if s is None else Sampling(*s),
                body["draft"], body["cached"], body.get("grammar") or [], body.get("stop_eos", True),
                body.get("points", []))

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
            self.cache = [c for c in self.cache if len(c[0]) < n and hit[0][:len(c[0])] == c[0]] + [hit]

    def _remember(self, ids: list[int], snap: dict) -> None:
        self.cache = [c for c in self.cache if c[0] != ids][-(KEEP_SERIAL - 1):] + [(ids, snap)]

    @property
    def supports_logprobs(self) -> bool:
        return self.tp == 1

    def _serial(self, prompt: list[int], max_tokens: int, sampling, on_tokens, constraint=None,
                stop_eos: bool = True, probabilities=None) -> dict[str, Any]:
        """One token a round from a fresh prefill in the serial engine's own state (no drafts, no kept states)."""

        import torch

        from .decode import prefill, serial_decode

        if self.serial is None:
            self.serial = self.e.twin()
        t0 = time.perf_counter()
        first = prefill(self.serial, prompt, sampling, mtp=False, constraint=constraint, probabilities=probabilities)
        torch.cuda.synchronize()
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": 0, "drafts": False}
        if (on_tokens is not None and on_tokens([first])) or (stop_eos and first in self.eos) or max_tokens <= 1:
            return stats
        res = serial_decode(self.serial, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                            constraint=constraint, probabilities=probabilities)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def _decode(self, prompt: list[int], max_tokens: int, sampling, on_tokens, hit, constraint=None,
                stop_eos: bool = True, probabilities=None, points=None) -> dict[str, Any]:
        import torch

        from .decode import entry_end, mtp_decode, prefill, serial_decode

        t0 = time.perf_counter()
        self._start_from(hit)
        from tensorfold.cuda.markers import MIN_GAP

        end, cached = entry_end(prompt), len(hit[0]) if hit else 0
        markers = getattr(self, "points", None)
        if points is None:
            points = markers(prompt) if markers is not None else []
        stops = [p for p in points if cached + MIN_GAP <= p < end]
        def keep(p, snap, tail):
            self._remember(list(prompt[:p]), {"state": snap, "tail": tail})
        first = prefill(self.e, prompt, sampling, resume=hit[1] if hit else None, constraint=constraint,
                        probabilities=probabilities, keep_at=end, stops=stops, keep=keep)
        # the state one token before the prompt's end, so the same prompt or a next turn resumes from it
        self._remember(list(prompt[:end]), self.e.kept)
        torch.cuda.synchronize()
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": len(hit[0]) if hit else 0,
                                 "drafts": True}
        if (on_tokens is not None and on_tokens([first])) or (stop_eos and first in self.eos) or max_tokens <= 1:
            return stats
        if self.depth > 0:
            res = mtp_decode(self.e, first, max_tokens, sampling, depth=self.depth, confidence=self.confidence,
                             stop_eos=stop_eos, on_tokens=on_tokens, constraint=constraint, probabilities=probabilities)
            stats.update(drafted=res.drafted, accepted=res.accepted, min_rows=min(res.widths, default=0))
        else:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                                constraint=constraint, probabilities=probabilities)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def generate(self, prompt: list[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None], draft: bool = True, constraint=None,
                 stop_eos: bool = True, background: bool = False, probabilities=None, *, vision=None) -> dict[str, Any]:
        """``draft=False``: one token a round, no MTP drafts; ``background``: last, yielding lanes to waiting ones."""

        max_tokens = self._limit(prompt, max_tokens)
        if probabilities is not None and not self.supports_logprobs:
            raise ValueError("logprobs are supported on one GPU only")
        if probabilities is not None and constraint is not None:
            raise ValueError("logprobs do not support structured output")
        if vision is not None and self.vision is None:
            raise ValueError("image inputs require starting this server with --vision")
        if vision is not None and background:
            raise ValueError("image requests cannot yield a background lane")
        if self.scheduler is not None:
            grammar = {} if constraint is None else {"constraint": constraint}
            return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                         **grammar, **({"background": True} if background else {}),
                                         probabilities=probabilities,
                                         **({"vision": vision} if vision is not None else {}))
        hit = self._resume(prompt) if draft else None
        points = None
        if self.tp == 2:                     # rank 0 decodes exactly what it hands rank 1
            prompt, max_tokens, sampling, draft, _, _, stop_eos, points = self._share(
                prompt, max_tokens, sampling, draft, len(hit[0]) if hit else 0, constraint, stop_eos)
            self.served += 1
            emit = on_tokens
            on_tokens = lambda new: (emit(new), False)[1]       # noqa: E731  both ranks decode to the end
        if not draft:
            return self._serial(prompt, max_tokens, sampling, on_tokens, constraint, stop_eos, probabilities=probabilities)
        return self._decode(prompt, max_tokens, sampling, on_tokens, hit, constraint, stop_eos,
                            probabilities=probabilities, points=points)

    def follow(self) -> None:
        """Rank 1: decode every request rank 0 serves, until rank 0 stops."""

        while True:
            request = self._receive()
            if request is None:
                return
            prompt, max_tokens, sampling, draft, cached, packed, stop_eos, points = request
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
                    self._decode(prompt, max_tokens, sampling, None, hit, constraint, stop_eos, points=points)
                else:
                    self._serial(prompt, max_tokens, sampling, None, constraint, stop_eos)
            except ValueError as exc:                       # rank 0 raised at the same point on the same input
                print(f"[tensorfold] request {self.served} failed on both ranks: {exc}", flush=True)
