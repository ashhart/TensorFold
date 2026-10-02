"""DeepSeek-V4.1-Flash served from two DGX Sparks: the serial engine behind ``tensorfold serve --tp 2``.

Rank 0 runs the HTTP server and calls ``generate``; rank 1 calls ``follow`` and mirrors every request. Each request's
header (lengths, flags, sampling) and prompt go to rank 1 through the NCCL all-gather; during decoding rank 0's stop
(client gone, stop string) rides on the per-round draft agreement, and every other stop follows from the tokens,
which both ranks compute alike.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from typing import Any

DEFAULT_CONTEXT = 40960          # long-context parity is verified to 40K; --context takes more where memory allows
DRAFTS = 5                       # the checkpoint's DSpark block (dspark_block_size) drafts up to 5 tokens a round
# memory a context token costs: compressed + indexer-key caches (3.2 KB), RoPE tables (0.8 KB), and the prompt
# selection's per-block maxima and flags (512 rows / 8-entry blocks: ~0.3 KB; scores stream in fixed segments)
def _cache_bytes() -> int:
    """A stream's per-token caches: compressed entries + indexer keys at 2.5 entries a token (ratio-2 sources 2, 8, 14
    and ratio-1 source 20): bf16 1024 + 256 B an entry, fp8 604 (448 e4m3 + 64 bf16 RoPE + 7 scales) + 134."""

    from .serial import KV_FP8

    return int(2.5 * ((604 + 134) if KV_FP8 else (1024 + 256)))


TOKEN_BYTES = _cache_bytes() + 768 + 512 * 4 // 8 + 512 // 8
NATIVE_CONTEXT = 1048576         # the model's window (config max_position_embeddings)
FIXED_GIB = float(os.environ.get("TF_DSV41_FIXED_GIB") or "4")      # engine buffers 2.7 + prompt transients 1.0 (measured)
RESERVE_GIB = float(os.environ.get("TF_DSV41_RESERVE_GIB") or "2.5")  # left to the OS (unified memory: an OOM wedges)
PROMPT_TRANSIENT_GIB = 1.5       # a prompt chunk's buffers beyond the context's (expert Z, GEMM workspace, ...)


def available_bytes() -> int:
    """What the system can still give us: MemAvailable (GB10 memory is unified; it counts the reclaimable page cache,
    which right after loading holds the weight files and which CUDA's free figure leaves out), else CUDA's free."""

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import torch

    return torch.cuda.mem_get_info()[0]


SLOT_BYTES = 40 * 256 * 512 * 2 + 3 * 256 * 1024 * 4 + 3 * 256 * 512 * 2   # a stream slot's decode rings (~15 MB)
CACHE_BYTES = _cache_bytes()     # a stream's per-token caches (compressed entries + indexer keys)


def _comp_bytes() -> int:
    """A stream's per-token compressed entries (the part a display carveout can hold)."""

    from .serial import KV_FP8

    return int(2.5 * (604 if KV_FP8 else 1024))


def largest_context(free: int, streams: int = 1, carve: int = 0) -> int:
    """The largest context (a multiple of 1,024) whose caches and prompt buffers fit ``free`` bytes, ``streams``
    stream slots each holding that context; ``carve`` bytes of display carveout take compressed entries first."""

    room = free - int((FIXED_GIB + RESERVE_GIB) * 2 ** 30) - (streams - 1) * SLOT_BYTES
    per = TOKEN_BYTES + (streams - 1) * CACHE_BYTES
    comp = _comp_bytes() * streams
    best = room // per                                   # no carveout help
    if carve:
        full = (room + carve) // per                     # the carveout holds comp up to its size
        if full * comp >= carve:
            best = max(best, full)
        best = max(best, min(room // max(per - comp, 1), carve // comp))   # all comp carved
    return min(NATIVE_CONTEXT, max(0, best // 1024 * 1024))


def _f64_ints(value: float) -> list[int]:
    lo, hi = struct.unpack("<ii", struct.pack("<d", float(value)))
    return [lo, hi]


def _ints_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<ii", int(lo), int(hi)))[0]


class Dsv41Engine:
    """Both ranks build the same engine (weights split by ``split.py``); ``generate`` on rank 0, ``follow`` on rank 1."""

    tp = 2

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, engram: Path, drafts: bool = True,
                 context: int | None = None, context_explicit: bool | None = None, warm: bool = True,
                 parallel: int = 1) -> None:
        import torch

        from tensorfold.cuda.comm import NCCL

        from . import weights as W
        from .serial import Comm, SerialEngine

        if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            # prompt chunks allocate indexer buffers of a new size each chunk: expandable segments return them
            # instead of keeping every size reserved (3.6 -> 1.6 GiB at a 38K prompt)
            if not torch.cuda.is_initialized():
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"    # read at the first allocation
            else:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        torch.cuda.set_device(0)
        self.torch = torch
        self.rank = rank
        self.model_dir = Path(model_dir)
        self.nccl = NCCL(rank, 2, master, port)
        self.nccl.barrier()
        if os.environ.get("TF_COMM", "nccl") == "rdma":            # small all-gathers over our RoCE transport
            from tensorfold.cuda.rdma import RdmaComm

            self.nccl = RdmaComm(self.nccl, rank)
            if rank == 0:
                print(f"[tensorfold] all-gathers up to {self.nccl.rdma.slot_bytes >> 10} KiB over RoCE "
                      f"({self.nccl.rdma.device}), larger ones over NCCL", flush=True)
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        cap = int(context) if context and explicit else DEFAULT_CONTEXT   # the CLI hands the native window otherwise
        if not (Path(engram) / "config.json").exists() and not any(Path(engram).glob("*.safetensors")):
            raise FileNotFoundError(f"no Engram tables in {engram}: put shards 47-48 of deepseek-ai/DeepSeek-V4.1-Flash "
                                    "there (or point TF_DSV41_ENGRAM_DIR at them)")
        self.streams = max(1, int(parallel))
        mine = [cap, int(bool(drafts)), int(explicit), self.streams]
        both = self._gather_ints(mine)
        if both[0] != both[1]:
            raise RuntimeError(f"the two ranks were started with different settings (context, drafts, parallel): "
                               f"rank 0 {both[0]}, rank 1 {both[1]}; give both the same flags")
        started = time.perf_counter()
        w = W.load(self.model_dir, rank=rank, log=lambda *a, **k: None, draft=bool(drafts))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()                     # the load's staging buffers back before memory is measured
        # admission, before any cache exists: both ranks' memory decides (a GB10 out of memory can wedge the node)
        free = available_bytes()
        if rank == 0:
            print(f"[tensorfold] memory left after the weights: {free / 2 ** 30:.1f} GiB", flush=True)
        from tensorfold.cuda import carveout

        carve = carveout.requested_bytes() if carveout.enabled() else 0
        largest = min(row[0] for row in self._gather_ints([largest_context(free, self.streams, carve)]))
        self.capacity_plan = {"largest_window": largest, "context_window": cap}
        if cap > largest:
            if explicit:
                raise ValueError(f"--context {cap} does not fit: after the weights, this memory admits up to {largest} "
                                 f"tokens on both ranks (TF_DSV41_RESERVE_GIB={RESERVE_GIB:g} kept for the system)")
            if rank == 0:
                print(f"[tensorfold] context {largest} (the default {cap} does not fit the memory left)", flush=True)
            cap = largest
        if cap < 4096:
            raise ValueError(f"only {largest} tokens of context fit after the weights: free memory first")
        self.capacity_plan["context_window"] = cap
        self.e = SerialEngine(w, Comm(self.nccl), str(engram), str(self.model_dir / "tokenizer.json"), cap=cap,
                              slots=self.streams)
        self.nccl.barrier()
        with torch.no_grad():
            if drafts:
                self.e.enable_dspark(DRAFTS)
            self.e.capture(1)
            # verify windows of one stream (1 + drafts rows); with --parallel, any round of up to 16 rows
            from .serial import PROMPT_ROWS

            top = PROMPT_ROWS if self.streams > 1 else (DRAFTS + 1 if drafts else 1)
            if drafts and self.streams == 1 and os.environ.get("TF_COPY_DRAFTS", "1") != "0":   # longer copy windows
                top = max(top, min(PROMPT_ROWS, int(os.environ.get("TF_COPY_MAX") or 15) + 1))
            for rows in range(2, top + 1):
                self.e.capture(rows)
            if drafts:
                self.e.drafter.capture()
                if self.streams > 1:                      # one drafting pass for several streams
                    from .serial import PROMPT_ROWS

                    self.e.drafter.capture_multi(min(self.streams, PROMPT_ROWS // DRAFTS))
        if hasattr(self.nccl, "settle"):                  # graphs captured: RoCE gathers go without a barrier first
            self.nccl.settle()
        self.limit = cap
        self.eos = (int(w.cfg.eos_token_id),)
        self.drafts = bool(drafts)
        if rank == 0 and getattr(self.e, "carved", 0):
            print(f"[tensorfold] compressed-KV pools in the display carveout: {self.e.carved / 2 ** 30:.2f} GiB",
                  flush=True)
        if rank == 0:
            print(f"[tensorfold] memory left after capture: {available_bytes() / 2 ** 30:.1f} GiB "
                  f"(largest admissible context {largest})", flush=True)
            print(f"[tensorfold] DeepSeek-V4.1-Flash ready in {time.perf_counter() - started:.0f}s: context {cap}, "
                  f"{'DSpark drafts up to ' + str(DRAFTS) if drafts else 'serial decoding'}, "
                  f"{torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB a rank", flush=True)
        if warm and os.environ.get("TF_DSV41_WARM", "1") != "0":
            self._warm()
        self._make_pool(cap)
        self.concurrent = self.streams > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder

            self.multi = MultiDecoder(self.e, self._share, rank=rank, drafts=DRAFTS if drafts else 0)
            self.multi.model_dir = self.model_dir
            self.multi.calibrate(self._gather_ints)
            if rank == 0:
                curve = " ".join(f"{v:.0f}" for v in self.multi.costs)
                print(f"[tensorfold] verify ms by rows 1..{len(self.multi.costs)}: {curve}; a draft "
                      f"{self.multi.draft_ms:.1f} ms", flush=True)
            if rank == 0:
                self.scheduler = Scheduler(self.multi, max_streams=self.streams)
                print(f"[tensorfold] {self.streams} concurrent streams of up to {cap} tokens each", flush=True)

    def _make_pool(self, cap: int) -> None:
        """Kept prompt states (``tensorfold.cuda.kv_pool``) in what the context's prompt buffers and the reserve
        leave; both ranks keep the smaller budget, so their pools stay identical."""

        from tensorfold.cuda.kv_pool import PrefixPool

        transient = int(PROMPT_TRANSIENT_GIB * 2 ** 30) + (TOKEN_BYTES - 3200 - 768) * cap
        mine = max(0, available_bytes() - int(RESERVE_GIB * 2 ** 30) - transient)
        asked = os.environ.get("TF_DSV41_POOL_GIB")
        if asked:
            mine = min(mine, int(float(asked) * 2 ** 30))
        budget = min(row[0] for row in self._gather_ints([mine]))
        self.e.pool = PrefixPool(budget, min_tokens=int(os.environ.get("TF_DSV41_POOL_MIN", "1024")))
        if self.rank == 0:
            per_token = CACHE_BYTES
            print(f"[tensorfold] kept prompt states: {budget / 2 ** 30:.1f} GiB (~{budget // per_token:,} tokens of "
                  f"conversation prefixes)", flush=True)

    # -- rank agreement -------------------------------------------------------------------------------------------
    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int64, device="cuda")
        got = torch.empty((2 * len(values),), dtype=torch.int64, device="cuda")
        self.nccl.all_gather(mine, got)
        return [got[:len(values)].tolist(), got[len(values):].tolist()]

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (its length first, then the values)."""

        count = self._gather_ints([len(values) if self.rank == 0 else 0])[0][0]
        if count == 0:
            return []
        torch = self.torch
        mine = (torch.tensor(values, dtype=torch.int64, device="cuda") if self.rank == 0
                else torch.zeros((count,), dtype=torch.int64, device="cuda"))
        got = torch.empty((2 * count,), dtype=torch.int64, device="cuda")
        self.nccl.all_gather(mine, got)
        return got[:count].tolist()

    # -- requests ------------------------------------------------------------------------------------------------
    def _warm(self) -> None:
        """Both ranks, before serving: a prompt of two chunks and a few rounds (Triton compiles, Engram pages)."""

        import random

        rng = random.Random(0)
        prompt = [rng.randrange(1000, 100000) for _ in range(2100)]
        t = time.perf_counter()
        for draft in ((True, False) if self.drafts else (False,)):
            self._run(prompt, 8, None, draft, True, None)
        torch = self.torch
        reserved = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()                     # the warm-up's prompt buffers back to the system
        if self.rank == 0:
            print(f"[tensorfold] warmed in {time.perf_counter() - t:.0f}s; GPU memory {torch.cuda.memory_allocated() / 2 ** 30:.1f} "
                  f"GiB in use, {reserved / 2 ** 30:.1f} GiB was reserved; {available_bytes() / 2 ** 30:.1f} GiB left",
                  flush=True)

    def _run(self, prompt: list[int], max_tokens: int, sampling, draft: bool, stop_eos: bool, on_tokens,
             constraint=None) -> dict:
        try:
            with self.torch.no_grad():
                return self.e.generate(list(prompt), max_tokens, sampling=sampling, on_tokens=on_tokens,
                                       draft=draft and self.drafts, stop_eos=stop_eos, constraint=constraint)
        finally:
            if len(prompt) > 4096:                   # a long prompt's transient buffers back to the system
                self.torch.cuda.empty_cache()

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 stop_eos: bool = True, constraint=None, background: bool = False) -> dict[str, Any]:
        """Rank 0: one reply, mirrored on rank 1; ``draft=False`` decodes serially (the reference drafts equal).
        With --parallel the request joins the scheduler's rounds (this call returns when it is done)."""

        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        if self.concurrent:
            stats = self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos,
                                          constraint=constraint, background=background)
            return {**stats, "prompt_tokens": len(prompt)}
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        header = [max_tokens, int(stop_eos), int(draft), seed & 0xFFFFFFFF, seed >> 32,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0), *_f64_ints(sampling.min_p if sampling else 0.0),
                  int(constraint is not None)]
        self._share(header)
        self._share(list(prompt))
        if constraint is not None:                     # the request's grammar: rank 1 compiles the same
            from tensorfold.engine.grammar import pack

            self._share(pack(constraint))
        res = self._run(prompt, max_tokens, sampling, draft, stop_eos, on_tokens, constraint)
        stats = {"prompt_tokens": len(prompt), "completion_tokens": len(res["tokens"]),
                 "prefill_s": round(res["prefill_s"], 3), "decode_s": round(res["decode_s"], 3),
                 "prefill_tps": round(res["prefill_tps"], 1), "decode_tps": round(res["decode_tps"], 1),
                 "drafts": bool(draft and self.drafts), "cached": int(res.get("cached", 0)),
                 "kept_states": len(self.e.pool.entries) if self.e.pool is not None else 0}
        for key in ("rounds", "accepted_per_round", "tokens_per_round", "k_histogram", "copy_rounds", "copy_accepted"):
            if key in res:
                stats[key] = res[key]
        if hasattr(self.nccl, "check"):                  # a RoCE wait that gave up inside a graph surfaces here
            self.nccl.check()
        return stats

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        if self.concurrent:
            self.multi.follow()
            return

        while True:
            (max_tokens, stop_eos, draft, s_lo, s_hi, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi,
             shaped) = self._share(None)
            prompt = self._share(None)
            constraint = None
            if shaped:                                 # compiled here as on rank 0
                from tensorfold.engine import grammar

                constraint = grammar.compiler(self, self.model_dir, self.eos).follow(self._share(None))
            temperature = _ints_f64(t_lo, t_hi)
            sampling = (Sampling(seed=(s_hi << 32) | s_lo, temperature=temperature, top_k=top_k,
                                 top_p=_ints_f64(p_lo, p_hi), min_p=_ints_f64(m_lo, m_hi))
                        if temperature > 0 else None)
            try:
                self._run(prompt, max_tokens, sampling, bool(draft), bool(stop_eos), None, constraint)
            except Exception as exc:
                from tensorfold.engine.grammar import GrammarError

                if not isinstance(exc, GrammarError):
                    raise
                print(f"[tensorfold] rank 1: the request's grammar ended it: {exc}", flush=True)
