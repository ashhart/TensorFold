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
# memory a context token costs: compressed + indexer-key caches (1.8 KB in fp8), RoPE tables (0.8 KB), and the prompt
# selection's per-block maxima and flags (512 rows / 8-entry blocks: ~0.3 KB; scores stream in fixed segments)
def _cache_bytes() -> int:
    """A stream's per-token caches: compressed entries + indexer keys at 2.5 entries a token (ratio-2 sources 2, 8, 14
    and ratio-1 source 20), ``serial.entry_bytes``: bf16 1024 + 256 B an entry, fp8 604 (448 e4m3 + 64 bf16 RoPE + 7
    scales) + 134, fp4 288 (NVFP4) + 68 (MXFP4)."""

    from .serial import entry_bytes

    return int(2.5 * sum(entry_bytes()))


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

    from .serial import entry_bytes

    return int(2.5 * entry_bytes()[0])


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


# --parallel above 1: the streams draw their per-token caches from one shared pool (``pool.py``) instead of each
# holding a fixed --context's worth; TF_DSV41_SHARED_POOL=0 keeps the fixed layout
SHARED_POOL = os.environ.get("TF_DSV41_SHARED_POOL", "1") != "0"
# kept prompts inside the shared pool: their per-token caches stay in the pool's free rows, their window rings (a
# slot's DRING rows of every ring, SLOT_BYTES) in a bank of this many entries
KEPT_ENTRIES = int(os.environ.get("TF_DSV41_KEPT_ENTRIES") or "32")


# the decode graphs' buffers that scale with a stream's limit ([R, limit // ratio] indexer scores and top-k scratch):
# on one shared graph pool the largest graph's, measured 0.27 GiB for 128 graphs (32 row counts x 4 key widths) at a
# 614400 limit; with a pool per graph (TF_DSV41_SHARED_GRAPHS=0) the sum, 4.05 GiB for the 32 full-width graphs
GRAPH_BYTES_PER_LIMIT_TOKEN = 512 if os.environ.get("TF_DSV41_SHARED_GRAPHS", "1") != "0" else 7168
# what an instantiated decode graph costs the driver (outside the CUDA allocator: ~5.1 GB of MemAvailable went for
# 128 graphs while the allocator grew 0.26 GB); FIXED_GIB covers the 32 full-width ones, this each narrower one
GRAPH_EXEC_BYTES = 40 * 2 ** 20


def _widths_words() -> list[int]:
    """The decode graphs' key widths and graph-pool flag as ints (both ranks must capture the same graphs)."""

    from .serial import SHARED_GRAPH_POOL, WIDTHS

    ws = sorted(WIDTHS)[:4]
    return [int(SHARED_GRAPH_POOL), len(WIDTHS), *ws, *[0] * (4 - len(ws))]


def _keep_words() -> list[int]:
    """The kept-prompt policy (each rank decides keeps from it on its own: the ranks must agree)."""

    from .multi import KEEP_MIN

    return [KEEP_MIN, int(os.environ.get("TF_DSV41_POOL_KEEP", "1") != "0")]


def _kv_words() -> list[int]:
    """The per-token cache format and its fp4 knobs (TF_DSV41_KV, TF_DSV41_IQ_FP4 / SWA_FP8 / COMP_BF16): a state's
    bytes and numerics follow them, and each rank sizes and keeps prompts from its own: the ranks must agree."""

    from .serial import COMP_BF16, IQ_FP4, KV_MODE, SWA_FP8

    return [("bf16", "fp8", "fp4").index(KV_MODE), int(IQ_FP4), int(SWA_FP8), int(COMP_BF16)]


def _disk_words() -> list[int]:
    """The NVMe tier of kept prompts (``kvdisk``): on, budget, minimum tokens, stage (both ranks' disk indexes change
    together and the stage is counted before the pool is sized: the ranks must agree)."""

    from .kvdisk import agreement_words

    return agreement_words()


def carved_bytes(pool: int, carve: int, ratios: tuple[int, ...] = (2, 2, 2, 1)) -> int:
    """Bytes of compressed-entry planes ``_comp_pools`` places in a ``carve``-byte display carveout for a ``pool``-
    token arena: whole planes (one per kv source, ``pool // ratio + 1`` rows), largest first, while each fits."""

    from .serial import entry_bytes

    row = entry_bytes()[0]
    placed, left = 0, carve
    for r in sorted(ratios):                             # smallest ratio = largest plane first
        need = (pool // r + 1) * row
        if need + 4096 <= left:
            placed, left = placed + need, left - need
    return placed


def narrow_graphs(streams: int, limit: int) -> int:
    """Decode graphs captured at narrower key widths than ``limit`` (32 row counts each with --parallel)."""

    from .serial import PROMPT_ROWS, WIDTHS

    return PROMPT_ROWS * len([w for w in WIDTHS if w < limit]) if streams > 1 else 0


def pool_tokens(free: int, streams: int, limit: int, carve: int = 0) -> int:
    """Shared-pool rows (a multiple of ``pool.ALIGN``) that fit ``free`` bytes beside ``streams`` slots' rings, a
    ``limit``-token window's buffers (RoPE tables, selection, the decode graphs' score buffers) and the kept-prompt
    reserve; whole compressed-entry planes go to a ``carve``-byte display carveout first. At most every stream at
    the full limit."""

    from .pool import ALIGN, align_up

    room = (free - int((FIXED_GIB + RESERVE_GIB) * 2 ** 30) - (streams - 1 + KEPT_ENTRIES) * SLOT_BYTES
            - narrow_graphs(streams, limit) * GRAPH_EXEC_BYTES
            - (TOKEN_BYTES - CACHE_BYTES + GRAPH_BYTES_PER_LIMIT_TOKEN) * limit)
    cap = streams * align_up(limit)
    best = max(0, min(room // CACHE_BYTES, cap)) // ALIGN * ALIGN
    if carve:                                            # grow while the carved planes pay for the extra rows
        step = best
        while step >= ALIGN:
            trial = min(cap, best + step) // ALIGN * ALIGN
            if trial > best and trial * CACHE_BYTES - carved_bytes(trial, carve) <= room:
                best = trial
            else:
                step //= 2
    return best


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
        self._use_rdma(rank)
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        cap = int(context) if context and explicit else DEFAULT_CONTEXT   # the CLI hands the native window otherwise
        if not (Path(engram) / "config.json").exists() and not any(Path(engram).glob("*.safetensors")):
            raise FileNotFoundError(f"no Engram tables in {engram}: put shards 47-48 of deepseek-ai/DeepSeek-V4.1-Flash "
                                    "there (or point TF_DSV41_ENGRAM_DIR at them)")
        self.streams = max(1, int(parallel))
        mine = [cap, int(bool(drafts)), int(explicit), self.streams, int(SHARED_POOL and self.streams > 1),
                KEPT_ENTRIES, *_widths_words(), *_keep_words(), *_disk_words(), *_kv_words()]
        both = self._gather_ints(mine)
        if both[0] != both[1]:
            raise RuntimeError(f"the two ranks were started with different settings (context, drafts, parallel, "
                               f"TF_DSV41_DISK*, TF_DSV41_KV and its knobs): rank 0 {both[0]}, rank 1 {both[1]}; "
                               "give both the same flags")
        started = time.perf_counter()
        self._boot = [("start", started)]
        w = W.load(self.model_dir, rank=rank, log=lambda *a, **k: None, draft=bool(drafts))
        self._mark("weights")
        # the NVMe tier of kept prompts (TF_DSV41_DISK; shared pool only): its pinned stage before memory is measured
        self.disk = None
        if SHARED_POOL and self.streams > 1 and _keep_words()[1]:
            from . import kvdisk

            self.disk = kvdisk.from_env(rank)
        elif rank == 0 and _disk_words()[0]:
            print("[tensorfold] TF_DSV41_DISK is for kept prompts in the shared pool (--parallel 2 or more, "
                  "TF_DSV41_POOL_KEEP on): off", flush=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()                     # the load's staging buffers back before memory is measured
        # admission, before any cache exists: both ranks' memory decides (a GB10 out of memory can wedge the node)
        free = available_bytes()
        if rank == 0:
            print(f"[tensorfold] memory left after the weights: {free / 2 ** 30:.1f} GiB", flush=True)
        from tensorfold.cuda import carveout

        carve = carveout.requested_bytes() if carveout.enabled() else 0
        self.shared = SHARED_POOL and self.streams > 1
        if self.shared:                              # one stream may reach the whole pool, less a reply's room
            limit_asked = min(cap, NATIVE_CONTEXT)
            pool = min(row[0] for row in self._gather_ints([pool_tokens(free, self.streams, limit_asked, carve)]))
            if os.environ.get("TF_DSV41_POOL_TOKENS"):         # (tests: a pool smaller than memory allows)
                from .pool import ALIGN

                pool = min(pool, int(os.environ["TF_DSV41_POOL_TOKENS"]) // ALIGN * ALIGN)
            largest = max(0, min(NATIVE_CONTEXT, pool - 4096) // 1024 * 1024)
        else:
            pool = None
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
        if self.shared:
            self.capacity_plan["pool_tokens"] = pool
            if rank == 0:
                print(f"[tensorfold] shared cache pool: {pool:,} tokens for {self.streams} streams of up to {cap:,} "
                      f"each; its free rows keep up to {KEPT_ENTRIES} prompt states", flush=True)
        self._memlog("before the caches")
        self.e = SerialEngine(w, Comm(self.nccl), str(engram), str(self.model_dir / "tokenizer.json"), cap=cap,
                              slots=self.streams, pool_tokens=pool)
        self.nccl.barrier()
        self._mark("caches")
        self._memlog("after the caches")
        with torch.no_grad():
            if drafts:
                self.e.enable_dspark(DRAFTS)
            if self.shared:
                self.e.make_bank(KEPT_ENTRIES)
            torch.cuda.synchronize()
            self._mark("drafter+bank")
            self._memlog("after drafter and bank")
            before, avail = torch.cuda.memory_reserved(), available_bytes()
            self.e.capture(1)
            # verify windows of one stream (1 + drafts rows); with --parallel, any round of up to 16 rows
            from .serial import PROMPT_ROWS

            top = PROMPT_ROWS if self.streams > 1 else (DRAFTS + 1 if drafts else 1)
            # copy-draft wiring follows MiaAI-Lab's GLM recipe patches 0007 / 0032 (Apache-2.0);
            # see THIRD_PARTY_NOTICES.md
            if drafts and self.streams == 1 and os.environ.get("TF_COPY_DRAFTS", "1") != "0":   # longer copy windows
                top = max(top, min(PROMPT_ROWS, int(os.environ.get("TF_COPY_MAX") or 15) + 1))
            for rows in range(2, top + 1):
                self.e.capture(rows)
            torch.cuda.synchronize()
            self._mark("decode graphs")
            self._memlog("after the decode graphs")
            if rank == 0:
                held = (torch.cuda.memory_reserved() - before) / 2 ** 30
                print(f"[tensorfold] decode graphs: {held:.2f} GiB of buffers + "
                      f"{max(0.0, (avail - available_bytes()) / 2 ** 30 - held):.1f} GiB of the driver's for "
                      f"{len(self.e.graphs) + len(self.e.narrow)} graphs (key widths {[*self.e.widths, cap]})",
                      flush=True)
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
            self._memlog("before warm-up")
            self._mark("drafter graphs")
            self._warm()
            self._mark("warm-up")
            self._memlog("after warm-up")
        if self.shared:                                 # kept prompts live in the shared pool (multi.Kept)
            self.e.pool = None
        else:
            self._make_pool(cap)
        self._warm_tool_grammar()
        self.concurrent = self.streams > 1
        self.multi = self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder
            from .pool import Pool

            if self.disk is not None:
                self._attach_disk(Path(engram))
            self.multi = MultiDecoder(self.e, self._share, rank=rank, drafts=DRAFTS if drafts else 0,
                                      pool=Pool(self.e.pool_tokens) if self.shared else None,
                                      gather=self._gather_ints, disk=self.disk)
            self.multi.model_dir = self.model_dir
            self.multi.calibrate(self._gather_ints)
            self._mark("calibration")
            self._memlog("after calibration")
            if rank == 0:
                curve = " ".join(f"{v:.0f}" for v in self.multi.costs)
                print(f"[tensorfold] verify ms by rows 1..{len(self.multi.costs)}: {curve}; a draft "
                      f"{self.multi.draft_ms:.1f} ms", flush=True)
            if rank == 0:
                self.scheduler = Scheduler(self.multi, max_streams=self.streams)
                print(f"[tensorfold] {self.streams} concurrent streams of up to {cap} tokens each", flush=True)
        self._mark("ready")
        if rank == 0:
            marks = self._boot
            print("[boot] " + ", ".join(f"{name} {t - prev:.1f}s" for (_, prev), (name, t) in zip(marks, marks[1:]))
                  + f"; total {marks[-1][1] - marks[0][1]:.1f}s", flush=True)

    def _attach_disk(self, engram: Path) -> None:
        """Both ranks: this build's directory of the NVMe tier indexed (``reconcile``), the entry layout checked
        equal, and only the entries both ranks hold kept, in rank 0's order (both indexes the same)."""

        from . import kvdisk
        from .pool import ALIGN

        d = self.disk
        t = time.perf_counter()
        mine = [0, 0, 0]
        try:                                       # (no exception may skip the gather: the other rank waits)
            d.attach(kvdisk.compat_ident(self.e, self.model_dir, engram))
            views = self.e.kept_views(0, ALIGN, 0)
            mine = [1, sum(kvdisk._nbytes(v) for _, v in views), len(views)]
        except Exception as exc:                   # noqa: BLE001
            print(f"[tensorfold] rank {self.rank}: kept prompts on NVMe unavailable: {exc}", flush=True)
        both = self._gather_ints(mine)
        if not (both[0][0] and both[1][0]) or both[0] != both[1]:
            if self.rank == 0:
                print(f"[tensorfold] kept prompts on NVMe: off (ranks' entry layouts {both})", flush=True)
            self.disk = None
            return
        kvdisk.intersect(d, self._gather_ints)
        if self.rank == 0:
            print(f"[tensorfold] {d.describe()}; {len(d.index)} kept on both ranks, ready in "
                  f"{time.perf_counter() - t:.1f}s", flush=True)

    def stop_serving(self) -> None:
        """Rank 0, the server stopping (SIGTERM / ``make down``): with the NVMe tier on, the scheduler ends the live
        streams and both ranks write their kept prompts to disk, then rank 1 leaves ``follow``. Off: nothing (the
        process exits as before)."""

        if self.scheduler is not None and getattr(self.multi, "disk", None) is not None:
            self.scheduler.shutdown()

    def _mark(self, stage: str) -> None:
        """A startup timeline mark (printed as one [boot] line when ready)."""

        if hasattr(self, "_boot"):
            self.torch.cuda.synchronize()
            self._boot.append((stage, time.perf_counter()))

    def _memlog(self, stage: str) -> None:
        """TF_DSV41_MEMLOG=1: what the system and the CUDA allocator hold at a startup stage (both ranks)."""

        if os.environ.get("TF_DSV41_MEMLOG") != "1":
            return
        import torch

        torch.cuda.synchronize()
        print(f"[memlog r{self.rank}] {stage}: available {available_bytes() / 2 ** 30:.2f} GiB, allocated "
              f"{torch.cuda.memory_allocated() / 2 ** 30:.2f}, reserved {torch.cuda.memory_reserved() / 2 ** 30:.2f}",
              flush=True)

    def _warm_tool_grammar(self) -> None:
        """TF_DSV41_TOOL_GRAMMAR on: both ranks build the tool-call compiler now. Built on rank 1's first such request
        instead, it would parse tokenizer.json inside ``follow`` while rank 0 waits in NCCL, stalling every stream."""

        from tensorfold.engine import grammar
        from tensorfold.server.errors import RequestError

        if grammar.tool_grammar_mode() == "off":
            return
        started = time.perf_counter()
        try:
            grammar.compiler(self, self.model_dir, self.eos).tools_compiler()
        except (ImportError, RequestError, ValueError) as exc:     # served without it: such requests are refused
            print(f"[tensorfold] rank {self.rank}: tool-call grammars unavailable: {exc}", flush=True)
            return
        if self.rank == 0:
            print(f"[tensorfold] tool-call grammars ({grammar.tool_grammar_mode()}): DSML compiler built in "
                  f"{time.perf_counter() - started:.1f}s", flush=True)

    def _make_pool(self, cap: int) -> None:
        """Kept prompt states (``tensorfold.cuda.kv_pool``) in what the context's prompt buffers and the reserve
        leave; both ranks keep the smaller budget, so their pools stay identical."""

        from tensorfold.cuda.kv_pool import PrefixPool

        transient = int(PROMPT_TRANSIENT_GIB * 2 ** 30) + (TOKEN_BYTES - CACHE_BYTES - 768) * cap   # selection rows
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
    def _use_rdma(self, rank: int) -> None:
        """Small all-gathers over our RoCE transport (``tensorfold.cuda.rdma``: the same bits, +1.7% serial) unless
        TF_COMM=nccl. Both ranks first check they can open it (the proxy library, an RDMA device) and agree; the setup
        itself fails on both ranks together (it votes), and any failure leaves NCCL in place."""

        if (os.environ.get("TF_COMM") or "rdma") != "rdma":
            return
        try:
            from tensorfold.cuda import rdma

            rdma.proxy()
            rdma._device()
            ok = 1
        except Exception as exc:  # noqa: BLE001 - reported below, then NCCL
            ok, why = 0, f"{type(exc).__name__}: {exc}"
        flags = self._gather_ints([ok])
        if not all(f[0] for f in flags):
            if rank == 0:
                print("[tensorfold] RoCE all-gathers unavailable on a rank"
                      + (f" ({why})" if not ok else "") + "; NCCL carries every gather", flush=True)
            return
        try:
            self.nccl = rdma.RdmaComm(self.nccl, rank)
        except RuntimeError as exc:                                 # (both ranks raise together)
            if rank == 0:
                print(f"[tensorfold] {exc}; NCCL carries every gather", flush=True)
            return
        if rank == 0:
            print(f"[tensorfold] all-gathers up to {self.nccl.rdma.slot_bytes >> 10} KiB over RoCE "
                  f"({self.nccl.rdma.device}), larger ones over NCCL", flush=True)

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
        # (with --parallel every request runs the multi path: the serial drafts-off pass would warm nothing it uses)
        for draft in ((True, False) if self.drafts and self.streams == 1 else (bool(self.drafts),)):
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
