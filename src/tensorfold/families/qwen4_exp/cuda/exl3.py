"""Flash Next weights from an EXL3 checkpoint, and the few kernels the EXL3 path adds.

An EXL3 pack of Qwen3.8 Flash Next (``quant_method: exl3``, any codebook, per-tensor widths mixed freely) fills the
same ``Weights`` dataclasses as the MLX checkpoint, with three kinds of matrices in place of ``qmm.Q4``:

- ``X3``: the dense trellis groups (attention and DeltaNet projections, the MTP head's fc layers, the head) on
  the row-invariant EXL3 linear (``tensorfold.cuda.exl3.linear``); projections that read the same input are
  ``Stack``-ed, each part writing its columns.
- ``F16``: the tensors the pack leaves unquantized (hyper-connection down / inject / up, DeltaNet in_proj_a/b,
  the n-gram key and value projections) on a plain fp16 matmul: one program per (row block, 128 outputs, K slice)
  with fp32 accumulation over K in a fixed order, slices summed in order. A function of the shape only, so a row's
  bits never depend on the rows beside it; no cuBLAS.
- the routed experts and the shared expert (expert 512 of the same table) on the grouped EXL3 expert kernel
  (``tensorfold.cuda.exl3.experts``), each expert matrix at its own width.

What the pack stores differently from the MLX checkpoint, checked tensor by tensor against it:

- every centred RMSNorm weight (hyper-connection norms, q/k norms, the indexer's norms, the n-gram branch's three
  norms, the MTP head's two input norms) is stored as gamma - 1 (the module's constant bias of 1 is not folded in):
  the loader adds 1, in fp32. The DeltaNet's gated norm, A_log, dt_bias, conv weights and the router are identical.
- the n-gram tables (``ngram_embedding.safetensors``) are not EXL3 tiles despite their ``.trellis`` name: each row
  is one fp16 scale word then a 160-value tail-biting bitstream of K bits per value (a trellis of the mul1 codebook,
  ExLlamaV3's ``ngram_codec``), plus a per-head fp16 bias. ``ple_rows`` decodes a window's rows exactly as
  ExLlamaV3's ``ngram_dequant`` does (fp16 codebook value * scale + head bias, rounded to fp16).
- turboderp's packs ship the MTP head's final mixer in ``mtp_hyper_connection_mixer_patch.safetensors``, outside
  the index; the reader takes names from it that the index lacks.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3.linear import Exl3Linear

ROWS = 128               # most rows one call of the EXL3 linear takes (wider inputs run in chunks of 128)
MOE_ROWS = 128           # most rows a MoE window takes
EXTRA_FILES = ("ngram_embedding.safetensors", "mtp_hyper_connection_mixer_patch.safetensors")
_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32,
       "I16": torch.int16, "U8": torch.uint8, "I8": torch.int8, "U16": torch.int16, "U32": torch.int32}


def is_exl3(model_dir: str | Path) -> bool:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    for source in (raw, raw.get("text_config") or {}):
        for key in ("quantization_config", "quantization"):
            block = source.get(key)
            if isinstance(block, dict) and str(block.get("quant_method", "")).lower() == "exl3":
                return True
    return False


# -- the pack's tensors -------------------------------------------------------------------------------------
class Pack:
    """Tensors by name from the index's shards and the extra files beside them, read with large sequential reads;
    ``release`` drops the read files' cached pages (unified memory: the bytes are on the GPU already)."""

    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        self.where: dict[str, str] = dict(json.loads((self.dir / "model.safetensors.index.json").read_text())
                                          ["weight_map"])
        self.headers: dict[str, tuple[int, dict]] = {}
        for extra in EXTRA_FILES:
            if (self.dir / extra).is_file():
                _, header = self.header(extra)
                for name in header:
                    if name != "__metadata__":
                        self.where.setdefault(name, extra)
        self.touched: set[str] = set()

    def header(self, file: str) -> tuple[int, dict]:
        got = self.headers.get(file)
        if got is None:
            with open(self.dir / file, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                got = (8 + n, json.loads(f.read(n)))
            self.headers[file] = got
        return got

    def has(self, name: str) -> bool:
        return name in self.where

    def entry(self, name: str) -> tuple[str, int, int, str, list[int]]:
        """(file, absolute begin, absolute end, dtype, shape)."""

        file = self.where[name]
        base, header = self.header(file)
        e = header[name]
        begin, end = e["data_offsets"]
        return file, base + begin, base + end, e["dtype"], list(e["shape"])

    def read(self, file: str, begin: int, end: int) -> torch.Tensor:
        raw = torch.empty((end - begin,), dtype=torch.uint8)
        view = memoryview(raw.numpy())
        with open(self.dir / file, "rb", buffering=0) as f:
            f.seek(begin)
            at = 0
            while at < len(view):
                got = f.readinto(view[at:at + (64 << 20)])
                if not got:
                    raise IOError(f"short read of {file} at {begin + at}")
                at += got
        self.touched.add(file)
        return raw

    def get(self, name: str) -> torch.Tensor:
        file, begin, end, dtype, shape = self.entry(name)
        return self.read(file, begin, end).view(_DT[dtype]).reshape(shape)

    def release(self) -> None:
        for file in list(self.touched):
            try:
                fd = os.open(self.dir / file, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except (OSError, AttributeError):
                pass
        self.touched.clear()

    def codebook(self, prefix: str) -> str:
        return "mul1" if self.has(prefix + ".mul1") else "mcg" if self.has(prefix + ".mcg") else "3inst"


# -- scratch shared by every matrix of the model -----------------------------------------------------------------
class Scratch:
    """One set of scratch buffers for all EXL3 / fp16 matrices (calls run one after another on one stream)."""

    def __init__(self, slots: int) -> None:
        self.slots = slots                 # routed experts a row, plus the shared expert
        self.users: list[Any] = []
        self.xh = self.z = self.tmp = self.part = None
        self.moe: x3experts.Scratch | None = None
        self.ple_host = self.ple_dev = self.ple_emb = None

    def allocate(self, device, *, experts: x3experts.Exl3RoutedExperts, ple_words: int, ple_heads: int,
                 ple_dim: int) -> None:
        xh = z = tmp = part = 1
        for u in self.users:
            if isinstance(u, X3):
                xh = max(xh, ROWS * u.lin.k)
                sk = u.lin.split[0]
                if sk > 1:
                    z = max(z, sk * ROWS * u.lin.n)
            elif isinstance(u, F16):
                part = max(part, u.sk * ROWS * u.n)
            elif isinstance(u, Stack):
                if len(u.parts) > 1:
                    tmp = max(tmp, ROWS * max(p.n for p in u.parts))
        self.xh = torch.empty((xh,), dtype=torch.float16, device=device)
        self.z = torch.empty((z,), dtype=torch.float32, device=device)
        self.tmp = torch.empty((tmp,), dtype=torch.bfloat16, device=device)
        self.part = torch.empty((part,), dtype=torch.float32, device=device)
        self.moe = x3experts.Scratch(experts, MOE_ROWS, self.slots, device=device)
        if ple_words:
            pin = torch.cuda.is_available()
            self.ple_host = torch.zeros((MOE_ROWS * ple_heads, ple_words), dtype=torch.int16, pin_memory=pin)
            self.ple_dev = torch.zeros((MOE_ROWS * ple_heads, ple_words), dtype=torch.int16, device=device)
            self.ple_emb = torch.empty((MOE_ROWS, ple_dim), dtype=torch.float16, device=device)


# -- the fp16 matmul (tensors the pack leaves unquantized) -----------------------------------------------------
@triton.jit(do_not_specialize=["M"])
def _f16_mm(X, W, OUT, M, N, x_stride, o_stride, K: tl.constexpr, KS: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
    """Program (m block, n block, k slice): OUT[slice][m, n] = fp32 sum over the slice's K (BK steps in order) of
    fp16(x[m, k]) * w[n, k]. With one slice, OUT is the result (bf16 or fp32); else fp32 partials [KS, M, N]."""

    pm = tl.program_id(0)
    pn = tl.program_id(1)
    ps = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = ps * KS
    for kk in range(0, KS, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + kk + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * K + (k0 + kk + rk)[None, :], mask=n_ok[:, None], other=0.0)
        acc = tl.dot(x.to(tl.float16), tl.trans(w), acc)
    if F32:
        tl.store(OUT + ps * M * N + rm[:, None] * N + rn[None, :], acc, mask=m_ok[:, None] & n_ok[None, :])
    else:
        tl.store(OUT + rm[:, None] * o_stride + rn[None, :], acc.to(OUT.dtype.element_ty),
                 mask=m_ok[:, None] & n_ok[None, :])


@triton.jit(do_not_specialize=["M"])
def _reduce(P, OUT, M, N, o_stride, SK: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SK):
        acc += tl.load(P + s * M * N + r * N + c, mask=ok, other=0.0)
    tl.store(OUT + r * o_stride + c, acc.to(OUT.dtype.element_ty), mask=ok)


def f16_split(n: int, k: int, target: int = 96) -> int:
    """K slices for an (n, k) fp16 matrix: a function of the shape only."""

    tiles = -(-n // 64)
    sk = 1
    while sk < 32 and tiles * sk < target and k % (sk * 2 * 64) == 0 and k // (sk * 2) >= 256:
        sk *= 2
    return sk


@dataclass
class F16:
    """y = x @ w.T for an unquantized [n, k] matrix (fp16), row-invariant (tile sizes fixed by the shape)."""

    w: torch.Tensor
    n: int
    k: int
    sk: int
    sc: Scratch = field(repr=False)

    def nbytes(self) -> int:
        return self.w.numel() * 2

    def partials(self, x: torch.Tensor) -> torch.Tensor:
        """Unreduced fp32 slices [sk, M, n] (in the shared scratch), for a consumer that sums them in order."""

        m = x.shape[0]
        if m > ROWS:
            raise ValueError(f"F16.partials takes at most {ROWS} rows")
        out = self.sc.part[:self.sk * m * self.n].view(self.sk, m, self.n)
        self._launch(x, out, True)
        return out

    def _launch(self, x: torch.Tensor, out: torch.Tensor, f32: bool) -> None:
        m = x.shape[0]
        if x.stride(1) != 1 or x.shape[1] != self.k:
            raise ValueError(f"F16: x {tuple(x.shape)} does not match K={self.k}")
        bm, bn, bk = 16, 64, 64
        grid = (triton.cdiv(m, bm), triton.cdiv(self.n, bn), self.sk)
        _f16_mm[grid](x, self.w, out, m, self.n, x.stride(0), self.n if f32 else out.stride(0),
                      K=self.k, KS=self.k // self.sk, BM=bm, BN=bn, BK=bk, F32=f32, num_warps=4, num_stages=3)

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        if out.stride(-1) != 1:
            raise ValueError("F16: output rows must be contiguous")
        if self.sk == 1:
            self._launch(x, out, False)
            return out
        if m > ROWS:
            raise ValueError(f"F16 with K slices takes at most {ROWS} rows")
        part = self.sc.part[:self.sk * m * self.n].view(self.sk, m, self.n)
        self._launch(x, part, True)
        _reduce[(m, triton.cdiv(self.n, 256))](part, out, m, self.n, out.stride(0), SK=self.sk, BLOCK=256,
                                               num_warps=2)
        return out


def f16(sc: Scratch, rows: list[torch.Tensor], device) -> F16:
    w = torch.cat([r.to(torch.float16) for r in rows]).to(device).contiguous()
    n, k = w.shape
    got = F16(w, n, k, f16_split(n, k), sc)
    sc.users.append(got)
    return got


# -- EXL3 linears -------------------------------------------------------------------------------------------
@dataclass
class X3:
    lin: Exl3Linear
    sc: Scratch = field(repr=False)

    @property
    def n(self) -> int:
        return self.lin.n

    @property
    def k(self) -> int:
        return self.lin.k

    def nbytes(self) -> int:
        return self.lin.nbytes()

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        lin = self.lin
        sk = lin.split[0]
        for r0 in range(0, m, ROWS):
            r1 = min(m, r0 + ROWS)
            xs = x[r0:r1]
            if not xs.is_contiguous():
                xs = xs.contiguous()
            rows = r1 - r0
            lin(xs, out=out[r0:r1], xh=self.sc.xh[:rows * lin.k].view(rows, lin.k),
                z=self.sc.z[:sk * rows * lin.n] if sk > 1 else None)
        return out


def x3(sc: Scratch, pk: Pack, prefix: str, device) -> X3:
    suh = pk.get(prefix + ".suh") if pk.has(prefix + ".suh") else pk.get(prefix + ".su")
    svh = pk.get(prefix + ".svh") if pk.has(prefix + ".svh") else pk.get(prefix + ".sv")
    bias = pk.get(prefix + ".bias") if pk.has(prefix + ".bias") else None
    lin = Exl3Linear.from_tensors(pk.get(prefix + ".trellis"), suh, svh, pk.codebook(prefix), bias, device)
    got = X3(lin, sc)
    sc.users.append(got)
    return got


@dataclass
class Stack:
    """Matrices reading the same input, written side by side into one output (the MLX path's stacked Q4)."""

    parts: list[Any]
    sc: Scratch = field(repr=False)

    @property
    def n(self) -> int:
        return sum(p.n for p in self.parts)

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if len(self.parts) == 1 and out.is_contiguous():
            return self.parts[0](x, out)
        m = x.shape[0]
        if m == 1:                    # one row: a part's slice of ``out`` needs no copy (its row stride is unused)
            at = 0
            for p in self.parts:
                p(x, out[:, at:at + p.n])
                at += p.n
            return out
        at = 0
        for p in self.parts:
            if m > ROWS:
                raise ValueError(f"a stacked projection takes at most {ROWS} rows")
            tmp = self.sc.tmp[:m * p.n].view(m, p.n)
            p(x, tmp)
            out[:, at:at + p.n].copy_(tmp)
            at += p.n
        return out


def stack(sc: Scratch, parts: list[Any]) -> Stack:
    got = Stack(parts, sc)
    sc.users.append(got)
    return got


def mm(q: Any, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return q(x, out)


# -- embedding --------------------------------------------------------------------------------------------
@triton.jit
def _embed(IDS, T, OUT, D: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = tl.load(IDS + r).to(tl.int64)
    v = tl.load(T + t * D + d).to(tl.bfloat16)
    for s in tl.static_range(S):
        tl.store(OUT + r * (S * D) + s * D + d, v)


def embed(ids: torch.Tensor, table: torch.Tensor, dims: int, copies: int, out: torch.Tensor) -> torch.Tensor:
    """ids (R,) int32 -> out (R, copies * dims) bf16: the unquantized embedding row, repeated per stream."""

    rows = ids.shape[0]
    _embed[(rows, dims // 256)](ids, table, out, D=dims, S=copies, BLOCK=256, num_warps=2)
    return out


# -- the n-gram tables -------------------------------------------------------------------------------------
@triton.jit
def _ple_rows(PK, HB, OUT, WORDS: tl.constexpr, KB: tl.constexpr, HEADS: tl.constexpr, DH: tl.constexpr,
              BLOCK: tl.constexpr, K_INV: tl.constexpr, K_BIAS: tl.constexpr):
    """Program (r, h): staged row r * HEADS + h (one fp16 scale word, then DH values of KB bits in a tail-biting
    ring: value i's 16-bit state takes stream bit m from ring position ((i - m // KB) mod DH) * KB + m % KB) ->
    OUT[r, h DH: (h + 1) DH] = fp16(fp16(mul1 codebook(state)) * scale + head bias[h])."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    row = (r * HEADS + h).to(tl.int64)
    base = PK + row * WORDS
    i = tl.arange(0, BLOCK)
    ok = i < DH
    scale = tl.load(base).to(tl.float16, bitcast=True).to(tl.float32)
    state = tl.zeros((BLOCK,), dtype=tl.int64)
    for m in tl.static_range(16):
        pos = i - (m // KB)
        pos = tl.where(pos < 0, pos + DH, pos)
        sb = pos * KB + (m % KB)
        word = tl.load(base + 1 + (sb >> 4), mask=ok, other=0).to(tl.int64) & 0xFFFF
        state |= ((word >> (sb & 15)) & 1) << m
    prod = (state * 0x83DCD12D) & 0xFFFFFFFF
    hs = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    cb = ((1024 + hs).to(tl.float32) * K_INV + K_BIAS).to(tl.float16).to(tl.float32)
    b = tl.load(HB + h * DH + i, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * (HEADS * DH) + h * DH + i, (cb * scale + b).to(tl.float16), mask=ok)


def _fp16_bits(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


K_INV, K_BIAS = _fp16_bits(0x1EEE), _fp16_bits(0xC931)


def ple_rows(rows: int, packed: torch.Tensor, head_bias: torch.Tensor, heads: int, dh: int, bits: int,
             out: torch.Tensor) -> torch.Tensor:
    """Staged table rows (row r * heads + h) -> out [rows, heads * dh] fp16."""

    _ple_rows[(rows, heads)](packed, head_bias, out, WORDS=packed.shape[1], KB=bits, HEADS=heads, DH=dh,
                             BLOCK=triton.next_power_of_2(dh), K_INV=K_INV, K_BIAS=K_BIAS, num_warps=4)
    return out


class NgramTable:
    """The n-gram embedding's 128 shards (ExLlamaV3's row codec), memory-mapped from the checkpoint; a token reads
    16 rows of ~120 bytes, which ``gather`` copies out."""

    def __init__(self, pk: Pack, base: str, shards: int, device) -> None:
        maps: dict[str, tuple[np.memmap, int]] = {}
        starts, offsets, fidx, files = [0], [], [], []
        words = None
        for i in range(shards):
            file, begin, end, dtype, shape = pk.entry(f"{base}shard_{i}.trellis")
            if dtype != "I16" or len(shape) != 2:
                raise ValueError(f"n-gram shard {i}: expected int16 [rows, words], got {dtype} {shape}")
            if words is None:
                words = shape[1]
            elif shape[1] != words:
                raise ValueError("n-gram shards of different widths")
            if file not in maps:
                maps[file] = (np.memmap(pk.dir / file, dtype=np.uint8, mode="r"), len(files))
                files.append(file)
            fidx.append(maps[file][1])
            offsets.append(begin)
            starts.append(starts[-1] + shape[0])
        self.words = int(words)
        self.dh = 160
        self.bits = (self.words - 1) * 16 // self.dh
        if 1 + self.dh * self.bits // 16 != self.words:
            raise ValueError(f"n-gram rows of {self.words} words are not one scale plus 160 values")
        self.maps = [maps[f][0] for f in files]
        self.fidx = np.array(fidx, dtype=np.int64)
        self.offsets = np.array(offsets, dtype=np.int64)
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.row_bytes = 2 * self.words
        self.head_bias = pk.get(base + "head_bias").to(torch.float16).to(device).contiguous()
        self.head_offsets = pk.get(base + "head_offsets").cpu().numpy()
        self.head_sizes = pk.get(base + "head_vocab_sizes").cpu().numpy()
        self.multipliers = pk.get(base + "layer_multipliers").cpu().numpy()
        self.paths = [pk.dir / f for f in files]
        self.spans = [(int(o), int(o) + (int(s1) - int(s0)) * self.row_bytes)
                      for o, s0, s1 in zip(self.offsets, self.starts[:-1], self.starts[1:])]

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (global) -> int16 [n, words]."""

        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        at = self.offsets[shard] + (flat - self.starts[shard]) * self.row_bytes
        where = self.fidx[shard]
        out = np.empty((len(flat), self.row_bytes), dtype=np.uint8)
        cols = np.arange(self.row_bytes)
        for f in np.unique(where):
            sel = np.nonzero(where == f)[0]
            out[sel] = self.maps[f][at[sel, None] + cols]
        return out.view(np.int16)

    def prefetch(self, workers: int = 8) -> float:
        import time
        from concurrent.futures import ThreadPoolExecutor

        def touch(job) -> None:
            f, (b, e) = job
            mm = self.maps[f]
            step = 64 << 20
            for i in range(b, e, step):
                np.asarray(mm[i:min(e, i + step)]).sum(dtype=np.uint64)

        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(touch, list(zip(self.fidx.tolist(), self.spans))))
        return time.time() - t0


def stage_ple(table: NgramTable, sc: Scratch, ids: np.ndarray) -> None:
    rows = table.gather(ids)
    n = rows.shape[0]
    sc.ple_host[:n].numpy()[:] = rows
    sc.ple_dev[:n].copy_(sc.ple_host[:n], non_blocking=True)


# -- routed experts ------------------------------------------------------------------------------------------
def expert_table(pk: Pack, prefix: str, count: int, shared: str, device) -> x3experts.Exl3RoutedExperts:
    """Experts ``prefix.{0..count-1}`` and the shared expert (as expert ``count``): every trellis read in one pass
    per file into one GPU buffer, referenced in place by the grouped kernel."""

    names = [f"{prefix}.{e}" for e in range(count)] + [shared]
    projs = ("gate_proj", "up_proj", "down_proj")
    parts = ("trellis", "suh", "svh")
    entries = {}
    for nm in names:
        for p in projs:
            for part in parts:
                key = f"{nm}.{p}.{part}"
                entries[key] = pk.entry(key)
    by_file: dict[str, list[str]] = {}
    for key, (file, *_rest) in entries.items():
        by_file.setdefault(file, []).append(key)
    trellis_keys = [k for k in entries if k.endswith(".trellis")]
    place, total = {}, 0
    for k in trellis_keys:
        place[k] = total
        total += -(-(entries[k][2] - entries[k][1]) // 256) * 256
    big = torch.empty((total,), dtype=torch.uint8, device=device)
    small: dict[str, torch.Tensor] = {}
    for file, keys in by_file.items():
        keys.sort(key=lambda k: entries[k][1])
        # read runs of at most ~2 GB spanning these tensors (they are laid out together in a shard)
        run: list[str] = []

        def flush(run: list[str]) -> None:
            if not run:
                return
            b0, b1 = entries[run[0]][1], max(entries[k][2] for k in run)
            host = pk.read(file, b0, b1)
            dev = host.to(device)
            for k in run:
                _, b, e, dtype, shape = entries[k]
                if k.endswith(".trellis"):
                    big[place[k]:place[k] + (e - b)].copy_(dev[b - b0:e - b0])
                else:
                    small[k] = dev[b - b0:e - b0].clone().view(_DT[dtype]).reshape(shape)
            del dev, host

        for k in keys:
            # a run spans at most ~2 GB and never skips more than 16 MB of other tensors
            if run and (entries[k][2] - entries[run[0]][1] > (2 << 30)
                        or entries[k][1] - max(entries[j][2] for j in run[-4:]) > (16 << 20)):
                flush(run)
                run = []
            run.append(k)
        flush(run)

    def trellis(k: str) -> torch.Tensor:
        _, b, e, dtype, shape = entries[k]
        if dtype != "I16":
            raise ValueError(f"{k}: trellis dtype {dtype}")
        return big[place[k]:place[k] + (e - b)].view(torch.int16).view(shape)

    cb = pk.codebook(f"{names[0]}.gate_proj")
    lists = {p: [(trellis(f"{nm}.{p}.trellis"), small[f"{nm}.{p}.suh"], small[f"{nm}.{p}.svh"]) for nm in names]
             for p in projs}
    ex = x3experts.prepare(lists["gate_proj"], lists["up_proj"], lists["down_proj"], cb, device=device)
    ex.keep.append(big)
    return ex


# -- the loader --------------------------------------------------------------------------------------------
def centred_offset(pk: Pack, names: list[str]) -> float:
    """1.0 when the pack stores the centred norms as gamma - 1 (every HF EXL3 pack seen), 0.0 when as gamma."""

    means = np.array([float(pk.get(n).float().mean()) for n in names if pk.has(n)])
    if not len(means):
        return 1.0
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    if around_zero == around_one:
        raise ValueError(f"cannot tell how the pack stores its norm weights (median mean {np.median(means):.3f})")
    return 1.0 if around_zero else 0.0


def requant_rows(head: X3, ids: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The head's rows ``ids`` (original basis, decoded through the EXL3 linear itself) as MLX 4-bit groups of 32:
    the MTP drafts' head over the draft vocabulary. Drafts change speed only, never the output."""

    k = head.k
    rows = torch.empty((len(ids), k), dtype=torch.float32, device=device)
    eye = torch.eye(ROWS, dtype=torch.bfloat16, device=device)
    out = torch.empty((ROWS, head.n), dtype=torch.float32, device=device)
    for k0 in range(0, k, ROWS):
        x = torch.zeros((ROWS, k), dtype=torch.bfloat16, device=device)
        x[:, k0:k0 + ROWS] = eye
        head(x, out)
        rows[:, k0:k0 + ROWS] = out[:, ids].t()
    g = rows.view(len(ids), k // 32, 32)
    lo = g.amin(dim=-1)
    hi = g.amax(dim=-1)
    scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
    bias = lo.to(torch.bfloat16)
    q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15).to(torch.int64)
    q = q.view(len(ids), k // 8, 8)
    words = (q << (torch.arange(8, device=device, dtype=torch.int64) * 4)).sum(dim=-1)
    words = (words & 0xFFFFFFFF).to(torch.int64)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.contiguous(), scale.contiguous(), bias.contiguous()


def load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True, tp: tuple[int, int] | None = None,
         draft_vocab: int | str | None = None):
    import time

    from .qmm import make_q4
    from .weights import GDNW, HC, AttnW, Config, LayerW, MoEW, MTPW, PLEW, Weights, draft_token_ids

    if tp is not None and tp[1] > 1:
        raise ValueError("EXL3 checkpoints of Flash Next run on one GPU (tensor parallel reads the MLX checkpoint)")
    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    pk = Pack(model_dir)
    sc = Scratch(cfg.top_k + 1)
    T = "model.language_model."
    t0 = time.time()
    offset = centred_offset(pk, [f"{T}layers.{i}.attn_hyper_connection.hc_norm.weight" for i in range(cfg.layers)])

    def plain(name: str) -> torch.Tensor:
        return pk.get(name).to(device)

    def centred(name: str) -> torch.Tensor:
        return (pk.get(name).float() + offset).to(device).contiguous()

    def hc(name: str, inject: bool) -> HC:
        rows = [pk.get(name + ".input_mix_weight_down.weight")]
        if inject:
            rows.append(pk.get(name + ".block_inject_weight.weight"))
        return HC(f16(sc, rows, device), f16(sc, [pk.get(name + ".input_mix_weight_up.weight")], device),
                  centred(name + ".hc_norm.weight"), inject)

    def moe(name: str) -> MoEW:
        router = torch.cat([pk.get(name + ".gate.weight").to(torch.bfloat16),
                            pk.get(name + ".shared_expert_gate.weight").to(torch.bfloat16)]).to(device).contiguous()
        return MoEW(router, expert_table(pk, name + ".experts", cfg.experts, name + ".shared_expert", device))

    def attention(name: str) -> AttnW:
        proj = stack(sc, [x3(sc, pk, name + p, device) for p in (".q_proj", ".k_proj", ".v_proj",
                                                                  ".indexer.index_qk_proj")])
        return AttnW(proj, centred(name + ".q_norm.weight"), centred(name + ".k_norm.weight"),
                     centred(name + ".indexer.q_layernorm.weight"), centred(name + ".indexer.k_layernorm.weight"),
                     x3(sc, pk, name + ".o_proj", device))

    def gdn(name: str) -> GDNW:
        proj = stack(sc, [x3(sc, pk, name + ".in_proj_qkv", device), x3(sc, pk, name + ".in_proj_z", device),
                          f16(sc, [pk.get(name + ".in_proj_b.weight"), pk.get(name + ".in_proj_a.weight")], device)])
        conv = plain(name + ".conv1d.weight").reshape(cfg.conv_dim, cfg.conv_kernel).to(torch.bfloat16).contiguous()
        return GDNW(proj, conv, plain(name + ".A_log").float().contiguous(),
                    plain(name + ".dt_bias").float().contiguous(),
                    plain(name + ".norm.weight").to(torch.bfloat16).contiguous(),
                    x3(sc, pk, name + ".out_proj", device))

    def ple_layer(name: str, ple_index: int) -> PLEW:
        ngram = cfg.ngram(ple_index)
        base = name + ".ple_embedding.ngram_embedding."
        table = NgramTable(pk, base, cfg.ngram_shards, device)
        ngram.check(table.multipliers, table.head_offsets, table.head_sizes)
        if table.rows != ngram.rows:
            raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
        if table.dh != ngram.dims:
            raise ValueError(f"n-gram rows of {table.dh} values, the config gives {ngram.dims}")
        conv = plain(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, f16(sc, [pk.get(name + ".key_proj.weight")], device),
                    f16(sc, [pk.get(name + ".value_proj.weight")], device), centred(name + ".norm_key.weight"),
                    centred(name + ".norm_query.weight"), centred(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def layer(i: int, base: str, kind: str, with_ple: bool) -> LayerW:
        linear = kind == "linear"
        entry = LayerW(i, linear, hc(base + ".attn_hyper_connection", True), hc(base + ".mlp_hyper_connection", True),
                       gdn(base + ".linear_attn") if linear else None,
                       None if linear else attention(base + ".self_attn"), moe(base + ".mlp"))
        if with_ple and i in cfg.ple_layers:
            entry.ple = ple_layer(base + ".ple", cfg.ple_layers.index(i))
        return entry

    embed = plain(T + "embed_tokens.weight")
    if embed.dtype not in (torch.bfloat16, torch.float16):
        embed = embed.to(torch.bfloat16)
    loaded = []
    for i in range(cfg.layers):
        loaded.append(layer(i, f"{T}layers.{i}", cfg.layer_types[i], True))
        pk.release()
        torch.cuda.empty_cache()
    mixer = hc(T + "hyper_connection_mixer", False)
    head = x3(sc, pk, "lm_head", device)
    w = Weights(cfg, (embed.contiguous(),), loaded, mixer, head, _inv_freq(cfg, device), around_one=True)
    w.quant = "exl3"
    w.meta.update(rank=0, world=1, vocab_offset=0, full=cfg, centred_offset=offset)
    ids = draft_token_ids(draft_vocab)
    if mtp and pk.has("mtp.fc_embedding.trellis"):
        w.mtp = MTPW(centred("mtp.pre_fc_norm_embedding.weight"), centred("mtp.pre_fc_norm_hidden.weight"),
                     x3(sc, pk, "mtp.fc_embedding", device), x3(sc, pk, "mtp.fc_hidden", device),
                     layer(-1, "mtp.layers.0", "attention", False), hc("mtp.hyper_connection_mixer", False))
    ple = next((lay.ple for lay in loaded if lay.ple is not None), None)
    sc.allocate(device, experts=loaded[0].moe.experts, ple_words=ple.table.words if ple else 0,
                ple_heads=ple.ngram.heads if ple else 0, ple_dim=cfg.ple_dim)
    w.x3 = sc
    if ids is not None and w.mtp is not None:
        ids = ids[ids < cfg.vocab]
        w.draft_ids = torch.from_numpy(ids).to(device)
        w.draft_head = make_q4(*requant_rows(head, w.draft_ids, device))
    pk.release()
    torch.cuda.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w


def _inv_freq(cfg, device) -> torch.Tensor:
    inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (
        -torch.arange(0, cfg.rotary_dim // 2, dtype=torch.float64) / (cfg.rotary_dim // 2))
    return inv.to(torch.float32).to(device)
