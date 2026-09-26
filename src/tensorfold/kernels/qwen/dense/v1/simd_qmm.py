"""A 4-bit matmul for 1 to 16 rows on Apple GPUs without the M5's tensor units (M1 to M4), whose rows have the
same bits at any row count.

Without the M5's tensor units a verify window's rows otherwise go through MLX's kernels, where a row's bits depend
on how many rows ride with it, or through ``row_qmv``, where each extra row costs about a third of a one-row step
(2026-09-26, M3 Ultra).

Arithmetic, the same for every row count: per 64-input group g, P = sum of x * q as one FMA chain in the order
(s = 0..7, i = 0..7) over k = 8 i + s (word i of the group, nibble s); the group's input sum by a fixed tree (eight
left-to-right sums of 8 inputs, then pairs); then acc = fma(scale, P, acc) and acc = fma(bias, sum, acc). The groups
go to S interleaved chunks (chunk c takes groups c, c + S, ...), each its own accumulator, and the chunks combine by
a fixed pairwise tree. S depends on the shape only.

Two kernels compute exactly that:

- ``mma`` (2 to 16 rows): simdgroup_multiply_accumulate on 8x8 fp32 tiles. On an M3 Ultra an fp32 MMA equals the
  forward FMA chain d = fma(a7, b7, ... fma(a0, b0, c)) bit for bit (2026-09-26, 5.12M outputs), so an MMA step is
  eight links of P's chain. The product is D^T = W X^T: lane (fm, fn) holds words fn and fn + 1 of weight row
  n + fm (8 bytes a group, each word loaded once a simdgroup, no shuffles) and inputs x[row fn / fn + 1][8 fm ..
  8 fm + 7]; step s multiplies nibble s of each word. Nibbles enter unshifted, q' = word & (0xF << 4 s), against
  x' = x * 2^-4s, so x' q' = x q exactly. No threadgroup staging or barriers in the K loop; a threadgroup is S
  simdgroups (one a chunk) that combine through threadgroup memory at the end.
- ``scalar`` (1 row): lane (chunk c, output j) holds whole groups of its outputs' weights in registers and runs
  the same chain with scalar FMAs, reading pre-scaled inputs that the threadgroup stages in chain order in
  threadgroup memory; chunks combine by butterfly shuffles in the same tree. Serial decoding goes through it, so it
  streams weights like MLX's qmv (M3 Ultra, 2026-09-26, dependent chains: 17408x5120 79.6 us vs MLX 77.3,
  248320x5120 933 vs 933) instead of paying for an 8-row MMA tile.

A chip whose MMA is not that FMA chain would give the two kernels different bits: ``install`` checks every shape
at load and sends one-row calls of any shape where they differ through the MMA kernel. Adds the compiler could
reassociate under fast math are written as fma(a, one, b) with ``one`` read from a buffer. Constants go into each
kernel's source rather than template arguments: MLX runs a std::regex over template arguments on every call.
Inputs need K % 64 == 0 (any number of groups) and N % 8 == 0.
"""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple, Sequence

import mlx.core as mx

MAX_ROWS = 1 << 16   # rows a call routed here (prompt chunks included); every row's bits are its one-row bits
RT_MAX = 2           # 8-row tiles a threadgroup: more rows than 8 RT_MAX spread over the grid's y axis
GROUP = 64

_HEADER = r"""
#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
// the bf16 at index e (0..7) of 8 packed bf16 as fp32
inline float bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
// a row's 8 inputs summed left to right
inline float sum8(uint4 v, float one) {
  float t = bf8(v, 0);
  for (int e = 1; e < 8; e++) t = fma(bf8(v, e), one, t);
  return t;
}
// 2^-4s
inline float pre(int s) { return as_type<float>(uint(127 - 4 * s) << 23); }
"""

_SCALAR = r"""
  // one row. Threadgroup of SGS simdgroups; lane (chunk c = lane % S, slot j = lane / S) runs chunk c of NR outputs
  // n0 + j + (32 / S) u, a whole group (8 words) of each in registers, the next group's loaded before this one is
  // used. The threadgroup stages XB groups of inputs at a time in threadgroup memory, pre-scaled and in chain order
  // (x'[g][8 s + i] = x[64 g + 8 i + s] * 2^-4s), with each group's 8 left-to-right sums. (Loading each block of a
  // row contiguously and swapping halves between lane pairs gave the same bits and ran 1.4% slower, 2026-09-26.)
  constexpr int XP = 76;                        // floats a staged group: 64 inputs, 8 sums, 4 pad
  threadgroup float xs[XB * XP];
  const uint lane = thread_index_in_simdgroup;
  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
  const int c = int(lane) % S;
  constexpr int SLOTS = 32 / S;
  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
                 + int(lane) / S;
  constexpr int G = K / 64;
  const float one = ONE[0];
  const device uint4* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = (const device uint4*)(W + size_t(nn) * (K / 8));
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    acc[u] = 0.0f;
  }
  uint4 na[NR], nb[NR];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) { na[u] = uint4(0); nb[u] = uint4(0); }
  if (c < G) {
    PRAGMA_UNROLL
    for (int u = 0; u < NR; u++) { na[u] = wr[u][2 * c]; nb[u] = wr[u][2 * c + 1]; }
  }
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < nbk * 8; idx += SGS * 32) {
      const int gl = idx / 8, i = idx % 8;
      const uint4 v = LOAD8(0, 8 * (b0 + gl) + i);
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) xs[gl * XP + 8 * s + i] = bf8(v, s) * pre(s);
      xs[gl * XP + 64 + i] = sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint4 wa[NR], wb[NR];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) { wa[u] = na[u]; wb[u] = nb[u]; }
      if (g + S < G) {
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) { na[u] = wr[u][2 * (g + S)]; nb[u] = wr[u][2 * (g + S) + 1]; }
      }
      const threadgroup float* xg = xs + (g - b0) * XP;
      const float4 p0 = *(const threadgroup float4*)(xg + 64), p1 = *(const threadgroup float4*)(xg + 68);
      const float xsum = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one,
                             fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x)));
      float P[NR];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) P[u] = 0.0f;
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) {
        const uint mask = 0xFu << (4 * s);
        const float4 lo = *(const threadgroup float4*)(xg + 8 * s), hi = *(const threadgroup float4*)(xg + 8 * s + 4);
        const float xq[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) {
          const uint wd[8] = {wa[u].x, wa[u].y, wa[u].z, wa[u].w, wb[u].x, wb[u].y, wb[u].z, wb[u].w};
          PRAGMA_UNROLL
          for (int i = 0; i < 8; i++) P[u] = fma(xq[i], float(wd[i] & mask), P[u]);
        }
      }
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) {
        acc[u] = fma(float(sr[u][g]), P[u], acc[u]);
        acc[u] = fma(float(br[u][g]), xsum, acc[u]);
      }
    }
  }
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    float v = acc[u];
    PRAGMA_UNROLL
    for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
    const int n = n0 + SLOTS * u;
    if (n < N && c == 0) OUT[n] = bfloat(v);
  }
"""

_MMA = r"""
  // R rows: threadgroup (x, y) takes rows 8 RT y .. 8 RT y + 8 RT - 1 in RT tiles of 8 (rows >= R read row R - 1;
  // their results are dropped) and S simdgroups, simdgroup c running chunk c for NT tiles of 8 outputs.
  const uint lane = thread_index_in_simdgroup;
  const int c = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int G = K / 64;
  const float one = ONE[0];
  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  const device uint2* W2 = (const device uint2*)W;
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
  float acc[RT][NT][2];
  for (int rt = 0; rt < RT; rt++)
    for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
  for (int g = c; g < G; g += S) {
    uint2 wv[NT];
    PRAGMA_UNROLL
    for (int t = 0; t < NT; t++) wv[t] = W2[size_t(wrow[t]) * (K / 16) + 4 * g + fn / 2];
    uint4 xa[RT], xb[RT];
    float xs0[RT], xs1[RT];
    PRAGMA_UNROLL
    for (int rt = 0; rt < RT; rt++) {
      xa[rt] = LOAD8(xr0[rt], 8 * g + fm);
      xb[rt] = LOAD8(xr1[rt], 8 * g + fm);
      float v = sum8(xa[rt], one), u = sum8(xb[rt], one);
      v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
      v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
      v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
      xs0[rt] = v; xs1[rt] = u;
    }
    simdgroup_matrix<float, 8, 8> P[RT][NT];
    PRAGMA_UNROLL
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    PRAGMA_UNROLL
    for (int s = 0; s < 8; s++) {
      const float ps = pre(s);
      const uint mask = 0xFu << (4 * s);
      simdgroup_matrix<float, 8, 8> bm[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        bm[rt].thread_elements()[0] = bf8(xa[rt], s) * ps;
        bm[rt].thread_elements()[1] = bf8(xb[rt], s) * ps;
      }
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        simdgroup_matrix<float, 8, 8> am;
        am.thread_elements()[0] = float(wv[t].x & mask);
        am.thread_elements()[1] = float(wv[t].y & mask);
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
      }
    }
    PRAGMA_UNROLL
    for (int t = 0; t < NT; t++) {
      const float sc = float(SC[size_t(wrow[t]) * G + g]);
      const float bi = float(BI[size_t(wrow[t]) * G + g]);
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
        acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
      }
    }
  }
  if (S == 1) {
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++)
        for (int e = 0; e < 2; e++) {
          const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
          if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
        }
    return;
  }
  for (int rt = 0; rt < RT; rt++)
    for (int t = 0; t < NT; t++)
      for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = c * 32 + int(lane); idx < RT * NT * 64; idx += S * 32) {
    float v[S];
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    for (int w = 1; w < S; w *= 2)
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
  }
"""

SGS = 2          # simdgroups a threadgroup in the scalar kernel
NR = 2           # outputs a lane in the scalar kernel
XB = 32          # scalar kernel: groups of inputs staged at a time
_kernels: dict[tuple, Any] = {}
_plans: dict[tuple, Any] = {}
_BF16 = [mx.bfloat16]
_one: Any = None
_ORIG: Any = None
enabled = False
# shapes (n, k) whose one-row calls must go through the MMA kernel (its bits differ from the scalar kernel's there)
mma_one_row: set[tuple[int, int]] = set()


class Prologue(NamedTuple):
    """Computes the kernels' inputs where they are loaded (a norm or activation folded into the matmul).

    ``load8`` is a Metal expression for ``LOAD8(r, j)``: the 8 inputs x[r][8 j .. 8 j + 7] as packed bf16 (uint4),
    bit for bit what the unfused op would have stored, from ``X`` and the extra inputs named in ``inputs`` (passed
    to ``qmm`` as ``extra``, in order). ``header`` holds its helper functions; ``name`` goes into the kernel's
    name. Everything after the load is unchanged, so a row's bits are the unfused call's. The one-row kernel runs
    it once per threadgroup for each input it stages (one row of K), the multi-row kernel once per simdgroup for
    each input it reads, so an expensive prologue costs N / 16 (one row) or N / (8 NT) (windows) evaluations a
    input."""

    name: str
    load8: str
    inputs: tuple[str, ...] = ()
    header: str = ""


_DEFAULT = Prologue("x", "(((const device uint4*)X)[size_t(r) * (K / 8) + (j)])")


def _compiled(kind: str, consts: tuple[tuple[str, int], ...], dep: bool = False, prologue: Prologue = _DEFAULT
              ) -> Any:
    """One kernel a (kind, constants, prologue): the constants are written into the source instead of passed as
    template arguments, because MLX runs a std::regex over template arguments on every call (~7 us of CPU a call)."""

    key = (kind, consts, dep, prologue.name)
    kernel = _kernels.get(key)
    if kernel is None:
        source = ("".join(f"  constexpr int {k} = {v};\n" for k, v in consts)
                  + f"  #define LOAD8(r, j) ({prologue.load8})\n" + (_SCALAR if kind == "scalar" else _MMA)
                  + "  #undef LOAD8\n")
        header = _HEADER + prologue.header
        name = (f"simd_qmm_{kind}_{prologue.name}_" + hashlib.sha256((header + source).encode()).hexdigest()[:16]
                + ("_dep" if dep else ""))
        inputs = ["X", "W", "SC", "BI", "ONE", *prologue.inputs] + (["DEP"] if dep else [])
        kernel = _kernels[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["OUT"],
                                                      source=source, header=header)
    return kernel


def splits(n: int, k: int) -> int:
    """Chunks the K groups split into for an [n, k] weight: a function of the shape only (it sets the bits). Small
    outputs get more chunks for parallelism, so stacking projections into one weight can change their bits."""

    return 32 if n <= 64 else (16 if n <= 2048 else 8)


def tiles(n: int, rows: int, s: int) -> int:
    """Tiles of 8 outputs a simdgroup in the MMA kernel (speed only: no effect on the bits), at most 16 KB of
    threadgroup memory for the split reduction."""

    nt = 4 if n % 32 == 0 else (2 if n % 16 == 0 else 1)
    while nt > 1 and s * ((rows + 7) // 8) * nt * 64 * 4 > 16384:
        nt //= 2
    return nt


def fits(module: Any) -> bool:
    """Whether a quantized linear has the layout the kernels read: 4-bit affine, groups of 64, bf16 scales,
    inputs a multiple of 64 and outputs a multiple of 8."""

    weight = module["weight"]
    return (module.bits == 4 and module.group_size == GROUP and module["scales"].dtype == mx.bfloat16
            and weight.ndim == 2 and (int(weight.shape[1]) * 8) % GROUP == 0 and int(weight.shape[0]) % 8 == 0
            and getattr(module, "mode", "affine") == "affine")


def _launch(kind: str, rows: int, n: int, dims: int) -> tuple:
    """(constants, grid, threadgroup, output shapes) of one call: constants depend on the shape and on the row tile
    count only, so a new row count never compiles a kernel."""

    s = splits(n, dims)
    if kind == "scalar":
        assert XB % s == 0
        nr = NR if n > 2048 else 1
        sgs = SGS if n > 2048 else 8        # small outputs: more simdgroups share one staging of the inputs
        per = sgs * (32 // s) * nr
        consts = (("K", dims), ("N", n), ("S", s), ("SGS", sgs), ("NR", nr), ("XB", XB))
        return consts, (-(-n // per) * sgs * 32, 1, 1), (sgs * 32, 1, 1), [(1, n)]
    rt = min(RT_MAX, (rows + 7) // 8)
    nt = tiles(n, rt * 8, s)
    consts = (("K", dims), ("N", n), ("S", s), ("NT", nt), ("RT", rt))
    return consts, (-(-n // (8 * nt)) * s * 32, -(-rows // (8 * rt)), 1), (s * 32, 1, 1), [(rows, n)]


def qmm(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int = GROUP, *,
        kind: str | None = None, dep: mx.array | None = None, prologue: Prologue | None = None,
        extra: Sequence[mx.array] = ()) -> mx.array:
    """x [..., K] bf16 (at most MAX_ROWS rows) @ W.T for 4-bit weights [N, K / 8] -> [..., N] bf16; a row's bits
    do not depend on how many rows ride with it. ``kind`` forces "scalar" (one row) or "mma"; ``dep`` (timing only)
    makes the call wait for that array. ``prologue`` computes the inputs in the load from ``x`` and ``extra``
    (see ``Prologue``); ``x`` then only supplies the row count and whatever the prologue reads from ``X``."""

    global _one
    assert group_size == GROUP
    if _one is None:
        _one = mx.array([1.0], dtype=mx.float32)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    if kind is None:
        kind = "scalar" if rows == 1 and (n, dims) not in mma_one_row else "mma"
    pro = prologue or _DEFAULT
    assert len(extra) == len(pro.inputs)
    inputs = [x2, weight, scales, biases, _one, *extra] + ([dep] if dep is not None else [])
    plan = _plans.get((kind, rows, n, dims))
    if plan is None:
        plan = _plans[(kind, rows, n, dims)] = _launch(kind, rows, n, dims)
    consts, grid, tg, oshape = plan
    out = _compiled(kind, consts, dep is not None, pro)(inputs=inputs, grid=grid, threadgroup=tg,
                                                        output_shapes=oshape, output_dtypes=_BF16)[0]
    return out.reshape(*shape[:-1], n)


def check(weight: mx.array, scales: mx.array, biases: mx.array, *, seed: int = 0) -> bool:
    """Whether one-row calls through the scalar kernel give the MMA kernel's rows bit for bit for this weight."""

    k = int(weight.shape[1]) * 8
    x = (mx.random.normal((8, k), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    full = qmm(x, weight, scales, biases, kind="mma")
    return all(bool(mx.array_equal(qmm(x[r:r + 1], weight, scales, biases, kind="scalar"), full[r:r + 1]).item())
               for r in range(8))


def _plan(module: Any, rows: int) -> tuple:
    """The kernel call for ``rows`` rows of this linear, built once (the per-call Python cost matters: ~500 calls a
    forward)."""

    global _one
    if _one is None:
        _one = mx.array([1.0], dtype=mx.float32)
    weight = module["weight"]
    n, dims = int(weight.shape[0]), int(weight.shape[1]) * 8
    kind = "scalar" if rows == 1 and (n, dims) not in mma_one_row else "mma"
    consts, grid, tg, oshape = _launch(kind, rows, n, dims)
    tail = [weight, module["scales"], module["biases"], _one]
    return _compiled(kind, consts), grid, tg, oshape, n, dims, tail


def _call(self: Any, x: mx.array) -> mx.array:
    plans = self.__dict__.get("_simd_qmm") if enabled else None
    if plans is None or x.dtype != mx.bfloat16:
        return _ORIG(self, x)
    dims = x.shape[-1]
    rows = x.size // dims
    if not 1 <= rows <= MAX_ROWS:
        return _ORIG(self, x)
    p = plans.get(rows)
    if p is None:
        p = plans[rows] = _plan(self, rows)
    kernel, grid, tg, oshape, n, _, tail = p
    y = kernel(inputs=[x.reshape(rows, dims), *tail], grid=grid, threadgroup=tg, output_shapes=oshape,
               output_dtypes=_BF16)[0]
    if x.ndim != 2:
        y = y.reshape(*x.shape[:-1], n)
    if "bias" in self:
        y = y + self["bias"]
    return y


def install(model: Any) -> int:
    """Route every call of at most MAX_ROWS rows of the model's fitting 4-bit linears through ``qmm``, one-row
    calls (serial decoding) included, after checking per shape that the scalar kernel's rows are the MMA kernel's.
    Returns how many linears it covers. Idempotent."""

    global _ORIG, enabled
    import mlx.nn as nn

    if _ORIG is None:
        _ORIG = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _call
    count = 0
    checked: set[tuple[int, int]] = set()
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and fits(module):
            object.__setattr__(module, "_simd_qmm", {})
            count += 1
            shape = (int(module["weight"].shape[0]), int(module["weight"].shape[1]) * 8)
            if shape not in checked:
                checked.add(shape)
                if not check(module["weight"], module["scales"], module["biases"]):
                    mma_one_row.add(shape)
    enabled = True
    return count


__all__ = ["MAX_ROWS", "Prologue", "check", "fits", "install", "qmm", "splits"]
