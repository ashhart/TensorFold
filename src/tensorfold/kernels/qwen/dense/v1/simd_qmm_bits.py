"""simd_qmm's row-exact arithmetic for 5-, 6- and 8-bit codes in groups of 64 (oQ formats, MLX 8-bit): the matrix
units from two rows, a scalar twin for one, every row's bits the same at any width.

A group's P is simd_qmm's MMA chain with integer codes and unscaled inputs: step s, k takes code 8 k + s times
x[64 g + 8 k + s], an fp32 fma chain over k, steps in order; then fma(bias, xsum, fma(scale, P, acc)) and the same
chunk tree. install() checks the twin against the matrix kernel per shape; a shape that differs takes affine_rows.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

from tensorfold.kernels import threads
from tensorfold.kernels.qwen.dense.v1 import simd_qmm

BITS = (5, 6, 8)
GROUP = 64
fallback: set[tuple[int, int, int]] = set()     # (n, k, bits) whose twin differs from the matrix kernel here

_HEADER = simd_qmm._HEADER + r"""
// code j of B-bit codes packed from bit 0 of v (j a compile-time constant after unrolling)
template <int B>
inline uint code_at(const thread uint* v, const int j) {
  const int bit = j * B, word = bit >> 5, shift = bit & 31;
  uint c = v[word] >> shift;
  if (shift + B > 32) c |= v[word + 1] << (32 - shift);
  return c & ((1u << B) - 1u);
}
// float(c) for c < 2^23 without a convert: the same value
inline float cf(uint c) { return as_type<float>(0x4B000000u | c) - 8388608.0f; }
"""

_MMA = r"""
  // R rows: threadgroup (x, y) takes rows 8 RT y .. in RT tiles of 8 (rows >= R read row R - 1, dropped); SGS
  // simdgroups compute the S chunks in order. A lane's A elements are codes 8 fn .. 8 fn + 15 of its group.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int G = K / 64, WPR = K * B / 32, LW = B == 8 ? 4 : 3;
  const float one = ONE[0];
  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
  for (int c = sg; c < S; c += SGS) {
    float acc[RT][NT][2];
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    for (int g = c; g < G; g += S) {
      const int bit0 = 64 * B * g + 8 * B * fn;
      const bool half_word = (bit0 & 31) != 0;               // 5-bit lanes fn = 2, 6 start 16 bits into a word
      uint v[NT][4];
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        const device uint* wp = W + size_t(wrow[t]) * WPR + (bit0 >> 5);
        uint w[4];
        PRAGMA_UNROLL
        for (int i = 0; i < 4; i++) w[i] = i < LW ? wp[i] : 0u;
        v[t][0] = half_word ? (w[0] >> 16) | (w[1] << 16) : w[0];
        v[t][1] = half_word ? (w[1] >> 16) | (w[2] << 16) : w[1];
        v[t][2] = half_word ? (w[2] >> 16) : w[2];
        v[t][3] = w[3];
      }
      uint4 xa[RT], xb[RT];
      float xs0[RT], xs1[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        xa[rt] = LOAD8(xr0[rt], 8 * g + fm);
        xb[rt] = LOAD8(xr1[rt], 8 * g + fm);
        float a = sum8(xa[rt], one), u = sum8(xb[rt], one);
        a = fma(simd_shuffle_xor(a, ushort(2)), one, a); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
        a = fma(simd_shuffle_xor(a, ushort(4)), one, a); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
        a = fma(simd_shuffle_xor(a, ushort(16)), one, a); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
        xs0[rt] = a; xs1[rt] = u;
      }
      simdgroup_matrix<float, 8, 8> P[RT][NT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) {
        simdgroup_matrix<float, 8, 8> bm[RT];
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = bf8(xa[rt], s);
          bm[rt].thread_elements()[1] = bf8(xb[rt], s);
        }
        PRAGMA_UNROLL
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          am.thread_elements()[0] = cf(code_at<B>(v[t], s));
          am.thread_elements()[1] = cf(code_at<B>(v[t], 8 + s));
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
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SGS * 32) {
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

_SCALAR = r"""
  // RS rows (1 to 4). Lane (chunk c = lane % S, slot j = lane / S) runs chunk c of NR outputs n0 + j + (32 / S) u,
  // a whole group (2 B words) of each in registers; the threadgroup stages XB groups of each row's inputs in chain
  // order (step s, k at 8 s + k). A row's chain is the same at any RS and the matrix kernel's.
  constexpr int GW = 2 * B, XP = 76, G = K / 64, WPR = K * B / 32;
  threadgroup float xs[RS * XB * XP];
  const uint lane = thread_index_in_simdgroup;
  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
  const int c = int(lane) % S;
  constexpr int SLOTS = 32 / S;
  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
                 + int(lane) / S;
  const float one = ONE[0];
  const device uint* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR][RS];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = W + size_t(nn) * WPR;
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) acc[u][r] = 0.0f;
  }
  uint nw[NR][GW];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) for (int h = 0; h < GW; h++) nw[u][h] = c < G ? wr[u][GW * c + h] : 0u;
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < RS * nbk * 8; idx += SGS * 32) {
      const int r = RS == 1 ? 0 : idx / (nbk * 8);
      const int gl = (RS == 1 ? idx : idx - r * (nbk * 8)) / 8, j = idx % 8;
      const uint4 v = LOAD8(r, 8 * (b0 + gl) + j);
      threadgroup float* xr = xs + r * (XB * XP) + gl * XP;
      PRAGMA_UNROLL
      for (int e = 0; e < 8; e++) xr[8 * e + j] = bf8(v, e);
      xr[64 + j] = sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint wv[NR][GW];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) for (int h = 0; h < GW; h++) wv[u][h] = nw[u][h];
      if (g + S < G) {
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) for (int h = 0; h < GW; h++) nw[u][h] = wr[u][GW * (g + S) + h];
      }
      float xsum[RS];
      float P[NR][RS];
      PRAGMA_UNROLL
      for (int r = 0; r < RS; r++) {
        const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP;
        const float4 p0 = *(const threadgroup float4*)(xg + 64), p1 = *(const threadgroup float4*)(xg + 68);
        xsum[r] = fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x));
        xsum[r] = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one, xsum[r]);
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) P[u][r] = 0.0f;
      }
      PRAGMA_UNROLL
      for (int s = 0; s < 8; s++) {
        float xq[RS][8];
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP + 8 * s;
          const float4 lo = *(const threadgroup float4*)(xg), hi = *(const threadgroup float4*)(xg + 4);
          xq[r][0] = lo.x; xq[r][1] = lo.y; xq[r][2] = lo.z; xq[r][3] = lo.w;
          xq[r][4] = hi.x; xq[r][5] = hi.y; xq[r][6] = hi.z; xq[r][7] = hi.w;
        }
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++)
          PRAGMA_UNROLL
          for (int k = 0; k < 8; k++) {
            const float q = cf(code_at<B>(wv[u], 8 * k + s));
            PRAGMA_UNROLL
            for (int r = 0; r < RS; r++) P[u][r] = fma(xq[r][k], q, P[u][r]);
          }
      }
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) {
        const float sc = float(sr[u][g]), bi = float(br[u][g]);
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          acc[u][r] = fma(sc, P[u][r], acc[u][r]);
          acc[u][r] = fma(bi, xsum[r], acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++)
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) {
      float v = acc[u][r];
      PRAGMA_UNROLL
      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
      const int n = n0 + SLOTS * u;
      if (n < N && c == 0) OUT[size_t(r) * N + n] = bfloat(v);
    }
"""

_kernels: dict[tuple, Any] = {}
_plans: dict[tuple, Any] = {}


def _compiled(kind: str, consts: tuple[tuple[str, int], ...]) -> Any:
    kernel = _kernels.get((kind, consts))
    if kernel is None:
        source = ("".join(f"  constexpr int {k} = {v};\n" for k, v in consts)
                  + f"  #define LOAD8(r, j) ({simd_qmm._DEFAULT.load8})\n" + {"scalar": _SCALAR, "mma": _MMA}[kind]
                  + "  #undef LOAD8\n")
        name = f"simd_qmm_bits_{kind}_" + hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        kernel = _kernels[(kind, consts)] = mx.fast.metal_kernel(
            name=name, input_names=["X", "W", "SC", "BI", "ONE"], output_names=["OUT"], source=source, header=_HEADER)
    return kernel


def fits(weight: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int) -> bool:
    """5/6/8-bit codes in groups of 64 with bf16 scales and biases, outputs in eights, not a shape that fell back."""

    if bits not in BITS or group_size != GROUP or scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        return False
    n, k = int(weight.shape[0]), int(weight.shape[1]) * 32 // bits
    return weight.ndim == 2 and n % 8 == 0 and k % GROUP == 0 and (n, k, bits) not in fallback


def _launch(kind: str, rows: int, n: int, dims: int, bits: int, most: int = simd_qmm.MMA_SGS) -> tuple:
    s = simd_qmm.splits(n, dims)
    if kind == "scalar":
        xb = simd_qmm.scalar_block(rows, s, GROUP)
        assert xb
        nr = 1 if bits == 8 or n <= 2048 else simd_qmm.NR
        sgs = max(1, 16 // ((32 // s) * nr)) if n > 2048 else 8
        per = sgs * (32 // s) * nr
        consts = (("K", dims), ("N", n), ("S", s), ("SGS", sgs), ("NR", nr), ("XB", xb), ("RS", rows), ("B", bits))
        return consts, (-(-n // per) * sgs * 32, 1, 1), (sgs * 32, 1, 1), [(rows, n)]
    rt = min(simd_qmm.RT_MAX, (rows + 7) // 8)
    nt = simd_qmm.tiles(n, rt * 8, s)
    sgs = min(s, most)
    consts = (("K", dims), ("N", n), ("S", s), ("SGS", sgs), ("NT", nt), ("RT", rt), ("B", bits))
    return consts, (-(-n // (8 * nt)) * sgs * 32, -(-rows // (8 * rt)), 1), (sgs * 32, 1, 1), [(rows, n)]


def _go(kind: str, plan: tuple, inputs: list) -> mx.array:
    consts, grid, tg, oshape = plan
    return _compiled(kind, consts)(inputs=inputs, grid=grid, threadgroup=tg, output_shapes=oshape,
                                   output_dtypes=[mx.bfloat16])[0]


def _run(kind: str, rows: int, n: int, dims: int, bits: int, inputs: list) -> mx.array:
    key = (kind, rows, n, dims, bits)
    plan = _plans.get(key)
    if plan is not None:
        return _go(kind, plan, inputs)
    if kind == "scalar":
        plan = _plans[key] = _launch(kind, rows, n, dims, bits)
        return _go(kind, plan, inputs)
    consts = _launch(kind, rows, n, dims, bits)[0]
    made: list[tuple] = []

    def launch(size: int) -> mx.array:
        made.append(_launch(kind, rows, n, dims, bits, size // 32))
        return _go(kind, made[-1], inputs)

    pipeline = ("simd_qmm_bits", tuple(c for c in consts if c[0] != "SGS"))
    out = threads.fit(pipeline, [32 * g for g in (16, 8, 4, 2, 1) if g <= dict(consts)["SGS"]], launch, inputs)
    if threads.fitted(pipeline) is not None:
        _plans[key] = made[-1]
    return out


def qmm(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, bits: int, *,
        kind: str | None = None) -> mx.array:
    """Row-exact bf16 x @ W.T for 5/6/8-bit codes in groups of 64: every row's bits its one-row call's."""

    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    if kind is None:     # the twin for one row; from two, the matrix kernel (cheaper here than the twin's two rows)
        kind = "scalar" if rows == 1 and simd_qmm.scalar_block(rows, simd_qmm.splits(n, dims), GROUP) else "mma"
    one = mx.array([1.0], dtype=mx.float32)
    return _run(kind, rows, n, dims, bits, [x2, weight, scales, biases, one]).reshape(*shape[:-1], n)


def check(weight: mx.array, scales: mx.array, biases: mx.array, bits: int, *, seed: int = 0) -> bool:
    """The scalar twin's 1-4-row calls against the matrix kernel's rows, bit for bit, for this weight here."""

    k, n = int(weight.shape[1]) * 32 // bits, int(weight.shape[0])
    x = (mx.random.normal((8, k), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    full = qmm(x, weight, scales, biases, bits, kind="mma")
    s = simd_qmm.splits(n, k)
    calls = [(r, 1) for r in range(8)]
    calls += [(r, m) for m in range(2, simd_qmm.SCALAR_ROWS + 1) if simd_qmm.scalar_block(m, s, GROUP) for r in (0, 8 - m)]
    return all(bool(mx.array_equal(qmm(x[r:r + m], weight, scales, biases, bits, kind="scalar"), full[r:r + m]).item())
               for r, m in calls)


__all__ = ["BITS", "GROUP", "check", "fallback", "fits", "qmm"]
