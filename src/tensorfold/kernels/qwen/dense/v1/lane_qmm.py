"""Batch-invariant 4-, 3- and 2-bit projections on the M5 tensor units: the lane matmul.

Byte-exact lanes need every row of a verify window to come out bit-identical
to the same row computed alone. MLX's quantized matmul picks a different
kernel by row count (a vector kernel below ~10 rows, tensor-unit tiles above,
and other tiles for narrow layers below 64 rows), each with its own summation
order, so rows only match up to 9 and a 9-row check costs 4.3x a 1-row step.

This kernel has one arithmetic for every row count. For weight group g (64
inputs, one scale s and bias b per output column):

    P[m, n, g] = x[m, g-block] . q[n, g-block]      tensor units, bf16 x uint4 -> fp32
    y[m, n]    = sum over g, in order, of  s[n,g] * P[m,n,g] + b[n,g] * xs[m,g]

where xs[m, g] is the fp32 sum of the group's 64 inputs (sequential). The K
groups are split into SK slices per weight shape (never per row count); the
slices are added in slice order. The tensor op reads MLX's packed 4-bit weights
as they are (same nibble order), so there is no repacked copy of the weights;
scales and biases are kept once more, group-major and interleaved, so a lane
fetches its four columns' (s, b) pairs in one 16-byte load.

A row's result never depends on the other rows: rows 1..M of any call equal
the same rows computed one at a time (tested for M = 1..48 and all projection
shapes of Qwen3.8-27B). Accuracy against an fp32 reference is the same as
MLX's own kernels (bf16 output rounding dominates).

Measured on the M5 Max (2026-09-23), one 17408x5120 projection:
    rows        1      4      16
    MLX       0.095  0.191  0.257 ms
    lane      0.129  0.142  0.136 ms
The tensor-op path tops out near 400 GB/s, so one row costs ~1.35x MLX's
vector kernel, while 16 rows cost the same as one.

3- and 2-bit weights (``_MAIN_LOWBIT``): the tensor op reads 4-bit but not
3- or 2-bit operands, so each simdgroup first widens its column tile's
64-input group from MLX's packing (3-bit: 8 values in 3 bytes; 2-bit: 4 a
byte) to nibbles in threadgroup memory, then runs the same op and the same
per-group arithmetic. The widening is exact, and as at 4 bits a row's bits
depend on the weight's shape, never on the row count. Mixed checkpoints use
2-bit for some layers; a whole Qwen3.8-27B at 2 bits loses too much
(docs/recipes/qwen3.8-27b.md, "3-bit weights").
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

MAX_ROWS = 128         # rows the lane kernel accepts in one call
ROW_BLOCK = 32         # rows per threadgroup above 32 rows (one 32-row op per weight group)
NT = 32                # output columns per simdgroup tile
BITS = (2, 3, 4)       # weight widths the lane matmul takes (affine, groups of 64)

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_XSUM = r"""
  const int M = mdims[0], MP = mdims[1];
  const uint m = thread_position_in_grid.y;
  const uint g = thread_position_in_grid.x;
  if (g >= K / 64 || int(m) >= MP) return;
  float acc = 0.0f;
  if (int(m) < M) for (int i = 0; i < 64; i++) acc += float(X[m * K + g * 64 + i]);
  XS[g * MP + m] = acc;
"""

_MAIN = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / 64;
  constexpr int NF = NT / 16;
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;   // first row of this threadgroup's row block
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;

  // one op for all TMR 16-row blocks: its destination is the blocks' fragments in order, and each
  // row's bits equal the 16-row op's (tested); two 16-row ops per group cost 1.2-1.5x as much
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq, dextents<int32_t, 2>(K, N));

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint4* sbv = (const device uint4*)SBt;   // (s, b) bf16 pairs, [g][n]
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
  for (int g = g_begin; g < g_end; g++) {
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * 64, 0);
    auto b = tB.slice(g * 64, n0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    for (int t = 0; t < TMR; t++) {
      const float xs0 = XS[g * MP + rb + t * 16 + fm];
      const float xs1 = XS[g * MP + rb + t * 16 + fm + 8];
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  // K slices are added in slice order, one 16-row block at a time
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < M && n < N)
            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""

# Tiled weights (``tile_weight``): column tile t's group g is one contiguous NT x 64 block, so an
# op reads 1 KB in one piece instead of 32 pieces of 32 bytes 2.5 KB apart. Same values, same op:
# every row's bits equal the MLX-layout kernel's (tested), 8-15% faster at 1-32 rows.
_MAIN_TILED = _MAIN.replace(
    "    auto b = tB.slice(g * 64, n0);\n",
    "    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(\n"
    "        (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (NT * 32), dextents<int32_t, 2>(64, NT));\n")
assert _MAIN_TILED != _MAIN




# Weights tiled 64 columns wide (``install(wide=True)``), up to 16 rows: two simdgroups run each group's 16x64
# tensor op together (execution_simdgroups<2>): ~50 TF/s against ~30 for one simdgroup's 16x32 op, with as
# many simdgroups in flight as the 32-column kernel. Each output is the same arithmetic: P from the tensor
# unit per 64-input group (the same bits whatever the op's shape and scope, tested), then C = fma(s, P,
# fma(b, xs, C)) in group order, the K slices added in slice order: every projection shape, 1-16 rows,
# bit-identical to ``_MAIN_TILED``, 1-128 rows. Live (2026-09-24): blocks of rounds alternating in one server,
# 57.7 -> 56.3 ms a round at 21k context and 75.1 -> 73.9 at 70k; an agent client's 70,510-token turn replayed to the same
# sha, 77.0 -> 74.6 ms a round (45.1 -> 46.6 tok/s) and its 49k-token prefill 122 -> 105 s.
# Above 16 rows the same kernel takes 32-row blocks (one 32x64 op per group, as ``_MAIN_TILED``'s TMR = 2).
_COOP = r"""
  const ushort sg = simdgroup_index_in_threadgroup;
  const ushort slice = sg >> 1;                                  // K slice: a pair of simdgroups each
  const ushort tip = ushort(thread_position_in_threadgroup.x) - slice * 64;   // thread within its pair
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / 64;
  const int n0 = threadgroup_position_in_grid.x * 64;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;       // first row of this threadgroup's row block
  const int g_begin = (slice * KG) / SK;
  const int g_end = ((slice + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<2>> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  auto a0 = tA.slice(0, 0);
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)Wq, dextents<int32_t, 2>(64, 64));
  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  constexpr int CAP = 16 * TMR;                                  // 16 TMR x 64 outputs over 64 threads
  short ecol[CAP], erow[CAP];
  for (int i = 0; i < CAP; i++) { auto ids = P.get_multidimensional_index(i); ecol[i] = ids[0]; erow[i] = ids[1]; }
  float C[CAP];
  for (int i = 0; i < CAP; i++) C[i] = 0.0f;
  const device uint* sbw = (const device uint*)SBt;              // (s, b) bf16 pairs, [g][n]
  for (int g = g_begin; g < g_end; g++) {
    auto a = tA.slice(g * 64, 0);
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (64 * 32), dextents<int32_t, 2>(64, 64));
    op.run(a, b, P);
    for (int i = 0; i < CAP; i++) {
      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(sbw[g * N + n0 + ecol[i]]);
      const float xs = XS[g * MP + rb + erow[i]];
      C[i] = fma(float(sb[0]), P[i], fma(float(sb[1]), xs, C[i]));
    }
  }
  // K slices added in slice order, 16 outputs a thread at a time (the buffer stays within 28 KB at 8 slices)
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 64];
  if (SK > 1)
    for (int c0 = 0; c0 < CAP; c0 += 16) {
      if (slice > 0) for (int i = 0; i < 16; i++) part[((slice - 1) * 16 + i) * 64 + tip] = C[c0 + i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (slice == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < 16; i++) C[c0 + i] += part[((s2 - 1) * 16 + i) * 64 + tip];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  if (slice == 0)
    for (int i = 0; i < CAP; i++) {
      const int m = rb + erow[i], n = n0 + ecol[i];
      if (m < M) Y[m * N + n] = static_cast<bfloat>(C[i]);
    }
"""
# 3- and 2-bit weights: ``_MAIN``'s tiles and arithmetic, with the weight operand widened to nibbles in
# threadgroup memory first (the tensor op has no 3- or 2-bit operand). NT is 32: lane l widens column n0 + l.
# MLX packs 3-bit values 8 to 3 bytes, value j of a pack at bits 3j..3j+2, so a column's group is 6 words and its
# pack c is the 24 bits from bit 24c; 2-bit values go 16 to a word, value j at bits 2j..2j+1 (4 words a group).
# TILED: ``tile_weight(bits=BITS)``'s layout, where a column tile's group is one contiguous 32 x 8*BITS-byte block.
_MAIN_LOWBIT = r"""
  static_assert(NT == 32, "one column per lane");
  static_assert(BITS == 2 || BITS == 3, "4-bit weights go to the tensor op as they are");
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / 64;
  constexpr int NF = NT / 16;
  constexpr int WPG = 2 * BITS;                          // words per column per group: 64 values x BITS bits
  constexpr int KW = K * BITS / 32;                      // words per column
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
  threadgroup uint stage_all[SK * NT * 8];               // per K slice: NT columns x 64 nibbles
  threadgroup uint* stage = stage_all + sg * NT * 8;
  tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> b((threadgroup uchar*)stage, dextents<int32_t, 2>(64, NT));
  const device uint* Wv = (const device uint*)Wq;
  const int n = n0 + lane;

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint4* sbv = (const device uint4*)SBt;
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
  for (int g = g_begin; g < g_end; g++) {
    uint w[WPG + 1];
    for (int i = 0; i <= WPG; i++) w[i] = 0;
    if (n < N) {
      const device uint* src = TILED ? Wv + ((int64_t)(threadgroup_position_in_grid.x * KG + g) * NT + lane) * WPG
                                     : Wv + (int64_t)n * KW + g * WPG;
      for (int i = 0; i < WPG; i++) w[i] = src[i];
    }
    if (BITS == 3) {
      for (int c = 0; c < 8; c++) {
        const int bit = 24 * c, i = bit >> 5, sh = bit & 31;
        uint pack = w[i] >> sh;
        if (sh > 8) pack |= w[i + 1] << (32 - sh);
        uint nib = 0;
        for (int j = 0; j < 8; j++) nib |= ((pack >> (3 * j)) & 7u) << (4 * j);
        stage[lane * 8 + c] = nib;
      }
    } else {
      for (int c = 0; c < 8; c++) {                     // word c/2's half c%2: 8 values of 2 bits -> 8 nibbles
        uint v = (w[c >> 1] >> (16 * (c & 1))) & 0xFFFFu;
        v = (v | (v << 8)) & 0x00FF00FFu;
        v = (v | (v << 4)) & 0x0F0F0F0Fu;
        v = (v | (v << 2)) & 0x33333333u;
        stage[lane * 8 + c] = v;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * 64, 0);
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    simdgroup_barrier(mem_flags::mem_threadgroup);   // the op has read the stage before the next group's widening
    for (int t = 0; t < TMR; t++) {
      // the last row block can run past MP (MP % 32 == 16): those rows are never stored, and XS ends at MP
      const bool live = rb + t * 16 < MP;
      const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
      const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
  }
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int nn = n0 + f * 16 + fn;
          if (m < M && nn < N)
            for (int j = 0; j < 4; j++) Y[m * N + nn + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""

# ``TF_KERNEL_AB`` (lane_engine) flips this every few rounds to A/B a bit-identical variant live. Tried and
# removed 2026-09-24: a fewer-loads epilogue (2 x 16-byte (s, b) loads and 2 row-sum loads per group, 7 KB
# split-K buffer), bit-identical, -1% in a paired microbenchmark but +0.3 ms a round live (47.5 against
# 47.2 ms, 1,500 rounds in 94 paired blocks); staging the next group's weights in threadgroup memory,
# bit-identical, 4-8% slower in the microbenchmark.
AB_FLAG = [False]                                    # a live A/B flips this every few rounds (engine side)

_kernels: dict[str, Any] = {}


def _named(base: str, source: str) -> str:
    """Kernel names carry a hash of their source: MLX caches compiled kernels by name."""

    import hashlib

    return f"{base}_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"


def _kernel(name: str) -> Any:
    if name not in _kernels:
        if name == "xsum":
            _kernels[name] = mx.fast.metal_kernel(name=_named("lane_qmm_xsum", _XSUM), input_names=["X", "mdims"], output_names=["XS"],
                                                  source=_XSUM, header=_HEADER)
        elif name == "coop":
            _kernels[name] = mx.fast.metal_kernel(name=_named("lane_qmm_coop", _COOP), input_names=["X", "XS", "Wq", "SBt", "mdims"],
                                                  output_names=["Y"], source=_COOP, header=_HEADER)
        elif name == "lowbit":
            _kernels[name] = mx.fast.metal_kernel(name=_named("lane_qmm_lowbit", _MAIN_LOWBIT),
                                                  input_names=["X", "XS", "Wq", "SBt", "mdims"], output_names=["Y"],
                                                  source=_MAIN_LOWBIT, header=_HEADER)

        else:
            source = _MAIN_TILED if name == "main_tiled" else _MAIN
            _kernels[name] = mx.fast.metal_kernel(name=_named("lane_qmm_" + name, source), input_names=["X", "XS", "Wq", "SBt", "mdims"],
                                                  output_names=["Y"], source=source, header=_HEADER)
    return _kernels[name]


_mdims_cache: dict[tuple[int, int], mx.array] = {}
_xs_cache: dict[int, tuple[mx.array, mx.array]] = {}


def _mdims(m: int, mp: int) -> mx.array:
    key = (m, mp)
    if key not in _mdims_cache:
        _mdims_cache[key] = mx.array([m, mp], dtype=mx.int32)
    return _mdims_cache[key]


def split_k(n: int, k: int) -> int:
    """K slices for an (n, k) weight: fixed by the shape, never by the row count."""

    tiles = -(-n // NT)
    sk = 1
    while sk < 8 and tiles * sk < 1024 and (k // 64) // (sk * 2) >= 8:
        sk *= 2
    return sk


def pack_scales(scales: mx.array, biases: mx.array) -> mx.array:
    """(N, K/64) scales and biases -> (K/64, N, 2) bf16 pairs, group-major."""

    return mx.stack([scales.T, biases.T], axis=-1).astype(mx.bfloat16)


def tile_weight(weight: mx.array, nt: int = NT, *, bits: int) -> mx.array:
    """MLX's packed (N, K*bits/32) weight -> the same shape and bytes, regrouped by column tile and group.

    The order is [N/nt][K/64][nt columns x 8*bits bytes].

    ``bits`` must be the weight's: the packed shape alone does not tell a 3-bit weight from a 4-bit one of
    another K, and a 3-bit weight tiled as 4-bit is misread by the 3-bit kernel.
    """

    n, kw = int(weight.shape[0]), int(weight.shape[1])
    wpg = 2 * bits                                        # words per column per group of 64
    return mx.contiguous(weight.reshape(n // nt, nt, kw // wpg, wpg).transpose(0, 2, 1, 3).reshape(n, kw))


def untile_weight(weight: mx.array, nt: int = NT, *, bits: int) -> mx.array:
    """``tile_weight`` undone: MLX's packed layout again."""

    n, kw = int(weight.shape[0]), int(weight.shape[1])
    wpg = 2 * bits
    return mx.contiguous(weight.reshape(n // nt, kw // wpg, nt, wpg).transpose(0, 2, 1, 3).reshape(n, kw))


def weight_bits(weight: mx.array, k: int) -> int:
    """The bit width of a packed ``weight`` for K inputs (words per row = K * bits / 32)."""

    words = int(weight.shape[1])
    if k <= 0 or (words * 32) % k:
        raise ValueError(f"a packed weight of {words} words a row does not fit K = {k}")
    return words * 32 // k


def supports(weight: mx.array, scales: mx.array, x: mx.array, bits: int, group_size: int, mode: str) -> bool:
    if bits not in BITS or group_size != 64 or mode != "affine":
        return False
    if x.dtype != mx.bfloat16 or scales.dtype != mx.bfloat16 or weight.dtype != mx.uint32 or weight.ndim != 2:
        return False
    k = int(x.shape[-1])
    n = int(weight.shape[0])
    return k % 64 == 0 and int(weight.shape[1]) * 32 == k * bits and n % 4 == 0


def lane_matmul(x: mx.array, weight: mx.array, sbt: mx.array, *, tiled: bool = False,
                sk: int | None = None, nt: int = NT) -> mx.array:
    """x (..., K) bf16 times the packed 4-, 3- or 2-bit ``weight`` (N, K*bits/32) transposed; rows <= MAX_ROWS.

    The width is read from the shapes (``weight_bits``). ``tiled``: ``weight`` is in ``tile_weight``'s layout
    (N a multiple of ``nt``: 32 or 64 for 4-bit, 32 for 3- and 2-bit); same bits either way.
    ``sk``: the K slices to use instead of ``split_k(N, K)``. A column's bits depend only on K and
    the slices, so several weights stacked into one call give each of them the bits of its own
    call when ``sk`` is that weight's ``split_k`` (stack only weights whose ``split_k`` agree).
    """

    K = int(x.shape[-1])
    N = int(weight.shape[0])
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = int(x2.shape[0])
    if M > MAX_ROWS:
        raise ValueError(f"lane_matmul takes at most {MAX_ROWS} rows, got {M}")
    if K % 64 or N % 4:
        raise ValueError(f"lane_matmul needs K a multiple of 64 and N a multiple of 4, got K={K}, N={N}")
    bits = weight_bits(weight, K)
    if bits not in BITS:
        raise ValueError(f"lane_matmul takes {', '.join(map(str, BITS))}-bit weights, got {bits}-bit")
    if bits < 4 and tiled and int(nt) != NT:
        raise ValueError(f"{bits}-bit weights tile {NT} columns wide, got {nt}")
    if bits < 4 and sk and int(sk) > 8:    # threadgroup memory: 1 KB of stage + 2 KB of partial sums a slice
        raise ValueError(f"{bits}-bit weights take at most 8 K slices (split_k's largest), got {sk}")
    MP = 16 * ((M + 15) // 16)
    KG = K // 64
    mdims = _mdims(M, MP)
    # q/k/v, the recurrent layer's four input projections and gate/up read the same x:
    # its group sums are computed once (arrays are immutable; the entry holds x alive)
    hit = _xs_cache.get(id(x))
    if hit is not None and hit[0] is x:
        xs = hit[1]
    else:
        # row counts are runtime values: a template per row count meant a compile per new window width
        xs = _kernel("xsum")(inputs=[x2, mdims], template=[("K", K)], grid=(KG, MP, 1),
                             threadgroup=(min(KG, 256), 1, 1), output_shapes=[(KG, MP)],
                             output_dtypes=[mx.float32])[0]
        _xs_cache[id(x)] = (x, xs)
        while len(_xs_cache) > 4:
            _xs_cache.pop(next(iter(_xs_cache)))
    sk = int(sk) if sk else split_k(N, K)
    if bits < 4:
        if tiled and N % NT:
            raise ValueError(f"tiled weights need N to be a multiple of {NT}, got {N}")
        block = MP if MP <= ROW_BLOCK else ROW_BLOCK
        y = _kernel("lowbit")(inputs=[x2, xs, weight, sbt, mdims],
                              template=[("TMR", block // 16), ("N", N), ("K", K), ("NT", NT), ("SK", sk),
                                        ("BITS", bits), ("TILED", int(bool(tiled)))],
                              grid=(-(-N // NT) * 32 * sk, -(-MP // block), 1), threadgroup=(32 * sk, 1, 1),
                              output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
        return y.reshape(*lead, N)
    nt = int(nt) if tiled else NT
    if nt == 64:
        block = MP if MP <= ROW_BLOCK else ROW_BLOCK
        y = _kernel("coop")(inputs=[x2, xs, weight, sbt, mdims], template=[("TMR", block // 16), ("N", N), ("K", K), ("SK", sk)],
                            grid=((N // 64) * 64 * sk, -(-MP // block), 1), threadgroup=(64 * sk, 1, 1),
                            output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
        return y.reshape(*lead, N)
    tiles = -(-N // nt)
    # up to 32 rows: one threadgroup per column tile holds them all; above, 32-row blocks over
    # grid.y (a 64-row op spilled: 4x the 32-row cost; two 32-row threadgroups cost ~1.3x)
    block = MP if MP <= ROW_BLOCK else ROW_BLOCK
    if tiled and N % nt:
        raise ValueError(f"tiled weights need N to be a multiple of {nt}, got {N}")
    y = _kernel("main_tiled" if tiled else "main")(inputs=[x2, xs, weight, sbt, mdims],
                        template=[("TMR", block // 16), ("N", N), ("K", K), ("NT", nt), ("SK", sk)],
                        grid=(tiles * 32 * sk, -(-MP // block), 1), threadgroup=(32 * sk, 1, 1),
                        output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(*lead, N)


# -- routing the model's projections --------------------------------------------------------
_ORIG: Any = None
enabled = False
max_rows = MAX_ROWS


_tiled_modules: list[Any] = []   # modules whose weight install() regrouped (uninstall() restores them)


def _call(self: Any, x: mx.array) -> mx.array:
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    tiled = getattr(self, "_lane_tiled", False)
    if enabled and rows <= max_rows and supports(self["weight"], self["scales"], x, self.bits,
                                                 self.group_size, getattr(self, "mode", "affine")):
        sbt = getattr(self, "_lane_sbt", None)
        if sbt is None:
            sbt = pack_scales(self["scales"], self["biases"])
            mx.eval(sbt)
            object.__setattr__(self, "_lane_sbt", sbt)
        y = lane_matmul(x, self["weight"], sbt, tiled=tiled, nt=getattr(self, "_lane_nt", NT))
    elif tiled:
        # wider than the lane kernel takes (MLX's chunked prefill): MLX's layout, rebuilt for this call
        weight = untile_weight(self["weight"], getattr(self, "_lane_nt", NT), bits=self.bits)
        y = mx.quantized_matmul(x, weight, self["scales"], self["biases"], transpose=True,
                                group_size=self.group_size, bits=self.bits)
    else:
        return _ORIG(self, x)
    if "bias" in self:
        y = y + self["bias"]
    return y


# Kept 32 columns wide under ``wide``: in_proj_z stacks with in_proj_b/a (48 rows each) into 6,240 rows,
# which only 32-column tiles divide (``lane_fuse``)
NARROW = ("in_proj_z",)


def takes(module: Any) -> bool:
    """Whether the lane matmul takes a QuantizedLinear: affine, 4, 3 or 2 bits in groups of 64, bf16 scales."""

    return (module.bits in BITS and module.group_size == 64 and getattr(module, "mode", "affine") == "affine"
            and module["scales"].dtype == mx.bfloat16)


def install(model: Any = None, *, rows: int = MAX_ROWS, tile: bool = True, wide: bool = False) -> None:
    """Route every 4-, 3- and 2-bit QuantizedLinear call of at most ``rows`` rows through the lane matmul.

    With ``model`` given, the interleaved scales are built up front (1/8 of 4-bit weights' size,
    1/6 of 3-bit, 1/4 of 2-bit) instead of on the first call, and (``tile``) each weight whose row count
    is a multiple of NT is regrouped in place into ``tile_weight``'s layout: no second copy.
    ``wide``: 4-bit weights whose row count is a multiple of 64 are tiled 64 columns wide (``_COOP``);
    3- and 2-bit weights tile 32 wide. Idempotent.
    """

    global _ORIG, enabled, max_rows
    import mlx.nn as nn

    if _ORIG is None:
        _ORIG = nn.QuantizedLinear.__call__
    nn.QuantizedLinear.__call__ = _call
    enabled = True
    max_rows = min(int(rows), MAX_ROWS)
    if model is not None:
        built, pending = [], 0
        for name, module in model.named_modules():
            if not (isinstance(module, nn.QuantizedLinear) and takes(module)):
                continue
            if getattr(module, "_lane_sbt", None) is None:
                sbt = pack_scales(module["scales"], module["biases"])
                object.__setattr__(module, "_lane_sbt", sbt)
                built.append(sbt)
                pending += sbt.nbytes
            weight = module["weight"]
            n_out, words = int(weight.shape[0]), int(weight.shape[-1])
            if (tile and not getattr(module, "_lane_tiled", False) and weight.dtype == mx.uint32
                    and weight.ndim == 2 and n_out % NT == 0 and words % (2 * module.bits) == 0):
                wide_ok = wide and module.bits == 4 and n_out % 64 == 0 and not name.endswith(NARROW)
                nt = 64 if wide_ok else NT
                module.weight = tile_weight(weight, nt, bits=module.bits)
                object.__setattr__(module, "_lane_tiled", True)
                object.__setattr__(module, "_lane_nt", nt)
                _tiled_modules.append(module)
                built.append(module["weight"])
                pending += 2 * weight.nbytes             # the old layout lives until this batch is evaluated
            if pending >= 2 * 1024**3:
                mx.eval(built)
                built, pending = [], 0
        if built:
            mx.eval(built)
        mx.clear_cache()      # the old layouts' buffers would otherwise sit in MLX's buffer cache


def uncovered(model: Any) -> dict[str, int]:
    """The model's linear layers the lane matmul does not take, by kind ({"6-bit g64": 17, "unquantized": 2}).

    MLX's own kernels run those, whose bits depend on the row count: with any, a drafted row is checked at
    width (still the model's own sample) rather than bit-identical to one-row decoding. Mixed checkpoints
    (per-layer overrides: 6-bit lm_head or down_proj, for instance) are the usual case. A tied embedding head
    (``QuantizedEmbedding.as_linear``) also runs MLX's kernel and is not counted here.
    """

    import mlx.nn as nn

    counts: dict[str, int] = {}
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            if takes(module):
                continue
            kind = f"{module.bits}-bit g{module.group_size}"
            mode = getattr(module, "mode", "affine")
            kind += "" if mode == "affine" else f" {mode}"
            kind += "" if module["scales"].dtype == mx.bfloat16 else f" {module['scales'].dtype} scales"
        elif isinstance(module, nn.Linear):
            kind = "unquantized"
        else:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def warm(model: Any, *, rows: tuple[int, ...] = (1, 17, 33)) -> int:
    """Compile every kernel variant the model's projections will use (one per shape and row tile)."""

    import mlx.nn as nn

    seen: set[tuple[int, int, int, bool, int]] = set()
    outs = []
    for _, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear) or getattr(module, "_lane_sbt", None) is None:
            continue
        n, k = int(module["weight"].shape[0]), int(module["weight"].shape[1]) * 32 // module.bits
        key = (n, k, module.bits, getattr(module, "_lane_tiled", False), getattr(module, "_lane_nt", NT))
        if key in seen:
            continue
        seen.add(key)
        for m in rows:
            outs.append(lane_matmul(mx.zeros((m, k), dtype=mx.bfloat16), module["weight"], module._lane_sbt,
                                    tiled=getattr(module, "_lane_tiled", False), nt=getattr(module, "_lane_nt", NT)))
    mx.eval(outs)
    return len(seen)


def uninstall() -> None:
    """MLX's own kernels again, with the weights back in MLX's layout."""

    global enabled
    import mlx.nn as nn

    enabled = False
    while _tiled_modules:
        module = _tiled_modules.pop()
        module.weight = untile_weight(module["weight"], getattr(module, "_lane_nt", NT), bits=module.bits)
        object.__setattr__(module, "_lane_tiled", False)
        object.__setattr__(module, "_lane_nt", NT)
        mx.eval(module["weight"])
    if _ORIG is not None:
        nn.QuantizedLinear.__call__ = _ORIG


__all__ = ["BITS", "MAX_ROWS", "install", "lane_matmul", "pack_scales", "split_k", "supports", "takes",
           "tile_weight", "uncovered", "uninstall", "untile_weight", "warm", "weight_bits"]
