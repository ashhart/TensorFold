"""Keep serial and window attention bit-identical with arithmetic fixed by absolute key position, ordered chunk merges and zero contribution from masked tiles."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

MAX_QUERIES = 128
TILES_PER_GROUP = int(os.environ.get("TF_ATTN_TILES", "16"))    # Each threadgroup shares key reads across 16-row tiles; 24 tiles overflow P staging.
CHUNK = 512            # keys per chunk (fixed: part of the arithmetic)
TILE = 64              # keys per tile (fixed: part of the arithmetic)

_HEADER = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"

_PARTIAL = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int tile = int(threadgroup_position_in_grid.z) * SG + sg;   // 16-row tile (SGA in all, SG per threadgroup)
  const uint hk = threadgroup_position_in_grid.x;              // key head
  const uint c = threadgroup_position_in_grid.y;               // chunk of CK keys
  const int L = dims[0], NCH = dims[1], NQ = dims[2], SGA = dims[4];   // runtime: one variant per SG
  const int RP = 16 * SGA;
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = tile * 16 + fm, r1 = r0 + 8;
  const bool causal = dims[3] != 0;
  const int n0 = (r0 < G * NQ) ? (causal ? L - NQ + r0 / G + 1 : L) : 0;
  const int n1 = (r1 < G * NQ) ? (causal ? L - NQ + r1 / G + 1 : L) : 0;
  // P in half (it lies in [0, 1]): the P x V op then runs ~2x the fp32 rate. With 16-bit operands the op's
  // destination interleaves rows fm and fm + 8 every 4 elements (fp32 P: elements 0-31 are row fm)
  threadgroup half Ps[SG * 16 * TK];
  threadgroup half* myP = Ps + sg * 16 * TK;
  if (tile >= SGA) return;                                      // the last threadgroup's spare simdgroups
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)Qp + (int64_t)hk * RP * D, dextents<int32_t, 2>(D, RP));
  // rows at their real stride (a cache buffer's rows are D apart; other layouts need not be)
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK((device bfloat*)K + (int64_t)hk * K_strides[1], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, int32_t(K_strides[2])}));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV((device bfloat*)V + (int64_t)hk * V_strides[1], dextents<int32_t, 2>(D, L), array<int32_t, 2>({1, int32_t(V_strides[2])}));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  constexpr auto dS = matmul2d_descriptor(16, TK, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto aQ = tQ.slice(0, tile * 16);
  auto bV0 = tV.slice(0, 0);
  // the op is exact up to 128 output columns: the head dimension goes in two halves
  auto Olo = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bV0), float>();
  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bV0), float>();
  for (int i = 0; i < 64; i++) { Olo[i] = 0.0f; Ohi[i] = 0.0f; }
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  const int kbeg = int(c) * CK;
  const int kend = min(kbeg + CK, L);
  for (int kt = kbeg; kt < kend; kt += TK) {
    auto bK = tK.slice(0, kt);
    auto S = opS.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), float>();
    opS.run(aQ, bK, S);
    float s[TK / 2];
    for (int i = 0; i < TK / 2; i++) {                         // 8 elements per 16-key block: 4 of row fm, then 4 of fm + 8
      const int key = kt + (i >> 3) * 16 + fn + (i & 3);
      s[i] = key < ((i & 4) ? n1 : n0) ? S[i] * scale[0] : -INFINITY;
    }
    float x0 = -INFINITY, x1 = -INFINITY;
    for (int i = 0; i < TK / 2; i++) { if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }
    x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
    x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
    const float nm0 = max(m0, x0), nm1 = max(m1, x1);
    const float f0 = (x0 == -INFINITY) ? 1.0f : fast::exp(m0 - nm0);
    const float f1 = (x1 == -INFINITY) ? 1.0f : fast::exp(m1 - nm1);
    float p[TK / 2];
    for (int i = 0; i < TK / 2; i++) p[i] = (s[i] == -INFINITY) ? 0.0f : fast::exp(s[i] - ((i & 4) ? nm1 : nm0));
    float y0 = 0.0f, y1 = 0.0f;
    for (int b = 0; b < TK / 16; b++) {
      y0 += (p[b * 8] + p[b * 8 + 1]) + (p[b * 8 + 2] + p[b * 8 + 3]);
      y1 += (p[b * 8 + 4] + p[b * 8 + 5]) + (p[b * 8 + 6] + p[b * 8 + 7]);
    }
    y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
    y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
    if (x0 != -INFINITY) { l0 = l0 * f0 + y0; m0 = nm0; }
    if (x1 != -INFINITY) { l1 = l1 * f1 + y1; m1 = nm1; }
    for (int f = 0; f < TK / 16; f++)
      for (int i = 0; i < 4; i++) {
        myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);
        myP[(fm + 8) * TK + f * 16 + fn + i] = half(p[f * 8 + 4 + i]);
      }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 64; i++) { const float f = (i & 4) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }
    auto bVlo = tV.slice(0, kt);
    auto bVhi = tV.slice(128, kt);
    opO.run(tP, bVlo, Olo);
    opO.run(tP, bVhi, Ohi);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int64_t base = ((int64_t)hk * NCH + c) * RP;
  // 16-bit operand layout: elements 4q..4q+3 are four consecutive columns of one row
  for (int q = 0; q < 16; q++) {
    device float* dst = PO + (base + tile * 16 + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;
    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);
    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);
  }
  if ((lane & 9) == 0) {
    PM[base + r0] = m0; PL[base + r0] = l0;
    PM[base + r1] = m1; PL[base + r1] = l1;
  }
"""

# Register P preserves _PARTIAL arithmetic; the decoder version hash includes the reference source.
_P_STORE = """    for (int f = 0; f < TK / 16; f++)
      for (int i = 0; i < 4; i++) {
        myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);
        myP[(fm + 8) * TK + f * 16 + fn + i] = half(p[f * 8 + 4 + i]);
      }
    simdgroup_barrier(mem_flags::mem_threadgroup);
"""
_P_REGISTERS = """    auto Pc = opS.template get_destination_cooperative_tensor<decltype(aQ), decltype(bK), half>();
    for (int i = 0; i < TK / 2; i++) Pc[i] = half(p[i]);
    auto Pin = opO.template get_left_input_cooperative_tensor<half, bfloat, float>(Pc);
"""
_PV_THREADGROUP = """    opO.run(tP, bVlo, Olo);
    opO.run(tP, bVhi, Ohi);
    simdgroup_barrier(mem_flags::mem_threadgroup);"""
_PV_REGISTERS = """    opO.run(Pin, bVlo, Olo);
    opO.run(Pin, bVhi, Ohi);"""
assert _P_STORE in _PARTIAL and _PV_THREADGROUP in _PARTIAL
_PARTIAL_DIRECT = _PARTIAL.replace(_P_STORE, _P_REGISTERS).replace(_PV_THREADGROUP, _PV_REGISTERS)
# TF_ATTN_DIRECT_P=0 uses threadgroup-memory P with the same bits.
DIRECT_P = os.environ.get("TF_ATTN_DIRECT_P", "1") != "0"

def _single_half(source: str) -> str:
    """Head dimension 128 (Nemotron-H): the output in one 128-column half; the 256 kernels run two."""

    out = source
    for old, new in (
        ("  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(bV0), float>();\n", ""),
        ("{ Olo[i] = 0.0f; Ohi[i] = 0.0f; }", "{ Olo[i] = 0.0f; }"),
        ("Olo[i] *= f; Ohi[i] *= f; }", "Olo[i] *= f; }"),
        ("    auto bVhi = tV.slice(128, kt);\n", ""),
        ("    opO.run(Pin, bVhi, Ohi);", ""),
        ("    opO.run(tP, bVhi, Ohi);\n", ""),
        ("    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);\n", ""),
    ):
        out = out.replace(old, new)
    if "Ohi" in out or "bVhi" in out:
        raise AssertionError("lane_attention: the 128 variant still refers to the second half")
    return out


_PARTIAL_128 = _single_half(_PARTIAL)
_PARTIAL_DIRECT_128 = _single_half(_PARTIAL_DIRECT)


_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;              // row t * G + g
  const int NCH = dims[1], NQ = dims[2], SGA = dims[4];
  const int RP = 16 * SGA;
  constexpr int DP = D / 32;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const int64_t row = ((int64_t)hk * NCH + c) * RP + r;
    const float mc = PM[row];
    if (mc == -INFINITY) continue;
    const float lc = PL[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + PO[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int t = r / G, g = r % G, h = hk * G + g;
  for (int i = 0; i < DP; i++) OUT[((int64_t)h * NQ + t) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
"""

_kernels: dict[str, Any] = {}


def _named(base: str, source: str) -> str:
    """Kernel names carry a hash of their source: MLX caches compiled kernels by name."""

    import hashlib

    return f"{base}_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"


def _kernel(name: str) -> Any:
    if name not in _kernels:
        if name in ("partial", "partial_direct", "partial_128", "partial_direct_128"):
            source = {"partial": _PARTIAL, "partial_direct": _PARTIAL_DIRECT, "partial_128": _PARTIAL_128,
                      "partial_direct_128": _PARTIAL_DIRECT_128}[name]
            _kernels[name] = mx.fast.metal_kernel(
                name=_named("lane_attention_" + name, source), input_names=["Qp", "K", "V", "scale", "dims"],
                output_names=["PO", "PM", "PL"], source=source, header=_HEADER,
                ensure_row_contiguous=False)
        else:
            _kernels[name] = mx.fast.metal_kernel(
                name=_named("lane_attention_merge", _MERGE), input_names=["PO", "PM", "PL", "dims"], output_names=["OUT"],
                source=_MERGE, header=_HEADER)
    return _kernels[name]


def _partial(qp: mx.array, keys: mx.array, values: mx.array, sc: mx.array, dims: mx.array, *, G: int, D: int, SG: int,
             SGA: int, HKV: int, nch: int, RP: int) -> Any:
    return _kernel(("partial_direct" if DIRECT_P else "partial") + ("_128" if D == 128 else ""))(
        inputs=[qp, keys, values, sc, dims], template=[("G", G), ("D", D), ("SG", SG), ("CK", CHUNK), ("TK", TILE)],
        grid=(HKV * 32 * SG, nch, -(-SGA // SG)), threadgroup=(32 * SG, 1, 1),
        output_shapes=[(HKV * nch * RP * D,), (HKV * nch * RP,), (HKV * nch * RP,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])


def lane_sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Causal attention for queries [1, H, T, D] and keys/values [1, Hkv, L, D], allowing cache views with contiguous D-element rows."""

    _, H, T, D = (int(s) for s in queries.shape)
    HKV, L = int(keys.shape[1]), int(keys.shape[2])
    if D not in (128, 256) or H % HKV or T > MAX_QUERIES or T > L:
        raise ValueError(f"lane_sdpa: unsupported shape q={queries.shape} k={keys.shape}")
    if queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
        raise ValueError("lane_sdpa: bf16 only")
    G = H // HKV
    R = G * T
    RP = 16 * ((R + 15) // 16)
    SGA = RP // 16
    SG = min(SGA, TILES_PER_GROUP)
    qp = queries.reshape(HKV, G, T, D).transpose(0, 2, 1, 3).reshape(HKV, R, D)
    if RP != R:
        qp = mx.concatenate([qp, mx.zeros((HKV, RP - R, D), dtype=queries.dtype)], axis=1)
    qp = mx.contiguous(qp)
    nch = -(-L // CHUNK)
    # Runtime key counts avoid compiling a kernel per token.
    dims = mx.array([L, nch, T, 1, SGA], dtype=mx.int32)
    po, pm, pl = _partial(qp, keys, values, mx.array([float(scale)], dtype=mx.float32), dims, G=G, D=D, SG=SG, SGA=SGA,
                          HKV=HKV, nch=nch, RP=RP)
    return _kernel("merge")(
        inputs=[po, pm, pl, dims], template=[("G", G), ("D", D)],
        grid=(HKV * 32, R, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, H, T, D)], output_dtypes=[mx.bfloat16])[0]


def warm(heads: int = 24, kv_heads: int = 4, max_queries: int = 16) -> None:
    """Compile the kernel variants (one per simdgroups-per-threadgroup count) before the first request."""

    group = heads // kv_heads
    k = mx.zeros((1, kv_heads, max_queries + 64, 256), dtype=mx.bfloat16)
    widths = sorted({t for t in range(1, max_queries + 1) if (16 * ((group * t + 15) // 16)) // 16 <= TILES_PER_GROUP}
                    | {max_queries})
    seen, outs = set(), []
    for t in widths:
        sg = min((group * t + 15) // 16, TILES_PER_GROUP)
        if sg in seen:
            continue
        seen.add(sg)
        outs.append(lane_sdpa(mx.zeros((1, heads, t, 256), dtype=mx.bfloat16), k[:, :, :t + 64], k[:, :, :t + 64], 0.0625))
    mx.eval(outs)


# -- routing the model's decode attention ---------------------------------------------
_STOCK: Any = None
enabled = False


def lane_attention(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float,
                   mask: Any, sinks: Any = None) -> mx.array:
    T = int(queries.shape[2])
    if (enabled and T <= MAX_QUERIES and sinks is None and not hasattr(cache, "bits")
            and int(queries.shape[0]) == 1 and not isinstance(mask, mx.array) and int(queries.shape[3]) == 256
            and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16):
        return lane_sdpa(queries, keys, values, scale)
    return _STOCK(queries, keys, values, cache, scale, mask, sinks)


def install() -> None:
    """Route one-stream decode attention (up to 32 queries, causal) through ``lane_sdpa``."""

    global _STOCK, enabled
    import mlx_lm.models.qwen3_next as qn

    current = qn.scaled_dot_product_attention
    if current is not lane_attention:
        _STOCK = current
        qn.scaled_dot_product_attention = lane_attention
    enabled = True


__all__ = ["CHUNK", "MAX_QUERIES", "install", "lane_attention", "lane_sdpa", "warm"]
