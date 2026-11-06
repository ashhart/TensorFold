// DeepSeek V4.1 routed experts for prompt chunks on simdgroup MMA, equal bit for bit to the decode kernels per row.
#include "experts.metal"

constant constexpr uint XM_RF = 2;           // 8-row fragments a simdgroup carries
constant constexpr uint XM_RT = 8 * XM_RF;   // rows a tile holds
constant constexpr uint XM_E = 8 * XM_RF;    // outputs a thread holds (4 column fragments)

struct XmArgs {
  uint K, N, topk, pairs;
  float limit;
};

// One tile per (expert, 16 consecutive pair positions of a slot): {expert, first position, rows}; entry 0 is the count.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xm_tiles(
    device const uint* slots [[buffer(0)]], device const uint* nslots [[buffer(1)]], device uint4* tiles [[buffer(2)]],
    uint t [[thread_index_in_threadgroup]]) {
  if (t != 0) return;
  uint n = 0;
  for (uint s = 0; s < nslots[0]; ++s)
    for (uint r = 0; r < slots[3 * s + 2]; r += XM_RT) tiles[1 + n++] = uint4(slots[3 * s], slots[3 * s + 1] + r, min(XM_RT, slots[3 * s + 2] - r), 0u);
  tiles[0] = uint4(n, 0u, 0u, 0u);
}

// Every source row once as act_quant leaves it, a block's E4M3 code for k = 8s + m at byte 4m + s (a lane's word).
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xm_quant(device const bfloat* X [[buffer(0)]], device uchar* xc [[buffer(1)]],
                                  device float* xs [[buffer(2)]], constant uint2& a [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  const uint K = a.x, rows = a.y;
  if (gid.x >= K / 32 || gid.y >= rows) return;
  device const uint4* src = (device const uint4*)(X + gid.y * K + 32 * gid.x);
  float v[32], amax = 0.0f;
  bool bad = false;
  for (uint q = 0; q < 4; ++q)
    for (uint m = 0; m < 4; ++m) {
      const float2 f = float2(as_type<bfloat2>(src[q][m]));
      v[8 * q + 2 * m] = f.x, v[8 * q + 2 * m + 1] = f.y;
    }
  for (uint j = 0; j < 32; ++j) {
    if (TF_CTL == 1 && !isfinite(v[j])) v[j] = 0.0f;
    bad |= !isfinite(v[j]);
    amax = max(amax, fabs(v[j]));
  }
  const int e = ds_exp(amax);
  xs[gid.y * (K / 32) + gid.x] = bad ? NAN : ds_pow2(e);
  device uint* dst = (device uint*)(xc + gid.y * K + 32 * gid.x);
  for (uint m = 0; m < 8; ++m) {
    uint w = 0;
    for (uint s = 0; s < 4; ++s) w |= ds_code(min(fabs(v[8 * s + m]) * ds_pow2(-e), 448.0f), signbit(v[8 * s + m])) << (8 * s);
    dst[m] = w;
  }
}

// The four E4M3 codes of a word (k steps 0..3) as ds_val gives them: half bits of value/256, times 256 in half.
inline float4 xm_x8(uint q) {
  const uint a = ((q & 0x007F007Fu) << 7) | ((q & 0x00800080u) << 8);
  const uint b = (((q >> 8) & 0x007F007Fu) << 7) | (((q >> 8) & 0x00800080u) << 8);
  const half2 ha = as_type<half2>(a) * half2(256.0h), hb = as_type<half2>(b) * half2(256.0h);
  return float4(float(ha.x), float(hb.x), float(ha.y), float(hb.y));
}

// The four bf16 activations of a row at k steps 0..3 (stride 8) from element `o`.
inline float4 xm_x16(device const ushort* x, uint o) {
  float4 r;
  for (int s = 0; s < 4; ++s) {
    const float v = as_type<float>(uint(x[o + 8 * s]) << 16);
    r[s] = (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v;
  }
  return r;
}

// A group's sums (times the row scale if fp8) into the chains by k3_scaled_fma, or its own fma when all lanes allow.
template <bool FP8>
inline void xm_scale(thread float (&cur)[XM_E], thread simdgroup_matrix<float, 8, 8> (&acc)[XM_RF][4],
                     device const uchar* sb, uint sj, uint g, device const float* xs, const thread uint (&so)[XM_RF][2]) {
  uchar sc[4];
  bool normal = true;
  _Pragma("unroll")
  for (int j = 0; j < 4; ++j) {
    sc[j] = sb[j * sj + g];
    normal = normal && (sc[j] - 1u < 240u);
  }
  float xsv[XM_RF][2];
  _Pragma("unroll")
  for (int t = 0; t < XM_RF; ++t)
    _Pragma("unroll")
    for (int e = 0; e < 2; ++e) xsv[t][e] = FP8 ? xs[so[t][e] + g] : 1.0f;
  if (simd_all(normal)) {
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j) {
      const float f = k3_e8m0(sc[j]);
      _Pragma("unroll")
      for (int t = 0; t < XM_RF; ++t)
        _Pragma("unroll")
        for (int e = 0; e < 2; ++e) {
          float p = acc[t][j].thread_elements()[e];
          if (TF_CTL == 3) p = float(half(p));
          if (FP8) p *= xsv[t][e];
          cur[(t * 4 + j) * 2 + e] = fma(p, f, cur[(t * 4 + j) * 2 + e]);
        }
    }
  } else {
    _Pragma("unroll")
    for (int j = 0; j < 4; ++j)
      _Pragma("unroll")
      for (int t = 0; t < XM_RF; ++t)
        _Pragma("unroll")
        for (int e = 0; e < 2; ++e) {
          float p = acc[t][j].thread_elements()[e];
          if (TF_CTL == 3) p = float(half(p));
          if (FP8) p *= xsv[t][e];
          cur[(t * 4 + j) * 2 + e] = k3_scaled_fma(p, sc[j], cur[(t * 4 + j) * 2 + e]);
        }
  }
}

// Kimi's output chains on MMA (groups c, c+8, ...; 32 products in k order from zero a group), then the decode tree.
template <bool FP8>
inline void xm_sums(thread float (&res)[XM_E], device const uchar* wb, device const uchar* sb, uint wj, uint sj,
                    device const uchar* x, const thread uint (&xo)[XM_RF][2], device const float* xs,
                    const thread uint (&so)[XM_RF][2], uint G) {
  float stack[3][XM_E];
  const uint Ge = TF_CTL == 5 ? G - 1 : G;
  _Pragma("unroll")
  for (int c = 0; c < 8; ++c) {
    float cur[XM_E];
    _Pragma("unroll")
    for (int i = 0; i < XM_E; ++i) cur[i] = 0.0f;
    for (uint g = c; g < Ge; g += 8) {
      float4 xv[XM_RF][2];
      _Pragma("unroll")
      for (int t = 0; t < XM_RF; ++t)
        _Pragma("unroll")
        for (int e = 0; e < 2; ++e)
          xv[t][e] = FP8 ? xm_x8(*(device const uint*)(x + xo[t][e] + 32 * g)) : xm_x16((device const ushort*)x, xo[t][e] + 32 * g);
      simdgroup_matrix<float, 8, 8> acc[XM_RF][4];
      _Pragma("unroll")
      for (int t = 0; t < XM_RF; ++t)
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) acc[t][j] = simdgroup_matrix<float, 8, 8>(0.0f);
      _Pragma("unroll")
      for (int s = 0; s < 4; ++s) {
        simdgroup_matrix<float, 8, 8> xb[XM_RF];
        _Pragma("unroll")
        for (int t = 0; t < XM_RF; ++t)
          _Pragma("unroll")
          for (int e = 0; e < 2; ++e) xb[t].thread_elements()[e] = xv[t][e][s];
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) {
          simdgroup_matrix<float, 8, 8> wa;
          const float2 f = k3_fp4_pair(wb[j * wj + g * 16 + 4 * s]);
          wa.thread_elements()[0] = f.x, wa.thread_elements()[1] = f.y;
          _Pragma("unroll")
          for (int t = 0; t < XM_RF; ++t) simdgroup_multiply_accumulate(acc[t][j], wa, xb[t], acc[t][j]);
        }
      }
      xm_scale<FP8>(cur, acc, sb, sj, g, xs, so);
    }
    int lv = 0;
    while (lv < 3 && ((c >> lv) & 1)) {
      _Pragma("unroll")
      for (int i = 0; i < XM_E; ++i) cur[i] = stack[lv][i] + cur[i];
      ++lv;
    }
    _Pragma("unroll")
    for (int i = 0; i < XM_E; ++i) {
      if (lv < 3) stack[lv][i] = cur[i];
      else res[i] = cur[i];
    }
  }
}

// A thread's fragment cell: output column 8j + fm, tile row 8t + fn + e, held at (t * 4 + j) * 2 + e.
struct XmCell {
  uint fm, fn;
};

inline XmCell xm_cell(uint lane) {
  const uint qid = lane / 4;
  return {(qid & 4) + ((lane / 2) % 4), (qid & 2) * 2 + (lane % 2) * 2};
}

// Gate (simdgroup 0) and up (1) of a tile's pairs, then the V4.1 epilogue (clamps, silu, weight, bf16) and act_quant.
[[kernel, max_total_threads_per_threadgroup(64)]] void tf_dsv41_xm_up(
    device const uchar* xc [[buffer(0)]], device const float* xs [[buffer(1)]], device const ExpertPtrs* table [[buffer(2)]],
    device const uint4* tiles [[buffer(3)]], device const uint* pairs [[buffer(4)]], device bfloat* act [[buffer(5)]],
    constant XmArgs& a [[buffer(6)]], device const float* wts [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float gv[XM_RT][32];
  threadgroup float vals[XM_RT][32];
  if (tg.y >= tiles[0].x) return;
  const uint4 tile = tiles[1 + tg.y];
  const ExpertPtrs w = table[tile.x];
  const XmCell cell = xm_cell(lane);
  const uint K = a.K, G = K / 32, n0 = tg.x * 32, first = tile.y, nr = tile.z, n = n0 + cell.fm;
  uint xo[XM_RF][2], so[XM_RF][2];
  for (int t = 0; t < XM_RF; ++t)
    for (int e = 0; e < 2; ++e) {
      const uint row = pairs[first + min(uint(8 * t + cell.fn + e), nr - 1)] / a.topk;
      xo[t][e] = row * K + 4 * cell.fm, so[t][e] = row * G;
    }
  float res[XM_E];
  device const uchar* wp = sg == 0 ? w.w1p : w.w3p;
  device const uchar* ws = sg == 0 ? w.w1s : w.w3s;
  xm_sums<true>(res, wp + n * (K / 2) + cell.fn / 2, ws + n * G, 8 * (K / 2), 8 * G, xc, xo, xs, so, G);
  if (sg == 0)
    for (int t = 0; t < XM_RF; ++t)
      for (int j = 0; j < 4; ++j)
        for (int e = 0; e < 2; ++e) gv[8 * t + cell.fn + e][8 * j + cell.fm] = float(bfloat(res[(t * 4 + j) * 2 + e]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) return;
  for (int t = 0; t < XM_RF; ++t)
    for (int j = 0; j < 4; ++j)
      for (int e = 0; e < 2; ++e) {
        const uint lr = 8 * t + cell.fn + e;
        const float g = ds_clip(gv[lr][8 * j + cell.fm], -INFINITY, a.limit);
        const float u = ds_clip(float(bfloat(res[(t * 4 + j) * 2 + e])), -a.limit, a.limit);
        const float v = g / (1.0f + precise::exp(-g)) * u * (lr < nr ? wts[pairs[first + lr]] : 0.0f);
        vals[lr][8 * j + cell.fm] = TF_CTL == 2 ? v : float(bfloat(v));
      }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 0; r < nr; ++r) {
    const float v = vals[r][lane];
    const float amax = simd_max(fabs(v));
    const bool bad = simd_any(!isfinite(v));
    const int s = ds_exp(amax);
    const float q = ds_val(ds_code(min(fabs(v) * ds_pow2(-s), 448.0f), signbit(v))) * ds_pow2(s);
    act[(first + r) * a.N + n0 + lane] = bfloat(bad ? NAN : q);
  }
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && lane == 0) act[a.pairs * a.N] = bfloat(1.0f);
}

// Down of a tile's pairs over bf16 activations by pair position: bf16 outputs by pair position.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_dsv41_xm_down(
    device const bfloat* act [[buffer(0)]], device const ExpertPtrs* table [[buffer(1)]],
    device const uint4* tiles [[buffer(2)]], device bfloat* out [[buffer(3)]], constant XmArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (tg.y >= tiles[0].x) return;
  const uint4 tile = tiles[1 + tg.y];
  const ExpertPtrs w = table[tile.x];
  const XmCell cell = xm_cell(lane);
  const uint K = a.K, G = K / 32, n0 = tg.x * 32, first = tile.y, nr = tile.z, n = n0 + cell.fm;
  uint xo[XM_RF][2];
  for (int t = 0; t < XM_RF; ++t)
    for (int e = 0; e < 2; ++e) xo[t][e] = (first + min(uint(8 * t + cell.fn + e), nr - 1)) * K + cell.fm;
  float res[XM_E];
  xm_sums<false>(res, w.w2p + n * (K / 2) + cell.fn / 2, w.w2s + n * G, 8 * (K / 2), 8 * G, (device const uchar*)act, xo,
                 nullptr, xo, G);
  for (int t = 0; t < XM_RF; ++t)
    for (int j = 0; j < 4; ++j)
      for (int e = 0; e < 2; ++e) {
        const uint lr = 8 * t + cell.fn + e;
        if (lr < nr) out[(first + lr) * a.N + n0 + 8 * j + cell.fm] = bfloat(res[(t * 4 + j) * 2 + e]);
      }
  if (TF_CTL == 4 && tg.x == 0 && tg.y == 0 && lane == 0) out[a.pairs * a.N] = bfloat(1.0f);
}
