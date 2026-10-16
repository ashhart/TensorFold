// Drama 3 MoE: the router, FP8 activation rows, the routing plan, and the FP8 block-scaled experts.
#include "d3_common.h"

struct RouterArgs {
  uint ss, r_off;
};

// fp32 softmax over the 256 router logits of a row, top 8 (ties to the lower expert), renormalized; 256 threads.
[[max_total_threads_per_threadgroup(256)]] kernel void d3_router(device const bf16* sp [[buffer(0)]], device int* picks [[buffer(1)]],
                      device float* weights [[buffer(2)]], constant RouterArgs& a [[buffer(3)]],
                      uint t [[threadgroup_position_in_grid]], uint e [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  threadgroup float wv[8], best_v[8];
  threadgroup int wi[8], best_i[8];
  const float lg = b2f(sp[(ulong)t * a.ss + a.r_off + e]);
  const float m = group_max(lg, scratch, sg, lane, 8);
  const float p = precise::exp(lg - m);
  const float sum = group_sum(p, scratch + 8, sg, lane, 8);
  float mine = p / sum;
  for (int k = 0; k < 8; ++k) {
    float v = mine;
    int idx = int(e);
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_xor(v, o);
      const int oi = simd_shuffle_xor(idx, o);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0) {
      wv[sg] = v;
      wi[sg] = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
      float bv = wv[0];
      int bi = wi[0];
      for (int j = 1; j < 8; ++j)
        if (wv[j] > bv || (wv[j] == bv && wi[j] < bi)) {
          bv = wv[j];
          bi = wi[j];
        }
      best_v[k] = bv;
      best_i[k] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(e) == best_i[k]) mine = -1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (e < 8) {
    float s = 0.0f;
    for (int j = 0; j < 8; ++j) s += best_v[j];
    picks[t * 8 + e] = best_i[e];
    weights[t * 8 + e] = best_v[e] / s;
  }
}

struct QuantArgs {
  uint xs, qs, groups;
};

// SGLang's per_token_group_quant_fp8: scale = max(amax, 1e-10) / 448, q = e4m3(x / scale) kept exactly in bf16.
[[max_total_threads_per_threadgroup(128)]] kernel void d3_act_quant(device const bf16* x [[buffer(0)]], device bf16* q [[buffer(1)]], device float* s [[buffer(2)]],
                         constant QuantArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint i [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  const uint r = tg.x, g = tg.y;
  const float v = b2f(x[(ulong)r * a.xs + g * 128 + i]);
  const float amax = group_max(fabs(v), scratch, sg, lane, 4);
  const float sc = max(amax, 1e-10f) / 448.0f;
  const float y = clamp(precise::divide(v, sc), -448.0f, 448.0f);
  q[(ulong)r * a.qs + g * 128 + i] = f2b(to_e4m3(y));
  if (i == 0) s[(ulong)r * a.groups + g] = sc;
}

struct PlanArgs {
  uint pairs, cap;
};

// Prompt rows: each expert's pairs (counts, members); their order varies but no pair's sums depend on it.
[[max_total_threads_per_threadgroup(1024)]] kernel void d3_plan(device const int* picks [[buffer(0)]], device int* counts [[buffer(1)]],
                    device int* members [[buffer(2)]], constant PlanArgs& a [[buffer(3)]],
                    uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_int n[256];
  if (t < 256) atomic_store_explicit(&n[t], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < a.pairs; p += 1024) {
    const int e = picks[p];
    const int at = atomic_fetch_add_explicit(&n[e], 1, memory_order_relaxed);
    members[(ulong)e * a.cap + at] = int(p);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 256) counts[t] = atomic_load_explicit(&n[t], memory_order_relaxed);
}

// An expert's FP8 matrices and their [N / 128, K / 128] fp32 block scales (the checkpoint's own bytes).
struct ExpertPtrs {
  device const uchar* g;
  device const float* gs;
  device const uchar* u;
  device const float* us;
  device const uchar* d;
  device const float* ds;
};

struct XpArgs {
  uint K, N, prompt, cap, slots;
};

// Two e4m3 bytes (bits 0-15 of w) as floats times 2^-8, exactly: the e4m3 fields placed in fp16 bits.
inline float2 e4m3x2(uint w) {
  const uint h = ((w & 0x7Fu) << 7) | ((w & 0x80u) << 8) | ((w & 0x7F00u) << 15) | ((w & 0x8000u) << 16);
  return float2(as_type<half2>(h));
}

// Sixteen e4m3 weights (8-byte aligned) against sixteen FP8-valued bf16 inputs: an fma chain in k order, times 2^-8.
inline float fp8_dot16(uint2 wa, uint2 wb, device const bf16* x) {
  const uint ws[4] = {wa.x, wa.y, wb.x, wb.y};
  float p = 0.0f;
  for (int q = 0; q < 4; ++q) {
    const float4 xv = bf4(((device const uint2*)x)[q]);
    const float2 lo = e4m3x2(ws[q]), hi = e4m3x2(ws[q] >> 16);
    p = fma(lo.x, xv.x, p);
    p = fma(lo.y, xv.y, p);
    p = fma(hi.x, xv.z, p);
    p = fma(hi.y, xv.w, p);
  }
  return p;
}

// C FP8 rows against FP8 x: per lane, each 16-input group sum (2^8 restored) times w scale * x scale, then simd_sum.
template <int C>
inline void fp8_rows(device const uchar* const* w, device const float* const* wsc, uint K, device const bf16* x,
                     device const float* xs, uint lane, thread float* out) {
  float acc[C];
  for (int c = 0; c < C; ++c) acc[c] = 0.0f;
  for (uint k0 = lane * 16; k0 < K; k0 += 512) {
    const uint gi = k0 / 128;
    uint2 wv[C][2];
    for (int c = 0; c < C; ++c) {
      wv[c][0] = ((device const uint2*)(w[c] + k0))[0];
      wv[c][1] = ((device const uint2*)(w[c] + k0))[1];
    }
    const float xg = xs[gi];
    for (int c = 0; c < C; ++c) acc[c] = fma(fp8_dot16(wv[c][0], wv[c][1], x + k0) * 256.0f, wsc[c][gi] * xg, acc[c]);
  }
  for (int c = 0; c < C; ++c) out[c] = simd_sum(acc[c]);
}

// Gate and up with the SwiGLU epilogue, bf16(silu(g) * u) on bf16 g, u; grid (N / 8, pairs or 256 experts).
[[max_total_threads_per_threadgroup(128)]] kernel void d3_expert_up(device const bf16* xq [[buffer(0)]], device const float* xsc [[buffer(1)]],
                         device const ExpertPtrs* table [[buffer(2)]], device const int* picks [[buffer(3)]],
                         device const int* counts [[buffer(4)]], device const int* members [[buffer(5)]],
                         device bf16* gu [[buffer(6)]], constant XpArgs& a [[buffer(7)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 4 + sg) * 2;
  const int e = a.prompt ? int(tg.y) : picks[tg.y];
  const int cnt = a.prompt ? counts[e] : 1;
  const ExpertPtrs w = table[e];
  const uint groups = a.K / 128;
  device const uchar* const rows[4] = {w.g + (ulong)n0 * a.K, w.g + (ulong)(n0 + 1) * a.K, w.u + (ulong)n0 * a.K,
                                       w.u + (ulong)(n0 + 1) * a.K};
  device const float* const sc[4] = {w.gs + (n0 / 128) * groups, w.gs + ((n0 + 1) / 128) * groups,
                                     w.us + (n0 / 128) * groups, w.us + ((n0 + 1) / 128) * groups};
  for (int j = 0; j < cnt; ++j) {
    const uint pair = a.prompt ? uint(members[(ulong)e * a.cap + j]) : tg.y;
    const uint row = pair / a.slots;
    float v[4];
    fp8_rows<4>(rows, sc, a.K, xq + (ulong)row * a.K, xsc + (ulong)row * groups, lane, v);
    if (lane == 0)
      for (int c = 0; c < 2; ++c) {
        const float g = rb(v[c]), u = rb(v[2 + c]);
        gu[(ulong)pair * a.N + n0 + c] = f2b(g / (1.0f + precise::exp(-g)) * u);
      }
  }
}

// Down of routed pairs to fp32 rows [pair, N] (the combine weights and sums them); 4 columns a simdgroup.
[[max_total_threads_per_threadgroup(128)]] kernel void d3_expert_down(device const bf16* gq [[buffer(0)]], device const float* gsc [[buffer(1)]],
                           device const ExpertPtrs* table [[buffer(2)]], device const int* picks [[buffer(3)]],
                           device const int* counts [[buffer(4)]], device const int* members [[buffer(5)]],
                           device float* down [[buffer(6)]], constant XpArgs& a [[buffer(7)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * 4 + sg) * 4;
  const int e = a.prompt ? int(tg.y) : picks[tg.y];
  const int cnt = a.prompt ? counts[e] : 1;
  const ExpertPtrs w = table[e];
  const uint groups = a.K / 128;
  device const uchar* rows[4];
  device const float* sc[4];
  for (uint c = 0; c < 4; ++c) {
    rows[c] = w.d + (ulong)(n0 + c) * a.K;
    sc[c] = w.ds + ((n0 + c) / 128) * groups;
  }
  for (int j = 0; j < cnt; ++j) {
    const uint pair = a.prompt ? uint(members[(ulong)e * a.cap + j]) : tg.y;
    float v[4];
    fp8_rows<4>(rows, sc, a.K, gq + (ulong)pair * a.K, gsc + (ulong)pair * groups, lane, v);
    if (lane == 0)
      for (int c = 0; c < 4; ++c) down[(ulong)pair * a.N + n0 + c] = v[c];
  }
}
