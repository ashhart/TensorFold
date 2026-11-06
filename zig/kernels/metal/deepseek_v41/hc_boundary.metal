// Hyper-connection block boundary in two launches, ported whole from BobClawblaw/TensorFold, branch
// deepseek-v41-main, commit 72a552ed (Apache-2.0; see THIRD_PARTY_NOTICES.md): ds_hc_mix_part,
// ds_hc_expand_mix and ds_hc_boundary, minus their fused-expert branch (this engine's MoE writes its own
// branch) and the XR-rows variant. The arithmetic, the thread mapping (1024 threads a row, five columns a
// thread) and the lagged collapse are the sibling's, unchanged.
#include <metal_stdlib>
using namespace metal;

constant constexpr int S = 4, D = 5120, F = S * D, MIX = (2 + S) * S, SLICES = F / 1024;

struct HcArgs { float eps; float hc_eps; int iters; };
static_assert(sizeof(HcArgs) == 12, "HcArgs must mirror mhc.zig's BoundaryArgs");

inline float hc_bf(float v) { return float(bfloat(v)); }

// The mixes' partial sums: a threadgroup a (1024-column slice, row), 24 products and a sum of squares to PART.
[[kernel]] void tf_hc_mix_part(const device bfloat* X [[buffer(0)]], const device float* FN [[buffer(1)]],
                               device float* PART [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][MIX + 1];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const device bfloat* x = X + size_t(r) * F + 1024 * j + 4 * int(t);
  float v[4];
  for (int i = 0; i < 4; i++) v[i] = float(x[i]);
  float acc[MIX + 1];
  acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int m = 0; m < MIX; m++) {
    const device float* f = FN + size_t(m) * F + 1024 * j + 4 * int(t);
    acc[m] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
  }
  for (int m = 0; m <= MIX; m++) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) red[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= MIX) {
    float s = 0.0f;
    for (int k = 0; k < 8; k++) s += red[k][t];
    PART[(size_t(r) * SLICES + j) * (MIX + 1) + t] = s;
  }
}

// The pending branch into XO and the slice's mix sums from the bf16 written: one pass over the streams.
[[kernel]] void tf_hc_expand_mix(const device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                                 const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                                 const device float* FN [[buffer(4)]], device float* PART [[buffer(5)]],
                                 device bfloat* XO [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][MIX + 1];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const int col = 1024 * j + 4 * int(t);
  const int k = col / D, d0 = col % D;
  const device bfloat* x = X + size_t(r) * F;
  const device float* cm = COMB + r * S * S;
  float v[4];
  for (int i = 0; i < 4; i++) {
    const int d = d0 + i;
    const float b = float(BR[size_t(r) * D + d]);
    float xs[S];
    for (int jj = 0; jj < S; jj++) xs[jj] = float(x[jj * D + d]);
    float mm = 0.0f;
    for (int jj = 0; jj < S; jj++) mm += cm[jj * S + k] * xs[jj];
    const bfloat o = bfloat(POST[r * S + k] * b + mm);
    XO[size_t(r) * F + col + i] = o;
    v[i] = float(o);
  }
  float acc[MIX + 1];
  acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int m = 0; m < MIX; m++) {
    const device float* f = FN + size_t(m) * F + col;
    acc[m] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
  }
  for (int m = 0; m <= MIX; m++) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) red[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= MIX) {
    float s = 0.0f;
    for (int q = 0; q < 8; q++) s += red[q][t];
    PART[(size_t(r) * SLICES + j) * (MIX + 1) + t] = s;
  }
}

// A threadgroup of 1024 a row: (MIXES) the boundary's pre, post and Sinkhorn comb; the streams collapsed and normed.
template <bool MIXES>
[[kernel]] void tf_hc_boundary(const device bfloat* X [[buffer(0)]], const device float* PART [[buffer(1)]],
                               const device float* BASE [[buffer(2)]], const device float* SCALE [[buffer(3)]],
                               const device float* PRE_IN [[buffer(4)]], const device bfloat* NORMW [[buffer(5)]],
                               constant HcArgs& a [[buffer(6)]], device float* PRE_OUT [[buffer(7)]],
                               device float* POST_OUT [[buffer(8)]], device float* COMB_OUT [[buffer(9)]],
                               device bfloat* NORMED [[buffer(10)]], uint r [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[32][MIX + 1];
  threadgroup float tot[MIX + 1];
  threadgroup float pre_in[S];
  const device bfloat* x = X + size_t(r) * F;
  // the mixes' totals beside the previous boundary's lagged collapse, which overlaps the split's rms reduction
  if (MIXES && int(t) <= MIX) {
    float s = 0.0f;
    for (int j = 0; j < SLICES; j++) s += PART[(size_t(r) * SLICES + j) * (MIX + 1) + t];
    tot[t] = s;
  }
  if (t < uint(S)) pre_in[t] = PRE_IN[r * S + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ys[5];
  float ss = 0.0f;
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    float y = 0.0f;
    for (int s = 0; s < S; s++) y += pre_in[s] * float(x[s * D + d]);
    ys[i] = hc_bf(y);
    ss += ys[i] * ys[i];
  }
  const float s1 = simd_sum(ss);
  if (lane == 0) red[sg][0] = s1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the split on simdgroup 0: a pre and post a lane (0-3), a comb cell a lane (0-15), row and column sums by xor
  if (MIXES && sg == 0) {
    const float rs = precise::rsqrt(tot[MIX] / float(F) + a.eps);
    if (lane < uint(S)) {
      PRE_OUT[r * S + lane] = 1.0f / (1.0f + precise::exp(-(tot[lane] * rs * SCALE[0] + BASE[lane]))) + a.hc_eps;
      POST_OUT[r * S + lane] = 2.0f / (1.0f + precise::exp(-(tot[S + lane] * rs * SCALE[1] + BASE[S + lane])));
    }
    const bool act = lane < uint(S * S);
    const uint cell = act ? lane : 0;
    const float v = tot[2 * S + cell] * rs * SCALE[2] + BASE[2 * S + cell];
    float mx = max(v, simd_shuffle_xor(v, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
    const float e = precise::exp(v - mx);
    float row = e + simd_shuffle_xor(e, ushort(1));
    row = row + simd_shuffle_xor(row, ushort(2));
    float cm = act ? e / row + a.hc_eps : 0.0f;
    for (int it = 0; it < a.iters; it++) {
      if (it > 0) {
        float rsum = cm + simd_shuffle_xor(cm, ushort(1));
        rsum = rsum + simd_shuffle_xor(rsum, ushort(2));
        cm = act ? cm / (rsum + a.hc_eps) : 0.0f;
      }
      float csum = cm + simd_shuffle_xor(cm, ushort(4));
      csum = csum + simd_shuffle_xor(csum, ushort(8));
      cm = act ? cm / (csum + a.hc_eps) : 0.0f;
    }
    if (act) COMB_OUT[r * S * S + lane] = cm;
  }
  if (t == 32) {  // simdgroup 1, while simdgroup 0 runs the split
    float s = 0.0f;
    for (int k = 0; k < 32; k++) s += red[k][0];
    pre_in[0] = precise::rsqrt(s / float(D) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = pre_in[0];
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    NORMED[size_t(r) * D + d] = bfloat(float(NORMW[d]) * (ys[i] * inv));
  }
}

template [[host_name("tf_hc_boundary_mix")]] [[kernel]] decltype(tf_hc_boundary<true>) tf_hc_boundary<true>;
template [[host_name("tf_hc_boundary_plain")]] [[kernel]] decltype(tf_hc_boundary<false>) tf_hc_boundary<false>;

// ROUND-407: THE LAUNCH-OVERHEAD PROBE. A dispatch that reads the boundary's own partial sums and
// writes them back, so the split that follows DEPENDS on it: a chain of them prices the GPU's
// dependent-dispatch latency -- the piece every glue fusion would save -- and the probe's threadgroup
// size sweeps that latency's shape. The values are unchanged, so the reply cannot move.
kernel void tf_hc_probe(const device float *part [[buffer(0)]], device float *sink [[buffer(1)]],
                        uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint at = g * (SLICES * (MIX + 1)) + tid;
  if (tid <= uint(MIX)) sink[at] = part[at];
}

// ROUND-407: THE INDEPENDENT-DISPATCH PROBE. The same work-free dispatch, but each one writes its OWN
// sink (the host binds a different buffer a dispatch), so the chain carries no memory dependency between
// the probe dispatches: the GPU may run them concurrently. The gap between this form and tf_hc_probe is
// the GPU's DEPENDENT-dispatch latency -- the part of the launch cost that only a FOLD (not a reorder)
// can remove. The values the boundary reads are untouched, so the reply cannot move.
kernel void tf_hc_probe_indep(const device float *part [[buffer(0)]], device float *sink [[buffer(1)]],
                              uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  if (tid <= uint(MIX)) sink[tid] = part[tid];
}
