// The MTP head's one-row kernels fused, each step's arithmetic as the kernels it replaces (bit-identical outputs).
#include <metal_stdlib>
using namespace metal;

// A row's RMS factor as tf_rms_mlx sums it: simd sums of each thread's squares, then of the simdgroups' sums.
inline float tf_head_inv(float acc, threadgroup float* sums, uint lane, uint sg, float dims, float eps) {
  acc = simd_sum(acc);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sums[lane]);
    if (lane == 0) sums[32] = metal::precise::rsqrt(all / dims + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return sums[32];
}

// tf_embed_q4, tf_rms_mlx (enorm, hnorm) and xsum over [enorm(embed(id)), hnorm(h)]: a threadgroup a row, D / 4 threads.
kernel void tf_head_prep(const device uint* ids [[buffer(0)]],
                         const device uchar* weight [[buffer(1)]],
                         const device bfloat* scales [[buffer(2)]],
                         const device bfloat* biases [[buffer(3)]],
                         const device bfloat* we [[buffer(4)]],
                         const device bfloat* wh [[buffer(5)]],
                         const device bfloat* h [[buffer(6)]],
                         device bfloat* cat [[buffer(7)]],
                         device float* xs [[buffer(8)]],
                         constant float& eps [[buffer(9)]],
                         constant uint2& dims [[buffer(10)]],  // D, MP
                         uint row [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums_e[33];
  threadgroup float sums_h[33];
  threadgroup bfloat out[2 * 4096];
  const uint D = dims.x, MP = dims.y;
  const size_t id = ids[row];
  float ve[4], vh[4];
  float acc_e = 0.0f, acc_h = 0.0f;
  for (int k = 0; k < 2; k++) {
    const uint b = 2 * t + k;
    const uchar v = weight[id * (D / 2) + b];
    const size_t g = id * (D / 64) + (2 * b) / 64;
    const float s = float(scales[g]), z = float(biases[g]);
    ve[2 * k] = float(bfloat(s * float(v & 0xf) + z));
    ve[2 * k + 1] = float(bfloat(s * float(v >> 4) + z));
  }
  for (int i = 0; i < 4; i++) {
    acc_e += ve[i] * ve[i];
    vh[i] = float(h[size_t(row) * D + 4 * t + i]);
    acc_h += vh[i] * vh[i];
  }
  const float inv_e = tf_head_inv(acc_e, sums_e, lane, sg, float(D), eps);
  const float inv_h = tf_head_inv(acc_h, sums_h, lane, sg, float(D), eps);
  for (int i = 0; i < 4; i++) {
    const uint at = 4 * t + i;
    const bfloat oe = bfloat(float(we[at]) * float(bfloat(ve[i] * inv_e)));
    const bfloat oh = bfloat(float(wh[at]) * float(bfloat(vh[i] * inv_h)));
    out[at] = oe;
    out[D + at] = oh;
    cat[size_t(row) * 2 * D + at] = oe;
    cat[size_t(row) * 2 * D + D + at] = oh;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 2 * D / 64) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(out[t * 64 + i]);
    xs[t * MP + row] = acc;
  }
}

// tf_coop_combine (split-K partials in slice order), tf_rms_mlx and xsum: the residual, its norm and the norm's sums.
kernel void tf_head_norm(const device float* part [[buffer(0)]],
                         const device bfloat* w [[buffer(1)]],
                         device bfloat* hout [[buffer(2)]],
                         device bfloat* x [[buffer(3)]],
                         device float* xs [[buffer(4)]],
                         constant float& eps [[buffer(5)]],
                         constant uint4& dims [[buffer(6)]],  // D, MP, slices
                         uint row [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[33];
  threadgroup bfloat out[4096];
  const uint D = dims.x, MP = dims.y, SK = dims.z;
  float v[4];
  float acc = 0.0f;
  for (int i = 0; i < 4; i++) {
    const uint at = 4 * t + i;
    float c = part[size_t(row) * D + at];
    for (uint s = 1; s < SK; s++) c += part[(size_t(s) * MP + row) * D + at];
    const bfloat hb = static_cast<bfloat>(c);
    hout[size_t(row) * D + at] = hb;
    v[i] = float(hb);
    acc += v[i] * v[i];
  }
  const float inv = tf_head_inv(acc, sums, lane, sg, float(D), eps);
  for (int i = 0; i < 4; i++) {
    const uint at = 4 * t + i;
    const bfloat o = bfloat(float(w[at]) * float(bfloat(v[i] * inv)));
    out[at] = o;
    x[size_t(row) * D + at] = o;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < D / 64) {
    float a = 0.0f;
    for (int i = 0; i < 64; i++) a += float(out[t * 64 + i]);
    xs[t * MP + row] = a;
  }
}
