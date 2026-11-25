// Fused DFlash2 decode kernels: K-norm+RoPE+append, QKV prep, and dynamic-conv2+residual+RMSNorm.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL_GLM_DFOPT
#define TF_CTL_GLM_DFOPT 0
#endif

struct GlmDfAheadStream {
  ulong first;
  uint row0, rows;
};

struct GlmDfAheadArgs {
  uint total, kv_heads, dim, kv_stride, kv_width, ring, streams;
  float eps, theta;
};

struct GlmDfQkvArgs {
  uint rows, heads, kv_heads, dim, qkv_stride;
  float eps, theta;
};

struct GlmDfConvNormArgs {
  uint rows, width, group, block, branch;
  float eps;
};

#define GLM_DF_RING(R, s) ((s) == 0 ? R##0 : (s) == 1 ? R##1 : (s) == 2 ? R##2 : (s) == 3 ? R##3 : \
                           (s) == 4 ? R##4 : (s) == 5 ? R##5 : (s) == 6 ? R##6 : R##7)
#define GLM_DF_RINGS(R, at) \
    device bfloat* R##0 [[buffer(at)]], device bfloat* R##1 [[buffer(at + 1)]], \
    device bfloat* R##2 [[buffer(at + 2)]], device bfloat* R##3 [[buffer(at + 3)]], \
    device bfloat* R##4 [[buffer(at + 4)]], device bfloat* R##5 [[buffer(at + 5)]], \
    device bfloat* R##6 [[buffer(at + 6)]], device bfloat* R##7 [[buffer(at + 7)]]

// Per-layer fused k_norm + full_rope + stream K/V ring scatter, matching draft_ops rounding.
[[max_total_threads_per_threadgroup(256)]]
kernel void glm_dflash_kv_append(
    const device bfloat* KV [[buffer(0)]], const device bfloat* KW [[buffer(1)]],
    constant GlmDfAheadArgs& a [[buffer(2)]], constant GlmDfAheadStream* streams [[buffer(3)]],
    GLM_DF_RINGS(K, 4), GLM_DF_RINGS(V, 12),
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= a.kv_heads || row >= a.total) return;
  if (TF_CTL_GLM_DFOPT == 1 && head == 0u && row == 0u && lane == 0u) ((device uint16_t*)KV)[1] ^= 1u;
  uint s = 0u;
  while (s + 1u < a.streams && row >= streams[s].row0 + streams[s].rows) ++s;
  const constant GlmDfAheadStream& st = streams[s];
  const ulong p = st.first + ulong(row - st.row0);
  const uint slot = uint(p % ulong(a.ring));
  const device bfloat* k_row = KV + size_t(row) * a.kv_stride + size_t(head) * a.dim;
  const device bfloat* v_row = KV + size_t(row) * a.kv_stride + a.kv_width + size_t(head) * a.dim;
  threadgroup bfloat nrm[128];
  float sum = 0.0f;
  const uint base = 4u * lane;
  if (base < a.dim) {
    for (uint j = 0u; j < 4u && base + j < a.dim; ++j) {
      const float x = float(k_row[base + j]);
      sum += x * x;
    }
  }
  const float inv = metal::precise::rsqrt(simd_sum(sum) / float(a.dim) + a.eps);
  if (base < a.dim) {
    for (uint j = 0u; j < 4u && base + j < a.dim; ++j) {
      const uint col = base + j;
      const float normalized = float(bfloat(float(k_row[col]) * inv));
      nrm[col] = bfloat(normalized * float(KW[col]));
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* Kd = GLM_DF_RING(K, s) + (size_t(slot) * a.kv_heads + head) * a.dim;
  device bfloat* Vd = GLM_DF_RING(V, s) + (size_t(slot) * a.kv_heads + head) * a.dim;
  const uint pairs = a.dim / 2u;
  for (uint col = lane; col < pairs; col += 32u) {
    if (TF_CTL_GLM_DFOPT == 2 && col + 1u == pairs) continue;
    const float angle = float(p) * pow(a.theta, -float(col) / float(pairs));
    const float cs = cos(angle), sn = sin(angle);
    const float left = float(nrm[col]), right = float(nrm[col + pairs]);
    Kd[col] = bfloat(left * cs - right * sn);
    Kd[col + pairs] = bfloat(right * cs + left * sn);
  }
  for (uint col = lane; col < a.dim; col += 32u) Vd[col] = v_row[col];
}

// Single-pass QKV prep: q_norm+RoPE -> Q, k_norm+RoPE -> K, V -> contiguous V.
[[max_total_threads_per_threadgroup(256)]]
kernel void glm_dflash_qkv_prep(
    const device bfloat* QKV [[buffer(0)]], const device ulong* POS [[buffer(1)]],
    const device bfloat* QW [[buffer(2)]], const device bfloat* KW [[buffer(3)]],
    device bfloat* Q [[buffer(4)]], device bfloat* K [[buffer(5)]], device bfloat* V [[buffer(6)]],
    constant GlmDfQkvArgs& a [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint unit = tg.x, row = tg.y;
  if (unit >= a.heads + a.kv_heads || row >= a.rows) return;
  if (TF_CTL_GLM_DFOPT == 1 && unit == 0u && row == 0u && lane == 0u) ((device uint16_t*)QKV)[1] ^= 1u;
  const bool is_q = unit < a.heads;
  const uint head = is_q ? unit : unit - a.heads;
  const uint q_w = a.heads * a.dim;
  const uint k_w = a.kv_heads * a.dim;
  const device bfloat* src = QKV + size_t(row) * a.qkv_stride + (is_q ? 0u : q_w) + size_t(head) * a.dim;
  const device bfloat* w = is_q ? QW : KW;
  threadgroup bfloat nrm[128];
  float sum = 0.0f;
  const uint base = 4u * lane;
  if (base < a.dim) {
    for (uint j = 0u; j < 4u && base + j < a.dim; ++j) {
      const float x = float(src[base + j]);
      sum += x * x;
    }
  }
  const float inv = metal::precise::rsqrt(simd_sum(sum) / float(a.dim) + a.eps);
  if (base < a.dim) {
    for (uint j = 0u; j < 4u && base + j < a.dim; ++j) {
      const uint col = base + j;
      const float normalized = float(bfloat(float(src[col]) * inv));
      nrm[col] = bfloat(normalized * float(w[col]));
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* dst = is_q ? Q + ((size_t)row * a.heads + head) * a.dim
                            : K + ((size_t)row * a.kv_heads + head) * a.dim;
  const float p = float(POS[row]);
  const uint pairs = a.dim / 2u;
  for (uint col = lane; col < pairs; col += 32u) {
    if (TF_CTL_GLM_DFOPT == 2 && col + 1u == pairs) continue;
    const float angle = p * pow(a.theta, -float(col) / float(pairs));
    const float cs = cos(angle), sn = sin(angle);
    const float left = float(nrm[col]), right = float(nrm[col + pairs]);
    dst[col] = bfloat(left * cs - right * sn);
    dst[col + pairs] = bfloat(right * cs + left * sn);
  }
  if (is_q) return;
  const device bfloat* vs = QKV + size_t(row) * a.qkv_stride + q_w + k_w + size_t(head) * a.dim;
  device bfloat* vd = V + ((size_t)row * a.kv_heads + head) * a.dim;
  for (uint col = lane; col < a.dim; col += 32u) vd[col] = vs[col];
}

// Fused dynamic conv2 (branch=1, residual=1) -> HIDDEN, then RMSNorm -> NORM in exact draft_ops order.
[[max_total_threads_per_threadgroup(1024)]]
kernel void glm_dflash_conv_norm(
    const device bfloat* X [[buffer(0)]], const device bfloat* D [[buffer(1)]],
    const device bfloat* B [[buffer(2)]], const device bfloat* R [[buffer(3)]],
    const device bfloat* NW [[buffer(4)]], device bfloat* HIDDEN [[buffer(5)]],
    device bfloat* NORM [[buffer(6)]], constant GlmDfConvNormArgs& a [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sg_sums[32], inv_rms[1];
  const uint width = a.width, groups = width / a.group;
  const uint dyn_stride = 4u * groups, first = 4u * t;
  if (TF_CTL_GLM_DFOPT == 1 && row == 0u && t == 0u) ((device uint16_t*)X)[1] ^= 1u;
  const uint active_width = (TF_CTL_GLM_DFOPT == 2) ? (width - 4u) : width;
  float hval[4];
  float sq = 0.0f;
  for (uint j = 0u; j < 4u && first + j < active_width; ++j) {
    const uint col = first + j, g = col / a.group;
    float total = 0.0f;
    for (uint lag = 0u; lag < 2u; ++lag) {
      const float x = (lag && (row % a.block == 0u)) ? 0.0f : float(X[size_t(row - lag) * width + col]);
      const float base = float(B[(a.branch * 2u + lag) * width + col]);
      const float dyn = float(D[size_t(row) * dyn_stride + (a.branch * 2u + lag) * groups + g]);
      total = float(bfloat(total + float(bfloat(base * x))));
      total = float(bfloat(total + float(bfloat(dyn * x))));
    }
    const float conv = float(bfloat(total));
    const bfloat h = bfloat(float(R[size_t(row) * width + col]) + conv);
    HIDDEN[size_t(row) * width + col] = h;
    hval[j] = float(h);
    sq += hval[j] * hval[j];
  }
  sq = simd_sum(sq);
  if (sg == 0u) sg_sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) sg_sums[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    const float all = simd_sum(sg_sums[lane]);
    if (lane == 0u) inv_rms[0] = metal::precise::rsqrt(all / float(width) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = inv_rms[0];
  for (uint j = 0u; j < 4u && first + j < active_width; ++j) {
    const uint col = first + j;
    const float normalized = float(bfloat(hval[j] * inv));
    NORM[size_t(row) * width + col] = bfloat(normalized * float(NW[col]));
  }
}
