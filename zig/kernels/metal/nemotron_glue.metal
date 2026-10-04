// Our kernels around Nemotron's generated ones: embedding, MLX's RMS arithmetic, argmax and layout moves.
#include <metal_stdlib>
using namespace metal;

// Embedding rows of the 4-bit table, bf16(scale * q + bias) rounded once as MLX's dequantize; a thread a byte.
kernel void tf_embed_q4(const device uint* ids [[buffer(0)]],
                        const device uchar* weight [[buffer(1)]],
                        const device bfloat* scales [[buffer(2)]],
                        const device bfloat* biases [[buffer(3)]],
                        device bfloat* out [[buffer(4)]],
                        constant uint& dims [[buffer(5)]],
                        uint2 pos [[thread_position_in_grid]]) {
  const uint b = pos.x, r = pos.y;
  const size_t id = ids[r];
  const uchar v = weight[id * (dims / 2) + b];
  const size_t g = id * (dims / 64) + (2 * b) / 64;
  const float s = float(scales[g]), z = float(biases[g]);
  out[size_t(r) * dims + 2 * b] = bfloat(s * float(v & 0xf) + z);
  out[size_t(r) * dims + 2 * b + 1] = bfloat(s * float(v >> 4) + z);
}

// MLX's one-row RMS arithmetic: 4 squares a thread, two simd_sums, precise rsqrt, out = bf16(w * bf16(x * inv)).
kernel void tf_rms_mlx(const device bfloat* x [[buffer(0)]],
                       const device bfloat* w [[buffer(1)]],
                       device bfloat* out [[buffer(2)]],
                       constant float& eps [[buffer(3)]],
                       constant uint& dims [[buffer(4)]],
                       constant uint& w_stride [[buffer(5)]],
                       constant uint2& strides [[buffer(6)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint t [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float inv[1];
  threadgroup float sums[32];
  const size_t base = size_t(row) * strides.x;
  const size_t obase = size_t(row) * strides.y;
  float v[4];
  float acc = 0.0f;
  for (int i = 0; i < 4; i++) {
    v[i] = 4 * t + i < dims ? float(x[base + 4 * t + i]) : 0.0f;
    acc += v[i] * v[i];
  }
  acc = simd_sum(acc);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sums[lane]);
    if (lane == 0) inv[0] = metal::precise::rsqrt(all / float(dims) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 4; i++)
    if (4 * t + i < dims) out[obase + 4 * t + i] = bfloat(float(w[w_stride * (4 * t + i)]) * float(bfloat(v[i] * inv[0])));
}

// The better of two (value, index) candidates: NaN first, then the larger value, ties to the lower index (MLX argmax).
inline bool tf_better(float a, uint ia, float b, uint ib) {
  if (isnan(a)) return !isnan(b) || ia < ib;
  if (isnan(b)) return false;
  return a > b || (a == b && ia < ib);
}

// Greedy draw: argmax of each bf16 logits row, one 1024-thread threadgroup a row; ids[r] = map[the winning column].
kernel void tf_argmax_bf16(const device bfloat* logits [[buffer(0)]],
                           device uint* ids [[buffer(1)]],
                           constant uint& vocab [[buffer(2)]],
                           const device uint* map [[buffer(3)]],
                           constant uint& mapped [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint t [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float vals[32];
  threadgroup uint idxs[32];
  const device bfloat* x = logits + size_t(row) * vocab;
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < vocab; i += 1024) {
    const float v = float(x[i]);
    if (tf_better(v, i, best, at)) { best = v; at = i; }
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    const float ov = simd_shuffle_xor(best, d);
    const uint oi = simd_shuffle_xor(at, d);
    if (tf_better(ov, oi, best, at)) { best = ov; at = oi; }
  }
  if (lane == 0) { vals[sg] = best; idxs[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    best = vals[lane];
    at = idxs[lane];
    for (ushort d = 16; d > 0; d >>= 1) {
      const float ov = simd_shuffle_xor(best, d);
      const uint oi = simd_shuffle_xor(at, d);
      if (tf_better(ov, oi, best, at)) { best = ov; at = oi; }
    }
    if (lane == 0) ids[row] = mapped != 0 ? map[at] : at;
  }
}

// The stacked q/k/v output's new K and V rows into the caches at row `start`; grid (HD, KVH, R).
kernel void tf_kv_write(const device bfloat* qkv [[buffer(0)]],
                        device bfloat* kc [[buffer(1)]],
                        device bfloat* vc [[buffer(2)]],
                        constant uint4& dims [[buffer(3)]],     // (QKV, KOFF, capacity, start)
                        uint3 pos [[thread_position_in_grid]],
                        uint3 size [[threads_per_grid]]) {
  const uint d = pos.x, h = pos.y, t = pos.z, hd = size.x, kvh = size.y;
  const size_t src = size_t(t) * dims.x + dims.y + h * hd + d;
  const size_t dst = (size_t(h) * dims.z + dims.w + t) * hd + d;
  kc[dst] = qkv[src];
  vc[dst] = qkv[src + kvh * hd];
}

// lane_sdpa's Qp [KVH, T * G, HD] from the stacked output's q columns; grid (HD, G * T, KVH).
kernel void tf_attn_q(const device bfloat* qkv [[buffer(0)]],
                      device bfloat* qp [[buffer(1)]],
                      constant uint2& dims [[buffer(2)]],       // (QKV, G)
                      uint3 pos [[thread_position_in_grid]],
                      uint3 size [[threads_per_grid]]) {
  const uint d = pos.x, rg = pos.y, hk = pos.z, hd = size.x, g_all = size.y;
  const uint t = rg / dims.y, g = rg % dims.y;
  qp[(size_t(hk) * g_all + rg) * hd + d] = qkv[size_t(t) * dims.x + (hk * dims.y + g) * hd + d];
}

// lane_sdpa's output [H, T, HD] to rows [T, H * HD] for o_proj. grid (HD, H, T).
kernel void tf_attn_out(const device bfloat* o [[buffer(0)]],
                        device bfloat* x [[buffer(1)]],
                        uint3 pos [[thread_position_in_grid]],
                        uint3 size [[threads_per_grid]]) {
  const uint d = pos.x, h = pos.y, t = pos.z, hd = size.x, heads = size.y, rows = size.z;
  x[(size_t(t) * heads + h) * hd + d] = o[(size_t(h) * rows + t) * hd + d];
}

// Rows of bf16 from one buffer to another: grid (D, R).
kernel void tf_copy_rows(const device bfloat* src [[buffer(0)]],
                         device bfloat* dst [[buffer(1)]],
                         uint2 pos [[thread_position_in_grid]],
                         uint2 size [[threads_per_grid]]) {
  dst[size_t(pos.y) * size.x + pos.x] = src[size_t(pos.y) * size.x + pos.x];
}

// u32 values from one buffer to another (window ids into a shared round's ids): grid (n).
kernel void tf_copy_u32(const device uint* src [[buffer(0)]],
                        device uint* dst [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  dst[i] = src[i];
}

// lane_qmm's split-K partials summed in slice order, as the coop kernel's threadgroup reduction does, then rounded to bf16: grid (N, M).
kernel void tf_coop_combine(const device float* part [[buffer(0)]],
                            const device int* dims [[buffer(1)]],
                            device bfloat* y [[buffer(2)]],
                            uint2 pos [[thread_position_in_grid]]) {
  const int n_cols = dims[0], m_rows = dims[1], mp = dims[2], sk = dims[3];
  if (int(pos.x) >= n_cols || int(pos.y) >= m_rows) return;
  float c = part[size_t(pos.y) * n_cols + pos.x];
  for (int s = 1; s < sk; s++) c += part[(size_t(s) * mp + pos.y) * n_cols + pos.x];
  y[size_t(pos.y) * n_cols + pos.x] = static_cast<bfloat>(c);
}
