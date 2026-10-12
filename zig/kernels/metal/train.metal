// Learning in the weights: a low-rank change at each layer's output, and the gradients through Nemotron's layers to it.
#include <metal_stdlib>
using namespace metal;

// The sum over a threadgroup of `groups` simdgroups, the same in every thread.
inline float group_sum(float v, threadgroup float* scratch, uint lane, uint sg, uint groups) {
  v = simd_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = simd_sum(lane < groups ? scratch[lane] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

inline float group_max(float v, threadgroup float* scratch, uint lane, uint sg, uint groups) {
  v = simd_max(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = simd_max(lane < groups ? scratch[lane] : -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}

// The 4-bit LM head [vocab, dim] dequantized into bf16 [dim, vocab] for the backward product: grid (vocab, dim).
kernel void tf_train_head_t(const device uint* w [[buffer(0)]],
                            const device bfloat* scales [[buffer(1)]],
                            const device bfloat* biases [[buffer(2)]],
                            device bfloat* out [[buffer(3)]],
                            uint2 pos [[thread_position_in_grid]],
                            uint2 size [[threads_per_grid]]) {
  const uint v = pos.x, d = pos.y, vocab = size.x, dim = size.y;
  const size_t i = size_t(v) * dim + d;
  const uint q = (w[i / 8] >> (4 * (i % 8))) & 0xf;
  out[size_t(d) * vocab + v] = bfloat(float(scales[i / 64]) * float(q) + float(biases[i / 64]));
}

// A row's softmax over bf16 logits: stats = (loss, target's probability), then in place (p - onehot) weight[r].
kernel void tf_train_softmax(device bfloat* logits [[buffer(0)]],
                             const device uint* targets [[buffer(1)]],
                             const device float* weights [[buffer(2)]],
                             device float2* stats [[buffer(3)]],
                             constant uint& vocab [[buffer(4)]],
                             uint r [[threadgroup_position_in_grid]],
                             uint t [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  device bfloat* l = logits + size_t(r) * vocab;
  float m = -INFINITY;
  for (uint v = t; v < vocab; v += 1024) m = max(m, float(l[v]));
  m = group_max(m, scratch, lane, sg, 32);
  float s = 0;
  for (uint v = t; v < vocab; v += 1024) s += exp(float(l[v]) - m);
  s = group_sum(s, scratch, lane, sg, 32);
  const uint target = targets[r];
  const float lt = float(l[target]);
  if (t == 0) stats[r] = float2(log(s) + m - lt, exp(lt - m) / s);
  threadgroup_barrier(mem_flags::mem_device);
  const float w = weights[r];
  for (uint v = t; v < vocab; v += 1024) l[v] = bfloat((exp(float(l[v]) - m) / s - (v == target ? 1.0f : 0.0f)) * w);
}

// g[r] += s w dx - s^3 h (w dx . h) / dim: a row's input RMS norm undone onto the residual's gradient; 256 a row.
kernel void tf_train_rms_back(const device bfloat* h [[buffer(0)]],
                              const device bfloat* w [[buffer(1)]],
                              const device float* dx [[buffer(2)]],
                              device float* g [[buffer(3)]],
                              constant uint& dim [[buffer(4)]],
                              constant float& eps [[buffer(5)]],
                              uint r [[threadgroup_position_in_grid]],
                              uint t [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  const device bfloat* hr = h + size_t(r) * dim;
  const device float* dr = dx + size_t(r) * dim;
  float sq = 0, dot = 0;
  for (uint j = t; j < dim; j += 256) {
    const float v = float(hr[j]);
    sq += v * v;
    dot += float(w[j]) * dr[j] * v;
  }
  sq = group_sum(sq, scratch, lane, sg, 8);
  dot = group_sum(dot, scratch, lane, sg, 8);
  const float s = rsqrt(sq / float(dim) + eps), k = s * s * s * dot / float(dim);
  for (uint j = t; j < dim; j += 256) g[size_t(r) * dim + j] += s * float(w[j]) * dr[j] - k * float(hr[j]);
}

// xa[r, q] = x[r] . a[q], the change's input directions a [rank, in], and xn[r] = |x[r]|^2; a simdgroup per (q, r).
kernel void tf_train_lora_in(const device bfloat* x [[buffer(0)]],
                             const device float* a [[buffer(1)]],
                             device float* xa [[buffer(2)]],
                             constant uint4& dims [[buffer(3)]],
                             device float* xn [[buffer(4)]],
                             uint2 pos [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = dims.x, n_in = dims.y, rank = dims.z, q = pos.x / 32, r = pos.y;
  if (q >= rank || r >= rows) return;
  float s = 0, n = 0;
  for (uint i = lane; i < n_in; i += 32) {
    const float v = float(x[size_t(r) * n_in + i]);
    s += v * a[size_t(q) * n_in + i];
    n += v * v;
  }
  s = simd_sum(s);
  n = simd_sum(n);
  if (lane == 0) xa[size_t(r) * rank + q] = s;
  if (lane == 0 && q == 0) xn[r] = n;
}

// h[i] += delta[i]: a nudge to one bf16 row of the residual (h bound at that row).
kernel void tf_train_nudge(device bfloat* h [[buffer(0)]], const device float* delta [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  h[i] = bfloat(float(h[i]) + delta[i]);
}

// a pocket's neurons as the shared expert's are: each input product squared where positive, else zero
kernel void tf_pocket_relu2(device float* z [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float v = max(z[i], 0.0f);
  z[i] = v * v;
}

// tf_train_lora_in's products with tf_pocket_relu2 at the store, z[r, q] = relu(x[r] . up[q])^2 (no row lengths): a
// pocket's input half in one dispatch, the same bits; a simdgroup per (q, r).
kernel void tf_pocket_in(const device bfloat* x [[buffer(0)]],
                         const device float* up [[buffer(1)]],
                         device float* z [[buffer(2)]],
                         constant uint4& dims [[buffer(3)]],
                         uint2 pos [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = dims.x, n_in = dims.y, n = dims.z, q = pos.x / 32, r = pos.y;
  if (q >= n || r >= rows) return;
  float s = 0;
  for (uint i = lane; i < n_in; i += 32) {
    const float v = float(x[size_t(r) * n_in + i]);
    s += v * up[size_t(q) * n_in + i];
  }
  s = simd_sum(s);
  const float v = max(s, 0.0f);
  if (lane == 0) z[size_t(r) * n + q] = v * v;
}

// tf_pocket_relu2 over z [rows, n] (dims: count, n, from), neurons from `from` on times their row's scale
kernel void tf_pocket_relu2_rows(device float* z [[buffer(0)]], constant uint4& dims [[buffer(1)]],
                                 const device float* scale [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= dims.x) return;
  const float v = max(z[i], 0.0f), a = v * v;
  z[i] = i % dims.y >= dims.z ? a * scale[i / dims.y] : a;
}

// dst[r, i] = src[r, i] for `rows` rows of `width` bf16 values, src_stride and dst_stride apart; grid (width, rows)
kernel void tf_train_copy_cols(const device bfloat* src [[buffer(0)]], device bfloat* dst [[buffer(1)]],
                               constant uint4& dims [[buffer(2)]], uint2 pos [[thread_position_in_grid]]) {
  if (pos.x >= dims.z) return;
  dst[size_t(pos.y) * dims.y + pos.x] = src[size_t(pos.y) * dims.x + pos.x];
}

// A pass's rows by stream: each segment's first row, rows, its decision row (an index in the pass) and its latch slot.
struct Route {
  uint count;
  uint known; // 1: every latch was set before the pass (a training example), so rows after its decision row may act
  uint whole; // 1: lessons edit the weights, acting on every row: a training pass's flagged ones, an inference pass's all
  uint pad1;
  int4 seg[8];
};

// Block b acts on row r after its stream's decision row when its lesson chose to; the decision row adds to its score.
kernel void tf_train_gate(device float* xa [[buffer(0)]],
                          const device float* xn [[buffer(1)]],
                          const device float* tau [[buffer(2)]],
                          device float* gates [[buffer(3)]],
                          constant uint4& dims [[buffer(4)]],
                          constant float& unit [[buffer(5)]],
                          constant Route& route [[buffer(6)]],
                          const device uint* latch [[buffer(7)]],
                          device float* score [[buffer(8)]],
                          uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, rank = dims.y, blocks = dims.z, stride = dims.w, b = pos.x, r = pos.y;
  if (b >= blocks || r >= rows) return;
  int rel = -1;
  uint slot = 0;
  bool inside = false, settled = false; // settled: the decision row was in an earlier pass, so its latch is set
  for (uint s = 0; s < route.count; s++) {
    const int4 g = route.seg[s];
    if (int(r) >= g.x && int(r) < g.x + g.y) {
      inside = true;
      rel = int(r) - g.z;
      slot = uint(g.w);
      settled = route.known != 0 || g.z < g.x;
    }
  }
  device float* row = xa + size_t(r) * rank + b * 16;
  bool open = false;
  if (route.whole != 0) {
    open = inside && (route.known == 0 || latch[slot * stride + b] != 0);
  } else if (rel == 0) {
    const float n = unit * xn[r];
    if (n > 0) score[slot * stride + b] += row[0] * rsqrt(n) - tau[b];
  } else if (rel > 0 && settled) {
    open = latch[slot * stride + b] != 0;
  }
  gates[size_t(r) * blocks + b] = open ? 1.0f : 0.0f;
  if (!open)
    for (uint q = 0; q < 16; q++) row[q] = 0;
}

// A stream's lessons at its decision row: cleared before a pass; after it, those whose score reached `need` (0: never).
kernel void tf_lesson_latch(device float* score [[buffer(0)]],
                            const device float* need [[buffer(1)]],
                            device uint* latch [[buffer(2)]],
                            constant uint4& at [[buffer(3)]],
                            uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  const uint slot = at.x, blocks = at.y, stride = at.w;
  for (uint b = 0; b < blocks; b++) {
    latch[slot * stride + b] = at.z != 0 && need[b] > 0 && score[slot * stride + b] >= need[b] ? 1u : 0u;
    if (at.z == 0) score[slot * stride + b] = 0;
  }
}

// p[r, j] = f[j] . x[r] / |x[r]|: a row's input, made unit, along each candidate direction; a simdgroup a (j, r).
kernel void tf_train_project(const device bfloat* x [[buffer(0)]],
                             const device float* f [[buffer(1)]],
                             device float* p [[buffer(2)]],
                             constant uint4& dims [[buffer(3)]],
                             uint2 pos [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = dims.x, n_in = dims.y, k = dims.z, j = pos.x / 32, r = pos.y;
  if (j >= k || r >= rows) return;
  float s = 0, n = 0;
  for (uint i = lane; i < n_in; i += 32) {
    const float v = float(x[size_t(r) * n_in + i]);
    s += v * f[size_t(j) * n_in + i];
    n += v * v;
  }
  s = simd_sum(s);
  n = simd_sum(n);
  if (lane == 0) p[size_t(r) * k + j] = n > 0 ? s * rsqrt(n) : 0;
}

// y[r, j] += scale xa[r] . b[:, j], a change's up factor b [rank, out] added to a site's bf16 output; grid (out, rows).
kernel void tf_train_lora_out(const device float* xa [[buffer(0)]],
                              const device float* b [[buffer(1)]],
                              device bfloat* y [[buffer(2)]],
                              constant uint4& dims [[buffer(3)]],
                              constant float& scale [[buffer(4)]],
                              uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, n_out = dims.y, rank = dims.z, j = pos.x, r = pos.y;
  if (j >= n_out || r >= rows) return;
  float s = 0;
  for (uint q = 0; q < rank; q++) s += xa[size_t(r) * rank + q] * b[size_t(q) * n_out + j];
  const size_t at = size_t(r) * n_out + j;
  y[at] = bfloat(float(y[at]) + scale * s);
}

// db[q, j] += scale sum_r xa[r, first + q] g[r, j] for the open block's ranks; dims (rows, out, rank, first).
kernel void tf_train_lora_db(const device float* xa [[buffer(0)]],
                             const device float* g [[buffer(1)]],
                             device float* db [[buffer(2)]],
                             constant uint4& dims [[buffer(3)]],
                             constant float& scale [[buffer(4)]],
                             uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, n_out = dims.y, rank = dims.z, first = dims.w, j = pos.x, q = pos.y;
  if (j >= n_out || first + q >= rank) return;
  float s = 0;
  for (uint r = 0; r < rows; r++) s += xa[size_t(r) * rank + first + q] * g[size_t(r) * n_out + j];
  db[size_t(q) * n_out + j] += scale * s;
}

// dxa[r, q] = scale g[r] . b[q] where q's block is open on row r, else 0; dims (rows, out, rank); a simdgroup a (q, r).
kernel void tf_train_lora_dxa(const device float* g [[buffer(0)]],
                              const device float* b [[buffer(1)]],
                              device float* dxa [[buffer(2)]],
                              constant uint4& dims [[buffer(3)]],
                              constant float& scale [[buffer(4)]],
                              const device float* gates [[buffer(5)]],
                              uint2 pos [[thread_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = dims.x, n_out = dims.y, rank = dims.z, q = pos.x / 32, r = pos.y;
  if (q >= rank || r >= rows) return;
  float s = 0;
  for (uint j = lane; j < n_out; j += 32) s += g[size_t(r) * n_out + j] * b[size_t(q) * n_out + j];
  s = simd_sum(s);
  if (lane == 0) dxa[size_t(r) * rank + q] = scale * s * gates[size_t(r) * (rank / 16) + q / 16];
}

// dx[r, i] += sum_q dxa[r, q] a[q, i]: the change's part of its site's input gradient; dims (rows, in, rank).
kernel void tf_train_lora_dx(const device float* dxa [[buffer(0)]],
                             const device float* a [[buffer(1)]],
                             device float* dx [[buffer(2)]],
                             constant uint4& dims [[buffer(3)]],
                             uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, n_in = dims.y, rank = dims.z, i = pos.x, r = pos.y;
  if (i >= n_in || r >= rows) return;
  float s = 0;
  for (uint q = 0; q < rank; q++) s += dxa[size_t(r) * rank + q] * a[size_t(q) * n_in + i];
  dx[size_t(r) * n_in + i] += s;
}

// A sign from a row's place and a sketch column, +1 or -1 with even odds.
inline float coin(uint row, uint j, uint seed) {
  uint h = row * 0x9E3779B9u + j * 0x7FEB352Du + seed;
  h ^= h >> 16;
  h *= 0x7FEB352Du;
  h ^= h >> 15;
  h *= 0x846CA68Bu;
  h ^= h >> 16;
  return (h & 1) ? 1.0f : -1.0f;
}

// y[j, i] += w sum_r coin(first + r, j) x[r, i]: a site's input rows into a random sketch; dims (rows, in, k, first).
kernel void tf_train_sketch(const device bfloat* x [[buffer(0)]],
                            device float* y [[buffer(1)]],
                            constant uint4& dims [[buffer(2)]],
                            constant uint& seed [[buffer(3)]],
                            constant float& w [[buffer(4)]],
                            uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, n_in = dims.y, k = dims.z, first = dims.w, i = pos.x, j = pos.y;
  if (i >= n_in || j >= k) return;
  float s = 0;
  for (uint r = 0; r < rows; r++) s += coin(first + r, j, seed) * float(x[size_t(r) * n_in + i]);
  y[size_t(j) * n_in + i] += w * s;
}

// du = da 2 relu(u): the squared ReLU's backward on its bf16 pre-activation.
kernel void tf_train_relu2_back(const device bfloat* u [[buffer(0)]],
                                const device float* da [[buffer(1)]],
                                device float* du [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
  du[i] = da[i] * 2.0f * max(float(u[i]), 0.0f);
}

// dy[s] = wt[p] g[p / top_k] at sorted slot s of pair p = order[s]: the routed experts' output gradients.
kernel void tf_train_pairs_in(const device uint* order [[buffer(0)]],
                              const device float* wt [[buffer(1)]],
                              const device float* g [[buffer(2)]],
                              device float* dy [[buffer(3)]],
                              constant uint2& dims [[buffer(4)]],
                              uint2 pos [[thread_position_in_grid]]) {
  const uint dim = dims.x, top_k = dims.y, j = pos.x, s = pos.y;
  if (j >= dim) return;
  const uint p = order[s];
  dy[size_t(s) * dim + j] = wt[p] * g[size_t(p / top_k) * dim + j];
}

// dx[t] += its top_k pairs' input gradients, summed in pair order through inv (pair to sorted slot); dims (dim, top_k).
kernel void tf_train_pairs_out(const device uint* inv [[buffer(0)]],
                               const device float* dxp [[buffer(1)]],
                               device float* dx [[buffer(2)]],
                               constant uint2& dims [[buffer(3)]],
                               uint2 pos [[thread_position_in_grid]]) {
  const uint dim = dims.x, top_k = dims.y, j = pos.x, t = pos.y;
  if (j >= dim) return;
  float s = 0;
  for (uint k = 0; k < top_k; k++) s += dxp[size_t(inv[t * top_k + k]) * dim + j];
  dx[size_t(t) * dim + j] += s;
}

// Adam on a change's factors (hp: lr, beta1, beta2, eps; corr: bias corrections); the gradient is then cleared.
kernel void tf_train_adam(device float* p [[buffer(0)]],
                          device float* g [[buffer(1)]],
                          device float* m [[buffer(2)]],
                          device float* v [[buffer(3)]],
                          constant float4& hp [[buffer(4)]],
                          constant float2& corr [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
  const float gi = g[i];
  const float mi = hp.y * m[i] + (1 - hp.y) * gi, vi = hp.z * v[i] + (1 - hp.z) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  p[i] -= hp.x * (mi * corr.x) / (sqrt(vi * corr.y) + hp.w);
  g[i] = 0;
}

// y[r] += R x[r], R bf16 [N, K]: a projection's weight beyond its 4-bit codes; a simdgroup an output, 8 rows a pass.
kernel void tf_residual(const device bfloat* x [[buffer(0)]],
                        const device bfloat* rest [[buffer(1)]],
                        device bfloat* y [[buffer(2)]],
                        constant uint3& dims [[buffer(3)]],
                        uint2 pos [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = dims.x, n = dims.y, width = dims.z, j = pos.x / 32, r0 = pos.y * 8;
  if (j >= n) return;
  const device bfloat4* w = (const device bfloat4*)(rest + size_t(j) * width);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint i = lane; i < width / 4; i += 32) {
    const float4 wi = float4(w[i]);
    for (uint r = 0; r < 8 && r0 + r < rows; r++) acc[r] += dot(wi, float4(((const device bfloat4*)(x + size_t(r0 + r) * width))[i]));
  }
  for (uint r = 0; r < 8; r++) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r0 + r < rows) y[size_t(r0 + r) * n + j] = bfloat(float(y[size_t(r0 + r) * n + j]) + s);
  }
}

// dx[r] += sum_j dz_j gate[e_j]: routing weights scale s_j / sum s (s = sigmoid(logit)) back into the router's input.
kernel void tf_train_route_back(const device bfloat* logits [[buffer(0)]],
                                const device uint* ids [[buffer(1)]],
                                const device bfloat* ys [[buffer(2)]],
                                const device float* g [[buffer(3)]],
                                const device bfloat* gate [[buffer(4)]],
                                device float* dx [[buffer(5)]],
                                constant uint4& dims [[buffer(6)]],
                                constant float& scale [[buffer(7)]],
                                uint r [[threadgroup_position_in_grid]],
                                uint t [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint experts = dims.y, k = dims.z, d = dims.w;
  threadgroup float scratch[32];
  threadgroup float dz[16];
  float a[16];
  for (uint j = 0; j < k; j++) {
    float s = 0;
    for (uint i = t; i < d; i += 256) s += g[size_t(r) * d + i] * float(ys[(size_t(r) * k + j) * d + i]);
    a[j] = group_sum(s, scratch, lane, sg, 8);
  }
  if (t == 0) {
    float p[16];
    float total = 0;
    for (uint j = 0; j < k; j++) {
      const float z = float(logits[size_t(r) * experts + ids[r * k + j]]);
      p[j] = 1 / (1 + exp(-z));
      total += p[j];
    }
    total += 1e-20f;
    float mix = 0;
    for (uint j = 0; j < k; j++) mix += a[j] * p[j] / total;
    for (uint j = 0; j < k; j++) dz[j] = scale / total * (a[j] - mix) * p[j] * (1 - p[j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = t; i < d; i += 256) {
    float s = 0;
    for (uint j = 0; j < k; j++) s += dz[j] * float(gate[size_t(ids[r * k + j]) * d + i]);
    dx[size_t(r) * d + i] += s;
  }
}

// dx[r, i] += sum_j g[r, j] rest[j, i]: a projection's bf16 residual [N, K] carried back to its input; grid (K, rows).
kernel void tf_train_rest_back(const device float* g [[buffer(0)]],
                               const device bfloat* rest [[buffer(1)]],
                               device float* dx [[buffer(2)]],
                               constant uint3& dims [[buffer(3)]],
                               uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, n = dims.y, width = dims.z, i = pos.x, r = pos.y;
  if (i >= width || r >= rows) return;
  float s = 0;
  for (uint j = 0; j < n; j++) s += g[size_t(r) * n + j] * float(rest[size_t(j) * width + i]);
  dx[size_t(r) * width + i] += s;
}

kernel void tf_train_zero(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  x[i] = 0;
}

// out = float(x): bf16 rows widened, for gradients that start from a bf16 product.
kernel void tf_train_widen(const device bfloat* x [[buffer(0)]], device float* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  out[i] = float(x[i]);
}

// A 4-bit [N, K] matrix (MLX's layout) as bf16 [K, N], for the products that read it the other way round: grid (N, K).
kernel void tf_train_dequant_t(const device uint* w [[buffer(0)]],
                               const device bfloat* scales [[buffer(1)]],
                               const device bfloat* biases [[buffer(2)]],
                               device bfloat* out [[buffer(3)]],
                               uint2 pos [[thread_position_in_grid]],
                               uint2 size [[threads_per_grid]]) {
  const uint n = pos.x, k = pos.y, rows = size.x, cols = size.y;
  const size_t i = size_t(n) * cols + k;
  const uint q = (w[i / 8] >> (4 * (i % 8))) & 0xf;
  out[size_t(k) * rows + n] = bfloat(float(scales[i / 64]) * float(q) + float(biases[i / 64]));
}

kernel void tf_train_narrow(const device float* x [[buffer(0)]], device bfloat* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  out[i] = bfloat(x[i]);
}

kernel void tf_train_add(const device float* x [[buffer(0)]], device float* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  y[i] += x[i];
}

// Each expert's pairs cut into tiles of 16 rows: tiles[t] = (expert, first row), then (experts, 0) up to `cap`.
kernel void tf_train_experts_tiles(const device int* offsets [[buffer(0)]],
                                   device uint2* tiles [[buffer(1)]],
                                   constant uint4& dims [[buffer(2)]]) {
  const uint pairs = dims.x, cap = dims.y, experts = dims.w;
  uint t = 0;
  for (uint e = 0; e < experts; e++) {
    const uint first = uint(offsets[e]), last = e + 1 < experts ? uint(offsets[e + 1]) : pairs;
    for (uint r = first; r < last && t < cap; r += 16) tiles[t++] = uint2(e, r);
  }
  for (; t < cap; t++) tiles[t] = uint2(experts, 0);
}

// y[s] = x[s] w[e] (f32), tiles of 16 pairs by 64 columns in 8x8 simdgroup products, w dequantized once a tile.
kernel void tf_train_experts_mm(const device float* x [[buffer(0)]],
                                const device uint* w [[buffer(1)]],
                                const device bfloat* scales [[buffer(2)]],
                                const device bfloat* biases [[buffer(3)]],
                                const device uint2* tiles [[buffer(4)]],
                                const device int* offsets [[buffer(5)]],
                                device float* y [[buffer(6)]],
                                constant uint4& dims [[buffer(7)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint tid [[thread_index_in_threadgroup]]) {
  const uint pairs = dims.x, n_in = dims.y, k_out = dims.z, experts = dims.w, words = k_out / 8, groups = k_out / 64;
  const uint2 tile = tiles[tg.y];
  if (tile.x >= experts) return;
  const uint e = tile.x, r0 = tile.y, k0 = tg.x * 64;
  const uint last = e + 1 < experts ? uint(offsets[e + 1]) : pairs;
  threadgroup float bs[32 * 64];
  // simdgroup sg: rows rb * 8 of the tile, columns ch * 32 of its 64
  const uint rb = sg & 1, ch = sg >> 1;
  simdgroup_float8x8 acc[4];
  for (uint j = 0; j < 4; j++) acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const device uint* we = w + size_t(e) * n_in * words + k0 / 8;
  const device bfloat* se = scales + size_t(e) * n_in * groups + k0 / 64;
  const device bfloat* be = biases + size_t(e) * n_in * groups + k0 / 64;
  const device float* xr = x + size_t(r0 + rb * 8) * n_in;
  for (uint n0 = 0; n0 < n_in; n0 += 32) {
    for (uint i = tid; i < 256; i += 128) {
      const uint n = n0 + i / 8, word = i % 8;
      const uint q = we[size_t(n) * words + word];
      const float sc = float(se[size_t(n) * groups]), bi = float(be[size_t(n) * groups]);
      for (uint j = 0; j < 8; j++) bs[(i / 8) * 64 + word * 8 + j] = fma(sc, float((q >> (4 * j)) & 0xf), bi);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32; kk += 8) {
      simdgroup_float8x8 a;
      simdgroup_load(a, xr + n0 + kk, n_in);
      for (uint j = 0; j < 4; j++) {
        simdgroup_float8x8 b;
        simdgroup_load(b, bs + kk * 64 + ch * 32 + j * 8, 64);
        simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // through threadgroup memory: only the expert's own rows reach y
  for (uint j = 0; j < 4; j++) simdgroup_store(acc[j], bs + rb * 8 * 64 + ch * 32 + j * 8, 64);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 16 * 64; i += 128) {
    const uint r = r0 + i / 64;
    if (r < last) y[size_t(r) * k_out + k0 + i % 64] = bs[i];
  }
}
