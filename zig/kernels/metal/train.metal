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

// Block b opens on row r if the row's cosine with b's first direction reaches tau, else its xa and gate are 0.
kernel void tf_train_gate(device float* xa [[buffer(0)]],
                          const device float* xn [[buffer(1)]],
                          const device float* tau [[buffer(2)]],
                          device float* gates [[buffer(3)]],
                          constant uint4& dims [[buffer(4)]],
                          constant float& unit [[buffer(5)]],
                          uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, rank = dims.y, blocks = dims.z, b = pos.x, r = pos.y;
  if (b >= blocks || r >= rows) return;
  device float* row = xa + size_t(r) * rank + b * 16;
  const float n = unit * xn[r];
  const bool open = n > 0 && row[0] >= tau[b] * sqrt(n);
  gates[size_t(r) * blocks + b] = open ? 1.0f : 0.0f;
  if (!open)
    for (uint q = 0; q < 16; q++) row[q] = 0;
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

// part[slice][s] = x[s] w[e] on a slice of w's rows, 4 rows of x and 4 of w at a time, 8 columns a thread.
kernel void tf_train_experts_part(const device float* x [[buffer(0)]],
                                  const device uint* w [[buffer(1)]],
                                  const device bfloat* scales [[buffer(2)]],
                                  const device bfloat* biases [[buffer(3)]],
                                  const device int* offsets [[buffer(4)]],
                                  device float* part [[buffer(5)]],
                                  constant uint4& dims [[buffer(6)]],
                                  constant uint& slices [[buffer(7)]],
                                  uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, n_in = dims.y, k_out = dims.z, experts = dims.w, words = k_out / 8;
  const uint word = pos.x, e = pos.y / slices, slice = pos.y % slices;
  if (word >= words || e >= experts) return;
  const uint first = uint(offsets[e]), last = e + 1 < experts ? uint(offsets[e + 1]) : rows;
  if (first >= last) return;
  const uint span = n_in / slices, n0 = slice * span, n1 = n0 + span;
  const uint groups = k_out / 64;
  const device uint* we = w + size_t(e) * n_in * words + word;
  const device bfloat* se = scales + size_t(e) * n_in * groups + word / 8;
  const device bfloat* be = biases + size_t(e) * n_in * groups + word / 8;
  for (uint r0 = first; r0 < last; r0 += 4) {
    const device float4* xr[4];
    for (uint r = 0; r < 4; r++) xr[r] = (const device float4*)(x + size_t(min(r0 + r, last - 1)) * n_in);
    float acc[4][8] = {};
    for (uint n = n0; n < n1; n += 4) {
      uint q[4];
      float sc[4], bi[4];
      float4 xv[4];
      for (uint u = 0; u < 4; u++) {
        q[u] = we[size_t(n + u) * words];
        sc[u] = float(se[size_t(n + u) * groups]);
        bi[u] = float(be[size_t(n + u) * groups]);
      }
      for (uint r = 0; r < 4; r++) xv[r] = xr[r][n / 4];
      for (uint u = 0; u < 4; u++) {
        float v[8];
        for (uint j = 0; j < 8; j++) v[j] = fma(sc[u], float((q[u] >> (4 * j)) & 0xf), bi[u]);
        for (uint r = 0; r < 4; r++)
          for (uint j = 0; j < 8; j++) acc[r][j] = fma(xv[r][u], v[j], acc[r][j]);
      }
    }
    for (uint r = 0; r < min(4u, last - r0); r++)
      for (uint j = 0; j < 8; j++) part[(size_t(slice) * rows + r0 + r) * k_out + word * 8 + j] = acc[r][j];
  }
}

// y[s, k] = the slices' partial sums in slice order; grid (K, pairs).
kernel void tf_train_experts_sum(const device float* part [[buffer(0)]],
                                 device float* y [[buffer(1)]],
                                 constant uint4& dims [[buffer(2)]],
                                 constant uint& slices [[buffer(3)]],
                                 uint2 pos [[thread_position_in_grid]]) {
  const uint rows = dims.x, k_out = dims.z, k = pos.x, r = pos.y;
  if (k >= k_out || r >= rows) return;
  float sum = 0;
  for (uint s = 0; s < slices; s++) sum += part[(size_t(s) * rows + r) * k_out + k];
  y[size_t(r) * k_out + k] = sum;
}
