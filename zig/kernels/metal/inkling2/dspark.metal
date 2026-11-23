#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

constant uint DSP_H = 2304;
constant uint DSP_Q = 3072;
constant uint DSP_KV = 1024;
constant uint DSP_HEAD = 128;
constant uint DSP_RANK = 256;
constant uint DSP_RNN = 768;
constant float DSP_EPS = 1e-6f;
constant float DSP_SCALE = 0x1.6a09e6p-4f;

struct DsparkPositionArgs { uint base; };
struct DsparkContextArgs { uint base, rows, ring; };
struct DsparkAttentionArgs { uint committed, ring, local, from; };
struct DsparkTokenArgs { uint token; };
struct DsparkResidualArgs { uint init; };
struct DsparkCountArgs { uint count; };
struct DsparkBaseArgs { uint rows, vocab; };

kernel void tf_ink2_dspark_init_rows(const device bfloat* anchor [[buffer(0)]], const device bfloat* mask [[buffer(1)]], device bfloat* rows [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  for (uint d = t; d < DSP_H; d += 256) {
    const device bfloat* src = row == 0 ? anchor : mask;
    rows[row * DSP_H + d] = src[d];
  }
}

kernel void tf_ink2_dspark_fuse_taps(const device bfloat* taps [[buffer(0)]], const device bfloat* fuse_weight [[buffer(1)]], const device bfloat* norm_weight [[buffer(2)]], device bfloat* c [[buffer(3)]], device bfloat* newest_f [[buffer(4)]], constant DsparkCountArgs& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint fusion = group / a.count, row = group % a.count;
  threadgroup float alpha[4], red[8], scale;
  if (t == 0) {
    float scores[4];
    float peak = -INFINITY;
    for (uint j = 0; j < 4; ++j) { scores[j] = float(fuse_weight[fusion * 4 + j]); peak = max(peak, scores[j]); }
    float sum = 0.0f;
    for (uint j = 0; j < 4; ++j) { scores[j] = exp(scores[j] - peak); sum += scores[j]; }
    for (uint j = 0; j < 4; ++j) alpha[j] = float(bfloat(scores[j] / sum));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float vals[9];
  float part = 0.0f;
  for (uint n = 0; n < 9; ++n) {
    const uint d = n * 256 + t;
    float mixed = 0.0f;
    for (uint j = 0; j < 4; ++j) mixed += alpha[j] * float(taps[(row * 4 + j) * DSP_H + d]);
    vals[n] = mixed;
    part += mixed * mixed;
  }
  const float sg_sum = simd_sum(part);
  if (lane == 0) red[t / 32] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float square = 0.0f;
    for (uint s = 0; s < 8; ++s) square += red[s];
    scale = rsqrt(square / float(DSP_H) + DSP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint n = 0; n < 9; ++n) {
    const uint d = n * 256 + t;
    const bfloat out = bfloat(vals[n] * scale * float(norm_weight[fusion * DSP_H + d]));
    c[(fusion * a.count + row) * DSP_H + d] = out;
    if (fusion == 3 && row + 1 == a.count) newest_f[d] = out;
  }
}

static inline float dspark_rope_angle(uint d, uint position) {
  const uint pair = d % 64;
  const float frequency = pow(1e6f, -float(pair) / 64.0f);
  return float(position) * frequency;
}

kernel void tf_ink2_dspark_block_qkv(const device bfloat* q_raw [[buffer(0)]], const device bfloat* k_raw [[buffer(1)]], const device bfloat* v_raw [[buffer(2)]], const device bfloat* q_weight [[buffer(3)]], const device bfloat* k_weight [[buffer(4)]], device bfloat* q [[buffer(5)]], device bfloat* k [[buffer(6)]], device bfloat* v [[buffer(7)]], constant DsparkPositionArgs& a [[buffer(8)]], uint group [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]]) {
  const uint row = group / 40, work = group % 40;
  threadgroup float scale;
  if (work < 32 && d == 0) {
    float sum = 0.0f;
    if (work < 24) {
      const uint h = work;
      for (uint i = 0; i < DSP_HEAD; ++i) { const float x = float(q_raw[row * DSP_Q + h * DSP_HEAD + i]); sum += x * x; }
    } else {
      const uint h = work - 24;
      for (uint i = 0; i < DSP_HEAD; ++i) { const float x = float(k_raw[row * DSP_KV + h * DSP_HEAD + i]); sum += x * x; }
    }
    scale = rsqrt(sum / float(DSP_HEAD) + DSP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (work >= 32) {
    const uint h = work - 32;
    v[row * DSP_KV + h * DSP_HEAD + d] = v_raw[row * DSP_KV + h * DSP_HEAD + d];
    return;
  }
  const bool query = work < 24;
  const uint h = query ? work : work - 24;
  const uint at = row * (query ? DSP_Q : DSP_KV) + h * DSP_HEAD + d;
  const float x = float(bfloat(float(query ? q_raw[at] : k_raw[at]) * scale * float(query ? q_weight[d] : k_weight[d])));
  const uint partner = d < 64 ? d + 64 : d - 64;
  const uint other_at = row * (query ? DSP_Q : DSP_KV) + h * DSP_HEAD + partner;
  const float other = float(bfloat(float(query ? q_raw[other_at] : k_raw[other_at]) * scale * float(query ? q_weight[partner] : k_weight[partner])));
  const float angle = dspark_rope_angle(d, a.base + row);
  const float cs = cos(angle), sn = sin(angle);
  const float rotated = d < 64 ? x * cs - other * sn : other * sn + x * cs;
  if (query) q[at] = bfloat(rotated); else k[at] = bfloat(rotated);
}

kernel void tf_ink2_dspark_context_kv(const device bfloat* k_raw [[buffer(0)]], const device bfloat* v_raw [[buffer(1)]], const device bfloat* weight [[buffer(2)]], device bfloat* keys [[buffer(3)]], device bfloat* values [[buffer(4)]], constant DsparkContextArgs& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]]) {
  const uint row = group / 8, h = group % 8;
  threadgroup float scale;
  if (d == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < DSP_HEAD; ++i) { const float x = float(k_raw[row * DSP_KV + h * DSP_HEAD + i]); sum += x * x; }
    scale = rsqrt(sum / float(DSP_HEAD) + DSP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint position = a.base + row;
  const uint at = (position % a.ring) * DSP_KV + h * DSP_HEAD + d;
  const uint src = row * DSP_KV + h * DSP_HEAD + d;
  const uint partner = d < 64 ? d + 64 : d - 64;
  const float x = float(bfloat(float(k_raw[src]) * scale * float(weight[d])));
  const float other = float(bfloat(float(k_raw[row * DSP_KV + h * DSP_HEAD + partner]) * scale * float(weight[partner])));
  const float angle = dspark_rope_angle(d, position);
  const float cs = cos(angle), sn = sin(angle);
  keys[at] = bfloat(d < 64 ? x * cs - other * sn : other * sn + x * cs);
  values[at] = v_raw[src];
}

static inline float dspark_attention_score(const device bfloat* q, const device bfloat* k, uint q_at, uint k_at) {
  float dot = 0.0f;
  for (uint d = 0; d < DSP_HEAD; ++d) dot += float(q[q_at + d]) * float(k[k_at + d]);
  return dot * DSP_SCALE;
}

// A threadgroup a (row, head): thread t scores keys t, t + 128, ...; the denominator and value sums stay in key order.
kernel void tf_ink2_dspark_attention(const device bfloat* q [[buffer(0)]], const device bfloat* block_k [[buffer(1)]], const device bfloat* block_v [[buffer(2)]], const device bfloat* context_k [[buffer(3)]], const device bfloat* context_v [[buffer(4)]], device float* scores [[buffer(5)]], device bfloat* out [[buffer(6)]], constant DsparkAttentionArgs& a [[buffer(7)]], uint group [[threadgroup_position_in_grid]], uint d [[thread_position_in_threadgroup]]) {
  const uint row = group / 24, h = group % 24, kv = h / 3;
  const uint q_at = row * DSP_Q + h * DSP_HEAD;
  const uint out_at = q_at + d;
  const uint first = max(a.from, a.local && a.committed > 1024 ? a.committed - 1024 : 0);
  const uint retained = a.committed - first;
  const uint count = retained + 8;
  const uint score_at = group * count;
  threadgroup float peaks[128], peak, denominator;
  float m = -INFINITY;
  for (uint j = d; j < count; j += 128) {
    const float s = j < retained
        ? dspark_attention_score(q, context_k, q_at, ((first + j) % a.ring) * DSP_KV + kv * DSP_HEAD)
        : dspark_attention_score(q, block_k, q_at, (j - retained) * DSP_KV + kv * DSP_HEAD);
    scores[score_at + j] = s;
    m = max(m, s);
  }
  peaks[d] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (d == 0) {
    float top = -INFINITY;
    for (uint t = 0; t < 128; ++t) top = max(top, peaks[t]);
    peak = top;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = d; j < count; j += 128) scores[score_at + j] = exp(scores[score_at + j] - peak);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (d == 0) {
    float sum = 0.0f;
    for (uint j = 0; j < count; ++j) sum += scores[score_at + j];
    denominator = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = d; j < count; j += 128) scores[score_at + j] = scores[score_at + j] / denominator;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  float value = 0.0f;
  for (uint j = 0; j < retained; ++j) {
    const uint position = first + j;
    const uint v_at = (position % a.ring) * DSP_KV + kv * DSP_HEAD + d;
    value += scores[score_at + j] * float(context_v[v_at]);
  }
  for (uint j = 0; j < 8; ++j) {
    const uint v_at = (j * DSP_KV + kv * DSP_HEAD) + d;
    value += scores[score_at + retained + j] * float(block_v[v_at]);
  }
  out[out_at] = bfloat(value);
}

kernel void tf_ink2_dspark_residual_norm(const device bfloat* x [[buffer(0)]], const device bfloat* residual [[buffer(1)]], const device bfloat* weight [[buffer(2)]], device bfloat* new_residual [[buffer(3)]], device bfloat* normalized [[buffer(4)]], constant DsparkResidualArgs& a [[buffer(5)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup float sq[DSP_H], scale;
  float sigmas[9];
  for (uint n = 0; n < 9; ++n) {
    const uint d = n * 256 + t;
    const uint at = row * DSP_H + d;
    const float v = a.init ? float(x[at]) : float(x[at]) + float(residual[at]);
    sigmas[n] = v;
    sq[d] = v * v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float sum = 0.0f;
    for (uint d = 0; d < DSP_H; ++d) sum += sq[d];
    scale = rsqrt(sum / float(DSP_H) + DSP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint n = 0; n < 9; ++n) {
    const uint d = n * 256 + t;
    const uint at = row * DSP_H + d;
    const float sigma = sigmas[n];
    new_residual[at] = bfloat(sigma);
    normalized[at] = bfloat(sigma * scale * float(weight[d]));
  }
}

kernel void tf_ink2_dspark_swiglu(const device bfloat* g [[buffer(0)]], const device bfloat* u [[buffer(1)]], device bfloat* out [[buffer(2)]], constant DsparkCountArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const float gate = float(g[i]);
  out[i] = bfloat((gate / (1.0f + exp(-gate))) * float(u[i]));
}

kernel void tf_ink2_dspark_kappa(const device bfloat* y [[buffer(0)]], const device bfloat* f [[buffer(1)]], const device bfloat* weight [[buffer(2)]], const device bfloat* bias [[buffer(3)]], device float* out [[buffer(4)]], constant DsparkCountArgs& a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  const uint row = i / DSP_RNN, channel = i % DSP_RNN;
  if (row >= a.count) return;
  float sum = 0.0f;
  for (uint d = 0; d < DSP_H; ++d) sum += float(y[row * DSP_H + d]) * float(weight[channel * 5120 + 512 + d]);
  for (uint d = 0; d < DSP_H; ++d) sum += float(f[d]) * float(weight[channel * 5120 + 2816 + d]);
  sum += float(bias[channel]);
  out[i] = sum;
}

static inline void dspark_recurrent_step(const device bfloat* state, const device float* kappa, const device bfloat* rnn_weight, const device bfloat* markov, device bfloat* z, device bfloat* next_state, device bfloat* lambda, uint token, uint i) {
  if (i >= DSP_RANK) return;
  float values[3];
  for (uint part = 0; part < 3; ++part) {
    const uint channel = part * DSP_RANK + i;
    float sum = 0.0f;
    for (uint d = 0; d < DSP_RANK; ++d) sum += float(state[d]) * float(rnn_weight[channel * 5120 + d]);
    for (uint d = 0; d < DSP_RANK; ++d) sum += float(markov[uint64_t(token) * DSP_RANK + d]) * float(rnn_weight[channel * 5120 + 256 + d]);
    sum += float(kappa[channel]);
    const bfloat rounded = bfloat(sum);
    z[channel] = rounded;
    values[part] = float(rounded);
  }
  const bfloat gamma = bfloat(1.0f / (1.0f + exp(-values[0])));
  const bfloat candidate = bfloat(tanh(values[1]));
  const bfloat keep = bfloat(1.0f - float(gamma));
  const bfloat left = bfloat(float(gamma) * float(state[i]));
  const bfloat right = bfloat(float(keep) * float(candidate));
  next_state[i] = bfloat(float(left) + float(right));
  lambda[i] = bfloat(tanh(values[2]));
}

kernel void tf_ink2_dspark_recurrent(const device bfloat* state [[buffer(0)]], const device float* kappa [[buffer(1)]], const device bfloat* rnn_weight [[buffer(2)]], const device bfloat* markov [[buffer(3)]], device bfloat* z [[buffer(4)]], device bfloat* next_state [[buffer(5)]], device bfloat* lambda [[buffer(6)]], constant DsparkTokenArgs& a [[buffer(7)]], uint i [[thread_position_in_grid]]) {
  dspark_recurrent_step(state, kappa, rnn_weight, markov, z, next_state, lambda, a.token, i);
}

// The recurrent step after held[a.token]: the chain's previous token read on the GPU.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_chain_step(const device bfloat* state [[buffer(0)]], const device float* kappa [[buffer(1)]], const device bfloat* rnn_weight [[buffer(2)]], const device bfloat* markov [[buffer(3)]], device bfloat* z [[buffer(4)]], device bfloat* next_state [[buffer(5)]], device bfloat* lambda [[buffer(6)]], constant DsparkTokenArgs& a [[buffer(7)]], const device uint* held [[buffer(8)]], uint i [[thread_position_in_grid]]) {
  dspark_recurrent_step(state, kappa, rnn_weight, markov, z, next_state, lambda, held[a.token], i);
}

// held[a.token + 1] = the first greatest of the 200,058 logits; a NaN is never chosen unless logit 0 is one.
[[max_total_threads_per_threadgroup(1024)]]
kernel void tf_ink2_dspark_chain_pick(const device bfloat* logits [[buffer(0)]], device uint* held [[buffer(1)]], constant DsparkTokenArgs& a [[buffer(2)]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float values[32];
  threadgroup uint places[32];
  float best = -INFINITY;
  uint at = 0xffffffffu;
  for (uint i = t; i < 200058u; i += 1024u) {
    const float raw = float(logits[i]);
    const float v = isnan(raw) ? -INFINITY : raw;
    if (v > best || (v == best && i < at)) { best = v; at = i; }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(best, o);
    const uint oi = simd_shuffle_xor(at, o);
    if (ov > best || (ov == best && oi < at)) { best = ov; at = oi; }
  }
  if (lane == 0) { values[sg] = best; places[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  for (uint g = 0; g < 32; ++g)
    if (values[g] > best || (values[g] == best && places[g] < at)) { best = values[g]; at = places[g]; }
  held[a.token + 1] = isnan(float(logits[0])) ? 0u : at;
}

// A simdgroup an id: eight columns a lane a step, every row sharing the read of unembed[id]; rounded once to BF16.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_base(const device bfloat* unembed [[buffer(0)]], const device bfloat* y [[buffer(1)]], device bfloat* out [[buffer(2)]], constant DsparkBaseArgs& a [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint id = tg * 8 + sg;
  if (id >= a.vocab) return;
  const device bfloat* w = unembed + size_t(id) * DSP_H;
  float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint i = 8 * lane; i < DSP_H; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    for (uint row = 0; row < a.rows; ++row) {
      const device bfloat* x = y + size_t(row) * DSP_H + i;
      acc[row] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));
    }
  }
  for (uint row = 0; row < a.rows; ++row) {
    const float sum = simd_sum(acc[row]);
    if (lane == 0) out[size_t(row) * a.vocab + id] = bfloat(sum);
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_logits(const device bfloat* base [[buffer(0)]], const device bfloat* phi [[buffer(1)]], const device bfloat* beta [[buffer(2)]], const device bfloat* lambda [[buffer(3)]], device bfloat* out [[buffer(4)]], constant DsparkCountArgs& a [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float lam[DSP_RANK];
  lam[t] = float(lambda[t]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint i = tg * 8u + sg;
  if (i >= a.count) return;
  const device bfloat* row = phi + size_t(i) * DSP_RANK + 8u * lane;
  const float4 p0 = float4(*(const device bfloat4*)row);
  const float4 p1 = float4(*(const device bfloat4*)(row + 4u));
  const float4 l0 = float4(lam[8u * lane], lam[8u * lane + 1u], lam[8u * lane + 2u], lam[8u * lane + 3u]);
  const float4 l1 = float4(lam[8u * lane + 4u], lam[8u * lane + 5u], lam[8u * lane + 6u], lam[8u * lane + 7u]);
  const float dot = simd_sum(metal::dot(p0, l0) + metal::dot(p1, l1));
  if (lane != 0u) return;
  const bfloat bias = bfloat(float(beta[i]) + dot);
  const bfloat logit = bfloat(float(base[i]) + float(bias));
  out[i] = bfloat(30.0f * tanh(float(logit) / 30.0f));
}

struct DsparkPackArgs { uint rows, ngroups; };

// One-time GPU 2-bit affine quantizer (group size 64): packs 16 codes per uint and half2(scale, lo) per group.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_q2_pack(const device bfloat* mat [[buffer(0)]], device uint* codes [[buffer(1)]], device half2* params [[buffer(2)]], constant DsparkPackArgs& a [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint g_idx = tg * 8u + sg;
  const uint total = a.rows * a.ngroups;
  if (g_idx >= total) return;
  const device bfloat* src = mat + size_t(g_idx) * 64u + 2u * lane;
  const float v0 = float(src[0]), v1 = float(src[1]);
  const float lo = simd_min(min(v0, v1));
  const float hi = simd_max(max(v0, v1));
  const float span = hi - lo;
  const float scale = span > 0.0f ? span / 3.0f : 0.0f;
  const float inv = span > 0.0f ? 3.0f / span : 0.0f;
  const uint q0 = uint(clamp(rint((v0 - lo) * inv), 0.0f, 3.0f));
  const uint q1 = uint(clamp(rint((v1 - lo) * inv), 0.0f, 3.0f));
  uint bits = (q0 | (q1 << 2u)) << (4u * (lane & 7u));
  bits |= simd_shuffle_xor(bits, 1);
  bits |= simd_shuffle_xor(bits, 2);
  bits |= simd_shuffle_xor(bits, 4);
  if ((lane & 7u) == 0u) codes[size_t(g_idx) * 4u + (lane >> 3u)] = bits;
  if (lane == 0u) params[g_idx] = half2(half(scale), half(lo));
}

// One-time GPU 4-bit affine quantizer (group size 64): packs 8 codes per uint and half2(scale, lo) per group.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_q4_pack(const device bfloat* mat [[buffer(0)]], device uint* codes [[buffer(1)]], device half2* params [[buffer(2)]], constant DsparkPackArgs& a [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint g_idx = tg * 8u + sg;
  const uint total = a.rows * a.ngroups;
  if (g_idx >= total) return;
  const device bfloat* src = mat + size_t(g_idx) * 64u + 2u * lane;
  const float v0 = float(src[0]), v1 = float(src[1]);
  const float lo = simd_min(min(v0, v1));
  const float hi = simd_max(max(v0, v1));
  const float span = hi - lo;
  const float scale = span > 0.0f ? span / 15.0f : 0.0f;
  const float inv = span > 0.0f ? 15.0f / span : 0.0f;
  const uint q0 = uint(clamp(rint((v0 - lo) * inv), 0.0f, 15.0f));
  const uint q1 = uint(clamp(rint((v1 - lo) * inv), 0.0f, 15.0f));
  uint bits = (q0 | (q1 << 4u)) << (8u * (lane & 3u));
  bits |= simd_shuffle_xor(bits, 1);
  bits |= simd_shuffle_xor(bits, 2);
  if ((lane & 3u) == 0u) codes[size_t(g_idx) * 8u + (lane >> 2u)] = bits;
  if (lane == 0u) params[g_idx] = half2(half(scale), half(lo));
}

// 2-bit screened unembed projection for greedy DSpark chain: factored half4 math at full DRAM bandwidth.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_q2_proj(const device uint* u_codes [[buffer(0)]], const device half2* u_params [[buffer(1)]], const device bfloat* y [[buffer(2)]], device half* out [[buffer(3)]], constant DsparkBaseArgs& a [[buffer(4)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint id = tg * 8u + sg;
  if (id >= a.vocab) return;
  const device uint* rc = u_codes + size_t(id) * 144u;
  const device half2* rp = u_params + size_t(id) * 36u;
  float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint w = lane; w < 144u; w += 32u) {
    const uint word = rc[w];
    const half2 pr = rp[w >> 2u];
    const half sc = pr.x, lo = pr.y;
    const half4 q0 = half4(half(word & 3u), half((word >> 2u) & 3u), half((word >> 4u) & 3u), half((word >> 6u) & 3u));
    const half4 q1 = half4(half((word >> 8u) & 3u), half((word >> 10u) & 3u), half((word >> 12u) & 3u), half((word >> 14u) & 3u));
    const half4 q2 = half4(half((word >> 16u) & 3u), half((word >> 18u) & 3u), half((word >> 20u) & 3u), half((word >> 22u) & 3u));
    const half4 q3 = half4(half((word >> 24u) & 3u), half((word >> 26u) & 3u), half((word >> 28u) & 3u), half((word >> 30u) & 3u));
    const uint col = w * 16u;
    for (uint row = 0u; row < a.rows; ++row) {
      const device bfloat* xr = y + size_t(row) * DSP_H + col;
      const half4 x0 = half4(float4(*(const device bfloat4*)xr));
      const half4 x1 = half4(float4(*(const device bfloat4*)(xr + 4u)));
      const half4 x2 = half4(float4(*(const device bfloat4*)(xr + 8u)));
      const half4 x3 = half4(float4(*(const device bfloat4*)(xr + 12u)));
      const half4 xs = (x0 + x1) + (x2 + x3);
      const half xsum = (xs.x + xs.y) + (xs.z + xs.w);
      const half qdot = metal::dot(q0, x0) + metal::dot(q1, x1) + metal::dot(q2, x2) + metal::dot(q3, x3);
      acc[row] += float(sc * qdot + lo * xsum);
    }
  }
  for (uint row = 0u; row < a.rows; ++row) {
    const float sum = simd_sum(acc[row]);
    if (lane == 0u) out[size_t(row) * a.vocab + id] = half(sum);
  }
}

inline half4 dspark_q2_unpack4(uint w) {
  const half2 h01 = as_type<half2>(0x64006400u | (w & 3u) | ((w & 12u) << 14u)) - half2(1024.0h);
  const half2 h23 = as_type<half2>(0x64006400u | ((w >> 4u) & 3u) | ((w & 0xc0u) << 10u)) - half2(1024.0h);
  return half4(h01, h23);
}

// 2-bit screened Phi dot + per-TG top-2 shortlist (128 TGs -> 256 candidate vocab IDs), factored half4 math.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_q2_screen(const device half* base_approx [[buffer(0)]], const device uint* p_codes [[buffer(1)]], const device half2* p_params [[buffer(2)]], const device bfloat* beta [[buffer(3)]], const device bfloat* lambda [[buffer(4)]], device uint* candidates [[buffer(5)]], constant DsparkCountArgs& a [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup half lam[DSP_RANK];
  threadgroup float tg_v[32];
  threadgroup uint tg_i[32];
  lam[t] = half(float(lambda[t]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint v0 = uint((size_t(tg) * a.count) / 128u);
  const uint v1 = uint((size_t(tg + 1u) * a.count) / 128u);
  const uint sw = (sg << 1u) | (lane >> 4u);
  const uint sl = lane & 15u;
  const uint col = sl * 16u;
  const half4 l0 = half4(lam[col], lam[col + 1u], lam[col + 2u], lam[col + 3u]);
  const half4 l1 = half4(lam[col + 4u], lam[col + 5u], lam[col + 6u], lam[col + 7u]);
  const half4 l2 = half4(lam[col + 8u], lam[col + 9u], lam[col + 10u], lam[col + 11u]);
  const half4 l3 = half4(lam[col + 12u], lam[col + 13u], lam[col + 14u], lam[col + 15u]);
  const half4 ls = (l0 + l1) + (l2 + l3);
  const half lsum = (ls.x + ls.y) + (ls.z + ls.w);
  float bv0 = -INFINITY, bv1 = -INFINITY;
  uint bi0 = 0u, bi1 = 0u;
  for (uint i = v0 + sw; i < v1; i += 16u) {
    const uint word = p_codes[size_t(i) * 16u + sl];
    const half2 pr = p_params[size_t(i) * 4u + (sl >> 2u)];
    const half4 q0 = dspark_q2_unpack4(word);
    const half4 q1 = dspark_q2_unpack4(word >> 8u);
    const half4 q2 = dspark_q2_unpack4(word >> 16u);
    const half4 q3 = dspark_q2_unpack4(word >> 24u);
    const half qdot = metal::dot(q0, l0) + metal::dot(q1, l1) + metal::dot(q2, l2) + metal::dot(q3, l3);
    float part = float(pr.x * qdot + pr.y * lsum);
    part += simd_shuffle_xor(part, 8);
    part += simd_shuffle_xor(part, 4);
    part += simd_shuffle_xor(part, 2);
    part += simd_shuffle_xor(part, 1);
    if (sl == 0u) {
      const float raw = float(base_approx[i]) + float(beta[i]) + part;
      const float v = isnan(raw) ? -INFINITY : raw;
      if (v > bv0 || (v == bv0 && i < bi0)) {
        bv1 = bv0; bi1 = bi0;
        bv0 = v; bi0 = i;
      } else if (v > bv1 || (v == bv1 && i < bi1)) {
        bv1 = v; bi1 = i;
      }
    }
  }
  if (sl == 0u) {
    tg_v[sw * 2u] = bv0; tg_i[sw * 2u] = bi0;
    tg_v[sw * 2u + 1u] = bv1; tg_i[sw * 2u + 1u] = bi1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0u) return;
  float wv0 = -INFINITY, wv1 = -INFINITY;
  uint wi0 = 0u, wi1 = 0u;
  for (uint j = 0u; j < 32u; ++j) {
    const float v = tg_v[j];
    const uint idx = tg_i[j];
    if (v > wv0 || (v == wv0 && idx < wi0)) {
      wv1 = wv0; wi1 = wi0;
      wv0 = v; wi0 = idx;
    } else if (v > wv1 || (v == wv1 && idx < wi1)) {
      wv1 = v; wi1 = idx;
    }
  }
  candidates[tg * 2u] = wi0;
  candidates[tg * 2u + 1u] = wi1;
}

// Exact BF16 unembed + Phi + beta + softcap30 evaluation over the 256 shortlist candidates plus token 0.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_q2_refine(const device uint* candidates [[buffer(0)]], const device bfloat* unembed [[buffer(1)]], const device bfloat* y_row [[buffer(2)]], const device bfloat* phi [[buffer(3)]], const device bfloat* beta [[buffer(4)]], const device bfloat* lambda [[buffer(5)]], device bfloat* cand_logits [[buffer(6)]], device uint* cand_ids [[buffer(7)]], uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float lam[DSP_RANK];
  lam[t] = float(lambda[t]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg * 8u + sg;
  if (c >= 257u) return;
  const uint id = (c == 256u) ? 0u : candidates[c];
  const device bfloat* uw = unembed + size_t(id) * DSP_H;
  float bacc = 0.0f;
  for (uint i = 8u * lane; i < DSP_H; i += 256u) {
    const float4 w0 = float4(*(const device bfloat4*)(uw + i)), w1 = float4(*(const device bfloat4*)(uw + i + 4u));
    const float4 x0 = float4(*(const device bfloat4*)(y_row + i)), x1 = float4(*(const device bfloat4*)(y_row + i + 4u));
    bacc += metal::dot(w0, x0) + metal::dot(w1, x1);
  }
  const float bsum = simd_sum(bacc);
  const device bfloat* pr = phi + size_t(id) * DSP_RANK + 8u * lane;
  const float4 p0 = float4(*(const device bfloat4*)pr), p1 = float4(*(const device bfloat4*)(pr + 4u));
  const float4 l0 = float4(lam[8u * lane], lam[8u * lane + 1u], lam[8u * lane + 2u], lam[8u * lane + 3u]);
  const float4 l1 = float4(lam[8u * lane + 4u], lam[8u * lane + 5u], lam[8u * lane + 6u], lam[8u * lane + 7u]);
  const float pdot = simd_sum(metal::dot(p0, l0) + metal::dot(p1, l1));
  if (lane != 0u) return;
  const bfloat rbase = bfloat(bsum);
  const bfloat rbias = bfloat(float(beta[id]) + pdot);
  const bfloat rlogit = bfloat(float(rbase) + float(rbias));
  cand_logits[c] = bfloat(30.0f * tanh(float(rlogit) / 30.0f));
  cand_ids[c] = id;
}

// Exact greedy pick over the 257 refined candidates into held[a.token + 1].
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_q2_pick(const device bfloat* cand_logits [[buffer(0)]], const device uint* cand_ids [[buffer(1)]], device uint* held [[buffer(2)]], constant DsparkTokenArgs& a [[buffer(3)]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float values[8];
  threadgroup uint places[8];
  const float raw = float(cand_logits[t]);
  float best = isnan(raw) ? -INFINITY : raw;
  uint at = cand_ids[t];
  if (t == 0u) {
    const float raw0 = float(cand_logits[256]);
    const float v0 = isnan(raw0) ? -INFINITY : raw0;
    if (v0 > best || (v0 == best && 0u < at)) { best = v0; at = 0u; }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float pv = simd_shuffle_xor(best, o);
    const uint pi = simd_shuffle_xor(at, o);
    if (pv > best || (pv == best && pi < at)) { best = pv; at = pi; }
  }
  if (lane == 0u) { values[sg] = best; places[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0u) return;
  for (uint g = 0u; g < 8u; ++g)
    if (values[g] > best || (values[g] == best && places[g] < at)) { best = values[g]; at = places[g]; }
  held[a.token + 1] = isnan(float(cand_logits[256])) ? 0u : at;
}


struct DsparkSampleArgs { uint step, position, seed_lo, seed_hi, top_k; float inv_t, top_p, min_log; };

inline ulong dspark_mix64(ulong x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ul; x ^= x >> 27; x *= 0x94D049BB133111EBul; return x ^ (x >> 31);
}
inline uint dspark_fkey(float v) { const uint b = as_type<uint>(v); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }

// SplitMix64 Gumbel-max GPU sampler over the 256 refined shortlist candidates into held[a.step + 1].
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_dspark_q2_sample(const device bfloat* cand_logits [[buffer(0)]], const device uint* cand_ids [[buffer(1)]], device uint* held [[buffer(2)]], constant DsparkSampleArgs& a [[buffer(3)]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint sk[256], si[256];
  threadgroup float sv[256];
  threadgroup uint keep_n;
  const float raw = float(cand_logits[t]);
  const float val = isnan(raw) ? -INFINITY : raw * a.inv_t;
  sk[t] = dspark_fkey(val); sv[t] = val; si[t] = cand_ids[t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2u; k <= 256u; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      const uint ixj = t ^ j;
      if (ixj > t) {
        const bool asc = (t & k) == 0u;
        const bool gt = (sk[t] > sk[ixj]) || (sk[t] == sk[ixj] && si[t] < si[ixj]);
        if (asc ? !gt : gt) {
          const uint tk = sk[t], ti = si[t]; const float tv = sv[t];
          sk[t] = sk[ixj]; si[t] = si[ixj]; sv[t] = sv[ixj];
          sk[ixj] = tk; si[ixj] = ti; sv[ixj] = tv;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (t == 0u) {
    const uint n = (a.top_k == 0u || a.top_k > 256u) ? 256u : a.top_k;
    const float m = sv[0];
    float norm = 0.0f;
    for (uint j = 0u; j < n; ++j) norm += precise::exp(sv[j] - m);
    uint keep = n;
    if (a.top_p > 0.0f && a.top_p < 1.0f && norm > 0.0f) {
      float cum = 0.0f;
      for (uint j = 0u; j < n; ++j) { cum += precise::exp(sv[j] - m) / norm; if (cum >= a.top_p) { keep = j + 1u; break; } }
    }
    if (a.min_log > -INFINITY) {
      const float fl = m + a.min_log; uint j = 0u;
      while (j < keep && sv[j] >= fl) ++j;
      keep = max(j, 1u);
    }
    keep_n = keep;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gscore = -INFINITY;
  const uint id = si[t];
  if (t < keep_n) {
    const ulong seed = (ulong(a.seed_hi) << 32) | ulong(a.seed_lo);
    const ulong base = dspark_mix64(dspark_mix64(seed + 0x9E3779B97F4A7C15ul) ^ (ulong(a.position) * 0xD1B54A32D192ED03ul));
    const float u = (float(uint(dspark_mix64(base ^ ulong(id)) >> 40)) + 0.5f) * (1.0f / 16777216.0f);
    gscore = sv[t] - precise::log(-precise::log(u));
  }
  sv[t] = gscore;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    uint best = 0u; float bv = sv[0];
    for (uint j = 1u; j < keep_n; ++j) if (sv[j] > bv) { bv = sv[j]; best = j; }
    held[a.step + 1u] = si[best];
  }
}
