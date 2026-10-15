// Step-5's sparse indexer and attention, as StepFun's kernel.py (rev 364b3b03) computes them.
#include <metal_stdlib>
using namespace metal;

struct s5_sparse_args {
  uint rows, pos, heads, dim, span, ring, window, regions, groups, topk, first, norm;
  float eps, theta, scale;
};

// x rounded to e4m3fn (round to nearest even, 3 mantissa bits, subnormals from 2^-6, max 448), as a float.
inline float s5_e4m3(float x) {
  const float a = fabs(x);
  if (a == 0.0f || isnan(x)) return x;
  if (a >= 464.0f) return copysign(448.0f, x); // past the rounding midpoint above 448 still saturates here
  const float e = max(floor(log2(a)), -6.0f);
  const float step = exp2(e - 3.0f);
  return copysign(rint(a / step) * step, x);
}

// The indexer's head weights: bf16(X . bf16(W[h])) / sqrt(heads per group), a simdgroup a (row, head).
[[kernel]] void s5_index_weights(const device bfloat* X [[buffer(0)]], const device float* W [[buffer(1)]],
    constant s5_sparse_args& a [[buffer(2)]], device float* out [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, row = tg.y;
  if (h >= a.heads || row >= a.rows) return;
  float sum = 0.0f;
  for (uint k = lane; k < a.dim; k += 32) sum = fma(float(X[(size_t)row * a.dim + k]), float(bfloat(W[(size_t)h * a.dim + k])), sum);
  sum = simd_sum(sum);
  if (lane == 0) out[row * a.heads + h] = float(bfloat(sum)) * rsqrt(float(a.heads / a.groups));
}

// A simdgroup a (row, head): RMSNorm (1 + w) for q, LayerNorm (w, b) for k, to bf16, then NEOX over the span.
[[kernel]] void s5_index_norm(const device bfloat* X [[buffer(0)]], const device float* w [[buffer(1)]],
    const device float* bias [[buffer(2)]], constant s5_sparse_args& a [[buffer(3)]], device bfloat* Y [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= a.heads || row >= a.rows) return;
  const bool layer_norm = a.norm != 0;
  const device bfloat* x = X + ((size_t)row * a.heads + head) * a.dim;
  float v[8], sq = 0.0f, sum = 0.0f;
  for (uint j = 0; j < 8; ++j) {
    v[j] = float(x[lane + 32 * j]);
    sq = fma(v[j], v[j], sq);
    sum += v[j];
  }
  sq = simd_sum(sq);
  sum = simd_sum(sum);
  const float mean = layer_norm ? sum / float(a.dim) : 0.0f;
  const float var = layer_norm ? sq / float(a.dim) - mean * mean : sq / float(a.dim);
  const float rstd = rsqrt(var + a.eps);
  threadgroup float n[256];
  for (uint j = 0; j < 8; ++j) {
    const uint d = lane + 32 * j;
    const float r = layer_norm ? (v[j] - mean) * rstd * w[d] + bias[d] : v[j] * rstd * (w[d] + 1.0f);
    n[d] = float(bfloat(r));
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint p = a.pos + row;
  const uint slot = a.ring ? p % a.ring : row;
  device bfloat* y = Y + ((size_t)slot * a.heads + head) * a.dim;
  const uint half_span = a.span / 2;
  for (uint j = 0; j < 8; ++j) {
    const uint d = lane + 32 * j;
    if (d >= a.span) { y[d] = bfloat(n[d]); continue; }
    const uint i = d % half_span;
    const float angle = float(p) * (1.0f / precise::pow(a.theta, float(2 * i) / float(a.span)));
    const float c = float(bfloat(precise::cos(angle))), s = float(bfloat(precise::sin(angle)));
    y[d] = bfloat(d < half_span ? n[i] * c - n[i + half_span] * s : n[i] * s + n[i + half_span] * c);
  }
}

// Region first + g's summary: per dim, sum_t exp(z_td - m) k_td / sum_t exp(z_td - m), m one max; bf16, then e4m3.
[[kernel]] void s5_summarize(const device bfloat* K [[buffer(0)]], const device bfloat* Z [[buffer(1)]],
    constant s5_sparse_args& a [[buffer(2)]], device bfloat* S [[buffer(3)]], uint g [[threadgroup_position_in_grid]],
    uint d [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint region = a.first + g;
  threadgroup float part[8];
  float z[8], k[8], m = -INFINITY;
  for (uint t = 0; t < 8; ++t) {
    const uint slot = (region * 8 + t) % a.ring;
    z[t] = float(Z[(size_t)slot * a.dim + d]);
    k[t] = float(K[(size_t)slot * a.dim + d]);
    m = max(m, z[t]);
  }
  m = simd_max(m);
  if (lane == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = part[0];
  for (uint s = 1; s < a.dim / 32; ++s) m = max(m, part[s]);
  float num = 0.0f, den = 0.0f;
  for (uint t = 0; t < 8; ++t) {
    const float e = precise::exp(z[t] - m);
    den += e;
    num += e * k[t];
  }
  const float s = den > 0.0f ? num / max(den, 1e-20f) : 0.0f;
  S[(size_t)region * a.dim + d] = bfloat(s5_e4m3(float(bfloat(s))));
}

// Scores [rows, groups, regions]: per group, sum_h relu(e4m3(q_h) . summary) * w_h over the regions a row sees.
[[kernel]] void s5_score(const device bfloat* Q [[buffer(0)]], const device float* Wt [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], constant s5_sparse_args& a [[buffer(3)]], device float* out [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 tp [[thread_position_in_threadgroup]]) {
  const uint t = tp.x;
  const uint row = tg.y, group = tg.z, region = tg.x * 256 + t;
  const uint per = a.heads / a.groups;
  threadgroup float q[4][256];
  for (uint h = 0; h < per; ++h) q[h][t] = s5_e4m3(float(Q[((size_t)row * a.heads + group * per + h) * a.dim + t]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint visible = (a.pos + row) / 8;
  if (region >= a.regions) return;
  float score = 0.0f;
  if (region < visible) {
    const device bfloat* s = S + (size_t)region * a.dim;
    for (uint h = 0; h < per; ++h) {
      float dot = 0.0f;
      for (uint d = 0; d < 256; ++d) dot = fma(q[h][d], float(s[d]), dot);
      score += max(dot, 0.0f) * Wt[row * a.heads + group * per + h];
    }
  }
  out[((size_t)row * a.groups + group) * a.regions + region] = score;
}

inline uint s5_key(float x) {
  const uint b = as_type<uint>(x);
  return x < 0.0f ? ~b : (b | 0x80000000u);
}

// A row's top-k regions by (-score, id), ascending: the cut found a bit at a time, ties taking the lowest ids.
[[kernel]] void s5_select(const device float* scores [[buffer(0)]], constant s5_sparse_args& a [[buffer(1)]],
    device int* sel [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint t = tp.x;
  const uint row = tg.x, group = tg.y;
  const uint visible = (a.pos + row) / 8;
  const device float* s = scores + ((size_t)row * a.groups + group) * a.regions;
  device int* out = sel + ((size_t)row * a.groups + group) * a.topk;
  if (visible <= a.topk) return; // the attention reads every region
  threadgroup uint counts[32];
  threadgroup uint total;
  uint threshold = 0;
  for (int bit = 31; bit >= 0; --bit) {
    const uint probe = threshold | (1u << bit);
    uint c = 0;
    for (uint r = t; r < visible; r += 1024) c += s5_key(s[r]) >= probe ? 1u : 0u;
    c = simd_sum(c);
    if (lane == 0) counts[sg] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint n = 0;
      for (uint i = 0; i < 32; ++i) n += counts[i];
      total = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (total >= a.topk) threshold = probe;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // threshold: the topk-th largest key; take every key above it and the lowest-id ties up to topk
  threadgroup uint above_total;
  uint c = 0;
  for (uint r = t; r < visible; r += 1024) c += s5_key(s[r]) > threshold ? 1u : 0u;
  c = simd_sum(c);
  if (lane == 0) counts[sg] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    uint n = 0;
    for (uint i = 0; i < 32; ++i) n += counts[i];
    above_total = n;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint ties_wanted = a.topk - above_total;
  threadgroup uint base_sel, base_tie;
  if (t == 0) { base_sel = 0; base_tie = 0; }
  threadgroup uint pre_sel[32], pre_tie[32];
  for (uint tile = 0; tile < visible; tile += 1024) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r = tile + t;
    const uint key = r < visible ? s5_key(s[r]) : 0u;
    const uint tie = r < visible && key == threshold ? 1u : 0u;
    const uint tie_before = simd_prefix_exclusive_sum(tie);
    if (lane == 31) pre_tie[sg] = tie_before + tie;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tie_rank = base_tie + tie_before;
    for (uint i = 0; i < sg; ++i) tie_rank += pre_tie[i];
    const uint pick = r < visible && (key > threshold || (tie && tie_rank < ties_wanted)) ? 1u : 0u;
    const uint pick_before = simd_prefix_exclusive_sum(pick);
    if (lane == 31) pre_sel[sg] = pick_before + pick;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = base_sel + pick_before;
    for (uint i = 0; i < sg; ++i) at += pre_sel[i];
    if (pick && at < a.topk) out[at] = int(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      for (uint i = 0; i < 32; ++i) { base_sel += pre_sel[i]; base_tie += pre_tie[i]; }
    }
  }
}

// Online fp32 softmax of one key into the running state (the shared attention's order of operations).
inline void s5_take(thread float* out, thread float& maximum, thread float& total, float score, thread const float* value,
    uint per) {
  const float next = max(maximum, score);
  const float old = maximum == -INFINITY ? 0.0f : precise::exp(maximum - next);
  const float weight = precise::exp(score - next);
  maximum = next;
  total = fma(total, old, weight);
  for (uint d = 0; d < per; ++d) out[d] = fma(out[d], old, weight * value[d]);
}

// A simdgroup a (row, head): a sliding layer's window (a ring), or a full layer's picks and own region (all, if few).
[[kernel]] void s5_attend(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], const device int* sel [[buffer(3)]], constant s5_sparse_args& a [[buffer(4)]],
    device bfloat* O [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y;
  if (head >= a.heads || row >= a.rows) return;
  const uint kvh = a.groups, group = head / (a.heads / kvh);
  const uint per = a.dim / 32;
  const size_t qi = ((size_t)row * a.heads + head) * a.dim;
  float q[8], out[8], value[8];
  for (uint d = 0; d < per; ++d) { q[d] = float(Q[qi + lane + 32 * d]); out[d] = 0.0f; }
  float maximum = -INFINITY, total = 0.0f;
  const uint p = a.pos + row;
  uint lo = 0, regions = 0, own = 0;
  const device int* picks = sel + ((size_t)row * kvh + group) * a.topk;
  bool all = true;
  if (a.window > 0) {
    lo = p + 1 > a.window ? p + 1 - a.window : 0;
  } else {
    regions = p / 8;
    own = regions * 8;
    all = regions <= a.topk;
  }
  const uint keys = a.window > 0 ? p + 1 - lo : (all ? p + 1 : a.topk * 8 + (p + 1 - own));
  for (uint i = 0; i < keys; ++i) {
    uint key;
    if (a.window > 0 || all) key = lo + i;
    else key = i < a.topk * 8 ? uint(picks[i / 8]) * 8 + i % 8 : own + (i - a.topk * 8);
    const uint slot = a.ring ? key % a.ring : key;
    const size_t ki = ((size_t)slot * kvh + group) * a.dim;
    float dot = 0.0f;
    for (uint d = 0; d < per; ++d) dot = fma(q[d], float(K[ki + lane + 32 * d]), dot);
    const float score = simd_sum(dot) * a.scale;
    for (uint d = 0; d < per; ++d) value[d] = float(V[ki + lane + 32 * d]);
    s5_take(out, maximum, total, score, value, per);
  }
  for (uint d = 0; d < per; ++d) O[qi + lane + 32 * d] = bfloat(total == 0.0f ? 0.0f : out[d] / total);
}
