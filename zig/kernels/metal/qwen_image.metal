// Qwen-Image-2.1's transformer steps after attention_nax.metal (its attention); projections are qwen_vision.metal's.

inline float qi_bf(float x) { return float(bfloat(x)); }

// One threadgroup's sum of every thread's `v` (at most 1,024 threads), returned to all of them.
inline float qi_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    s = simd_sum(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

struct QiRows {
  uint width;
  float eps;
};

// y = bf16(bf16(layer_norm(x)) * s), s being one plus a modulation scale; one threadgroup a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qi_mod_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* s [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant QiRows& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]);
  const float mean = qi_sum(a, part, n, lane, sg) / float(p.width);
  float v = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float d = float(xr[i]) - mean;
    v += d * d;
  }
  const float inv = metal::precise::rsqrt(qi_sum(v, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat(qi_bf((float(xr[i]) - mean) * inv) * float(s[i]));
}

// The prompt rows' RMSNorm with a zero-centred weight: y = bf16(x * rsqrt(mean(x^2) + eps) * (w + 1)).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qi_text_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant QiRows& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]) * float(xr[i]);
  const float inv = metal::precise::rsqrt(qi_sum(a, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat(float(xr[i]) * inv * (float(w[i]) + 1.0f));
}

// y = gelu(y) in place, the tanh form, fp32 with one rounding.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qi_gelu(
    device bfloat* y [[buffer(0)]], constant uint& width [[buffer(1)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  const float v = float(y[i]);
  y[i] = bfloat(0.5f * v * (1.0f + metal::precise::tanh(0.79788456080286536f * (v + 0.044715f * v * v * v))));
}

struct QiHeads {
  uint rows;
  uint heads;
  uint keys;
  uint base;
  float eps;
};

// q and k take their head's RMSNorm and the rotation of pairs (2c, 2c + 1); one simdgroup a (row, head), 8 a group.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qi_qk(
    const device bfloat* qr [[buffer(0)]], const device bfloat* kr [[buffer(1)]], const device bfloat* vr [[buffer(2)]],
    const device bfloat* nq [[buffer(3)]], const device bfloat* nk [[buffer(4)]], const device float* cs [[buffer(5)]],
    const device float* sn [[buffer(6)]], device bfloat* Q [[buffer(7)]], device bfloat* K [[buffer(8)]],
    device bfloat* V [[buffer(9)]], constant QiHeads& p [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r = tg.x, h = tg.y * 8 + sg;
  if (h >= p.heads) return;
  const ulong from = (ulong(r) * p.heads + h) * 128 + lane * 4;
  float q[4], k[4];
  float qq = 0.0f, kk = 0.0f;
  for (uint i = 0; i < 4; i++) {
    q[i] = float(qr[from + i]);
    k[i] = float(kr[from + i]);
    qq += q[i] * q[i];
    kk += k[i] * k[i];
  }
  const float qn = metal::precise::rsqrt(simd_sum(qq) / 128.0f + p.eps);
  const float kn = metal::precise::rsqrt(simd_sum(kk) / 128.0f + p.eps);
  for (uint i = 0; i < 4; i++) {
    q[i] = qi_bf(float(nq[lane * 4 + i]) * qi_bf(q[i] * qn));
    k[i] = qi_bf(float(nk[lane * 4 + i]) * qi_bf(k[i] * kn));
  }
  device bfloat* qo = Q + (ulong(h) * p.rows + r) * 128 + lane * 4;
  const ulong to = (ulong(h) * p.keys + p.base + r) * 128 + lane * 4;
  for (uint j = 0; j < 2; j++) {
    const float c = cs[ulong(r) * 64 + lane * 2 + j], s = sn[ulong(r) * 64 + lane * 2 + j];
    qo[2 * j] = bfloat(q[2 * j] * c - q[2 * j + 1] * s);
    qo[2 * j + 1] = bfloat(q[2 * j] * s + q[2 * j + 1] * c);
    K[to + 2 * j] = bfloat(k[2 * j] * c - k[2 * j + 1] * s);
    K[to + 2 * j + 1] = bfloat(k[2 * j] * s + k[2 * j + 1] * c);
  }
  for (uint i = 0; i < 4; i++) V[to + i] = vr[from + i];
}

struct QiPrefix {
  uint text;
  uint keys;
};

// A block's stored prompt keys and values [heads, text, 128] to the first text rows of K, V [heads, keys, 128].
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qi_prefix(
    const device bfloat* kt [[buffer(0)]], const device bfloat* vt [[buffer(1)]], device bfloat* K [[buffer(2)]],
    device bfloat* V [[buffer(3)]], constant QiPrefix& p [[buffer(4)]], uint3 at [[thread_position_in_grid]]) {
  const ulong from = (ulong(at.z) * p.text + at.y) * 128 + at.x;
  const ulong to = (ulong(at.z) * p.keys + at.y) * 128 + at.x;
  K[to] = kt[from];
  V[to] = vt[from];
}

// x = bf16(x + bf16(g * y)) by column, g being the gate's tanh in bf16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qi_gate_add(
    device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]], const device bfloat* g [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  x[i] = bfloat(float(x[i]) + qi_bf(float(g[at.x]) * float(y[i])));
}

// a = bf16(silu(a) * b) in place, silu as x * sigmoid(x) with each stage rounded to bf16.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void qi_swiglu(
    device bfloat* a [[buffer(0)]], const device bfloat* b [[buffer(1)]], constant uint& width [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  const float v = float(a[i]);
  const float act = qi_bf(v * qi_bf(1.0f / (1.0f + metal::precise::exp(-v))));
  a[i] = bfloat(act * float(b[i]));
}

// tfp::attention_nax (MLX's steel attention on the tensor units): causal for the prompt rows, full for the image rows.
#define QI_ATTENTION(NAME, AQ, AK, CAUSAL)                                                                         \
  [[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME(                                               \
      const device bfloat16_t* Q [[buffer(0)]], const device bfloat16_t* K [[buffer(1)]],                       \
      const device bfloat16_t* V [[buffer(2)]], const device int32_t* P [[buffer(3)]],                          \
      const device float* F [[buffer(4)]], device bfloat16_t* O [[buffer(5)]],                                   \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                       \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                               \
    tfp::attention_nax<bfloat16_t, AQ, AK, CAUSAL>(Q, K, V, O, P, F, tg, sg, lane);                              \
  }
QI_ATTENTION(qi_attention_full_00, false, false, false)
QI_ATTENTION(qi_attention_full_01, false, true, false)
QI_ATTENTION(qi_attention_full_10, true, false, false)
QI_ATTENTION(qi_attention_full_11, true, true, false)
QI_ATTENTION(qi_attention_causal_00, false, false, true)
QI_ATTENTION(qi_attention_causal_01, false, true, true)
QI_ATTENTION(qi_attention_causal_10, true, false, true)
QI_ATTENTION(qi_attention_causal_11, true, true, true)
