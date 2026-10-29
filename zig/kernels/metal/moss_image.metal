// MOSS-Image's steps after attention_nax.metal, each rounded to bf16 where the publisher's torch code rounds.

inline float mi_bf(float x) { return float(bfloat(x)); }

// One threadgroup's sum of every thread's `v` (at most 1,024 threads), returned to all of them.
inline float mi_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
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

struct MiRows {
  uint width;
  float eps;
  uint modulated;
};

// y = bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w), then if modulated bf16(bf16(y * s) + b); a threadgroup a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void mi_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device bfloat* s [[buffer(2)]],
    const device bfloat* b [[buffer(3)]], device bfloat* y [[buffer(4)]], constant MiRows& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint n [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]) * float(xr[i]);
  const float inv = metal::precise::rsqrt(mi_sum(a, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) {
    float v = mi_bf(mi_bf(float(xr[i]) * inv) * float(w[i]));
    if (p.modulated) v = mi_bf(v * float(s[i])) + float(b[i]);
    yr[i] = bfloat(v);
  }
}

struct MiHeads {
  uint rows;
  uint heads;
  uint stride;
  float eps;
};

// q, k: head RMSNorm, then halves (c, c + 64) rotated by cs, sn; q, k, v to [heads, rows, 128]. A simdgroup a head.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void mi_qk(
    const device bfloat* qr [[buffer(0)]], const device bfloat* kr [[buffer(1)]], const device bfloat* vr [[buffer(2)]],
    const device bfloat* nq [[buffer(3)]], const device bfloat* nk [[buffer(4)]], const device float* cs [[buffer(5)]],
    const device float* sn [[buffer(6)]], device bfloat* Q [[buffer(7)]], device bfloat* K [[buffer(8)]],
    device bfloat* V [[buffer(9)]], constant MiHeads& p [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r = tg.x, h = tg.y * 8 + sg;
  if (h >= p.heads) return;
  const ulong from = ulong(r) * p.stride + h * 128 + lane * 4;
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
    q[i] = mi_bf(mi_bf(q[i] * qn) * float(nq[lane * 4 + i]));
    k[i] = mi_bf(mi_bf(k[i] * kn) * float(nk[lane * 4 + i]));
  }
  // lane l holds channels 4l..4l+3; its partner half sits 16 lanes away. The first half turns by -partner.
  const float sign = lane < 16 ? -1.0f : 1.0f;
  const ulong to = (ulong(h) * p.rows + r) * 128 + lane * 4;
  for (uint i = 0; i < 4; i++) {
    const float qp = sign * simd_shuffle_xor(q[i], ushort(16));
    const float kp = sign * simd_shuffle_xor(k[i], ushort(16));
    const uint j = (lane % 16) * 4 + i;
    const float c = cs[ulong(r) * 64 + j], s = sn[ulong(r) * 64 + j];
    Q[to + i] = bfloat(mi_bf(q[i] * c) + mi_bf(qp * s));
    K[to + i] = bfloat(mi_bf(k[i] * c) + mi_bf(kp * s));
    V[to + i] = vr[from + i];
  }
}

struct MiSitu {
  uint width;
  uint stride;
};

// o = situ(h, g) = (4 tanh(g / 4) sigmoid(g)) (25 tanh(h / 25)), each op rounded to bf16; h and g rows of `stride`.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void mi_situ(
    const device bfloat* h [[buffer(0)]], const device bfloat* g [[buffer(1)]], device bfloat* o [[buffer(2)]],
    constant MiSitu& p [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * p.stride + at.x;
  const float gv = float(g[i]), hv = float(h[i]);
  const float sig = mi_bf(1.0f / (1.0f + metal::precise::exp(-gv)));
  const float left = mi_bf(mi_bf(4.0f * mi_bf(metal::precise::tanh(mi_bf(gv * 0.25f)))) * sig);
  // torch divides by a host scalar as a product with its fp32 reciprocal
  const float right = mi_bf(25.0f * mi_bf(metal::precise::tanh(mi_bf(hv * (1.0f / 25.0f)))));
  o[ulong(at.y) * p.width + at.x] = bfloat(left * right);
}

// x = bf16(x + bf16(g * y)) by column; g is a modulation gate, or ones for an ungated residual.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void mi_gate_add(
    device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]], const device bfloat* g [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  x[i] = bfloat(float(x[i]) + mi_bf(float(g[at.x]) * float(y[i])));
}

// y = bf16(c + b) by column: a projection's fp32 sums and its bias, rounded once (the bias epilogue of a bf16 GEMM).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void mi_bias(
    const device float* c [[buffer(0)]], const device bfloat* b [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  y[i] = bfloat(c[i] + float(b[at.x]));
}

// tfp::attention_nax, full (every row sees every row): forms for L % 32 > 0, L % 32 = 0, and L % 64 = 0.
#define MI_ATTENTION(NAME, AQ, AK)                                                                                 \
  [[max_total_threads_per_threadgroup(128)]] [[kernel]] void NAME(                                               \
      const device bfloat16_t* Q [[buffer(0)]], const device bfloat16_t* K [[buffer(1)]],                       \
      const device bfloat16_t* V [[buffer(2)]], const device int32_t* P [[buffer(3)]],                          \
      const device float* F [[buffer(4)]], device bfloat16_t* O [[buffer(5)]],                                   \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                       \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                               \
    tfp::attention_nax<bfloat16_t, AQ, AK, false>(Q, K, V, O, P, F, tg, sg, lane);                               \
  }
MI_ATTENTION(mi_attention_00, false, false)
MI_ATTENTION(mi_attention_01, false, true)
MI_ATTENTION(mi_attention_11, true, true)
