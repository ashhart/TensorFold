// MOSS-Image's prompt encoder (Qwen3.5) after moss_image.metal, each step rounded where the publisher's torch rounds.

struct MeRows {
  uint width;
  uint in_stride;
  uint out_stride;
  float eps;
};

// y = bf16(x rsqrt(mean x^2 + eps) (1 + w)) in fp32 (Qwen3.5's zero-centered RMSNorm); a threadgroup a row.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void me_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant MeRows& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.in_stride;
  device bfloat* yr = y + ulong(row) * p.out_stride;
  float a = 0.0f;
  for (uint i = t; i < p.width; i += n) a += float(xr[i]) * float(xr[i]);
  const float inv = metal::precise::rsqrt(mi_sum(a, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat(float(xr[i]) * inv * (1.0f + float(w[i])));
}

struct MeHeads {
  uint rows;
  uint heads;
  uint in_stride;
  uint head_stride;
  float eps;
  uint normed;
};

// 256-wide heads to [heads, rows, 256], normed and the first 64 channels turned (else copied); a simdgroup each.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void me_head(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], const device float* cs [[buffer(2)]],
    const device float* sn [[buffer(3)]], device bfloat* y [[buffer(4)]], constant MeHeads& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r = tg.x, h = tg.y * 8 + sg;
  if (h >= p.heads) return;
  const ulong from = ulong(r) * p.in_stride + h * p.head_stride + lane * 8;
  const ulong to = (ulong(h) * p.rows + r) * 256 + lane * 8;
  if (p.normed == 0) {
    for (uint i = 0; i < 8; i++) y[to + i] = x[from + i];
    return;
  }
  float v[8];
  float ss = 0.0f;
  for (uint i = 0; i < 8; i++) {
    v[i] = float(x[from + i]);
    ss += v[i] * v[i];
  }
  const float inv = metal::precise::rsqrt(simd_sum(ss) / 256.0f + p.eps);
  for (uint i = 0; i < 8; i++) v[i] = mi_bf(v[i] * inv * (1.0f + float(w[lane * 8 + i])));
  // lanes 0-7 hold the rotated 64 channels; a channel's partner is 32 channels, four lanes, away
  const float sign = (lane & 4) != 0 ? 1.0f : -1.0f;
  for (uint i = 0; i < 8; i++) {
    const float pv = sign * simd_shuffle_xor(v[i], ushort(4));
    float o = v[i];
    if (lane < 8) {
      const uint j = (lane & 3) * 8 + i;
      o = mi_bf(v[i] * cs[ulong(r) * 32 + j]) + mi_bf(pv * sn[ulong(r) * 32 + j]);
    }
    y[to + i] = bfloat(o);
  }
}

struct MeAttn {
  uint rows;
  uint heads;
  uint kv_heads;
  float scale;
};

// o [rows, heads * 256] = causal softmax(q k^T scale) v in fp32, keys in order; a threadgroup a query row and head.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void me_attention(
    const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]], const device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]], constant MeAttn& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint2 dt [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const uint t = tg.x, h = tg.y, d = dt.x, kvh = h / (p.heads / p.kv_heads);
  const float q = float(Q[(ulong(h) * p.rows + t) * 256 + d]);
  float m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (uint j = 0; j <= t; j++) {
    const ulong at = (ulong(kvh) * p.rows + j) * 256 + d;
    const float s = mi_sum(q * float(K[at]), part, 256, lane, sg) * p.scale;
    const float top = max(m, s);
    const float old = metal::precise::exp(m - top), e = metal::precise::exp(s - top);
    l = l * old + e;
    acc = acc * old + e * float(V[at]);
    m = top;
  }
  O[(ulong(t) * p.heads + h) * 256 + d] = bfloat(acc / l);
}

// o = bf16(a bf16(sigmoid(g))): the attention rows gated by the second half of each 512-wide q_proj head.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void me_gate(
    const device bfloat* a [[buffer(0)]], const device bfloat* qg [[buffer(1)]], device bfloat* o [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  const float g = float(qg[ulong(at.y) * 2 * width + (at.x / 256) * 512 + 256 + at.x % 256]);
  o[i] = bfloat(float(a[i]) * mi_bf(1.0f / (1.0f + metal::precise::exp(-g))));
}

// o = bf16(bf16(silu(g)) u): the MLP's product.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void me_silu_mul(
    const device bfloat* g [[buffer(0)]], const device bfloat* u [[buffer(1)]], device bfloat* o [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  const float gv = float(g[i]);
  o[i] = bfloat(mi_bf(gv / (1.0f + metal::precise::exp(-gv))) * float(u[i]));
}

// y = bf16(silu(bf16(the 4-tap causal depthwise conv of x))) by channel, the taps in order, rows before 0 zero.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void me_conv(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant uint& width [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  float s = 0.0f;
  for (uint k = 0; k < 4; k++) {
    if (at.y + k >= 3) s += float(w[at.x * 4 + k]) * float(x[ulong(at.y + k - 3) * width + at.x]);
  }
  const float c = mi_bf(s);
  y[ulong(at.y) * width + at.x] = bfloat(c / (1.0f + metal::precise::exp(-c)));
}

struct MeGdn {
  uint rows;
  uint kheads;
  uint vheads;
  uint mix_stride;
  float scale;
};

// The DeltaNet's fp32 inputs: q, k L2-normed (q scaled), beta = bf16(sigmoid(b)), g = -exp(A_log) softplus(a + dt).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void me_gdn_prep(
    const device bfloat* mix [[buffer(0)]], const device bfloat* a [[buffer(1)]], const device bfloat* b [[buffer(2)]],
    const device bfloat* a_log [[buffer(3)]], const device bfloat* dt_bias [[buffer(4)]], device float* qn [[buffer(5)]],
    device float* kn [[buffer(6)]], device float* g [[buffer(7)]], device float* beta [[buffer(8)]],
    constant MeGdn& p [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r = tg.x, h = tg.y * 8 + sg;
  if (h >= p.kheads) return;
  const ulong row = ulong(r) * p.mix_stride;
  const uint kdim = p.kheads * 128;
  float q[4], k[4];
  float qq = 0.0f, kk = 0.0f;
  for (uint i = 0; i < 4; i++) {
    q[i] = float(mix[row + h * 128 + lane * 4 + i]);
    k[i] = float(mix[row + kdim + h * 128 + lane * 4 + i]);
    qq += q[i] * q[i];
    kk += k[i] * k[i];
  }
  const float qi = metal::precise::rsqrt(simd_sum(qq) + 1e-6f), ki = metal::precise::rsqrt(simd_sum(kk) + 1e-6f);
  const ulong to = (ulong(r) * p.kheads + h) * 128 + lane * 4;
  for (uint i = 0; i < 4; i++) {
    qn[to + i] = q[i] * qi * p.scale;
    kn[to + i] = k[i] * ki;
  }
  const uint per = p.vheads / p.kheads;
  if (lane < per) {
    const uint v = h * per + lane;
    const ulong at = ulong(r) * p.vheads + v;
    beta[at] = mi_bf(1.0f / (1.0f + metal::precise::exp(-float(b[at]))));
    const float x = float(a[at]) + float(dt_bias[v]);
    const float soft = x > 20.0f ? x : metal::precise::log(1.0f + metal::precise::exp(x));
    g[at] = -metal::precise::exp(float(a_log[v])) * soft;
  }
}

// S = S exp(g); d = (v - S^T k) beta; S += k d^T; o = S^T q over the rows in order, a thread a value column in fp32.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void me_gdn(
    const device float* qn [[buffer(0)]], const device float* kn [[buffer(1)]], const device bfloat* mix [[buffer(2)]],
    const device float* g [[buffer(3)]], const device float* beta [[buffer(4)]], device bfloat* out [[buffer(5)]],
    constant MeGdn& p [[buffer(6)]], uint h [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]]) {
  threadgroup float kq[256];
  float s[128];
  for (uint i = 0; i < 128; i++) s[i] = 0.0f;
  const uint kh = h / (p.vheads / p.kheads);
  const uint voff = 2 * p.kheads * 128 + h * 128 + j;
  for (uint t = 0; t < p.rows; t++) {
    const ulong at = (ulong(t) * p.kheads + kh) * 128 + j;
    kq[j] = kn[at];
    kq[128 + j] = qn[at];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float decay = metal::precise::exp(g[ulong(t) * p.vheads + h]);
    const float v = float(mix[ulong(t) * p.mix_stride + voff]);
    float kv = 0.0f;
    for (uint i = 0; i < 128; i++) {
      s[i] *= decay;
      kv += s[i] * kq[i];
    }
    const float delta = (v - kv) * beta[ulong(t) * p.vheads + h];
    float o = 0.0f;
    for (uint i = 0; i < 128; i++) {
      s[i] += kq[i] * delta;
      o += s[i] * kq[128 + i];
    }
    out[(ulong(t) * p.vheads + h) * 128 + j] = bfloat(o);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// y = bf16(bf16(w bf16(x rsqrt(mean x^2 + eps))) silu(z)) over each 128-wide head; a simdgroup a row and head.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void me_gated_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* z [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    device bfloat* y [[buffer(3)]], constant MeRows& p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r = tg.x, h = tg.y * 8 + sg;
  if (h * 128 >= p.width) return;
  const ulong at = ulong(r) * p.width + h * 128 + lane * 4;
  float v[4];
  float ss = 0.0f;
  for (uint i = 0; i < 4; i++) {
    v[i] = float(x[at + i]);
    ss += v[i] * v[i];
  }
  const float inv = metal::precise::rsqrt(simd_sum(ss) / 128.0f + p.eps);
  for (uint i = 0; i < 4; i++) {
    const float n = mi_bf(float(w[lane * 4 + i]) * mi_bf(v[i] * inv));
    const float zv = float(z[at + i]);
    y[at + i] = bfloat(n * (zv / (1.0f + metal::precise::exp(-zv))));
  }
}
