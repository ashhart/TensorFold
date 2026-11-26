// TFQ3 (2D-AffineCoset/half4) shared structs, FWHT-128 helpers, and decoders.
#define PW_UNROLL _Pragma("clang loop unroll(full)")

inline short2 pw_frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

struct PwExpertMeta {
  uint  b_bits;
  uint  heat_rank;
  ulong gate_word_off;
  ulong up_word_off;
  ulong down_word_off;
  uint  gate_sc_off;
  uint  up_sc_off;
  uint  down_sc_off;
  float gate_snr_db;
  float up_snr_db;
  float down_snr_db;
  uint  reserved[2];
};

struct PwDims {
  uint  rows;
  uint  slots;
  uint  topk;
  uint  experts;
  float lim;
  float routed_scale;
  uint  flags;
  uint  pad;
};

struct PwDenseDims {
  uint  rows;
  uint  n;
  uint  k;
  uint  ldx;
  uint  ldo;
  float lim;
  uint  ksplit;
  uint  pad1;
};

struct PwV3Tables {
  uint  magic;
  uint  version;
  float a3, b3, a4, b4, a5, b5;
  uint  pad0[8];
  uint  lut3[256];   // off 64   (1 KB)
  uint  lut4[1024];  // off 1088 (4 KB)
  uint  lut5[1024];  // off 5184 (4 KB)
  half4 c64_3[64];   // off 9280 (512 B)
  half4 c64_4[64];   // off 9792 (512 B)
  half4 c64_5[64];   // off 10304 (512 B)
};

inline float pw_sgn(uint seed, uint i) {
  uint h = (seed ^ i) * 2654435761u;
  return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline bfloat pw_glu(float g, float u, float limit) {
  const float lim = float(bfloat(limit));
  const bfloat gt = bfloat(metal::min(float(bfloat(g)), lim));
  const bfloat up = bfloat(metal::min(metal::max(float(bfloat(u)), -lim), lim));
  return (gt * sigmoid_fast(gt)) * up;
}

inline void pw_fwht128_simd4(thread float4& v, ushort lane) {
  PW_UNROLL
  for (ushort d = 1; d <= 16; d <<= 1) {
    float4 o = simd_shuffle_xor(v, d);
    v = ((lane & d) == 0u) ? (v + o) : (o - v);
  }
  float a0 = v.x + v.y, b0 = v.x - v.y;
  float a1 = v.z + v.w, b1 = v.z - v.w;
  v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

// === B=1 Fused Route-Select (bit-exact match to moe_route.metal tf_route_picks + tf_route_weights!) ===
inline float pw_sigmoid_precise(float x) {
  float e = metal::precise::exp(metal::abs(x));
  float y = metal::precise::divide(1.0f, 1.0f + e);
  return (x < 0.0f) ? y : (1.0f - y);
}
inline void pw_route_select_b1(
    const device float* logits, const device float* bias,
    device int* pick, device float* wts, uint experts, float routed_scale, ushort lane
) {
  float sc[9]; float c[9];
  for (int t = 0; t < 9; ++t) {
    uint id = uint(lane) + uint(t) * 32u;
    if (id < experts) {
      float sig = pw_sigmoid_precise(logits[id]);
      sc[t] = sig;
      c[t]  = sig + bias[id];
    } else {
      sc[t] = 0.0f; c[t] = -INFINITY;
    }
  }
  float ws[8];
  for (int k = 0; k < 8; ++k) {
    float bv = -INFINITY; int bi = int(experts); float bs = 0.0f;
    for (int t = 0; t < 9; ++t) {
      int id = int(lane) + t * 32;
      if (uint(id) < experts && (c[t] > bv || (c[t] == bv && id < bi))) { bv = c[t]; bi = id; bs = sc[t]; }
    }
    float mv = simd_max(bv);
    int win_id = simd_min((bv == mv) ? bi : int(experts));
    float win_s = simd_shuffle(bs, ushort(win_id & 31));
    for (int t = 0; t < 9; ++t) {
      if (int(lane) + t * 32 == win_id) c[t] = -INFINITY;
    }
    if (lane == 0u) pick[k] = win_id;
    ws[k] = (win_id < int(experts)) ? win_s : 0.0f;
  }
  if (lane == 0u) {
    float total = ws[0];
    for (int k = 1; k < 8; ++k) total += ws[k];
    for (int k = 0; k < 8; ++k) wts[k] = metal::precise::divide(ws[k], total) * routed_scale;
  }
}

// === Forward FWHT-128 on X[rows, 4096] (+ optional fused B=1 route_select at gid.x == 32) ===
[[kernel]] void pw_fwht_x_fwd(
    const device bfloat* X      [[buffer(0)]],
    device half* X_rot          [[buffer(1)]],
    constant PwDims& d          [[buffer(2)]],
    const device float* logits  [[buffer(3)]],
    const device float* bias    [[buffer(4)]],
    device int* pick            [[buffer(5)]],
    device float* wts           [[buffer(6)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
  if (gid.x == 32u) {
    if (gid.y == 0u && (d.flags & 2u) != 0u) pw_route_select_b1(logits, bias, pick, wts, d.experts, d.routed_scale, lane);
    return;
  }
  uint row = gid.y, c0 = gid.x * 128u;
  const device bfloat* src = X + ulong(row) * 4096ul + c0;
  float4 v;
  v.x = float(src[lane +  0u]) * pw_sgn(0x12345678u, c0 + lane +  0u);
  v.y = float(src[lane + 32u]) * pw_sgn(0x12345678u, c0 + lane + 32u);
  v.z = float(src[lane + 64u]) * pw_sgn(0x12345678u, c0 + lane + 64u);
  v.w = float(src[lane + 96u]) * pw_sgn(0x12345678u, c0 + lane + 96u);
  pw_fwht128_simd4(v, lane);
  device half* dst = X_rot + ulong(row) * 4096ul + c0;
  dst[lane +  0u] = half(v.x); dst[lane + 32u] = half(v.y);
  dst[lane + 64u] = half(v.z); dst[lane + 96u] = half(v.w);
}

[[kernel]] void pw_fwht_glu_fwht(
    const device float* GU_rot  [[buffer(0)]],
    device half* ACT_rot        [[buffer(1)]],
    constant PwDims& d          [[buffer(2)]],
    device half* SACT_rot       [[buffer(3)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
  uint slot = gid.y, r0 = gid.x * 128u;
  const device float* g_src = GU_rot + ulong(slot * 2u + 0u) * 2048ul + r0;
  const device float* u_src = GU_rot + ulong(slot * 2u + 1u) * 2048ul + r0;
  float4 gv = float4(g_src[lane], g_src[lane+32u], g_src[lane+64u], g_src[lane+96u]);
  float4 uv = float4(u_src[lane], u_src[lane+32u], u_src[lane+64u], u_src[lane+96u]);
  pw_fwht128_simd4(gv, lane); pw_fwht128_simd4(uv, lane);
  float4 z;
  z.x = float(pw_glu(gv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), d.lim)) * pw_sgn(0x12345678u, r0 + lane +  0u);
  z.y = float(pw_glu(gv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), d.lim)) * pw_sgn(0x12345678u, r0 + lane + 32u);
  z.z = float(pw_glu(gv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), d.lim)) * pw_sgn(0x12345678u, r0 + lane + 64u);
  z.w = float(pw_glu(gv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), d.lim)) * pw_sgn(0x12345678u, r0 + lane + 96u);
  pw_fwht128_simd4(z, lane);
  device half* dst = (slot < d.slots)
      ? (ACT_rot + ulong(slot) * 2048ul + r0)
      : (SACT_rot + ulong(slot - d.slots) * 2048ul + r0);
  dst[lane +  0u] = half(z.x); dst[lane + 32u] = half(z.y);
  dst[lane + 64u] = half(z.z); dst[lane + 96u] = half(z.w);
}

[[kernel]] void pw_fwht_y_inv(
    const device float* YR      [[buffer(0)]],
    device bfloat* YE           [[buffer(1)]],
    constant PwDims& d          [[buffer(2)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
  uint slot = gid.y, r0 = gid.x * 128u;
  const device float* src = YR + ulong(slot) * 4096ul + r0;
  float4 v = float4(src[lane], src[lane+32u], src[lane+64u], src[lane+96u]);
  pw_fwht128_simd4(v, lane);
  device bfloat* dst = YE + ulong(slot) * 4096ul + r0;
  dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u));
  dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u));
  dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u));
  dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u));
}

[[kernel]] void pw_fwht_y_combine(
    const device float* YR      [[buffer(0)]],
    const device bfloat* YS     [[buffer(1)]],
    const device float* wts     [[buffer(2)]],
    device bfloat* BRANCH       [[buffer(3)]],
    constant PwDims& d          [[buffer(4)]],
    uint2 gid                   [[threadgroup_position_in_grid]],
    ushort lane                 [[thread_index_in_simdgroup]]
) {
  uint row = gid.y, r0 = gid.x * 128u;
  uint base = row * d.topk;
  float4 acc = float4(0.0f);
  for (uint k = 0u; k < d.topk; ++k) {
    float wt = wts[base + k];
    const device float* src = YR + ulong(base + k) * 4096ul + r0;
    acc += float4(src[lane], src[lane+32u], src[lane+64u], src[lane+96u]) * wt;
  }
  if ((d.flags & 4u) != 0u) {
    const device float* sh = YR + ulong(d.slots + row) * 4096ul + r0;
    acc += float4(sh[lane], sh[lane+32u], sh[lane+64u], sh[lane+96u]);
    pw_fwht128_simd4(acc, lane);
    device bfloat* dst = BRANCH + ulong(row) * 4096ul + r0;
    dst[lane +  0u] = bfloat(acc.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u));
    dst[lane + 32u] = bfloat(acc.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u));
    dst[lane + 64u] = bfloat(acc.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u));
    dst[lane + 96u] = bfloat(acc.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u));
  } else {
    pw_fwht128_simd4(acc, lane);
    const device bfloat* ys = YS + ulong(row) * 4096ul + r0;
    device bfloat* dst = BRANCH + ulong(row) * 4096ul + r0;
    dst[lane +  0u] = bfloat(acc.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u) + float(ys[lane +  0u]));
    dst[lane + 32u] = bfloat(acc.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u) + float(ys[lane + 32u]));
    dst[lane + 64u] = bfloat(acc.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u) + float(ys[lane + 64u]));
    dst[lane + 96u] = bfloat(acc.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u) + float(ys[lane + 96u]));
  }
}

// === General Dense FWHT Helpers (with L2-fused 2D SPLIT_K reduction!) ===
[[kernel]] void pw_fwht_x_gen(
    const device bfloat* X  [[buffer(0)]],
    device half* X_rot      [[buffer(1)]],
    constant PwDenseDims& d [[buffer(2)]],
    uint2 gid               [[threadgroup_position_in_grid]],
    ushort lane             [[thread_index_in_simdgroup]]
) {
  uint row = gid.y, c0 = gid.x * 128u;
  const device bfloat* src = X + ulong(row) * ulong(d.ldx) + c0;
  float4 v;
  v.x = float(src[lane +  0u]) * pw_sgn(0x12345678u, c0 + lane +  0u);
  v.y = float(src[lane + 32u]) * pw_sgn(0x12345678u, c0 + lane + 32u);
  v.z = float(src[lane + 64u]) * pw_sgn(0x12345678u, c0 + lane + 64u);
  v.w = float(src[lane + 96u]) * pw_sgn(0x12345678u, c0 + lane + 96u);
  pw_fwht128_simd4(v, lane);
  device half* dst = X_rot + ulong(row) * ulong(d.k) + c0;
  dst[lane +  0u] = half(v.x); dst[lane + 32u] = half(v.y);
  dst[lane + 64u] = half(v.z); dst[lane + 96u] = half(v.w);
}

[[kernel]] void pw_fwht_x_plus_q8(
    const device bfloat* X   [[buffer(0)]],
    device half* X_rot       [[buffer(1)]],
    const device uchar* W8   [[buffer(2)]],
    const device bfloat* S8  [[buffer(3)]],
    const device bfloat* B8  [[buffer(4)]],
    device bfloat* OUT       [[buffer(5)]],
    constant PwDenseDims& d  [[buffer(6)]],
    uint2 gid                [[threadgroup_position_in_grid]],
    ushort lane              [[thread_index_in_simdgroup]]
) {
  uint row = gid.y;
  if (gid.x < 32u) {
    uint c0 = gid.x * 128u;
    const device bfloat* src = X + ulong(row) * ulong(d.ldx) + c0;
    float4 v;
    v.x = float(src[lane +  0u]) * pw_sgn(0x12345678u, c0 + lane +  0u);
    v.y = float(src[lane + 32u]) * pw_sgn(0x12345678u, c0 + lane + 32u);
    v.z = float(src[lane + 64u]) * pw_sgn(0x12345678u, c0 + lane + 64u);
    v.w = float(src[lane + 96u]) * pw_sgn(0x12345678u, c0 + lane + 96u);
    pw_fwht128_simd4(v, lane);
    device half* dst = X_rot + ulong(row) * 4096ul + c0;
    dst[lane +  0u] = half(v.x); dst[lane + 32u] = half(v.y);
    dst[lane + 64u] = half(v.z); dst[lane + 96u] = half(v.w);
    return;
  }
  uint r = gid.x - 32u;
  const device uint* w4 = (const device uint*)(W8 + ulong(r) * 4096ul);
  const device bfloat* sc = S8 + ulong(r) * 64ul;
  const device bfloat* bi = B8 + ulong(r) * 64ul;
  const device bfloat* x  = X + ulong(row) * ulong(d.ldx);
  float acc = 0.0f;
  for (uint g = 0u; g < 64u; g += 2u) {
    uint gi = g + (lane >> 4);
    uint c  = gi * 64u + (lane & 15u) * 4u;
    uint pk = w4[c >> 2];
    float s = float(sc[gi]), b = float(bi[gi]);
    float4 xv = float4(float(x[c]), float(x[c + 1u]), float(x[c + 2u]), float(x[c + 3u]));
    float4 wv = float4(float(pk & 255u), float((pk >> 8) & 255u), float((pk >> 16) & 255u), float(pk >> 24));
    acc += dot(fma(wv, s, b), xv);
  }
  acc = simd_sum(acc);
  if (lane == 0u) OUT[ulong(row) * ulong(d.ldo) + 24576u + r] = bfloat(acc);
}

inline uint pw_sr0(uint r0, uint n, uint k) {
  if (n == 20480u && k == 1536u) return r0 & 16383u; // MLA q_b (16384) + indexer.wq_b (4096)
  if (n > 16384u) return r0 & 8191u;                 // KDA qkv (3x8192) & lm_head (chunked 8192)
  if (n == 4096u && k == 4096u) return r0 & 2047u;   // Shared gate+up (2x2048)
  return r0;
}

[[kernel]] void pw_fwht_y_gen(
    const device float* YR  [[buffer(0)]],
    device bfloat* OUT      [[buffer(1)]],
    constant PwDenseDims& d [[buffer(2)]],
    uint2 gid               [[threadgroup_position_in_grid]],
    ushort lane             [[thread_index_in_simdgroup]]
) {
  uint row = gid.y, r0 = gid.x * 128u;
  float4 v = float4(0.0f);
  uint ks = metal::max(1u, d.ksplit);
  for (uint kp = 0u; kp < ks; ++kp) {
    const device float* src = YR + (ulong(kp) * ulong(d.rows) + ulong(row)) * ulong(d.n) + r0;
    v += float4(src[lane], src[lane+32u], src[lane+64u], src[lane+96u]);
  }
  pw_fwht128_simd4(v, lane);
  uint sr0 = pw_sr0(r0, d.n, d.k);
  device bfloat* dst = OUT + ulong(row) * ulong(d.ldo) + r0;
  dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
  dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
  dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
  dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
}

[[kernel]] void pw_fwht_glu_unrot(
    const device float* YR  [[buffer(0)]],
    device bfloat* OUT      [[buffer(1)]],
    constant PwDenseDims& d [[buffer(2)]],
    uint2 gid               [[threadgroup_position_in_grid]],
    ushort lane             [[thread_index_in_simdgroup]]
) {
  uint row = gid.y, half_n = d.n >> 1, r0 = gid.x * 128u;
  float4 gv = float4(0.0f), uv = float4(0.0f);
  uint ks = metal::max(1u, d.ksplit);
  for (uint kp = 0u; kp < ks; ++kp) {
    const device float* g_src = YR + (ulong(kp) * ulong(d.rows) + ulong(row)) * ulong(d.n) + r0;
    const device float* u_src = g_src + ulong(half_n);
    gv += float4(g_src[lane], g_src[lane+32u], g_src[lane+64u], g_src[lane+96u]);
    uv += float4(u_src[lane], u_src[lane+32u], u_src[lane+64u], u_src[lane+96u]);
  }
  pw_fwht128_simd4(gv, lane); pw_fwht128_simd4(uv, lane);
  device bfloat* dst = OUT + ulong(row) * ulong(d.ldo) + r0;
  dst[lane +  0u] = pw_glu(gv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), uv.x * pw_sgn(0x9E3779B9u, r0 + lane +  0u), d.lim);
  dst[lane + 32u] = pw_glu(gv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), uv.y * pw_sgn(0x9E3779B9u, r0 + lane + 32u), d.lim);
  dst[lane + 64u] = pw_glu(gv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), uv.z * pw_sgn(0x9E3779B9u, r0 + lane + 64u), d.lim);
  dst[lane + 96u] = pw_glu(gv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), uv.w * pw_sgn(0x9E3779B9u, r0 + lane + 96u), d.lim);
}

// CORE v3 2D-AffineCoset/half4 DECODERS (Bit-Exact Match to Encoder!)
inline void pw_v3_q8_4b(thread uint& st, uint wc, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = st;
  uint h0 = s0 * 0x9E3779B1u;
  uint h1 = ((s0 <<  8) | (wc >> 24)) * 0x9E3779B1u;
  uint h2 = ((s0 << 16) | (wc >> 16)) * 0x9E3779B1u;
  uint h3 = ((s0 << 24) | (wc >>  8)) * 0x9E3779B1u;
  st = wc;
  half4 c0 = T->c64_4[(h0 >> 16) & 63u], c2 = T->c64_4[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut4[(((wc >> 24)        << 2) ^ (h0 >> 22)) & 1023u]),
                 as_type<half2>(T->lut4[((((wc >> 16) & 255u) << 2) ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut4[((((wc >>  8) & 255u) << 2) ^ (h2 >> 22)) & 1023u]),
                 as_type<half2>(T->lut4[((( wc        & 255u) << 2) ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

inline void pw_v3_q8_3b(thread uint& st, uint m24, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = st;
  uint h0 = s0 * 0x9E3779B1u;
  uint h1 = ((s0 <<  6) | (m24 >> 18)) * 0x9E3779B1u;
  uint h2 = ((s0 << 12) | (m24 >> 12)) * 0x9E3779B1u;
  uint h3 = ((s0 << 18) | (m24 >>  6)) * 0x9E3779B1u;
  st = (s0 << 24) | m24;
  half4 c0 = T->c64_3[(h0 >> 16) & 63u], c2 = T->c64_3[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut3[(((m24 >> 16) & 0xFCu) ^ (h0 >> 24)) & 255u]),
                 as_type<half2>(T->lut3[(((m24 >> 10) & 0xFCu) ^ (h1 >> 24)) & 255u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut3[(((m24 >>  4) & 0xFCu) ^ (h2 >> 24)) & 255u]),
                 as_type<half2>(T->lut3[(((m24 <<  2) & 0xFCu) ^ (h3 >> 24)) & 255u])), c2.xyxy, c2.zwzw);
}

inline void pw_v3_q8_5b(thread uint& st, uint hi8, uint lo32, constant PwV3Tables* T, thread half4& q0, thread half4& q1) {
  uint s0 = st;
  uint b0 = ((hi8 << 2) | (lo32 >> 30)) & 1023u;
  uint b1 = (lo32 >> 20) & 1023u;
  uint b2 = (lo32 >> 10) & 1023u;
  uint b3 =  lo32        & 1023u;
  uint h0 = s0 * 0x9E3779B1u;
  uint h1 = ((s0 << 10) | b0) * 0x9E3779B1u;
  uint h2 = ((s0 << 20) | (hi8 << 12) | (lo32 >> 20)) * 0x9E3779B1u;
  uint h3 = ((s0 << 30) | (hi8 << 22) | (lo32 >> 10)) * 0x9E3779B1u;
  st = lo32;
  half4 c0 = T->c64_5[(h0 >> 16) & 63u], c2 = T->c64_5[(h2 >> 16) & 63u];
  q0 = fma(half4(as_type<half2>(T->lut5[(b0 ^ (h0 >> 22)) & 1023u]),
                 as_type<half2>(T->lut5[(b1 ^ (h1 >> 22)) & 1023u])), c0.xyxy, c0.zwzw);
  q1 = fma(half4(as_type<half2>(T->lut5[(b2 ^ (h2 >> 22)) & 1023u]),
                 as_type<half2>(T->lut5[(b3 ^ (h3 >> 22)) & 1023u])), c2.xyxy, c2.zwzw);
}

// === Lean 1-xv Native half4 SIMD FMA (Dense) + Lean 8-reg hardware dot4 (MoE -> 860 GB/s at B=4!) ===
inline float pw_h4sum(half4 p) { return (float(p.x) + float(p.y)) + (float(p.z) + float(p.w)); }
inline float pw_fdot(half4 q, half4 x) { return float(dot(q, x)); }
struct PwH2 { half4 x, y; };
struct PwH4 { half4 x, y, z, w; };
#define PW_DOT1(Q0, Q1, X, CB, BA0, BA1) { \
  half4 xv = *(const device half4*)((X) + (CB)); \
  (BA0) = fma((Q0), xv, (BA0)); (BA1) = fma((Q1), xv, (BA1)); }
#define PW_DOT2(Q0, Q1, X0, X1, CB, BA0, BA1) { \
  { half4 xv = *(const device half4*)((X0) + (CB)); (BA0).x = fma((Q0), xv, (BA0).x); (BA1).x = fma((Q1), xv, (BA1).x); } \
  { half4 xv = *(const device half4*)((X1) + (CB)); (BA0).y = fma((Q0), xv, (BA0).y); (BA1).y = fma((Q1), xv, (BA1).y); } }
#define PW_DOT4(Q0, Q1, X0, X1, X2, X3, CB, BA0, BA1) { \
  { half4 xv = *(const device half4*)((X0) + (CB)); (BA0).x = fma((Q0), xv, (BA0).x); (BA1).x = fma((Q1), xv, (BA1).x); } \
  { half4 xv = *(const device half4*)((X1) + (CB)); (BA0).y = fma((Q0), xv, (BA0).y); (BA1).y = fma((Q1), xv, (BA1).y); } \
  { half4 xv = *(const device half4*)((X2) + (CB)); (BA0).z = fma((Q0), xv, (BA0).z); (BA1).z = fma((Q1), xv, (BA1).z); } \
  { half4 xv = *(const device half4*)((X3) + (CB)); (BA0).w = fma((Q0), xv, (BA0).w); (BA1).w = fma((Q1), xv, (BA1).w); } }
#define PW_MDOT1(Q0, Q1, X, CB, BA0, BA1) { \
  half4 xv = *(const device half4*)((X) + (CB)); \
  (BA0) += pw_fdot((Q0), xv); (BA1) += pw_fdot((Q1), xv); }
#define PW_MDOT2(Q0, Q1, X0, X1, CB, BA0, BA1) { \
  { half4 xv = *(const device half4*)((X0) + (CB)); (BA0).x += pw_fdot((Q0), xv); (BA1).x += pw_fdot((Q1), xv); } \
  { half4 xv = *(const device half4*)((X1) + (CB)); (BA0).y += pw_fdot((Q0), xv); (BA1).y += pw_fdot((Q1), xv); } }
#define PW_MDOT4(Q0, Q1, X0, X1, X2, X3, CB, BA0, BA1) { \
  { half4 xv = *(const device half4*)((X0) + (CB)); (BA0).x += pw_fdot((Q0), xv); (BA1).x += pw_fdot((Q1), xv); } \
  { half4 xv = *(const device half4*)((X1) + (CB)); (BA0).y += pw_fdot((Q0), xv); (BA1).y += pw_fdot((Q1), xv); } \
  { half4 xv = *(const device half4*)((X2) + (CB)); (BA0).z += pw_fdot((Q0), xv); (BA1).z += pw_fdot((Q1), xv); } \
  { half4 xv = *(const device half4*)((X3) + (CB)); (BA0).w += pw_fdot((Q0), xv); (BA1).w += pw_fdot((Q1), xv); } }
inline float  pw_fin(half4  a) { return pw_h4sum(a); }
inline float2 pw_fin(PwH2   a) { return float2(pw_h4sum(a.x), pw_h4sum(a.y)); }
inline float4 pw_fin(PwH4   a) { return float4(pw_h4sum(a.x), pw_h4sum(a.y), pw_h4sum(a.z), pw_h4sum(a.w)); }
inline float  pw_fin(float  a) { return a; }
inline float2 pw_fin(float2 a) { return a; }
inline float4 pw_fin(float4 a) { return a; }
