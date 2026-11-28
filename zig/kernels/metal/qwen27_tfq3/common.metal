// Qwen27 TFQ3 (2D-AffineCoset/half4) shared structs, FWHT-128 helpers, decoders, and BF16 linears.
#define Q27_PW_UNROLL _Pragma("clang loop unroll(full)")

inline short2 pw_frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

struct Q27PwDenseDims {
  uint rows;
  uint n;
  uint k;
  uint ldx;
  uint ldo;
  uint sr_split;
  uint ksplit;
  uint pad;
};

struct Q27PwV3Tables {
  uint magic;
  uint version;
  float a3, b3, a4, b4, a5, b5;
  uint pad0[8];
  uint lut3[256];
  uint lut4[1024];
  uint lut5[1024];
  half4 c64_3[64];
  half4 c64_4[64];
  half4 c64_5[64];
};

inline float pw_sgn(uint seed, uint i) {
  uint h = (seed ^ i) * 2654435761u;
  return ((h >> 31) & 1u) ? -1.0f : 1.0f;
}

inline uint q27_pw_sr0(uint r0, uint sr_split) {
  return (sr_split != 0u) ? (r0 % sr_split) : r0;
}

inline void pw_fwht128_simd4(thread float4& v, ushort lane) {
  Q27_PW_UNROLL
  for (ushort d = 1; d <= 16; d <<= 1) {
    float4 o = simd_shuffle_xor(v, d);
    v = ((lane & d) == 0u) ? (v + o) : (o - v);
  }
  float a0 = v.x + v.y, b0 = v.x - v.y;
  float a1 = v.z + v.w, b1 = v.z - v.w;
  v = float4(a0 + a1, b0 + b1, a0 - a1, b0 - b1) * 0.08838834764831845f;
}

[[kernel]] void q27_pw_fwht_x(
    const device bfloat* X     [[buffer(0)]],
    device half* X_rot         [[buffer(1)]],
    constant Q27PwDenseDims& d [[buffer(2)]],
    uint2 gid                  [[threadgroup_position_in_grid]],
    ushort lane                [[thread_index_in_simdgroup]]
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

[[kernel]] void q27_pw_fwht_y(
    const device float* YR     [[buffer(0)]],
    device bfloat* OUT         [[buffer(1)]],
    constant Q27PwDenseDims& d [[buffer(2)]],
    uint2 gid                  [[threadgroup_position_in_grid]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint row = gid.y, r0 = gid.x * 128u;
  float4 v = float4(0.0f);
  uint ks = metal::max(1u, d.ksplit);
  for (uint kp = 0u; kp < ks; ++kp) {
    const device float* src = YR + (ulong(kp) * ulong(d.rows) + ulong(row)) * ulong(d.n) + r0;
    v += float4(src[lane], src[lane + 32u], src[lane + 64u], src[lane + 96u]);
  }
  pw_fwht128_simd4(v, lane);
  uint sr0 = q27_pw_sr0(r0, d.sr_split);
  device bfloat* dst = OUT + ulong(row) * ulong(d.ldo) + r0;
  dst[lane +  0u] = bfloat(v.x * pw_sgn(0x9E3779B9u, sr0 + lane +  0u));
  dst[lane + 32u] = bfloat(v.y * pw_sgn(0x9E3779B9u, sr0 + lane + 32u));
  dst[lane + 64u] = bfloat(v.z * pw_sgn(0x9E3779B9u, sr0 + lane + 64u));
  dst[lane + 96u] = bfloat(v.w * pw_sgn(0x9E3779B9u, sr0 + lane + 96u));
}

inline void pw_v3_q8_4b(thread uint& st, uint wc, constant Q27PwV3Tables* T, thread half4& q0, thread half4& q1) {
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

inline void pw_v3_q8_3b(thread uint& st, uint m24, constant Q27PwV3Tables* T, thread half4& q0, thread half4& q1) {
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

inline void pw_v3_q8_5b(thread uint& st, uint hi8, uint lo32, constant Q27PwV3Tables* T, thread half4& q0, thread half4& q1) {
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

inline float pw_h4sum(half4 p) { return (float(p.x) + float(p.y)) + (float(p.z) + float(p.w)); }
struct PwH2 { half4 x, y; };
struct PwH4 { half4 x, y, z, w; };
#define Q27_PW_DOT1(Q0, Q1, X, CB, BA0, BA1) { \
  half4 xv = *(const device half4*)((X) + (CB)); \
  (BA0) = fma((Q0), xv, (BA0)); (BA1) = fma((Q1), xv, (BA1)); }
#define Q27_PW_DOT2(Q0, Q1, X0, X1, CB, BA0, BA1) { \
  { half4 xv = *(const device half4*)((X0) + (CB)); (BA0).x = fma((Q0), xv, (BA0).x); (BA1).x = fma((Q1), xv, (BA1).x); } \
  { half4 xv = *(const device half4*)((X1) + (CB)); (BA0).y = fma((Q0), xv, (BA0).y); (BA1).y = fma((Q1), xv, (BA1).y); } }
#define Q27_PW_DOT4(Q0, Q1, X0, X1, X2, X3, CB, BA0, BA1) { \
  { half4 xv = *(const device half4*)((X0) + (CB)); (BA0).x = fma((Q0), xv, (BA0).x); (BA1).x = fma((Q1), xv, (BA1).x); } \
  { half4 xv = *(const device half4*)((X1) + (CB)); (BA0).y = fma((Q0), xv, (BA0).y); (BA1).y = fma((Q1), xv, (BA1).y); } \
  { half4 xv = *(const device half4*)((X2) + (CB)); (BA0).z = fma((Q0), xv, (BA0).z); (BA1).z = fma((Q1), xv, (BA1).z); } \
  { half4 xv = *(const device half4*)((X3) + (CB)); (BA0).w = fma((Q0), xv, (BA0).w); (BA1).w = fma((Q1), xv, (BA1).w); } }
inline float  pw_fin(half4 a) { return pw_h4sum(a); }
inline float2 pw_fin(PwH2  a) { return float2(pw_h4sum(a.x), pw_h4sum(a.y)); }
inline float4 pw_fin(PwH4  a) { return float4(pw_h4sum(a.x), pw_h4sum(a.y), pw_h4sum(a.z), pw_h4sum(a.w)); }

// BF16 linear projections for residual BF16 modules (in_proj_a/b, optional BF16 lm_head).
[[kernel]] void q27_bf16_mv1(
    const device bfloat* W     [[buffer(0)]],
    const device bfloat* X     [[buffer(1)]],
    device bfloat* OUT         [[buffer(2)]],
    constant Q27PwDenseDims& d [[buffer(3)]],
    uint gid                   [[threadgroup_position_in_grid]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint r = gid;
  if (r >= d.n) return;
  const device bfloat4* wr = (const device bfloat4*)(W + ulong(r) * ulong(d.k));
  const device bfloat4* xv = (const device bfloat4*)X;
  uint n4 = d.k >> 2;
  float acc = 0.0f;
  for (uint c = uint(lane); c < n4; c += 32u) {
    acc += dot(float4(wr[c]), float4(xv[c]));
  }
  acc = simd_sum(acc);
  if (lane == 0u) OUT[r] = bfloat(acc);
}

[[kernel]] void q27_bf16_mm(
    const device bfloat* W     [[buffer(0)]],
    const device bfloat* X     [[buffer(1)]],
    device bfloat* OUT         [[buffer(2)]],
    constant Q27PwDenseDims& d [[buffer(3)]],
    uint2 gid                  [[threadgroup_position_in_grid]],
    ushort sg                  [[simdgroup_index_in_threadgroup]],
    ushort lane                [[thread_index_in_simdgroup]]
) {
  uint r = gid.x * 4u + uint(sg);
  uint m0 = gid.y * 4u;
  if (r >= d.n || m0 >= d.rows) return;
  uint nc = metal::min(4u, d.rows - m0);
  const device bfloat* wr = W + ulong(r) * ulong(d.k);
  const device bfloat* x0 = X + ulong(m0 + 0u) * ulong(d.ldx);
  const device bfloat* x1 = X + ulong(m0 + ((nc > 1u) ? 1u : 0u)) * ulong(d.ldx);
  const device bfloat* x2 = X + ulong(m0 + ((nc > 2u) ? 2u : 0u)) * ulong(d.ldx);
  const device bfloat* x3 = X + ulong(m0 + ((nc > 3u) ? 3u : 0u)) * ulong(d.ldx);
  float4 acc = float4(0.0f);
  for (uint c = uint(lane) * 4u; c < d.k; c += 128u) {
    float4 wv = float4(float(wr[c]), float(wr[c + 1u]), float(wr[c + 2u]), float(wr[c + 3u]));
    acc.x += dot(wv, float4(float(x0[c]), float(x0[c + 1u]), float(x0[c + 2u]), float(x0[c + 3u])));
    if (nc > 1u) acc.y += dot(wv, float4(float(x1[c]), float(x1[c + 1u]), float(x1[c + 2u]), float(x1[c + 3u])));
    if (nc > 2u) acc.z += dot(wv, float4(float(x2[c]), float(x2[c + 1u]), float(x2[c + 2u]), float(x2[c + 3u])));
    if (nc > 3u) acc.w += dot(wv, float4(float(x3[c]), float(x3[c + 1u]), float(x3[c + 2u]), float(x3[c + 3u])));
  }
  acc.x = simd_sum(acc.x);
  if (nc > 1u) acc.y = simd_sum(acc.y);
  if (nc > 2u) acc.z = simd_sum(acc.z);
  if (nc > 3u) acc.w = simd_sum(acc.w);
  if (lane == 0u) {
    OUT[ulong(m0 + 0u) * ulong(d.ldo) + r] = bfloat(acc.x);
    if (nc > 1u) OUT[ulong(m0 + 1u) * ulong(d.ldo) + r] = bfloat(acc.y);
    if (nc > 2u) OUT[ulong(m0 + 2u) * ulong(d.ldo) + r] = bfloat(acc.z);
    if (nc > 3u) OUT[ulong(m0 + 3u) * ulong(d.ldo) + r] = bfloat(acc.w);
  }
}
