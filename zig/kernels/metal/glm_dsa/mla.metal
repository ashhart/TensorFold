// GLM-5.3's MLA for lane rows: latent and rope keys in a bf16 cache, queries absorbed into latent space.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint LAT = 512, ROPE = 64, NOPE = 192, KEY = LAT + ROPE, QK = NOPE + ROPE;

// Per row: its stream's cache slot, its position, and where its key list starts in `keys` (count in `nkeys`).
struct MlaRow {
  uint slot, pos, list, nkeys;
};
struct MlaArgs {
  uint rows, heads, head0, slot_keys;
  uint proj_stride, q_stride, pad2, kvb_in_blocks;
  float eps, scale, pad0, pad1;
  ulong kvb, kvb_scale;
};

// The sum of a value over the simdgroup in one fixed butterfly.
inline float glm_lanes(float v) {
  for (ushort d = 16; d > 0; d >>= 1) v += simd_shuffle_xor(v, d);
  return v;
}

// Pair i of a 64-wide interleaved rope at position pos: (x[2i], x[2i+1]) -> out[i], out[32 + i].
inline float2 glm_rope(float x0, float x1, uint pos, const device float* inv, uint i) {
  const float a = float(pos) * inv[i];
  const float c = precise::cos(a), s = precise::sin(a);
  return float2(fma(x0, c, -(x1 * s)), fma(x1, c, x0 * s));
}

// One row a threadgroup: q_resid = rms(q_a), the cached latent bf16(rms(kv_a[:512])), rope key bf16(rope(rest)).
kernel void glm_mla_prep(const device float* proj [[buffer(0)]], const device bfloat* w_qa [[buffer(1)]],
                         const device bfloat* w_kva [[buffer(2)]], const device float* inv [[buffer(3)]],
                         const device MlaRow* rows [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                         device float* q_resid [[buffer(6)]], device bfloat* cache [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[2][8];
  const device float* x = proj + ulong(row) * a.proj_stride;
  float sq = 0.0f, sk = 0.0f;
  for (uint i = t; i < 2048; i += 256) sq = fma(x[i], x[i], sq);
  for (uint i = t; i < LAT; i += 256) sk = fma(x[2048 + i], x[2048 + i], sk);
  sq = glm_lanes(sq);
  sk = glm_lanes(sk);
  if (lane == 0) part[0][sg] = sq, part[1][sg] = sk;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float vq = 0.0f, vk = 0.0f;
  for (uint g = 0; g < 8; ++g) vq += part[0][g], vk += part[1][g];
  const float rq = precise::rsqrt(vq / 2048.0f + a.eps), rk = precise::rsqrt(vk / float(LAT) + a.eps);
  for (uint i = t; i < 2048; i += 256) q_resid[ulong(row) * a.q_stride + i] = float(w_qa[i]) * (x[i] * rq);
  const MlaRow r = rows[row];
  device bfloat* key = cache + (ulong(r.slot) * a.slot_keys + r.pos) * KEY;
  for (uint i = t; i < LAT; i += 256) key[i] = bfloat(float(w_kva[i]) * (x[2048 + i] * rk));
  if (t < ROPE / 2) {
    const float2 o = glm_rope(x[2048 + LAT + 2 * t], x[2048 + LAT + 2 * t + 1], r.pos, inv, t);
    key[LAT + t] = bfloat(o.x);
    key[LAT + ROPE / 2 + t] = bfloat(o.y);
  }
}

// Head h, latent columns 128 cb..: qlat[r][h][c] = sum_d W_uk[h][d][c] q_nope[r][h][d] in d order; cb 0 ropes q_rope.
template <uint R>
inline void glm_mla_absorb_impl(const device float* q, const device uchar* W, const device MlaRow* rows,
                                constant MlaArgs& a, device float* qlat, device float* qrope, const device float* inv,
                                uint h, uint cb, uint t, threadgroup float* qs) {
  const uint n = min(a.rows, R), hg = a.head0 + h, c = 128 * cb + t;
  for (uint i = t; i < R * NOPE; i += 128) qs[i] = i / NOPE < n ? q[ulong(i / NOPE) * a.q_stride + h * QK + i % NOPE] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[R];
  for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
  const device uchar* fp8 = W + a.kvb;
  const device float* scale = reinterpret_cast<const device float*>(W + a.kvb_scale);
  for (uint d = 0; d < NOPE; ++d) {
    const uint o = 448 * hg + d;
    const uint code = fp8[ulong(o) * LAT + c];
    const float v = float(as_type<half>(ushort((code & 0x7Fu) << 7))) * 256.0f;
    const float w = ((code & 0x80u) ? -v : v) * scale[(o / 128) * a.kvb_in_blocks + c / 128];
    for (uint r = 0; r < R; ++r) acc[r] = fma(w, qs[r * NOPE + d], acc[r]);
  }
  for (uint r = 0; r < n; ++r) qlat[(ulong(r) * a.heads + h) * LAT + c] = acc[r];
  if (cb != 0 || t >= ROPE / 2) return;
  for (uint r = 0; r < n; ++r) {
    const device float* x = q + ulong(r) * a.q_stride + h * QK + NOPE;
    const float2 o = glm_rope(x[2 * t], x[2 * t + 1], rows[r].pos, inv, t);
    qrope[(ulong(r) * a.heads + h) * ROPE + t] = o.x;
    qrope[(ulong(r) * a.heads + h) * ROPE + ROPE / 2 + t] = o.y;
  }
}

#define GLM_MLA_ABSORB(R)                                                                                             \
  kernel void glm_mla_absorb_##R(const device float* q [[buffer(0)]], const device uchar* W [[buffer(1)]],          \
                                 const device MlaRow* rows [[buffer(2)]], constant MlaArgs& a [[buffer(3)]],        \
                                 device float* qlat [[buffer(4)]], device float* qrope [[buffer(5)]],               \
                                 const device float* inv [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], \
                                 uint t [[thread_index_in_threadgroup]]) {                                          \
    threadgroup float qs[R * NOPE];                                                                                 \
    glm_mla_absorb_impl<R>(q, W, rows, a, qlat, qrope, inv, tg.y, tg.x, t, qs);                                     \
  }
GLM_MLA_ABSORB(1)
GLM_MLA_ABSORB(2)
GLM_MLA_ABSORB(4)
GLM_MLA_ABSORB(8)
GLM_MLA_ABSORB(16)

// Keys a split takes: splits of 128 list entries, widened by 128 until 16 cover the list (fixed by the count alone).
constant constexpr uint SPLITS = 16;
inline uint2 glm_split(uint n, uint s) {
  const uint span = 128 * ((n + 128 * SPLITS - 1) / (128 * SPLITS));
  return uint2(min(s * span, n), min((s + 1) * span, n));
}

// One (row, head) a simdgroup over a split of its keys: online softmax, partial (m, l, o[512]); a lane 16 + 2 dims.
kernel void glm_mla_attend(const device float* qlat [[buffer(0)]], const device float* qrope [[buffer(1)]],
                           const device MlaRow* rows [[buffer(2)]], const device uint* keys [[buffer(3)]],
                           const device bfloat* cache [[buffer(4)]], constant MlaArgs& a [[buffer(5)]],
                           device float* partial [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.z, s = tg.y, h = tg.x * 4 + sg;
  if (h >= a.heads) return;
  const MlaRow row = rows[r];
  const uint2 span = glm_split(row.nkeys, s);
  device float* out = partial + ((ulong(r) * a.heads + h) * SPLITS + s) * (LAT + 2);
  if (span.x >= span.y) return;
  float qv[16], qr[2], o[16], m = -INFINITY, l = 0.0f;
  for (uint i = 0; i < 16; ++i) qv[i] = qlat[(ulong(r) * a.heads + h) * LAT + 16 * lane + i], o[i] = 0.0f;
  for (uint i = 0; i < 2; ++i) qr[i] = qrope[(ulong(r) * a.heads + h) * ROPE + 2 * lane + i];
  const device bfloat* base = cache + ulong(row.slot) * a.slot_keys * KEY;
  for (uint j = span.x; j < span.y; ++j) {
    const device bfloat* key = base + ulong(keys[row.list + j]) * KEY;
    float p = 0.0f;
    for (uint i = 0; i < 16; ++i) p = fma(qv[i], float(key[16 * lane + i]), p);
    for (uint i = 0; i < 2; ++i) p = fma(qr[i], float(key[LAT + 2 * lane + i]), p);
    const float sc = glm_lanes(p) * a.scale;
    const float mn = max(m, sc), corr = precise::exp(m - mn), e = precise::exp(sc - mn);
    l = fma(l, corr, e);
    for (uint i = 0; i < 16; ++i) o[i] = fma(e, float(key[16 * lane + i]), o[i] * corr);
    m = mn;
  }
  if (lane == 0) out[0] = m, out[1] = l;
  for (uint i = 0; i < 16; ++i) out[2 + 16 * lane + i] = o[i];
}

// A (row, head)'s splits merged in order: o = sum o_s exp(m_s - M) / sum l_s exp(m_s - M), fp32 latent.
kernel void glm_mla_merge(const device float* partial [[buffer(0)]], const device MlaRow* rows [[buffer(1)]],
                          constant MlaArgs& a [[buffer(2)]], device float* olat [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint r = tg.y, h = tg.x, n = rows[r].nkeys;
  const device float* p = partial + (ulong(r) * a.heads + h) * SPLITS * (LAT + 2);
  float M = -INFINITY;
  for (uint s = 0; s < SPLITS; ++s)
    if (glm_split(n, s).x < glm_split(n, s).y) M = max(M, p[s * (LAT + 2)]);
  float L = 0.0f, acc[2] = {0.0f, 0.0f};
  for (uint s = 0; s < SPLITS; ++s) {
    if (glm_split(n, s).x >= glm_split(n, s).y) continue;
    const float f = precise::exp(p[s * (LAT + 2)] - M);
    L = fma(p[s * (LAT + 2) + 1], f, L);
    for (uint i = 0; i < 2; ++i) acc[i] = fma(p[s * (LAT + 2) + 2 + 2 * t + i], f, acc[i]);
  }
  for (uint i = 0; i < 2; ++i) olat[(ulong(r) * a.heads + h) * LAT + 2 * t + i] = precise::divide(acc[i], L);
}
