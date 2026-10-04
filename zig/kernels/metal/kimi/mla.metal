// MLA in absorbed form (NoPE): latent cache rows, q into latent space, split-key attention, merge, W_uv and gate.
#include "kimi_common.h"

constant constexpr uint LAT = 512, ROPE = 64, KEY = LAT + ROPE, NOPE = 128, VD = 128, QD = NOPE + ROPE;
constant constexpr uint SPLITS = 16, TILE = 16;

// Where each row of the round lives: its stream's cache slot and its position there.
struct MlaRow {
  uint slot, pos;
};

struct MlaArgs {
  device bfloat* cache;
  uint slot_keys, heads, head0, rows, q_stride, kv_stride, gate_stride, out_stride;
  float eps, scale;
};

inline device bfloat* k3_keys(constant MlaArgs& a, MlaRow r) {
  return a.cache + ulong(r.slot) * a.slot_keys * KEY;
}

// The row's latent into its cache: KimiRMSNorm of kv_a's first 512, then its 64 rope values as they are.
kernel void k3_mla_cache(device const bfloat* kv [[buffer(0)]], device const bfloat* w [[buffer(1)]],
                         device const MlaRow* rows [[buffer(2)]], constant MlaArgs& a [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  device const bfloat* x = kv + row * a.kv_stride;
  device bfloat* dst = k3_keys(a, rows[row]) + rows[row].pos * KEY;
  float ss = 0.0f;
  for (uint i = t; i < LAT; i += 256) ss = fma(float(x[i]), float(x[i]), ss);
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (uint g = 0; g < 8; ++g) var += part[g];
  const float r = precise::rsqrt(var / float(LAT) + a.eps);
  for (uint i = t; i < LAT; i += 256) dst[i] = bfloat(float(w[i]) * float(bfloat(float(x[i]) * r)));
  if (t < ROPE) dst[LAT + t] = x[LAT + t];
}

// qlat[r, h, c] = sum over d of q_nope[r, h, d] kv_b[h, d, c]: a thread a latent column, rows in passes of 16.
kernel void k3_mla_qlat(device const bfloat* q [[buffer(0)]], device const bfloat* kvb [[buffer(1)]],
                        device float* qlat [[buffer(2)]], constant MlaArgs& a [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint hl = tg.y, c = tg.x * 256 + t;
  device const bfloat* wk = kvb + (a.head0 + hl) * (NOPE + VD) * LAT + c;
  for (uint r0 = 0; r0 < a.rows; r0 += 16) {
    float acc[16];
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
    for (uint d = 0; d < NOPE; ++d) {
      const float wv = float(wk[d * LAT]);
      for (int r = 0; r < 16; ++r)
        if (r0 + r < a.rows) acc[r] = fma(float(q[(r0 + r) * a.q_stride + hl * QD + d]), wv, acc[r]);
    }
    for (int r = 0; r < 16; ++r)
      if (r0 + r < a.rows) qlat[((r0 + r) * a.heads + hl) * LAT + c] = acc[r];
  }
}

// Key range of split s for a row with n keys: splits of 512 keys, widened by 512 until 16 cover the row.
inline uint2 k3_split(uint n, uint s) {
  const uint span = 512 * ((n + 512 * SPLITS - 1) / (512 * SPLITS));
  return uint2(min(s * span, n), min((s + 1) * span, n));
}

// One (row, head) a simdgroup over one key split: online softmax, partial (m, l, o[512]) out.
kernel void k3_mla_attend(device const bfloat* q [[buffer(0)]], device const float* qlat [[buffer(1)]],
                          device const MlaRow* rows [[buffer(2)]], device float* partial [[buffer(3)]],
                          constant MlaArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat tile[TILE * KEY];
  const uint r = tg.z, s = tg.y, hl = tg.x * nsg + sg;
  const uint n = rows[r].pos + 1;
  const uint2 span = k3_split(n, s);
  if (span.x >= span.y) return;
  device const bfloat* keys = k3_keys(a, rows[r]);
  const bool live = hl < a.heads;
  float qv[16], qr[2], o[16], m = -INFINITY, l = 0.0f;
  for (int i = 0; i < 16; ++i) qv[i] = live ? qlat[(r * a.heads + hl) * LAT + 16 * lane + i] : 0.0f, o[i] = 0.0f;
  for (int i = 0; i < 2; ++i) qr[i] = live ? float(q[r * a.q_stride + hl * QD + NOPE + 2 * lane + i]) : 0.0f;
  for (uint k0 = span.x; k0 < span.y; k0 += TILE) {
    const uint kn = min(TILE, span.y - k0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < kn * KEY / 4; i += nsg * 32)
      ((threadgroup uint2*)tile)[i] = ((device const uint2*)(keys + k0 * KEY))[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kn; ++j) {
      threadgroup const bfloat* key = tile + j * KEY;
      float p = 0.0f;
      for (int i = 0; i < 16; ++i) p = fma(qv[i], float(key[16 * lane + i]), p);
      for (int i = 0; i < 2; ++i) p = fma(qr[i], float(key[LAT + 2 * lane + i]), p);
      const float sc = simd_sum(p) * a.scale;
      const float mn = max(m, sc), corr = precise::exp(m - mn), e = precise::exp(sc - mn);
      l = fma(l, corr, e);
      for (int i = 0; i < 16; ++i) o[i] = fma(e, float(key[16 * lane + i]), o[i] * corr);
      m = mn;
    }
  }
  if (!live) return;
  device float* out = partial + ((r * a.heads + hl) * SPLITS + s) * (LAT + 2);
  if (lane == 0) out[0] = m, out[1] = l;
  for (int i = 0; i < 16; ++i) out[2 + 16 * lane + i] = o[i];
}

// Merge a (row, head)'s splits in order: o = sum o_s exp(m_s - M) / sum l_s exp(m_s - M), fp32 latent.
kernel void k3_mla_merge(device const float* partial [[buffer(0)]], device const MlaRow* rows [[buffer(1)]],
                         device float* olat [[buffer(2)]], constant MlaArgs& a [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint r = tg.y, hl = tg.x, n = rows[r].pos + 1;
  device const float* p = partial + (r * a.heads + hl) * SPLITS * (LAT + 2);
  float M = -INFINITY;
  for (uint s = 0; s < SPLITS && k3_split(n, s).x < k3_split(n, s).y; ++s) M = max(M, p[s * (LAT + 2)]);
  float L = 0.0f, acc[2] = {0.0f, 0.0f};
  for (uint s = 0; s < SPLITS && k3_split(n, s).x < k3_split(n, s).y; ++s) {
    const float f = precise::exp(p[s * (LAT + 2)] - M);
    L = fma(p[s * (LAT + 2) + 1], f, L);
    for (int i = 0; i < 2; ++i) acc[i] = fma(p[s * (LAT + 2) + 2 + 2 * t + i], f, acc[i]);
  }
  for (int i = 0; i < 2; ++i) olat[(r * a.heads + hl) * LAT + 2 * t + i] = acc[i] / L;
}

// attn[r, h, v] = bf16(kv_b's W_uv rows . olat) times bf16(sigmoid(gate)), rounded to bf16: o_proj's input.
template <int RB>
kernel void k3_mla_uv(device const float* olat [[buffer(0)]], device const bfloat* kvb [[buffer(1)]],
                      device const bfloat* gate [[buffer(2)]], device bfloat* y [[buffer(3)]],
                      constant MlaArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const uint hl = tg.y, v = tg.x * nsg + sg;
  device const bfloat* wv = kvb + ((a.head0 + hl) * (NOPE + VD) + NOPE + v) * LAT;
  for (uint r0 = 0; r0 < a.rows; r0 += RB) {
    float acc[RB];
    for (int r = 0; r < RB; ++r) acc[r] = 0.0f;
    for (uint c = lane * 4; c < LAT; c += 128) {
      const float4 w4 = k3_bf16x4(*(device const uint2*)(wv + c));
      for (int r = 0; r < RB; ++r) {
        if (r0 + r >= a.rows) break;
        const float4 x4 = *(device const float4*)(olat + ((r0 + r) * a.heads + hl) * LAT + c);
        for (int j = 0; j < 4; ++j) acc[r] = fma(x4[j], w4[j], acc[r]);
      }
    }
    for (int r = 0; r < RB; ++r) {
      const float sv = simd_sum(acc[r]);
      if (lane == 0 && r0 + r < a.rows) {
        const uint row = r0 + r, col = hl * VD + v;
        const float g = float(bfloat(k3_sigmoid(float(gate[row * a.gate_stride + col]))));
        y[row * a.out_stride + col] = bfloat(float(bfloat(sv)) * g);
      }
    }
  }
}

template [[host_name("k3_mla_uv")]] kernel void k3_mla_uv<16>(device const float*, device const bfloat*,
    device const bfloat*, device bfloat*, constant MlaArgs&, uint2, uint, uint, uint);
