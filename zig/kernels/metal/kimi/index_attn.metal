// K3's absorbed MLA over an explicit chronological index list; copied from our own dense MLA arithmetic.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint K31_LAT = 512, K31_ROPE = 64, K31_KEY = 576, K31_NOPE = 128, K31_QD = 192;
constant constexpr uint K31_SPLITS = 16, K31_TILE = 16;
struct K31AttnRow { uint slot, pos, list, nkeys; };
struct K31MlaArgs {
  device bfloat* cache;
  uint slot_keys, heads, head0, rows, q_stride, kv_stride, gate_stride, out_stride;
  float eps, scale;
};

inline uint2 k31_split(uint n, uint s) {
  const uint span = 512 * ((n + 512 * K31_SPLITS - 1) / (512 * K31_SPLITS));
  return uint2(min(s * span, n), min((s + 1) * span, n));
}

// Only key addressing differs from k3_mla_attend; lane, key, split and FMA orders stay the same.
kernel void k3_index_attend(const device bfloat* q [[buffer(0)]], const device float* qlat [[buffer(1)]],
                            const device K31AttnRow* rows [[buffer(2)]], device float* partial [[buffer(3)]],
                            constant K31MlaArgs& a [[buffer(4)]], const device uint* list [[buffer(5)]],
                            uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat tile[K31_TILE * K31_KEY];
  const uint r = tg.z, s = tg.y, hl = tg.x * nsg + sg;
  const K31AttnRow row = rows[r];
  const uint2 span = k31_split(row.nkeys, s);
  if (span.x >= span.y) return;
  const device bfloat* keys = a.cache + ulong(row.slot) * a.slot_keys * K31_KEY;
  const bool live = hl < a.heads;
  float qv[16], qr[2], o[16], m = -INFINITY, l = 0.0f;
  for (int i = 0; i < 16; ++i) qv[i] = live ? qlat[(r * a.heads + hl) * K31_LAT + 16 * lane + i] : 0.0f, o[i] = 0.0f;
  for (int i = 0; i < 2; ++i) qr[i] = live ? float(q[r * a.q_stride + hl * K31_QD + K31_NOPE + 2 * lane + i]) : 0.0f;
  for (uint k0 = span.x; k0 < span.y; k0 += K31_TILE) {
    const uint kn = min(K31_TILE, span.y - k0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < kn * K31_KEY / 4; i += nsg * 32) {
      const uint j = i / (K31_KEY / 4), col = i % (K31_KEY / 4);
      const uint key = list[row.list + k0 + j];
      ((threadgroup uint2*)tile)[i] = ((const device uint2*)(keys + ulong(key) * K31_KEY))[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < kn; ++j) {
      const threadgroup bfloat* key = tile + j * K31_KEY;
      float p = 0.0f;
      for (int i = 0; i < 16; ++i) p = fma(qv[i], float(key[16 * lane + i]), p);
      for (int i = 0; i < 2; ++i) p = fma(qr[i], float(key[K31_LAT + 2 * lane + i]), p);
      const float sc = simd_sum(p) * a.scale;
      const float mn = max(m, sc), corr = precise::exp(m - mn), e = precise::exp(sc - mn);
      l = fma(l, corr, e);
      for (int i = 0; i < 16; ++i) o[i] = fma(e, float(key[16 * lane + i]), o[i] * corr);
      m = mn;
    }
  }
  if (!live) return;
  device float* out = partial + ((r * a.heads + hl) * K31_SPLITS + s) * (K31_LAT + 2);
  if (lane == 0) out[0] = m, out[1] = l;
  for (int i = 0; i < 16; ++i) out[2 + 16 * lane + i] = o[i];
}

// Identical split merge to K3, with the selected count replacing the dense visible-key count.
[[kernel, max_total_threads_per_threadgroup(256)]] void k3_index_merge(const device float* partial [[buffer(0)]], const device K31AttnRow* rows [[buffer(1)]],
                           device float* olat [[buffer(2)]], constant K31MlaArgs& a [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  const uint r = tg.y, hl = tg.x, n = rows[r].nkeys;
  const device float* p = partial + (r * a.heads + hl) * K31_SPLITS * (K31_LAT + 2);
  float M = -INFINITY;
  for (uint s = 0; s < K31_SPLITS && k31_split(n, s).x < k31_split(n, s).y; ++s) M = max(M, p[s * (K31_LAT + 2)]);
  float L = 0.0f, acc[2] = {0.0f, 0.0f};
  for (uint s = 0; s < K31_SPLITS && k31_split(n, s).x < k31_split(n, s).y; ++s) {
    const float f = precise::exp(p[s * (K31_LAT + 2)] - M);
    L = fma(p[s * (K31_LAT + 2) + 1], f, L);
    for (int i = 0; i < 2; ++i) acc[i] = fma(p[s * (K31_LAT + 2) + 2 + 2 * t + i], f, acc[i]);
  }
  for (int i = 0; i < 2; ++i) olat[(r * a.heads + hl) * K31_LAT + 2 * t + i] = acc[i] / L;
}
