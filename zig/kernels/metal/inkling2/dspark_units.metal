//! Mini DSpark's block attention on the tensor units; dspark_sources prepends the tensor-fragment helpers.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)

constant constexpr uint DSU_D = 128, DSU_KV = 1024, DSU_KEYS = 64, DSU_PAIRS = 24, DSU_GROUP = 3;
constant constexpr float DSU_SCALE = 0x1.6a09e6p-4f, DSU_LOG2E = 1.4426950408889634f;
struct DsparkUnitsArgs { uint committed, ring, first, chunk, chunks; };

// The 16x16 block at key row r, column c of a tile in a context ring; a key outside [lo, hi) reads as zero.
inline void dsu_tile_get(thread tfp::frag<bfloat>& f, const device bfloat* tile, int r, int c, short2 home, int lo, int hi) {
  const device bfloat* p = tile + (r + home.y) * int(DSU_KV) + (c + home.x);
  const bfloat none = bfloat(0);
  const bfloat4 zero = bfloat4(none, none, none, none);
  const int x = r + home.y, y = x + 8;
  f = tfp::frag<bfloat>(x >= lo && x < hi ? bfloat4(*(const device bfloat4*)p) : zero,
                        y >= lo && y < hi ? bfloat4(*(const device bfloat4*)(p + 8 * int(DSU_KV))) : zero);
}

inline void dsu_rows(thread const tfp::frag<float>* s, thread float (&r)[2], bool maximum) {
  for (uint f = 0; f < 4; ++f) for (uint h = 0; h < 2; ++h) {
    const uint b = 4 * h;
    float v = maximum ? max(max(s[f][b], s[f][b + 1]), max(s[f][b + 2], s[f][b + 3]))
                      : (s[f][b] + s[f][b + 1]) + (s[f][b + 2] + s[f][b + 3]);
    const float x = simd_shuffle_xor(v, ushort(1));
    v = maximum ? max(v, x) : v + x;
    const float y = simd_shuffle_xor(v, ushort(8));
    v = maximum ? max(v, y) : v + y;
    r[h] = maximum ? max(r[h], v) : r[h] + v;
  }
}

// A key head's 24 (row, head) pairs and one chunk of the retained context a 64-thread group; a simdgroup owns 16 pairs.
[[max_total_threads_per_threadgroup(64)]]
kernel void tf_ink2_dspark_attn_units(const device bfloat* q [[buffer(0)]], const device bfloat* kc [[buffer(1)]],
    const device bfloat* vc [[buffer(2)]], device float* part_o [[buffer(3)]], device float2* part_ml [[buffer(4)]],
    constant DsparkUnitsArgs& a [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint kvh = tg.y, b_lo = (a.first / a.chunk + tg.x) * a.chunk, k_hi = min(b_lo + a.chunk, a.committed) - 1;
  const uint m = 16 * sg;
  const short2 home = tfp::frag_home(ushort(lane));
  const uint row_a = m + uint(home.y), row_b = row_a + 8;
  const uint pair_a = min(row_a, DSU_PAIRS - 1), pair_b = min(row_b, DSU_PAIRS - 1);
  const device bfloat* qb0 = q + ((pair_a / DSU_GROUP) * DSU_PAIRS + kvh * DSU_GROUP + pair_a % DSU_GROUP) * DSU_D;
  const device bfloat* qb1 = q + ((pair_b / DSU_GROUP) * DSU_PAIRS + kvh * DSU_GROUP + pair_b % DSU_GROUP) * DSU_D;
  tfp::frag<float> acc[8];
  for (uint d = 0; d < 8; ++d) acc[d] = tfp::frag<float>(0);
  float top[2] = {-INFINITY, -INFINITY}, mass[2] = {0.0f, 0.0f};
  for (uint kt = b_lo; kt <= k_hi; kt += DSU_KEYS) {
    // A tile is 64 keys from a multiple of 64 and a ring is a multiple of 64 rows, so a tile never wraps its ring.
    const device bfloat* tile_k = kc + size_t(kt % a.ring) * DSU_KV + kvh * DSU_D;
    const device bfloat* tile_v = vc + size_t(kt % a.ring) * DSU_KV + kvh * DSU_D;
    const int lo = a.first > kt ? int(a.first - kt) : 0, hi = int(min(kt + DSU_KEYS, a.committed) - kt);
    if (lo >= hi) continue;
    tfp::frag<float> score[4] = {tfp::frag<float>(0), tfp::frag<float>(0), tfp::frag<float>(0), tfp::frag<float>(0)};
    for (uint d = 0; d < 8; ++d) {
      tfp::frag<bfloat> qa, kb0, kb1, kb2, kb3;
      tfp::frag_get_rows(qa, qb0, qb1, 16 * d, home, row_a < DSU_PAIRS, row_b < DSU_PAIRS);
      dsu_tile_get(kb0, tile_k, 0, 16 * d, home, lo, hi);
      dsu_tile_get(kb1, tile_k, 16, 16 * d, home, lo, hi);
      dsu_tile_get(kb2, tile_k, 32, 16 * d, home, lo, hi);
      dsu_tile_get(kb3, tile_k, 48, 16 * d, home, lo, hi);
      tfp::mma_16x32<false, true>(score[0], score[1], qa, kb0, kb1);
      tfp::mma_16x32<false, true>(score[2], score[3], qa, kb2, kb3);
    }
    float peak[2] = {top[0], top[1]};
    for (uint f = 0; f < 4; ++f) for (uint e = 0; e < 8; ++e) {
      const uint row = m + uint(home.y) + (e >> 2) * 8;
      const int key = int(16 * f) + home.x + int(TF_COL(e));
      const bool live = row < DSU_PAIRS && key >= lo && key < hi;
      score[f][e] = live ? score[f][e] * DSU_SCALE * DSU_LOG2E : -INFINITY;
    }
    dsu_rows(score, peak, true);
    float keep[2];
    for (uint h = 0; h < 2; ++h) {
      const float mu = peak[h] == -INFINITY ? 0.0f : peak[h];
      keep[h] = precise::exp2(top[h] - mu);
      top[h] = peak[h]; mass[h] *= keep[h];
    }
    tfp::frag<bfloat> p[4];
    for (uint f = 0; f < 4; ++f) for (uint e = 0; e < 8; ++e) {
      const float mu = peak[e >> 2] == -INFINITY ? 0.0f : peak[e >> 2];
      p[f][e] = bfloat(precise::exp2(score[f][e] - mu));
    }
    // Reduce the rounded P, not its pre-bf16 exponential.
    tfp::frag<float> rounded[4];
    for (uint f = 0; f < 4; ++f) for (uint e = 0; e < 8; ++e) rounded[f][e] = float(p[f][e]);
    dsu_rows(rounded, mass, false);
    for (uint d = 0; d < 8; ++d) for (uint e = 0; e < 8; ++e) acc[d][e] *= keep[e >> 2];
    for (uint f = 0; f < 4; ++f) for (uint d = 0; d < 8; d += 2) {
      tfp::frag<bfloat> v0, v1;
      dsu_tile_get(v0, tile_v, 16 * f, 16 * d, home, lo, hi);
      dsu_tile_get(v1, tile_v, 16 * f, 16 * d + 16, home, lo, hi);
      tfp::mma_16x32<false, false>(acc[d], acc[d + 1], p[f], v0, v1);
    }
  }
  for (uint d = 0; d < 8; ++d) for (uint e = 0; e < 8; ++e) {
    const uint row = m + uint(home.y) + (e >> 2) * 8, col = uint(home.x) + TF_COL(e);
    if (row >= DSU_PAIRS) continue;
    const size_t slot = size_t((row / DSU_GROUP) * DSU_PAIRS + kvh * DSU_GROUP + row % DSU_GROUP) * a.chunks + tg.x;
    part_o[slot * DSU_D + 16 * d + col] = acc[d][e];
  }
  if (home.x == 0) for (uint h = 0; h < 2; ++h) {
    const uint row = m + uint(home.y) + 8 * h;
    if (row >= DSU_PAIRS) continue;
    const size_t slot = size_t((row / DSU_GROUP) * DSU_PAIRS + kvh * DSU_GROUP + row % DSU_GROUP) * a.chunks + tg.x;
    part_ml[slot] = float2(top[h], mass[h]);
  }
}

// A (row, head) a 128-thread group: the block's eight keys, then the context's chunks in order, rounded once to bf16.
[[max_total_threads_per_threadgroup(128)]]
kernel void tf_ink2_dspark_attn_merge(const device bfloat* q [[buffer(0)]], const device bfloat* block_k [[buffer(1)]],
    const device bfloat* block_v [[buffer(2)]], const device float* part_o [[buffer(3)]],
    const device float2* part_ml [[buffer(4)]], device bfloat* out [[buffer(5)]], constant DsparkUnitsArgs& a [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint kv = (group % DSU_PAIRS) / DSU_GROUP, q_at = group * DSU_D;
  threadgroup float bs[8];
  if (d < 8) {
    float dot = 0.0f;
    for (uint i = 0; i < DSU_D; ++i) dot += float(q[q_at + i]) * float(block_k[d * DSU_KV + kv * DSU_D + i]);
    bs[d] = dot * DSU_SCALE * DSU_LOG2E;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mm = bs[0];
  for (uint j = 1; j < 8; ++j) mm = max(mm, bs[j]);
  float ll = 0.0f, o = 0.0f;
  for (uint j = 0; j < 8; ++j) {
    const float w = float(bfloat(precise::exp2(bs[j] - mm)));
    ll += w;
    o += w * float(block_v[j * DSU_KV + kv * DSU_D + d]);
  }
  for (uint c = 0; c < a.chunks; ++c) {
    const size_t at = size_t(group) * a.chunks + c;
    const float2 ml = part_ml[at];
    const float mn = max(mm, ml.x), x = precise::exp2(mm - mn), y = precise::exp2(ml.x - mn);
    ll = ll * x + ml.y * y;
    o = o * x + part_o[at * DSU_D + d] * y;
    mm = mn;
  }
  out[q_at + d] = bfloat(o / ll);
}
