// SAM3 attention one simdgroup per (row, head), keys in order, with an optional additive bias [head, row, key].
#include <metal_stdlib>
using namespace metal;

struct Sam3RowsAttention {
  uint rows;
  uint keys;
  uint heads;
  uint head_width;
  uint has_bias;
};

// Lane l owns channels l, l + 32, … of a head of at most 256 channels.
constant constexpr uint sam3_rows_lanes = 8;

[[max_total_threads_per_threadgroup(32)]] [[kernel]] void sam3_attention_rows(
    const device float* q [[buffer(0)]], const device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    const device float* bias [[buffer(3)]], device float* out [[buffer(4)]],
    constant Sam3RowsAttention& p [[buffer(5)]], uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint r = group.x, h = group.y, d = p.head_width;
  if (r >= p.rows || h >= p.heads || d > 32 * sam3_rows_lanes) return;
  const ulong q_row = (ulong(r) * p.heads + h) * d;
  float own[sam3_rows_lanes];
  float acc[sam3_rows_lanes];
  for (uint i = 0; i < sam3_rows_lanes; i++) {
    const uint c = lane + 32 * i;
    own[i] = c < d ? q[q_row + c] : 0.0f;
    acc[i] = 0.0f;
  }
  float m = -INFINITY;
  float total = 0.0f;
  for (uint j = 0; j < p.keys; j++) {
    const ulong k_row = (ulong(j) * p.heads + h) * d;
    float z = 0.0f;
    for (uint i = 0; i < sam3_rows_lanes; i++) {
      const uint c = lane + 32 * i;
      if (c < d) z += own[i] * k[k_row + c];
    }
    z = simd_sum(z) * fast::rsqrt(float(d));
    if (p.has_bias != 0) z = z + bias[(ulong(h) * p.rows + r) * p.keys + j];
    const float top = max(m, z);
    const float o = fast::exp(m - top);
    const float e = fast::exp(z - top);
    for (uint i = 0; i < sam3_rows_lanes; i++) {
      const uint c = lane + 32 * i;
      if (c < d) acc[i] = acc[i] * o + e * v[k_row + c];
    }
    total = total * o + e;
    m = top;
  }
  for (uint i = 0; i < sam3_rows_lanes; i++) {
    const uint c = lane + 32 * i;
    if (c < d) out[q_row + c] = acc[i] / total;
  }
}
