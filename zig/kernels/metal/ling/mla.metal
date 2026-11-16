// Ling MLA extras around k3: RoPE (GPT-J pairs then halves) and per-head gate expand.

// k3_mla_* kernels are unchanged. Layout matches ling_forward.py rot() and HF apply_rotary_pos_emb_interleave.

// Pair (z[2i], z[2i+1]) rotates to (out[i], out[32+i]).
#include <metal_stdlib>
using namespace metal;

constant constexpr uint LAT = 512, ROPE = 64, KEY = LAT + ROPE, NOPE = 128, VD = 128, QD = NOPE + ROPE;

struct MlaRow {
  uint slot, pos;
};

struct MlaArgs {
  device bfloat* cache;
  uint slot_keys, heads, head0, rows, q_stride, kv_stride, gate_stride, out_stride;
  float eps, scale;
};

// cs is (rows, 32, 2) f32: host f64 cos/sin of pos*inv, rounded to f32. Kernels apply only.
inline float2 ling_apply(float x0, float x1, const device float* cs, uint row, uint i) {
  const uint at = (row * (ROPE / 2) + i) * 2;
  const float c = cs[at], s = cs[at + 1];
  return float2(fma(x0, c, -(x1 * s)), fma(x1, c, x0 * s));
}

inline device bfloat* ling_keys(constant MlaArgs& a, MlaRow r) {
  return a.cache + ulong(r.slot) * a.slot_keys * KEY;
}

// In-place GPT-J rotate of the 64 rope columns k3_mla_cache just copied.

// 32 threads, no early return: the threadgroup temp needs every lane at the barrier.

// Cache address is rows[row].pos.
kernel void tf_ling_mla_rope_cache(device const MlaRow* rows [[buffer(0)]], constant MlaArgs& a [[buffer(1)]],
                                   device const float* cs [[buffer(2)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[ROPE];
  device bfloat* dst = ling_keys(a, rows[row]) + ulong(rows[row].pos) * KEY + LAT;
  const float2 o = ling_apply(float(dst[2 * t]), float(dst[2 * t + 1]), cs, row, t);
  tmp[t] = o.x;
  tmp[t + ROPE / 2] = o.y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  dst[t] = bfloat(tmp[t]);
  dst[t + ROPE / 2] = bfloat(tmp[t + ROPE / 2]);
}

// Same rotate on q's rope region (offset NOPE in each head) in place. Angles from cs[row].
kernel void tf_ling_mla_rope_q(device bfloat* q [[buffer(0)]], constant MlaArgs& a [[buffer(1)]],
                               device const float* cs [[buffer(2)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[ROPE];
  const uint hl = tg.x, row = tg.y;
  device bfloat* z = q + ulong(row) * a.q_stride + hl * QD + NOPE;
  const float2 o = ling_apply(float(z[2 * t]), float(z[2 * t + 1]), cs, row, t);
  tmp[t] = o.x;
  tmp[t + ROPE / 2] = o.y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  z[t] = bfloat(tmp[t]);
  z[t + ROPE / 2] = bfloat(tmp[t + ROPE / 2]);
}

// Broadcast each per-head gate to 128 value columns so k3_mla_uv keeps Kimi's stride.
kernel void tf_ling_mla_gate_expand(device const bfloat* g [[buffer(0)]], device bfloat* g2 [[buffer(1)]],
                                    constant MlaArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const uint col = gid.x, row = gid.y;
  if (row >= a.rows || col >= a.heads * VD) return;
  g2[row * a.gate_stride + col] = g[row * a.heads + col / VD];
}
