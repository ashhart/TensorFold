#include <metal_stdlib>
using namespace metal;

// tf_lane_reg's weight order (regBlock) made in place from tile order: a threadgroup per 32x64-code block.
kernel void tf_lane_order(device uint* W [[buffer(0)]], uint t [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint tiled[256];
  device uint* b = W + (size_t)block * 256;
  tiled[t] = b[t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint lane = (t >> 2) & 31, w = 4 * (t >> 7) + (t & 3);
  const uint at = (((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1)) * 8 + ((lane >> 3) & 1) + 4 * (w & 1), shift = 16 * (lane & 1);
  const uint a = (tiled[at] >> shift) & 0xFFFFu, c = (tiled[at + 2] >> shift) & 0xFFFFu;
  // codes a0 a2 c0 c2 a1 a3 c1 c3, low nibble first
  b[t] = (a & 0xFu) | ((a >> 4) & 0xF0u) | ((c & 0xFu) << 8) | ((c << 4) & 0xF000u) | (((a >> 4) & 0xFu) << 16) | (((a >> 8) & 0xF0u) << 16) | (((c >> 4) & 0xFu) << 24) | (((c >> 8) & 0xF0u) << 24);
}

inline uint tf_extract_tile_code(const threadgroup uint* tiled, uint row_words, uint col, uint k, uint bits) {
  const uint b = k * bits, idx = col * row_words + (b >> 5), sh = b & 31;
  uint v = tiled[idx] >> sh;
  if (sh + bits > 32) v |= tiled[idx + 1] << (32 - sh);
  return v & ((1u << bits) - 1u);
}

inline uint tf_reg_coord_k(uint lane, uint w, uint j) {
  const uint k0 = 4 * (lane & 1) + 8 * ((lane >> 3) & 1) + 32 * (w & 1);
  return k0 + (j & 1) * 2 + ((j >> 1) & 1) * 16 + (j >> 2);
}

inline uint tf_reg_coord_col(uint lane, uint w) {
  return ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1);
}

// Split-plane (4+1) in-place rewrite for 5-bit g64: words 0..255 low nibbles in regBlock order, 256..319 high bits.
kernel void tf_lane_order_q5(device uint* W [[buffer(0)]], uint t [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint tiled[320];
  device uint* b = W + (size_t)block * 320;
  tiled[t] = b[t];
  if (t < 64) tiled[256 + t] = b[256 + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint lane = (t >> 2) & 31, w = 4 * (t >> 7) + (t & 3), col = tf_reg_coord_col(lane, w);
  uint lo4 = 0u;
  for (uint j = 0; j < 8; j++) lo4 |= (tf_extract_tile_code(tiled, 10, col, tf_reg_coord_k(lane, w, j), 5) & 0xFu) << (4 * j);
  b[t] = lo4;
  if (t < 64) {
    const uint hl = t >> 1, hp = t & 1;
    uint hi_w = 0u;
    for (uint s = 0; s < 4; s++) {
      const uint hw = 4 * hp + s, hcol = tf_reg_coord_col(hl, hw);
      for (uint j = 0; j < 4; j++) {
        const uint b0 = (tf_extract_tile_code(tiled, 10, hcol, tf_reg_coord_k(hl, hw, j), 5) >> 4) & 1u;
        const uint b1 = (tf_extract_tile_code(tiled, 10, hcol, tf_reg_coord_k(hl, hw, j + 4), 5) >> 4) & 1u;
        hi_w |= (b0 << (4 * s + j)) | (b1 << (16 + 4 * s + j));
      }
    }
    b[256 + t] = hi_w;
  }
}

// Split-plane (4+2) rewrite for 6-bit g64: words 0..255 low nibbles in regBlock order, 256..383 high 2-bit pairs.
kernel void tf_lane_order_q6(device uint* W [[buffer(0)]], uint t [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
  threadgroup uint tiled[384];
  device uint* b = W + (size_t)block * 384;
  tiled[t] = b[t];
  if (t < 128) tiled[256 + t] = b[256 + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint lane = (t >> 2) & 31, w = 4 * (t >> 7) + (t & 3), col = tf_reg_coord_col(lane, w);
  uint lo4 = 0u;
  for (uint j = 0; j < 8; j++) lo4 |= (tf_extract_tile_code(tiled, 12, col, tf_reg_coord_k(lane, w, j), 6) & 0xFu) << (4 * j);
  b[t] = lo4;
  if (t < 128) {
    const uint hl = t >> 2, p = t & 3, hcol = tf_reg_coord_col(hl, 2 * p);
    uint hi_w = 0u;
    for (uint s = 0; s < 2; s++) {
      const uint hw = 2 * p + s;
      for (uint j = 0; j < 4; j++) {
        const uint c0 = (tf_extract_tile_code(tiled, 12, hcol, tf_reg_coord_k(hl, hw, j), 6) >> 4) & 3u;
        const uint c1 = (tf_extract_tile_code(tiled, 12, hcol, tf_reg_coord_k(hl, hw, j + 4), 6) >> 4) & 3u;
        hi_w |= (c0 << (8 * s + 2 * j)) | (c1 << (16 + 8 * s + 2 * j));
      }
    }
    b[256 + t] = hi_w;
  }
}
