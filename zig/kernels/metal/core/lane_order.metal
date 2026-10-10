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
