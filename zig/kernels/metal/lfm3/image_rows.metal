// Image rows replace placeholder embeddings without changing any BF16 bits.
#include <metal_stdlib>
using namespace metal;
[[kernel, max_total_threads_per_threadgroup(256)]] void lfm3_image_rows(
    const device ushort* src [[buffer(0)]], device ushort* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count) dst[i] = src[i];
}
