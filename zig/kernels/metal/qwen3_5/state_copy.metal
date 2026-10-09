// Qwen3.5 DeltaNet recurrence from a window snapshot row back into a stream's cache, 16 bytes a thread.
#include <metal_stdlib>
using namespace metal;
[[kernel]] void qwen35_state_copy(const device float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
  dst[i] = src[i];
}
