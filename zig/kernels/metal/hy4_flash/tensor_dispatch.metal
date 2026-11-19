// Bit-preserving row and route layouts for the TensorOps dispatches.
#include <metal_stdlib>
using namespace metal;

kernel void tf_hy4_qkv_split(
    const device ushort* input [[buffer(0)]], device ushort* q [[buffer(1)]],
    device ushort* kv [[buffer(2)]], device ushort* gate [[buffer(3)]],
    constant uint3& dims [[buffer(4)]], uint2 tid [[thread_position_in_grid]]) {
    const uint col = tid.x, row = tid.y;
    const uint qn = dims.x, kn = dims.y, gn = dims.z;
    const ushort v = input[ulong(row) * (qn + kn + gn) + col];
    if (col < qn) q[ulong(row) * qn + col] = v;
    else if (col < qn + kn) kv[ulong(row) * kn + col - qn] = v;
    else gate[ulong(row) * gn + col - qn - kn] = v;
}

// The router has already grouped picks by active expert. Write only their compact offsets and pair IDs.
kernel void tf_hy4_active_experts(
    const device int* members [[buffer(0)]], const device int* active [[buffer(1)]],
    device int* offsets [[buffer(2)]], device int* order [[buffer(3)]],
    constant uint2& shape [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint counts[8];
    uint n = 0;
    if (int(t) < active[0]) {
        for (uint i = 0; i < shape.y && members[t * shape.y + i] >= 0; ++i) ++n;
    }
    const uint local = simd_prefix_exclusive_sum(n);
    if (lane == 31) counts[sg] = local + n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= shape.x) return;
    uint base = local;
    for (uint i = 0; i < sg; ++i) base += counts[i];
    offsets[t] = int(base);
    for (uint i = 0; i < n; ++i) order[base + i] = members[t * shape.y + i];
}
