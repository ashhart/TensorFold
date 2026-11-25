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

// A prompt chunk's (row, pick) pairs grouped by expert for the gathers
kernel void tf_hy4_group_pairs(
    const device int* pick [[buffer(0)]], device int* offsets [[buffer(1)]], device int* order [[buffer(2)]],
    device int* ids [[buffer(3)]], constant uint3& a [[buffer(4)]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint counts[8][512]; // each block's count of an expert, then where its pairs start
    threadgroup uint bits[64][16];
    threadgroup uint carry[32];
    const uint pairs = a.x, experts = a.y, topk = a.z, rows = pairs / topk, blocks = (rows + 63) / 64;
    for (uint i = t; i < 8 * 512; i += 1024) atomic_store_explicit(&counts[i / 512][i % 512], 0u, memory_order_relaxed);
    for (uint e = t; e < experts; e += 1024) ids[e] = int(e);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = t; p < pairs; p += 1024) atomic_fetch_add_explicit(&counts[(p / topk) / 64][pick[p]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = 0;
    if (t < experts) for (uint b = 0; b < blocks; ++b) total += atomic_load_explicit(&counts[b][t], memory_order_relaxed);
    const uint local = simd_prefix_exclusive_sum(total);
    if (lane == 31) carry[sg] = local + total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < experts) {
        uint base = local;
        for (uint i = 0; i < sg; ++i) base += carry[i];
        offsets[t] = int(base);
        for (uint b = 0; b < blocks; ++b) {
            const uint n = atomic_load_explicit(&counts[b][t], memory_order_relaxed);
            atomic_store_explicit(&counts[b][t], base, memory_order_relaxed);
            base += n;
        }
    }
    for (uint b = 0; b < blocks; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = t; i < 64 * 16; i += 1024) bits[i / 16][i % 16] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint first = b * 64, last = min(rows, first + 64);
        for (uint p = first * topk + t; p < last * topk; p += 1024) {
            const uint e = uint(pick[p]);
            atomic_fetch_or_explicit((threadgroup atomic_uint *)&bits[p / topk - first][e / 32], 1u << (e % 32), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = first * topk + t; p < last * topk; p += 1024) {
            const uint e = uint(pick[p]), r = p / topk - first;
            uint rank = 0;
            for (uint q = 0; q < r; ++q) rank += (bits[q][e / 32] >> (e % 32)) & 1u;
            order[atomic_load_explicit(&counts[b][e], memory_order_relaxed) + rank] = int(p);
        }
    }
}
