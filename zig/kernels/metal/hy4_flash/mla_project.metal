// Hy4-flash projections around absorbed MLA. The rotary lanes are filled separately.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Hy4MlaProjectArgs { uint rows; uint heads; };
struct Hy4RopeParameters { float frequencies[32]; uint axes[32]; };
struct Hy4RopeView { uint rows, heads, input_stride, output_stride; };
struct Hy4CacheWrite { uint rows, capacity; float epsilon; };

inline float hy4_mla_value(ushort word) { return as_type<float>(uint(word) << 16); }

inline ushort hy4_mla_round(float value) {
    const uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x0040u);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Match the checkpoint's BF16 trigonometric coefficients before the FP32 pair rotation.
kernel void tf_hy4_rotary_table(
    const device int *positions [[buffer(0)]],
    device ushort *table [[buffer(1)]],
    constant Hy4RopeParameters &p [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint pair = tid.x, row = tid.y;
    const float phase = float(positions[ulong(row) * 3 + p.axes[pair]]) * p.frequencies[pair];
    table[ulong(row) * 64 + 2 * pair] = hy4_mla_round(precise::cos(phase));
    table[ulong(row) * 64 + 2 * pair + 1] = hy4_mla_round(precise::sin(phase));
}

kernel void tf_hy4_rotate(
    const device ushort *input [[buffer(0)]],
    const device ushort *table [[buffer(1)]],
    device ushort *output [[buffer(2)]],
    constant Hy4RopeView &v [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]) {
    const ulong row_head = ulong(tid.z) * v.heads + tid.y;
    const uint col = 2 * tid.x;
    const ulong i = row_head * v.input_stride + col, o = row_head * v.output_stride + col;
    const ulong base = ulong(tid.z) * 64 + col;
    const float a = hy4_mla_value(input[i]), b = hy4_mla_value(input[i + 1]);
    const float c = hy4_mla_value(table[base]), s = hy4_mla_value(table[base + 1]);
    output[o] = hy4_mla_round(a * c - b * s);
    output[o + 1] = hy4_mla_round(b * c + a * s);
}

kernel void tf_hy4_cache_keys(
    const device ushort *input [[buffer(0)]],
    const device ushort *gain [[buffer(1)]],
    const device ushort *table [[buffer(2)]],
    const device int *slots [[buffer(3)]],
    device ushort *cache [[buffer(4)]],
    constant Hy4CacheWrite &a [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const int slot = slots[row];
    if (slot < 0 || uint(slot) >= a.capacity) return;
    const device ushort *x = input + ulong(row) * 576;
    device ushort *y = cache + ulong(slot) * 576;
    float v[4], sq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        v[i] = hy4_mla_value(x[t + i * 128]);
        sq = fma(v[i], v[i], sq);
    }
    for (ushort d = 16; d > 0; d >>= 1) sq += simd_shuffle_xor(sq, d);
    threadgroup float part[4];
    if (lane == 0) part[simd] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum = (part[0] + part[2]) + (part[1] + part[3]);
    const float inverse = precise::rsqrt(sum / 512.0f + a.epsilon);
    for (uint i = 0; i < 4; ++i) {
        const uint col = t + i * 128;
        const float normalized = hy4_mla_value(hy4_mla_round(v[i] * inverse));
        y[col] = hy4_mla_round(normalized * hy4_mla_value(gain[col]));
    }
    if (t < 32) {
        const uint col = 2 * t;
        const float u = hy4_mla_value(x[512 + col]), v1 = hy4_mla_value(x[513 + col]);
        const float c = hy4_mla_value(table[ulong(row) * 64 + col]);
        const float s = hy4_mla_value(table[ulong(row) * 64 + col + 1]);
        y[512 + col] = hy4_mla_round(u * c - v1 * s);
        y[513 + col] = hy4_mla_round(v1 * c + u * s);
    }
}

kernel void tf_hy4_gate_values(
    device ushort *values [[buffer(0)]],
    const device ushort *logits [[buffer(1)]],
    constant uint &columns [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
    const ulong i = ulong(tid.y) * columns + tid.x;
    const float gate = 1.0f / (1.0f + precise::exp(-hy4_mla_value(logits[i])));
    values[i] = hy4_mla_round(hy4_mla_value(values[i]) * hy4_mla_value(hy4_mla_round(gate)));
}

kernel void tf_hy4_absorb_q(
    const device ushort *query [[buffer(0)]],
    const device ushort *key_projection [[buffer(1)]],
    device ushort *absorbed [[buffer(2)]],
    constant Hy4MlaProjectArgs &args [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint column = tid.x;
    const uint row_head = tid.y;
    if (column >= 512 || row_head >= ulong(args.rows) * args.heads) return;

    const ulong q = ulong(row_head) * 256;
    const ulong wk = ulong(row_head % args.heads) * 192 * 512 + column;
    float sum = 0.0f;
    for (uint i = 0; i < 192; ++i) {
        sum = fma(hy4_mla_value(query[q + i]), hy4_mla_value(key_projection[wk + i * 512]), sum);
    }
    absorbed[ulong(row_head) * 576 + column] = hy4_mla_round(sum);
}

kernel void tf_hy4_expand_v(
    const device ushort *latent [[buffer(0)]],
    const device ushort *value_projection [[buffer(1)]],
    device ushort *expanded [[buffer(2)]],
    constant Hy4MlaProjectArgs &args [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint column = tid.x;
    const uint row_head = tid.y;
    if (column >= 256 || row_head >= ulong(args.rows) * args.heads) return;

    const ulong from = ulong(row_head) * 512;
    const ulong wv = (ulong(row_head % args.heads) * 256 + column) * 512;
    float sum = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        sum = fma(hy4_mla_value(latent[from + i]), hy4_mla_value(value_projection[wv + i]), sum);
    }
    expanded[ulong(row_head) * 256 + column] = hy4_mla_round(sum);
}

// tf_hy4_expand_v then tf_hy4_gate_values in one launch: the expanded value rounded as before, then gated by the
// rounded sigmoid of its logit exactly as the gate pass does, so the same bits without the second pass.
kernel void tf_hy4_expand_v_gated(
    const device ushort *latent [[buffer(0)]],
    const device ushort *value_projection [[buffer(1)]],
    device ushort *expanded [[buffer(2)]],
    constant Hy4MlaProjectArgs &args [[buffer(3)]],
    const device ushort *logits [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint column = tid.x;
    const uint row_head = tid.y;
    if (column >= 256 || row_head >= ulong(args.rows) * args.heads) return;

    const ulong from = ulong(row_head) * 512;
    const ulong wv = (ulong(row_head % args.heads) * 256 + column) * 512;
    float sum = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        sum = fma(hy4_mla_value(latent[from + i]), hy4_mla_value(value_projection[wv + i]), sum);
    }
    const ulong at = ulong(row_head) * 256 + column;
    const ushort value = hy4_mla_round(sum);
    const float gate = 1.0f / (1.0f + precise::exp(-hy4_mla_value(logits[at])));
    expanded[at] = hy4_mla_round(hy4_mla_value(value) * hy4_mla_value(hy4_mla_round(gate)));
}
