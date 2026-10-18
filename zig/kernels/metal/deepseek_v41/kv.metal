// DeepSeek V4.1 cache quantization and pooling, following the official inference kernels.
#include <metal_stdlib>
using namespace metal;
inline float dsv41_bf(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort dsv41_round(float x) {
    uint u = as_type<uint>(x);
    return ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
inline float dsv41_e4(uchar c) {
    uint e = (c >> 3) & 15, m = c & 7;
    float v = e == 0 ? float(m) * 0x1p-9f : ldexp(1.0f + float(m) / 8.0f, int(e) - 7);
    return (c & 128) ? -v : v;
}
inline uchar dsv41_e4code(float x) {
    float a = min(abs(x), 448.0f);
    if (a == 0) return signbit(x) ? 128 : 0;
    int e = max(-6, int((as_type<uint>(a) >> 23) & 255) - 127);
    int q = int(rint(ldexp(a, 3 - e)));
    uint c = e == -6 && q < 8 ? uint(q) : uint((e + 7) * 8 + q - 8);
    return uchar(min(c, 126u) | (signbit(x) ? 128u : 0u));
}
inline float dsv41_e2(uchar c) {
    const float v[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}
inline uchar dsv41_e2code(float x) {
    float a = min(abs(x), 6.0f);
    uint pick = a <= .25f ? 0 : a < .75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3
        : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
    return uchar(pick | (signbit(x) ? 8 : 0));
}
inline float dsv41_unpack(device const uchar *p, uint d, uint width, uint block, uint kind) {
    if (kind == 0) {
        float s = as_type<float>(uint(p[width + d / 32]) << 23);
        return dsv41_bf(dsv41_round(dsv41_e4(p[d]) * s));
    }
    uchar c = p[d / 2];
    float s = kind == 1 ? dsv41_e4(p[width / 2 + d / block])
                       : as_type<float>(uint(p[width / 2 + d / block]) << 23);
    return dsv41_bf(dsv41_round(dsv41_e2((d & 1) ? c >> 4 : c & 15) * s));
}
struct Dsv41Pack { uint rows, width, stride, packed_stride, block, kind, first, ring, inplace; };
kernel void dsv41_pack(device ushort *x [[buffer(0)]], device uchar *out [[buffer(1)]],
                       constant Dsv41Pack &p [[buffer(2)]], uint2 g [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]]) {
    uint row = g.y, group = g.x, d = group * p.block + lane;
    if (row >= p.rows) return;
    float v = lane < p.block ? dsv41_bf(x[row * p.stride + d]) : 0;
    float a = simd_max(abs(v));
    float scale; uchar sc;
    if (p.kind == 1) {
        sc = dsv41_e4code(max(a, 6.0f * 0x1p-9f) / 6.0f); scale = dsv41_e4(sc);
    } else {
        float minimum = p.kind == 0 ? 1e-4f : 6.0f * 0x1p-126f;
        float z = max(a, minimum) / (p.kind == 0 ? 448.0f : 6.0f);
        uint bits = as_type<uint>(z);
        int exp = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
        sc = uchar(exp + 127); scale = ldexp(1.0f, exp);
    }
    uint dst = p.ring ? (p.first + row) % p.ring : p.first + row;
    device uchar *packed = out + dst * p.packed_stride;
    if (lane == 0) packed[(p.kind == 0 ? p.width : p.width / 2) + group] = sc;
    if (lane >= p.block) return;
    if (p.kind == 0) {
        uchar c = dsv41_e4code(v / scale); packed[d] = c;
        if (p.inplace) x[row * p.stride + d] = dsv41_round(dsv41_e4(c) * scale);
    } else {
        uchar c = dsv41_e2code(v / scale);
        uchar other = uchar(simd_shuffle_xor(uint(c), 1));
        if (!(lane & 1)) packed[d / 2] = c | (other << 4);
        if (p.inplace) x[row * p.stride + d] = dsv41_round(dsv41_e2(c) * scale);
    }
}
struct Dsv41Pool { uint rows, width, ratio, position, stride, input_f32; float eps; };
kernel void dsv41_pool(device const uchar *values [[buffer(0)]], device const float *gate [[buffer(1)]],
                       device float *partial [[buffer(2)]], device const ushort *weight [[buffer(3)]],
                       device ushort *latent [[buffer(4)]], constant Dsv41Pool &p [[buffer(5)]],
                       uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    // A group reads the prior chunk's tail before the independent tail-update dispatch.
    uint start = (p.position / p.ratio + group) * p.ratio;
    threadgroup float pooled[512];
    float square = 0;
    for (uint d = lane; d < p.width; d += 128) {
        float mx = -INFINITY, sum = 0, total = 0;
        for (uint j = 0; j < p.ratio; ++j) {
            uint pos = start + j;
            float score = p.ratio == 1 ? 0 : (pos < p.position ? partial[p.ratio * p.width + (pos % p.ratio) * p.width + d]
                : gate[(pos - p.position) * p.stride + d]);
            mx = max(mx, score);
        }
        float slots[2], exponent[2];
        for (uint j = 0; j < p.ratio; ++j) {
            uint pos = start + j;
            slots[j] = pos < p.position ? partial[(pos % p.ratio) * p.width + d]
                : (p.input_f32 ? reinterpret_cast<device const float *>(values)[(pos - p.position) * p.stride + d]
                              : dsv41_bf(reinterpret_cast<device const ushort *>(values)[(pos - p.position) * p.stride + d]));
            float score = p.ratio == 1 ? 0 : (pos < p.position ? partial[p.ratio * p.width + (pos % p.ratio) * p.width + d]
                : gate[(pos - p.position) * p.stride + d]);
            exponent[j] = exp(score - mx); sum += exponent[j];
        }
        for (uint j = 0; j < p.ratio; ++j) total += slots[j] * (exponent[j] / sum);
        float v = dsv41_bf(dsv41_round(total)); pooled[d] = v; square += v * v;
    }
    threadgroup float sums[4];
    float ss = simd_sum(square);
    if (!(lane % 32)) sums[lane / 32] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((sums[0] + sums[1] + sums[2] + sums[3]) / p.width + p.eps);
    for (uint d = lane; d < p.width; d += 128)
        latent[group * p.width + d] = dsv41_round(pooled[d] * inv * dsv41_bf(weight[d]));
}
kernel void dsv41_pool_tail(device const float *values [[buffer(0)]], device const float *gate [[buffer(1)]],
                            device float *partial [[buffer(2)]], constant Dsv41Pool &p [[buffer(3)]],
                            uint d [[thread_position_in_grid]]) {
    if (d >= p.width || p.ratio == 1) return;
    uint end = p.position + p.rows, tail = end % p.ratio;
    for (uint j = 0; j < tail; ++j) {
        uint pos = end - tail + j;
        if (pos < p.position) continue;
        partial[j * p.width + d] = values[(pos - p.position) * p.stride + d];
        partial[p.ratio * p.width + j * p.width + d] = gate[(pos - p.position) * p.stride + d];
    }
}
kernel void dsv41_copy(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                       constant uint &bytes [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < bytes) dst[i] = src[i];
}
kernel void dsv41_publish_window(device const uchar *chunk [[buffer(0)]], device uchar *ring [[buffer(1)]],
    constant uint4 &p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
    uint count = min(p.x, 128u), first = p.x - count;
    if (i.x >= p.z || i.y >= count) return;
    uint row = first + i.y, slot = (p.y + row) % 128;
    ring[slot * p.w + i.x] = chunk[row * p.z + i.x];
}
kernel void dsv41_unpack_rows(device const uchar *packed [[buffer(0)]], device ushort *out [[buffer(1)]],
    constant Dsv41Pack &p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
    if (i.x >= p.width || i.y >= p.rows) return;
    uint row = p.ring ? (p.first + i.y) % p.ring : p.first + i.y;
    out[i.y * p.stride + i.x] = dsv41_round(dsv41_unpack(packed + row * p.packed_stride,
        i.x, p.width, p.block, p.kind));
}
