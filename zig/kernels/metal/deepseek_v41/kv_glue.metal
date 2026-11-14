// DeepSeek V4.1 served engine: the window KV's RMS norm + RoPE + fp8 cache pack as ONE dispatch.
#include <metal_stdlib>
using namespace metal;

struct Dsv41KvGlue {
    uint rows, width, stride, packed_stride, block, position, rotary, write_y;
    float eps;
};
static_assert(sizeof(Dsv41KvGlue) == 36, "Dsv41KvGlue must mirror kv.zig's Glue");

inline float dsv41kg_bf(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort dsv41kg_round(float x) {
    uint u = as_type<uint>(x);
    return ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
inline uchar dsv41kg_e4code(float x) {
    float a = min(abs(x), 448.0f);
    if (a == 0) return signbit(x) ? 128 : 0;
    int e = max(-6, int((as_type<uint>(a) >> 23) & 255) - 127);
    int q = int(rint(ldexp(a, 3 - e)));
    uint c = e == -6 && q < 8 ? uint(q) : uint((e + 7) * 8 + q - 8);
    return uchar(min(c, 126u) | (signbit(x) ? 128u : 0u));
}

// One row's norm + rope + pack; `position` is the row's absolute position (its rope factor row).
inline void dsv41_kv_glue_body(const device ushort* X, const device ushort* W, const device float2* F, device uchar* PACKED,
                               device ushort* Y, constant Dsv41KvGlue& p, threadgroup float* part, uint2 g, uint t,
                               uint lane, uint sg, uint position) {
    const device ushort* x = X + ulong(g.y) * p.stride;
    // dsv41_rms: s = fma(v1, v1, fma(v0, v0, 0)) over i = t, t+256, then the xor butterfly and part[0..7].
    const float v0 = dsv41kg_bf(x[t]), v1 = dsv41kg_bf(x[t + 256u]);
    float s = fma(v0, v0, 0.0f);
    s = fma(v1, v1, s);
    for (uint m = 16; m > 0; m >>= 1) s += simd_shuffle_xor(s, m);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = part[0];
    for (uint j = 1; j < 8; ++j) total += part[j];
    const float r = precise::rsqrt(precise::divide(total, float(p.width)) + p.eps);
    // Only the first simdgroup packs this threadgroup's block; the rest have done their part of the sum.
    if (sg != 0u) return;
    const uint d = g.x * p.block + lane;
    // dsv41_rms's output expression, rounded to bf16 exactly once.
    ushort yb = dsv41kg_round(dsv41kg_bf(W[d]) * (dsv41kg_bf(x[d]) * r));
    // rope.metal, applied to the pair's two pre-rotation bf16 values.
    if (d >= p.width - p.rotary) {
        const float a0 = dsv41kg_bf(yb);
        const float other = dsv41kg_bf(ushort(simd_shuffle_xor(uint(yb), 1)));
        const uint pr = (d - (p.width - p.rotary)) / 2u;
        const float2 cs = F[ulong(position) * (p.rotary / 2u) + pr];
        uint u;
        if ((d & 1u) == 0u) u = as_type<uint>(a0 * cs.x - other * cs.y);
        else u = as_type<uint>(other * cs.y + a0 * cs.x);
        yb = ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    if (p.write_y != 0u) Y[ulong(g.y) * p.width + d] = yb;
    // dsv41_pack (window_fp8): one 32-lane simdgroup a block, the same max, scale exponent and code.
    const float v = dsv41kg_bf(yb);
    const float a = simd_max(abs(v));
    const float z = max(a, 1e-4f) / 448.0f;
    const uint bits = as_type<uint>(z);
    const int exp = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
    const uchar sc = uchar(exp + 127);
    const float scale = ldexp(1.0f, exp);
    const uchar c = dsv41kg_e4code(v / scale);
    device uchar* out = PACKED + ulong(g.y) * p.packed_stride;
    out[d] = c;
    if (lane == 0) out[p.width + g.x] = sc;
}

kernel void dsv41_kv_glue(
    const device ushort* X [[buffer(0)]],      // raw [rows, stride] (kv projection output)
    const device ushort* W [[buffer(1)]],      // kv_norm weight [width] (bf16)
    const device float2* F [[buffer(2)]],      // rope factors, selected spectrum, position-indexed
    device uchar* PACKED [[buffer(3)]],        // [rows, packed_stride] window_fp8
    device ushort* Y [[buffer(4)]],            // optional norm+rope writeback (write_y); never X
    constant Dsv41KvGlue& p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]],  // g.x block, g.y row
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    if (g.y >= p.rows) return;
    threadgroup float part[8];
    dsv41_kv_glue_body(X, W, F, PACKED, Y, p, part, g, t, lane, sg, p.position + g.y);
}

// A batched decode window: row g.y is one slot's one row, at that slot's position `pos[g.y]`; p.position is unused.
kernel void dsv41_kv_glue_rows(
    const device ushort* X [[buffer(0)]], const device ushort* W [[buffer(1)]], const device float2* F [[buffer(2)]],
    device uchar* PACKED [[buffer(3)]], device ushort* Y [[buffer(4)]], constant Dsv41KvGlue& p [[buffer(5)]],
    const device uint* pos [[buffer(6)]], uint2 g [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    if (g.y >= p.rows) return;
    threadgroup float part[8];
    dsv41_kv_glue_body(X, W, F, PACKED, Y, p, part, g, t, lane, sg, pos[g.y]);
}
