// Hy4-flash's absorbed-MLA BF16 products on the tensor units (nax.h's 16x32x16 op): every row count runs the same
// tiles, so prompt chunks and decode rows get the same bits. One head a threadgroup column, 16 rows by 128 outputs a
// threadgroup (4 simdgroups of 16 x 32).
#include "../nax.h"
using namespace tfp;

struct Hy4MlaProjectArgs { uint rows; uint heads; };

inline float hy4n_value(bfloat w) { return float(w); }
inline bfloat hy4n_round(float value) {
    const uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return as_type<bfloat>(ushort((bits >> 16) | 0x0040u));
    return as_type<bfloat>(ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16));
}

// absorbed[r][h][0, 512) = q[r][h][0, 192) W_UK[h], W_UK[h] row-major [192][512]; rows 576 apart a head.
[[kernel]] void tf_hy4_absorb_nax(const device bfloat* query [[buffer(0)]], const device bfloat* wk [[buffer(1)]],
                                  device bfloat* absorbed [[buffer(2)]], constant Hy4MlaProjectArgs& args [[buffer(3)]],
                                  uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const short2 home = frag_home(ushort(lane));
    const int heads = int(args.heads), h = int(group.y), r0 = int(group.z) * 16, n0 = int(group.x) * 128 + int(sg) * 32;
    const int live = int(args.rows) - r0;
    const device bfloat* a_rows = query + long(r0) * heads * 256 + h * 256;
    const device bfloat* b = wk + long(h) * 192 * 512;
    frag<float> lo = 0.0f, hi = 0.0f;
    for (int k = 0; k < 192; k += 16) {
        frag<bfloat> a, b0, b1;
        frag_get_in(a, a_rows, heads * 256, 0, k, home, live, 192);
        frag_get(b0, b, 512, k, n0, home);
        frag_get(b1, b, 512, k, n0 + 16, home);
        mma_16x32<false, false>(lo, hi, a, b0, b1);
    }
    device bfloat* c = absorbed + long(r0) * heads * 576 + h * 576;
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
        const int rr = home.y + (e >> 2) * 8, cc = home.x + TF_COL(e);
        if (rr < live) {
            c[long(rr) * heads * 576 + n0 + cc] = hy4n_round(lo[e]);
            c[long(rr) * heads * 576 + n0 + 16 + cc] = hy4n_round(hi[e]);
        }
    }
}

// values[r][h][0, 256) = gate(latent[r][h][0, 512) W_UV[h]^T), W_UV[h] stored [256][512]; the gate as
// tf_hy4_gate_values: the value rounded, times its rounded sigmoid, rounded.
[[kernel]] void tf_hy4_expand_gated_nax(const device bfloat* latent [[buffer(0)]], const device bfloat* wv [[buffer(1)]],
                                        device bfloat* values [[buffer(2)]], constant Hy4MlaProjectArgs& args [[buffer(3)]],
                                        const device bfloat* logits [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const short2 home = frag_home(ushort(lane));
    const int heads = int(args.heads), h = int(group.y), r0 = int(group.z) * 16, n0 = int(group.x) * 128 + int(sg) * 32;
    const int live = int(args.rows) - r0;
    const device bfloat* a_rows = latent + long(r0) * heads * 512 + h * 512;
    const device bfloat* bt = wv + long(h) * 256 * 512;
    frag<float> lo = 0.0f, hi = 0.0f;
    for (int k = 0; k < 512; k += 16) {
        frag<bfloat> a, b0, b1;
        frag_get_in(a, a_rows, heads * 512, 0, k, home, live, 512);
        frag_get_t(b0, bt, 512, n0, k, home);
        frag_get_t(b1, bt, 512, n0 + 16, k, home);
        mma_16x32<false, true>(lo, hi, a, b0, b1);
    }
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
        const int rr = home.y + (e >> 2) * 8, cc = home.x + TF_COL(e);
        if (rr < live) {
            for (int side = 0; side < 2; ++side) {
                const long at = long(r0 + rr) * heads * 256 + h * 256 + n0 + 16 * side + cc;
                const bfloat v = hy4n_round(side ? hi[e] : lo[e]);
                const float gate = 1.0f / (1.0f + precise::exp(-float(logits[at])));
                values[at] = hy4n_round(float(v) * float(hy4n_round(gate)));
            }
        }
    }
}

// Absorbed MLA attention on the tensor units, flash style: one head and 16 query rows a threadgroup, 32-key tiles at
// absolute key indices (so a row's tiles and their order are its own, whatever rows share its block; tiles outside a
// row's range are exact no-ops for it). S = Q K^T with the 576 key columns split over the 4 simdgroups and their
// partials added in a fixed order; an online softmax a row; O += P V with each simdgroup owning 128 latent columns;
// the learned sink joins at the end. ranges[r] = [lo, hi) key indices; tiles[t] = the slot of key index 32 t (a
// tile's 32 slots are consecutive).
constant constexpr int hy4_tk = 32, hy4_kw = 576, hy4_lw = 512;
struct Hy4FlashShape { uint rows, heads; float scale; uint pad; };

// SPLIT (decode passes, one row block): threadgroup z of `a.pad` takes the z-th run of the tiles and stores its
// unnormalized O, running max and sum (fp32) for tf_hy4_attend_combine instead of the sink and the division.
template <bool SPLIT>
inline void hy4_attend_nax(const device bfloat* query, const device bfloat* cache, const device uint2* ranges,
                           const device uint* tiles, const device float* sinks, device bfloat* out,
                           constant Hy4FlashShape& a, uint3 group, uint sg, uint lane, device float* o_part,
                           device float* ml_part, threadgroup float (&part)[4][16][hy4_tk]) {
    const short2 home = frag_home(ushort(lane));
    const int heads = int(a.heads), h = int(group.x), r0 = int(group.y) * 16;
    const int live = min(16, int(a.rows) - r0);
    uint lo[2] = {0u, 0u}, hi[2] = {0u, 0u};
    for (int i = 0; i < 2; ++i) {
        const int rr = home.y + 8 * i;
        if (rr < live) { const uint2 g = ranges[r0 + rr]; lo[i] = g.x; hi[i] = g.y; }
    }
    uint blo = 0xffffffffu, bhi = 0u;
    for (int r = 0; r < live; ++r) { const uint2 g = ranges[r0 + r]; blo = min(blo, g.x); bhi = max(bhi, g.y); }
    const device bfloat* q = query + (long(r0) * heads + h) * hy4_kw;
    const int ldq = heads * hy4_kw;
    frag<float> o[4][2];
    TF_UNROLL
    for (int j = 0; j < 4; ++j) { o[j][0] = 0.0f; o[j][1] = 0.0f; }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    int t0 = int(blo / hy4_tk), t1 = int((bhi + hy4_tk - 1) / hy4_tk);
    if (SPLIT) {
        const int per = (t1 - t0 + int(a.pad) - 1) / int(a.pad);
        t0 = min(t1, t0 + int(group.z) * per);
        t1 = min(t1, t0 + per);
    }
    for (int t = t0; t < t1; ++t) {
        const device bfloat* k = cache + long(tiles[t]) * hy4_kw;
        frag<float> slo = 0.0f, shi = 0.0f;
        for (int d = int(sg) * 144; d < int(sg) * 144 + 144; d += 16) {
            frag<bfloat> qa, k0, k1;
            frag_get_in(qa, q, ldq, 0, d, home, live, hy4_kw);
            frag_get_t(k0, k, hy4_kw, 0, d, home);
            frag_get_t(k1, k, hy4_kw, 16, d, home);
            mma_16x32<false, true>(slo, shi, qa, k0, k1);
        }
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
            const int rr = home.y + (e >> 2) * 8, cc = home.x + TF_COL(e);
            part[sg][rr][cc] = slo[e];
            part[sg][rr][16 + cc] = shi[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float p[2][8];
        float tmax[2] = {-INFINITY, -INFINITY};
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
            const int i = e >> 2, rr = home.y + i * 8, cc = home.x + TF_COL(e);
            for (int side = 0; side < 2; ++side) {
                const int col = side * 16 + cc;
                float s = part[0][rr][col];
                s += part[1][rr][col];
                s += part[2][rr][col];
                s += part[3][rr][col];
                const uint key = uint(t * hy4_tk + col);
                s = (key >= lo[i] && key < hi[i]) ? s * a.scale : -INFINITY;
                p[side][e] = s;
                tmax[i] = max(tmax[i], s);
            }
        }
        float alpha[2];
        for (int i = 0; i < 2; ++i) {
            tmax[i] = max(tmax[i], simd_shuffle_xor(tmax[i], ushort(1)));
            tmax[i] = max(tmax[i], simd_shuffle_xor(tmax[i], ushort(8)));
            const float next = max(m[i], tmax[i]);
            alpha[i] = next == -INFINITY ? 1.0f : precise::exp(m[i] - next);
            m[i] = next;
        }
        float sum[2] = {0.0f, 0.0f};
        frag<bfloat> plo, phi;
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
            const int i = e >> 2;
            for (int side = 0; side < 2; ++side) {
                const float v = m[i] == -INFINITY ? 0.0f : precise::exp(p[side][e] - m[i]);
                p[side][e] = v;
                sum[i] += v;
            }
            plo[e] = bfloat(p[0][e]);
            phi[e] = bfloat(p[1][e]);
        }
        for (int i = 0; i < 2; ++i) {
            sum[i] += simd_shuffle_xor(sum[i], ushort(1));
            sum[i] += simd_shuffle_xor(sum[i], ushort(8));
            l[i] = l[i] * alpha[i] + sum[i];
        }
        TF_UNROLL
        for (int j = 0; j < 4; ++j) {
            TF_UNROLL
            for (short e = 0; e < 8; e++) {
                o[j][0][e] *= alpha[e >> 2];
                o[j][1][e] *= alpha[e >> 2];
            }
            const int n = int(sg) * 128 + j * 32;
            frag<bfloat> v0, v1;
            frag_get(v0, k, hy4_kw, 0, n, home);
            frag_get(v1, k, hy4_kw, 0, n + 16, home);
            mma_16x32<false, false>(o[j][0], o[j][1], plo, v0, v1);
            frag_get(v0, k, hy4_kw, 16, n, home);
            frag_get(v1, k, hy4_kw, 16, n + 16, home);
            mma_16x32<false, false>(o[j][0], o[j][1], phi, v0, v1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (SPLIT) {
        const long z = long(group.z), rows = long(a.rows);
        for (int i = 0; i < 2; ++i) {
            const int rr = home.y + 8 * i;
            if (rr < live && sg == 0) {
                device float* ml = ml_part + ((z * rows + r0 + rr) * heads + h) * 2;
                ml[0] = m[i];
                ml[1] = l[i];
            }
        }
        TF_UNROLL
        for (int j = 0; j < 4; ++j) {
            TF_UNROLL
            for (short e = 0; e < 8; e++) {
                const int i = e >> 2, rr = home.y + i * 8, cc = home.x + TF_COL(e);
                if (rr < live) {
                    device float* row = o_part + ((z * rows + r0 + rr) * heads + h) * hy4_lw + int(sg) * 128 + j * 32 + cc;
                    row[0] = o[j][0][e];
                    row[16] = o[j][1][e];
                }
            }
        }
        return;
    }
    const float sink = sinks[h];
    float keep[2], denom[2];
    for (int i = 0; i < 2; ++i) {
        const float top = max(m[i], sink);
        keep[i] = m[i] == -INFINITY ? 0.0f : precise::exp(m[i] - top);
        denom[i] = l[i] * keep[i] + precise::exp(sink - top);
    }
    TF_UNROLL
    for (int j = 0; j < 4; ++j) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
            const int i = e >> 2, rr = home.y + i * 8, cc = home.x + TF_COL(e);
            if (rr < live) {
                device bfloat* row = out + (long(r0 + rr) * heads + h) * hy4_lw + int(sg) * 128 + j * 32 + cc;
                row[0] = bfloat(precise::divide(o[j][0][e] * keep[i], denom[i]));
                row[16] = bfloat(precise::divide(o[j][1][e] * keep[i], denom[i]));
            }
        }
    }
}

[[kernel]] void tf_hy4_attend_nax(const device bfloat* query [[buffer(0)]], const device bfloat* cache [[buffer(1)]],
                                  const device uint2* ranges [[buffer(2)]], const device uint* tiles [[buffer(3)]],
                                  const device float* sinks [[buffer(4)]], device bfloat* out [[buffer(5)]],
                                  constant Hy4FlashShape& a [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4][16][hy4_tk];
    hy4_attend_nax<false>(query, cache, ranges, tiles, sinks, out, a, group, sg, lane, nullptr, nullptr, part);
}

[[kernel]] void tf_hy4_attend_nax_split(const device bfloat* query [[buffer(0)]], const device bfloat* cache [[buffer(1)]],
                                        const device uint2* ranges [[buffer(2)]], const device uint* tiles [[buffer(3)]],
                                        const device float* sinks [[buffer(4)]], device bfloat* out [[buffer(5)]],
                                        constant Hy4FlashShape& a [[buffer(6)]], device float* o_part [[buffer(7)]],
                                        device float* ml_part [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4][16][hy4_tk];
    hy4_attend_nax<true>(query, cache, ranges, tiles, sinks, out, a, group, sg, lane, o_part, ml_part, part);
}

// The split runs of tf_hy4_attend_nax_split joined with the learned sink: out = sum_z O_z e^(m_z - M) / L,
// L = sum_z l_z e^(m_z - M) + e^(sink - M), M the largest of the maxima and the sink. Grid (heads, rows), 128 threads.
[[kernel]] void tf_hy4_attend_combine(const device float* o_part [[buffer(0)]], const device float* ml_part [[buffer(1)]],
                                      const device float* sinks [[buffer(2)]], device bfloat* out [[buffer(3)]],
                                      constant Hy4FlashShape& a [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                                      uint t [[thread_index_in_threadgroup]]) {
    const int h = int(group.x), r = int(group.y), heads = int(a.heads), splits = min(int(a.pad), 8);
    const long rows = long(a.rows);
    const float sink = sinks[h];
    float top = sink;
    for (int z = 0; z < splits; ++z) top = max(top, ml_part[((z * rows + r) * heads + h) * 2]);
    float w[8];
    float denom = precise::exp(sink - top);
    for (int z = 0; z < 8; ++z) {
        w[z] = 0.0f;
        if (z >= splits) continue;
        const device float* ml = ml_part + ((z * rows + r) * heads + h) * 2;
        w[z] = ml[0] == -INFINITY ? 0.0f : precise::exp(ml[0] - top);
        denom += ml[1] * w[z];
    }
    for (int c = int(t) * 4; c < int(t) * 4 + 4; ++c) {
        float v = 0.0f;
        for (int z = 0; z < splits; ++z) v += o_part[((z * rows + r) * heads + h) * hy4_lw + c] * w[z];
        out[(long(r) * heads + h) * hy4_lw + c] = bfloat(precise::divide(v, denom));
    }
}
