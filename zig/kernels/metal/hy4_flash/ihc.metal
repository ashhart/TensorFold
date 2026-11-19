// Hy4-flash reads independent stream gates in FP32 and stores BF16 hidden rows.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Hy4IhcShape {
    uint rows, width, gates;
    float eps, gate_eps, post_magnitude;
};

inline float ihc_simd_sum(float score) {
    for (ushort s = 16; s > 0; s >>= 1) score += simd_shuffle_xor(score, s);
    return score;
}

kernel void tf_hy4_ihc_collapse(
    const device bfloat *hidden [[buffer(0)]],
    const device float *project [[buffer(1)]],
    const device float *bias [[buffer(2)]],
    const device float *scale [[buffer(3)]],
    device bfloat *collapsed [[buffer(4)]],
    device float *post [[buffer(5)]],
    constant Hy4IhcShape &a [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[9][8], pre[4];
    const uint wide = 4 * a.width;
    const device bfloat *x = hidden + ulong(row) * wide;
    float sq = 0.0f, logits[8] = {};
    for (uint i = t; i < wide; i += 256) {
        const float v = float(x[i]);
        sq = fma(v, v, sq);
        for (uint j = 0; j < a.gates; ++j) logits[j] = fma(v, project[j * wide + i], logits[j]);
    }
    sq = ihc_simd_sum(sq);
    if (lane == 0) partial[8][simd] = sq;
    for (uint j = 0; j < a.gates; ++j) {
        const float score = ihc_simd_sum(logits[j]);
        if (lane == 0) partial[j][simd] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < a.gates) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            sum += partial[8][i];
            dot += partial[t][i];
        }
        const float mix = dot * precise::rsqrt(sum / float(wide) + a.eps);
        const float gate = 1.0f / (1.0f + precise::exp(-(mix * scale[t / 4] + bias[t])));
        if (t < 4) pre[t] = gate + a.gate_eps;
        else post[ulong(row) * 4 + t - 4] = a.post_magnitude * gate + a.gate_eps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < a.width; i += 256) {
        float v = 0.0f;
        for (uint stream = 0; stream < 4; ++stream) v += pre[stream] * float(x[stream * a.width + i]);
        collapsed[ulong(row) * a.width + i] = bfloat(v);
    }
}

// tf_hy4_ihc_collapse with the gate count fixed at compile time: the same thread partition, fma chains, simd
// butterflies and partial sums in the same order, so the same bits, but the eight gate sums live in registers
// and the loads of an iteration issue together instead of through an indexed stack array.
template <uint G>
inline void ihc_collapse_fixed(
    const device bfloat *hidden, const device float *project, const device float *bias, const device float *scale,
    device bfloat *collapsed, device float *post, constant Hy4IhcShape &a,
    uint row, uint t, uint lane, uint simd, threadgroup float (&partial)[9][8], threadgroup float (&pre)[4]) {
    const uint wide = 4 * a.width;
    const device bfloat *x = hidden + ulong(row) * wide;
    float sq = 0.0f, logits[G];
#pragma unroll
    for (uint j = 0; j < G; ++j) logits[j] = 0.0f;
    for (uint i = t; i < wide; i += 256) {
        const float v = float(x[i]);
        sq = fma(v, v, sq);
#pragma unroll
        for (uint j = 0; j < G; ++j) logits[j] = fma(v, project[j * wide + i], logits[j]);
    }
    sq = ihc_simd_sum(sq);
    if (lane == 0) partial[8][simd] = sq;
#pragma unroll
    for (uint j = 0; j < G; ++j) {
        const float score = ihc_simd_sum(logits[j]);
        if (lane == 0) partial[j][simd] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < G) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            sum += partial[8][i];
            dot += partial[t][i];
        }
        const float mix = dot * precise::rsqrt(sum / float(wide) + a.eps);
        const float gate = 1.0f / (1.0f + precise::exp(-(mix * scale[t / 4] + bias[t])));
        if (t < 4) pre[t] = gate + a.gate_eps;
        else post[ulong(row) * 4 + t - 4] = a.post_magnitude * gate + a.gate_eps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < a.width; i += 256) {
        float v = 0.0f;
        for (uint stream = 0; stream < 4; ++stream) v += pre[stream] * float(x[stream * a.width + i]);
        collapsed[ulong(row) * a.width + i] = bfloat(v);
    }
}

#define HY4_IHC_FIXED(G)                                                                                         \
    kernel void tf_hy4_ihc_collapse_g##G(                                                                        \
        const device bfloat *hidden [[buffer(0)]], const device float *project [[buffer(1)]],                    \
        const device float *bias [[buffer(2)]], const device float *scale [[buffer(3)]],                         \
        device bfloat *collapsed [[buffer(4)]], device float *post [[buffer(5)]], constant Hy4IhcShape &a [[buffer(6)]], \
        uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],                       \
        uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {                  \
        threadgroup float partial[9][8], pre[4];                                                                 \
        ihc_collapse_fixed<G>(hidden, project, bias, scale, collapsed, post, a, row, t, lane, simd, partial, pre); \
    }
HY4_IHC_FIXED(4)
HY4_IHC_FIXED(8)

// The same collapse in two launches. tf_hy4_ihc_partials runs each of the 256-thread version's eight simdgroups as
// its own threadgroup (simdgroup s on core s instead of all eight on one core): thread t = 32 s + lane keeps its
// stride-256 fma chain and butterfly, and lane 0 stores the simdgroup's sums as part[row][gate][s] (gate 8: sum of
// squares). tf_hy4_ihc_finish then adds the eight partials in the same order and writes the collapsed row.
template <uint G>
inline void ihc_partials_fixed(const device bfloat *hidden, const device float *project, device float *part,
                               constant Hy4IhcShape &a, uint row, uint simd, uint lane) {
    const uint wide = 4 * a.width, t = 32 * simd + lane;
    const device bfloat *x = hidden + ulong(row) * wide;
    float sq = 0.0f, logits[G];
#pragma unroll
    for (uint j = 0; j < G; ++j) logits[j] = 0.0f;
#pragma unroll 4
    for (uint i = t; i < wide; i += 256) {
        const float v = float(x[i]);
        sq = fma(v, v, sq);
#pragma unroll
        for (uint j = 0; j < G; ++j) logits[j] = fma(v, project[j * wide + i], logits[j]);
    }
    device float *p = part + ulong(row) * 72;
    sq = ihc_simd_sum(sq);
    if (lane == 0) p[64 + simd] = sq;
#pragma unroll
    for (uint j = 0; j < G; ++j) {
        const float score = ihc_simd_sum(logits[j]);
        if (lane == 0) p[8 * j + simd] = score;
    }
}

// Threadgroup `slice` of a row writes columns [256 slice, 256 slice + 256): every slice derives the same gates from
// the same partials in the same order, and slice 0 alone stores the post gates.
template <uint G>
inline void ihc_finish_fixed(const device bfloat *hidden, const device float *part, const device float *bias,
                             const device float *scale, device bfloat *collapsed, device float *post,
                             constant Hy4IhcShape &a, uint row, uint t, uint slice, threadgroup float (&pre)[4]) {
    const uint wide = 4 * a.width;
    const device bfloat *x = hidden + ulong(row) * wide;
    const device float *p = part + ulong(row) * 72;
    if (t < G && (t < 4 || slice == 0)) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            sum += p[64 + i];
            dot += p[8 * t + i];
        }
        const float mix = dot * precise::rsqrt(sum / float(wide) + a.eps);
        const float gate = 1.0f / (1.0f + precise::exp(-(mix * scale[t / 4] + bias[t])));
        if (t < 4) pre[t] = gate + a.gate_eps;
        else post[ulong(row) * 4 + t - 4] = a.post_magnitude * gate + a.gate_eps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = 256 * slice + t;
    if (i < a.width) {
        float v = 0.0f;
        for (uint stream = 0; stream < 4; ++stream) v += pre[stream] * float(x[stream * a.width + i]);
        collapsed[ulong(row) * a.width + i] = bfloat(v);
    }
}

#define HY4_IHC_SPLIT(G)                                                                                         \
    kernel void tf_hy4_ihc_partials_g##G(                                                                        \
        const device bfloat *hidden [[buffer(0)]], const device float *project [[buffer(1)]],                    \
        device float *part [[buffer(2)]], constant Hy4IhcShape &a [[buffer(3)]],                                 \
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {                  \
        ihc_partials_fixed<G>(hidden, project, part, a, group.y, group.x, lane);                                 \
    }                                                                                                            \
    kernel void tf_hy4_ihc_finish_g##G(                                                                          \
        const device bfloat *hidden [[buffer(0)]], const device float *part [[buffer(1)]],                       \
        const device float *bias [[buffer(2)]], const device float *scale [[buffer(3)]],                         \
        device bfloat *collapsed [[buffer(4)]], device float *post [[buffer(5)]], constant Hy4IhcShape &a [[buffer(6)]], \
        uint2 group [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                  \
        threadgroup float pre[4];                                                                                \
        ihc_finish_fixed<G>(hidden, part, bias, scale, collapsed, post, a, group.y, t, group.x, pre);            \
    }
HY4_IHC_SPLIT(4)
HY4_IHC_SPLIT(8)

kernel void tf_hy4_ihc_add(
    const device bfloat *residual [[buffer(0)]],
    const device bfloat *branch [[buffer(1)]],
    const device float *post [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant Hy4IhcShape &a [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
    const ulong full = ulong(tid.y) * 4 * a.width + tid.x;
    const ulong local = ulong(tid.y) * a.width + tid.x % a.width;
    output[full] = bfloat(float(residual[full]) + post[ulong(tid.y) * 4 + tid.x / a.width] * float(branch[local]));
}

// The fusion is attached only to slot zero of every DFly2 block.
kernel void tf_hy4_draft_fuse(
    device bfloat *embedded [[buffer(0)]],
    const device bfloat *delta [[buffer(1)]],
    const device bfloat *gain [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const ulong anchor = ulong(tid.y) * shape.x * shape.y + tid.x;
    const bfloat update = bfloat(float(delta[ulong(tid.y) * shape.x + tid.x]) * float(gain[tid.x]));
    embedded[anchor] = bfloat(float(embedded[anchor]) + float(update));
}

kernel void tf_hy4_draft_residual(
    const device bfloat *base [[buffer(0)]],
    const device bfloat *branch [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const ulong o = ulong(tid.y) * shape.x + tid.x;
    const ulong i = ulong(tid.y) * shape.y + tid.x;
    out[o] = bfloat(float(base[i]) + float(branch[o]));
}

// ---- Sub-block seams in two launches instead of five (decode and prompt alike) ----
// tail: a sub-block's branch (one input, or two added in bfloat as the expert branches are) through its output RMS
// norm (bf16(x * inv) * gain, as the norm kernel rounds), mixed into the four residual streams with the post gates
// (rounded as tf_hy4_ihc_add), and, when the next collapse's weights are given, that collapse's fp32 partial sums
// over the mixed streams. 12 threadgroups of 256 threads a row, each element once; every threadgroup derives the norm
// scale itself from the branch row (6 KB) and stores its 19 sums at part[row][sum][tg] (16 slots a sum).
// mode 0: partials only of the given streams (no branch; after an Engram layer).
constant constexpr uint seam_tgs = 12;
struct Hy4TailShape { uint rows, width, two, partials; float eps; uint mode; };

inline float seam_tg_sum(float v, uint lane, uint sg, threadgroup float *scratch) {
    v = ihc_simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// The 10 stream products sum_i m[s][i] m[u][i] (s <= u) a row, so the head gets the collapsed row's sum of squares
// as pre^T Gram pre instead of a second pass over the row: part[row][9 + k][tg].
constant constexpr uint seam_gram = 10;
constant constexpr uint seam_slots = 16;

template <uint G>
inline void ihc_tail(const device bfloat *branch, const device bfloat *branch2, const device bfloat *gain,
                     const device bfloat *residual, const device float *post, device bfloat *out,
                     const device float *project, device float *part, constant Hy4TailShape &a, uint row, uint tg,
                     uint t, uint lane, uint sg, threadgroup float *scratch, threadgroup float (&sums)[9 + seam_gram][8]) {
    const uint W = a.width, wide = 4 * W;
    const device bfloat *b0 = branch + ulong(row) * W;
    const device bfloat *b1 = branch2 + ulong(row) * W;
    float inv = 0.0f;
    if (a.mode != 0) {
        float sq = 0.0f;
        for (uint i = t; i < W; i += 256) {
            const float v = a.two ? float(bfloat(b0[i] + b1[i])) : float(b0[i]);
            sq = fma(v, v, sq);
        }
        inv = precise::rsqrt(seam_tg_sum(sq, lane, sg, scratch) / float(W) + a.eps);
    }
    float msq = 0.0f, logits[G], gram[seam_gram];
#pragma unroll
    for (uint j = 0; j < G; ++j) logits[j] = 0.0f;
#pragma unroll
    for (uint k = 0; k < seam_gram; ++k) gram[k] = 0.0f;
    // a column's four streams together (for the 3072 width, thread t of threadgroup tg owns one column; its stream
    // order is the flat order's, so the gate sums keep their bits)
    for (uint col = tg * 256 + t; col < W; col += seam_tgs * 256) {
        float m[4];
        float v = 0.0f, normed = 0.0f;
        if (a.mode != 0) {
            v = a.two ? float(bfloat(b0[col] + b1[col])) : float(b0[col]);
            normed = float(bfloat(float(bfloat(v * inv)) * float(gain[col])));
        }
#pragma unroll
        for (uint stream = 0; stream < 4; ++stream) {
            const ulong i = ulong(row) * wide + stream * W + col;
            if (a.mode != 0) {
                m[stream] = float(bfloat(float(residual[i]) + post[ulong(row) * 4 + stream] * normed));
                out[i] = bfloat(m[stream]);
            } else {
                m[stream] = float(residual[i]);
            }
        }
        if (a.partials) {
#pragma unroll
            for (uint stream = 0; stream < 4; ++stream) {
                msq = fma(m[stream], m[stream], msq);
#pragma unroll
                for (uint j = 0; j < G; ++j) logits[j] = fma(m[stream], project[j * wide + stream * W + col], logits[j]);
            }
            uint k = 0;
#pragma unroll
            for (uint s = 0; s < 4; ++s)
#pragma unroll
                for (uint u = s; u < 4; ++u, ++k) gram[k] = fma(m[s], m[u], gram[k]);
        }
    }
    if (!a.partials) return;
    // every sum's simdgroup totals in one threadgroup step, then each added over the 8 simdgroups in order
#pragma unroll
    for (uint j = 0; j < G; ++j) {
        const float v = ihc_simd_sum(logits[j]);
        if (lane == 0) sums[j][sg] = v;
    }
    {
        const float v = ihc_simd_sum(msq);
        if (lane == 0) sums[8][sg] = v;
    }
#pragma unroll
    for (uint k = 0; k < seam_gram; ++k) {
        const float v = ihc_simd_sum(gram[k]);
        if (lane == 0) sums[9 + k][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *p = part + ulong(row) * (9 + seam_gram) * seam_slots;
    if (t < G || t == 8 || (t >= 9 && t < 9 + seam_gram)) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += sums[t][i];
        p[t * seam_slots + tg] = total;
    }
}

// head: the collapse's gates from the tail's partials (added over the 12 threadgroups in order), the collapsed row
// (as tf_hy4_ihc_finish), then the sub-block's input RMS norm with the row's sum of squares from the tail's stream
// products: 12 threadgroups of 256 a row, a column a thread, no pass over the whole row.
template <uint G>
inline void ihc_head(const device bfloat *hidden, const device float *part, const device float *bias,
                     const device float *scale, const device bfloat *gain, device bfloat *collapsed,
                     device bfloat *normalized, device float *post, constant Hy4IhcShape &a, float norm_eps, uint row,
                     uint slice, uint t, threadgroup float (&pre)[4], threadgroup float (&gram)[seam_gram]) {
    const uint W = a.width, wide = 4 * W;
    const device bfloat *x = hidden + ulong(row) * wide;
    const device float *p = part + ulong(row) * (9 + seam_gram) * seam_slots;
    if (t < G && (t < 4 || slice == 0)) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < seam_tgs; ++i) {
            sum += p[8 * seam_slots + i];
            dot += p[t * seam_slots + i];
        }
        const float mix = dot * precise::rsqrt(sum / float(wide) + a.eps);
        const float gate = 1.0f / (1.0f + precise::exp(-(mix * scale[t / 4] + bias[t])));
        if (t < 4) pre[t] = gate + a.gate_eps;
        else post[ulong(row) * 4 + t - 4] = a.post_magnitude * gate + a.gate_eps;
    } else if (t >= 32 && t < 32 + seam_gram) {
        float g = 0.0f;
        for (uint i = 0; i < seam_tgs; ++i) g += p[(9 + t - 32) * seam_slots + i];
        gram[t - 32] = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sq = 0.0f;
    uint k = 0;
    for (uint s = 0; s < 4; ++s)
        for (uint u = s; u < 4; ++u, ++k) sq = fma((s == u ? 1.0f : 2.0f) * pre[s] * pre[u], gram[k], sq);
    const float inv = precise::rsqrt(sq / float(W) + norm_eps);
    for (uint i = slice * 256 + t; i < W; i += seam_tgs * 256) {
        float v = 0.0f;
        for (uint stream = 0; stream < 4; ++stream) v += pre[stream] * float(x[stream * W + i]);
        const float c = float(bfloat(v));
        collapsed[ulong(row) * W + i] = bfloat(v);
        normalized[ulong(row) * W + i] = bfloat(float(bfloat(c * inv)) * float(gain[i]));
    }
}

// ---- tail and the next head in one launch (decode passes): the row's 12 threadgroups meet at an in-kernel barrier
// between them (all are resident: at most 16 rows a launch), so the head's gates and Gram sums come from every
// threadgroup's partials without a second launch. A thread owns one column (width 3072 = 12 x 256) and keeps its
// four mixed stream values in registers across the barrier. The partials cross threadgroups as device atomics.
// The same arithmetic as ihc_tail then ihc_head (the same sums in the same orders), so the same bits.
struct Hy4SeamBarrier { atomic_uint count; atomic_uint gen; };

inline void seam_barrier(device Hy4SeamBarrier* b, uint t) {
    // every thread's stores published, then one thread a threadgroup arrives and waits for the row's other 11
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
        const uint gen = atomic_load_explicit(&b->gen, memory_order_relaxed);
        // the generation read stays ahead of the arrival (else the last arrival's bump could come first)
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint seen = atomic_fetch_add_explicit(&b->count, 1u, memory_order_relaxed);
        if (seen + 1 == seam_tgs) {
            atomic_store_explicit(&b->count, 0u, memory_order_relaxed);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            atomic_fetch_add_explicit(&b->gen, 1u, memory_order_relaxed);
        } else {
            while (atomic_load_explicit(&b->gen, memory_order_relaxed) == gen) {
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

template <uint G>
inline void ihc_seam(const device bfloat *branch, const device bfloat *branch2, const device bfloat *gain,
                     const device bfloat *residual, device float *post, device bfloat *out,
                     const device float *project, device atomic_float *part, constant Hy4TailShape &a,
                     const device float *bias, const device float *scale, const device bfloat *next_gain,
                     device bfloat *collapsed, device bfloat *normalized, constant Hy4IhcShape &n, float norm_eps,
                     device Hy4SeamBarrier *bars, uint row, uint tg, uint t, uint lane, uint sg,
                     threadgroup float *scratch, threadgroup float (&sums)[9 + seam_gram][8],
                     threadgroup float (&pre)[4], threadgroup float (&gram)[seam_gram]) {
    const uint W = a.width, wide = 4 * W;
    const device bfloat *b0 = branch + ulong(row) * W;
    const device bfloat *b1 = branch2 + ulong(row) * W;
    float sq = 0.0f;
    for (uint i = t; i < W; i += 256) {
        const float v = a.two ? float(bfloat(b0[i] + b1[i])) : float(b0[i]);
        sq = fma(v, v, sq);
    }
    const float inv = precise::rsqrt(seam_tg_sum(sq, lane, sg, scratch) / float(W) + a.eps);
    const uint col = tg * 256 + t;
    float m[4];
    const float v = a.two ? float(bfloat(b0[col] + b1[col])) : float(b0[col]);
    const float normed = float(bfloat(float(bfloat(v * inv)) * float(gain[col])));
#pragma unroll
    for (uint stream = 0; stream < 4; ++stream) {
        const ulong i = ulong(row) * wide + stream * W + col;
        m[stream] = float(bfloat(float(residual[i]) + post[ulong(row) * 4 + stream] * normed));
        out[i] = bfloat(m[stream]);
    }
    float msq = 0.0f, logits[G], gr[seam_gram];
#pragma unroll
    for (uint j = 0; j < G; ++j) logits[j] = 0.0f;
#pragma unroll
    for (uint k = 0; k < seam_gram; ++k) gr[k] = 0.0f;
#pragma unroll
    for (uint stream = 0; stream < 4; ++stream) {
        msq = fma(m[stream], m[stream], msq);
#pragma unroll
        for (uint j = 0; j < G; ++j) logits[j] = fma(m[stream], project[j * wide + stream * W + col], logits[j]);
    }
    {
        uint k = 0;
#pragma unroll
        for (uint s0 = 0; s0 < 4; ++s0)
#pragma unroll
            for (uint u = s0; u < 4; ++u, ++k) gr[k] = fma(m[s0], m[u], gr[k]);
    }
#pragma unroll
    for (uint j = 0; j < G; ++j) {
        const float x = ihc_simd_sum(logits[j]);
        if (lane == 0) sums[j][sg] = x;
    }
    {
        const float x = ihc_simd_sum(msq);
        if (lane == 0) sums[8][sg] = x;
    }
#pragma unroll
    for (uint k = 0; k < seam_gram; ++k) {
        const float x = ihc_simd_sum(gr[k]);
        if (lane == 0) sums[9 + k][sg] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_float *p = part + ulong(row) * (9 + seam_gram) * seam_slots;
    if (t < G || t == 8 || (t >= 9 && t < 9 + seam_gram)) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += sums[t][i];
        atomic_store_explicit(p + t * seam_slots + tg, total, memory_order_relaxed);
    }
    // every threadgroup has read this sub-block's post gates above; the head may now overwrite them
    seam_barrier(bars + row, t);
    if (t < G && (t < 4 || tg == 0)) {
        float sum = 0.0f, dot = 0.0f;
        for (uint i = 0; i < seam_tgs; ++i) {
            sum += atomic_load_explicit(p + 8 * seam_slots + i, memory_order_relaxed);
            dot += atomic_load_explicit(p + t * seam_slots + i, memory_order_relaxed);
        }
        const float mix = dot * precise::rsqrt(sum / float(wide) + n.eps);
        const float gate = 1.0f / (1.0f + precise::exp(-(mix * scale[t / 4] + bias[t])));
        if (t < 4) pre[t] = gate + n.gate_eps;
        else post[ulong(row) * 4 + t - 4] = n.post_magnitude * gate + n.gate_eps;
    } else if (t >= 32 && t < 32 + seam_gram) {
        float g = 0.0f;
        for (uint i = 0; i < seam_tgs; ++i) g += atomic_load_explicit(p + (9 + t - 32) * seam_slots + i, memory_order_relaxed);
        gram[t - 32] = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float hsq = 0.0f;
    uint k = 0;
    for (uint s0 = 0; s0 < 4; ++s0)
        for (uint u = s0; u < 4; ++u, ++k) hsq = fma((s0 == u ? 1.0f : 2.0f) * pre[s0] * pre[u], gram[k], hsq);
    const float hinv = precise::rsqrt(hsq / float(W) + norm_eps);
    float c = 0.0f;
    for (uint stream = 0; stream < 4; ++stream) c += pre[stream] * m[stream];
    const float cb = float(bfloat(c));
    collapsed[ulong(row) * W + col] = bfloat(c);
    normalized[ulong(row) * W + col] = bfloat(float(bfloat(cb * hinv)) * float(next_gain[col]));
}

#define HY4_IHC_SEAM_FUSED(G)                                                                                    \
    kernel void tf_hy4_seam_g##G(                                                                                \
        const device bfloat *branch [[buffer(0)]], const device bfloat *branch2 [[buffer(1)]],                   \
        const device bfloat *gain [[buffer(2)]], const device bfloat *residual [[buffer(3)]],                    \
        device float *post [[buffer(4)]], device bfloat *out [[buffer(5)]],                                      \
        const device float *project [[buffer(6)]], device atomic_float *part [[buffer(7)]],                      \
        constant Hy4TailShape &a [[buffer(8)]], const device float *bias [[buffer(9)]],                          \
        const device float *scale [[buffer(10)]], const device bfloat *next_gain [[buffer(11)]],                 \
        device bfloat *collapsed [[buffer(12)]], device bfloat *normalized [[buffer(13)]],                       \
        constant Hy4IhcShape &n [[buffer(14)]], constant float &norm_eps [[buffer(15)]],                         \
        device Hy4SeamBarrier *bars [[buffer(16)]], uint2 group [[threadgroup_position_in_grid]],                \
        uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                         \
        uint sg [[simdgroup_index_in_threadgroup]]) {                                                            \
        threadgroup float scratch[8], sums[9 + seam_gram][8], pre[4], gram[seam_gram];                           \
        ihc_seam<G>(branch, branch2, gain, residual, post, out, project, part, a, bias, scale, next_gain,        \
                    collapsed, normalized, n, norm_eps, bars, group.y, group.x, t, lane, sg, scratch, sums, pre,  \
                    gram);                                                                                       \
    }
HY4_IHC_SEAM_FUSED(4)
HY4_IHC_SEAM_FUSED(8)

#define HY4_IHC_SEAMS(G)                                                                                         \
    kernel void tf_hy4_tail_g##G(                                                                                \
        const device bfloat *branch [[buffer(0)]], const device bfloat *branch2 [[buffer(1)]],                   \
        const device bfloat *gain [[buffer(2)]], const device bfloat *residual [[buffer(3)]],                    \
        const device float *post [[buffer(4)]], device bfloat *out [[buffer(5)]],                                \
        const device float *project [[buffer(6)]], device float *part [[buffer(7)]],                             \
        constant Hy4TailShape &a [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],                    \
        uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                         \
        uint sg [[simdgroup_index_in_threadgroup]]) {                                                            \
        threadgroup float scratch[8], sums[9 + seam_gram][8];                                                    \
        ihc_tail<G>(branch, branch2, gain, residual, post, out, project, part, a, group.y, group.x, t, lane, sg,  \
                    scratch, sums);                                                                              \
    }                                                                                                            \
    kernel void tf_hy4_head_g##G(                                                                                \
        const device bfloat *hidden [[buffer(0)]], const device float *part [[buffer(1)]],                       \
        const device float *bias [[buffer(2)]], const device float *scale [[buffer(3)]],                         \
        const device bfloat *gain [[buffer(4)]], device bfloat *collapsed [[buffer(5)]],                         \
        device bfloat *normalized [[buffer(6)]], device float *post [[buffer(7)]],                               \
        constant Hy4IhcShape &a [[buffer(8)]], constant float &norm_eps [[buffer(9)]],                           \
        uint2 group [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {                  \
        threadgroup float pre[4], gram[seam_gram];                                                               \
        ihc_head<G>(hidden, part, bias, scale, gain, collapsed, normalized, post, a, norm_eps, group.y, group.x, \
                    t, pre, gram);                                                                               \
    }
HY4_IHC_SEAMS(4)
HY4_IHC_SEAMS(8)
