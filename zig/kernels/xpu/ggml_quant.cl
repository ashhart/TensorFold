// ggml block-quantized weights (llama.cpp GGUF) on Xe2: Q2_K, Q4_K, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_M.
//
// Row-interleaved layout (ggml.zig packRows): rows go in groups of 16; for a group and a 256-weight block b the 16 rows' blocks are stored
// transposed, so that a 16-lane sub-group owning one group (lane = row) loads its block with fully coalesced 256-byte (uint4 a lane) and
// 64-byte (dword a lane) messages and streams the group's data in address order. A block is PB = 16*A + 4*C bytes (the file's block with the
// fp16 `d` moved to the end for the five types that lead with it, padded to 4 bytes): A uint4 pieces [piece][lane][16 B], then C dwords [dword][lane].
//
// Arithmetic (llama.cpp mmvq style): the activation x (bf16) is quantized to int8 per 32 values inside the kernel (d = amax * fl(1/127),
// q = rne(x * (127 / amax))) into local memory; every lane decodes its row's 16 units of 16 weights (dec_<T> -> DL: four packed int8 words +
// scales) and accumulates dp4a(q, xq) * a * d_x (Q4_K/Q2_K: a * dp4a(q, xq) - b * sum(xq)) over the blocks of its K range. No cross-lane
// reduction (K is split over `ksplit` sub-groups of a work-group only to fill the machine on small matrices; their partials are added in order).
// dq_<T> / embed_<T> use the same decoders and convert to fp32 in ggml-quants.c's operation order (bit-exact vs libggml).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#include "ggml_tables.h"

#ifndef NSGW
#define NSGW 8  // sub-groups a work-group
#endif
#define MAXK 17408

// 4 sign bits -> a mask byte (0xFF) per set bit
__constant uint kmask16[16] = {0x0u, 0xFFu, 0xFF00u, 0xFFFFu, 0xFF0000u, 0xFF00FFu, 0xFFFF00u, 0xFFFFFFu, 0xFF000000u, 0xFF0000FFu, 0xFF00FF00u, 0xFF00FFFFu, 0xFFFF0000u, 0xFFFF00FFu, 0xFFFFFF00u, 0xFFFFFFFFu};

typedef __global const uchar *P;

typedef struct {
    __local const ulong *g8;  // iq2*/iq1 grids, 8 weights an entry
    __local const uint *g4;   // iq3 grids, 4 weights an entry
    __local const ulong *k6;  // ksigns64: 7-bit sign code -> 8 byte masks
    __local const uint *m16;  // 4 sign bits -> 4 byte masks
    __local const ushort *tk; // iq4_xs: byte -> kvalues[low nibble] | kvalues[high nibble] << 8
} Tab;

// 16 weights: q0..q3 four packed int8 each; w_i = a * q_i [- b0] (U: unsigned bytes with min), or a * (q_i + delta) (iq1_m: b0 for the first 8, b1 for the next 8).
typedef struct { uint q0, q1, q2, q3; float a, b0, b1; } DL;

inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float h2f(uint u) { return vload_half(0, (__private const half *)&u); }
inline uint neg4(uint x, uint m) { return (x ^ m) + (m & 0x01010101u); }  // per-byte -x for masked bytes (grid bytes are nonzero: no carry)
inline uint lo32(ulong v) { return (uint)v; }
inline uint hi32(ulong v) { return (uint)(v >> 32); }
inline int dp_ss(uint a, uint b, int c) { return dot_acc_sat_4x8packed_ss_int(a, b, c); }
inline int dp_us(uint a, uint b, int c) { return dot_acc_sat_4x8packed_us_int(a, b, c); }

// Fields of a block held in registers as dwords w[] (byte offsets are compile-time constants once the unit loops are unrolled).
#define D32(o) w[(o) >> 2]
#define B8(o) ((w[(o) >> 2] >> (8 * ((o) & 3))) & 255u)
#define B16(o) ((w[(o) >> 2] >> (8 * ((o) & 2))) & 0xffffu)

// 84 bytes (A=5 C=1): scales[16] qs[64] d dmin
inline DL dec_q2_k(const uint *w, uint u, Tab t) {
    const uint n = u >> 3, j = (u & 7) >> 1, h = u & 1;
    const uint sc = B8(n * 8 + j * 2 + h), dm = D32(80), sh = 2 * j, qi = 4 + 8 * n + 4 * h;
    DL o;
    o.q0 = (w[qi] >> sh) & 0x03030303u;
    o.q1 = (w[qi + 1] >> sh) & 0x03030303u;
    o.q2 = (w[qi + 2] >> sh) & 0x03030303u;
    o.q3 = (w[qi + 3] >> sh) & 0x03030303u;
    o.a = h2f(dm & 0xffffu) * (float)(sc & 15u);
    o.b0 = h2f(dm >> 16) * (float)(sc >> 4);
    o.b1 = 0.0f;
    return o;
}

// 144 bytes (A=9 C=0): d dmin scales[12] qs[128]
inline DL dec_q4_k(const uint *w, uint u, Tab t) {
    const uint g = u >> 2, hi = (u >> 1) & 1, is = g * 2 + hi, jj = 8 * (is & 3), dm = D32(0);
    const uint A = (D32(4) >> jj) & 255u, B = (D32(8) >> jj) & 255u, C = (D32(12) >> jj) & 255u;  // get_scale_min_k4 on the 12 scale bytes
    const uint s = is < 4 ? (A & 63u) : ((C & 15u) | ((A >> 6) << 4));
    const uint m = is < 4 ? (B & 63u) : ((C >> 4) | ((B >> 6) << 4));
    const uint qi = 4 + 8 * g + 4 * (u & 1), sh = 4 * hi;
    DL o;
    o.q0 = (w[qi] >> sh) & 0x0F0F0F0Fu;
    o.q1 = (w[qi + 1] >> sh) & 0x0F0F0F0Fu;
    o.q2 = (w[qi + 2] >> sh) & 0x0F0F0F0Fu;
    o.q3 = (w[qi + 3] >> sh) & 0x0F0F0F0Fu;
    o.a = h2f(dm & 0xffffu) * (float)s;
    o.b0 = h2f(dm >> 16) * (float)m;
    o.b1 = 0.0f;
    return o;
}

// 176 bytes (A=11 C=0): d dmin scales[12] qh[32] qs[128]; Q4_K's nibbles plus a fifth bit from qh (bit 2g + hi of byte l for the weights of 64-group g)
inline DL dec_q5_k(const uint *w, uint u, Tab t) {
    const uint g = u >> 2, hi = (u >> 1) & 1, is = g * 2 + hi, jj = 8 * (is & 3), dm = D32(0);
    const uint A = (D32(4) >> jj) & 255u, B = (D32(8) >> jj) & 255u, C = (D32(12) >> jj) & 255u;
    const uint s = is < 4 ? (A & 63u) : ((C & 15u) | ((A >> 6) << 4));
    const uint m = is < 4 ? (B & 63u) : ((C >> 4) | ((B >> 6) << 4));
    const uint qi = 12 + 8 * g + 4 * (u & 1), hi_i = 4 + 4 * (u & 1), sh = 4 * hi, hs = is;
    DL o;
    o.q0 = ((w[qi] >> sh) & 0x0F0F0F0Fu) | (((w[hi_i] >> hs) & 0x01010101u) << 4);
    o.q1 = ((w[qi + 1] >> sh) & 0x0F0F0F0Fu) | (((w[hi_i + 1] >> hs) & 0x01010101u) << 4);
    o.q2 = ((w[qi + 2] >> sh) & 0x0F0F0F0Fu) | (((w[hi_i + 2] >> hs) & 0x01010101u) << 4);
    o.q3 = ((w[qi + 3] >> sh) & 0x0F0F0F0Fu) | (((w[hi_i + 3] >> hs) & 0x01010101u) << 4);
    o.a = h2f(dm & 0xffffu) * (float)s;
    o.b0 = h2f(dm >> 16) * (float)m;
    o.b1 = 0.0f;
    return o;
}

// 136 bytes (A=8 C=2): d scales_h scales_l[4] qs[128]; a sub-block (32 weights: low nibbles = first 16, high nibbles = next 16) yields two units
inline void dec2_iq4_xs(const uint *w, uint ib, Tab t, DL *lo, DL *hi) {
    const uint dsh = D32(0);
    const uint ls = ((B8(4 + (ib >> 1)) >> (4 * (ib & 1))) & 15u) | ((((dsh >> 16) >> (2 * ib)) & 3u) << 4);
    const float a = h2f(dsh & 0xffffu) * (float)((int)ls - 32);
    uint l[4], h[4];
    for (int k = 0; k < 4; k++) {
        const uint v = w[2 + 4 * ib + k];
        const uint t0 = t.tk[v & 255u], t1 = t.tk[(v >> 8) & 255u], t2 = t.tk[(v >> 16) & 255u], t3 = t.tk[v >> 24];
        l[k] = (t0 & 255u) | ((t1 & 255u) << 8) | ((t2 & 255u) << 16) | (t3 << 24);
        h[k] = (t0 >> 8) | (t1 & 0xFF00u) | ((t2 & 0xFF00u) << 8) | ((t3 & 0xFF00u) << 16);
    }
    lo->q0 = l[0]; lo->q1 = l[1]; lo->q2 = l[2]; lo->q3 = l[3];
    hi->q0 = h[0]; hi->q1 = h[1]; hi->q2 = h[2]; hi->q3 = h[3];
    lo->a = a; hi->a = a;
    lo->b0 = 0.0f; lo->b1 = 0.0f; hi->b0 = 0.0f; hi->b1 = 0.0f;
}

// 144 bytes = 8 blocks of 32 weights (A=9 C=0): qs[128] (16 bytes a block, low nibbles = first 16 weights, high = next 16) d[8] (fp16); w = d * kvalues[nibble]
inline void dec2_iq4_nl(const uint *w, uint ib, Tab t, DL *lo, DL *hi) {
    const float a = h2f(B16(128 + 2 * ib));
    uint l[4], h[4];
    for (int k = 0; k < 4; k++) {
        const uint v = w[4 * ib + k];
        const uint t0 = t.tk[v & 255u], t1 = t.tk[(v >> 8) & 255u], t2 = t.tk[(v >> 16) & 255u], t3 = t.tk[v >> 24];
        l[k] = (t0 & 255u) | ((t1 & 255u) << 8) | ((t2 & 255u) << 16) | (t3 << 24);
        h[k] = (t0 >> 8) | (t1 & 0xFF00u) | ((t2 & 0xFF00u) << 8) | ((t3 & 0xFF00u) << 16);
    }
    lo->q0 = l[0]; lo->q1 = l[1]; lo->q2 = l[2]; lo->q3 = l[3];
    hi->q0 = h[0]; hi->q1 = h[1]; hi->q2 = h[2]; hi->q3 = h[3];
    lo->a = a; hi->a = a;
    lo->b0 = 0.0f; lo->b1 = 0.0f; hi->b0 = 0.0f; hi->b1 = 0.0f;
}

// packed 68 bytes (A=4 C=1): qs[64] (per 32 weights: u32 of 4 grid bytes, u32 of 4 x 7 sign bits + 4-bit scale) d
inline DL dec_iq2_xxs(const uint *w, uint u, Tab t) {
    const uint ib = u >> 1, h = u & 1, a0 = D32(8 * ib), a1 = D32(8 * ib + 4), l0 = 2 * h;
    const ulong g0 = t.g8[(a0 >> (8 * l0)) & 255u], g1 = t.g8[(a0 >> (8 * l0 + 8)) & 255u];
    const ulong m0 = t.k6[(a1 >> (7 * l0)) & 127u], m1 = t.k6[(a1 >> (7 * l0 + 7)) & 127u];
    DL o;
    o.q0 = neg4(lo32(g0), lo32(m0));
    o.q1 = neg4(hi32(g0), hi32(m0));
    o.q2 = neg4(lo32(g1), lo32(m1));
    o.q3 = neg4(hi32(g1), hi32(m1));
    o.a = h2f(B16(64)) * (0.5f + (float)(a1 >> 28)) * 0.25f;
    o.b0 = 0.0f;
    o.b1 = 0.0f;
    return o;
}

// packed 76 bytes (A=4 C=3): qs[32] (u16: 9-bit grid index, 7-bit sign code) scales[8] d
inline DL dec_iq2_xs(const uint *w, uint u, Tab t) {
    const uint ib = u >> 1, h = u & 1, q = D32(8 * ib + 4 * h);
    const ulong g0 = t.g8[q & 511u], g1 = t.g8[(q >> 16) & 511u];
    const ulong m0 = t.k6[(q >> 9) & 127u], m1 = t.k6[q >> 25];
    DL o;
    o.q0 = neg4(lo32(g0), lo32(m0));
    o.q1 = neg4(hi32(g0), hi32(m0));
    o.q2 = neg4(lo32(g1), lo32(m1));
    o.q3 = neg4(hi32(g1), hi32(m1));
    o.a = h2f(B16(72)) * (0.5f + (float)((B8(64 + ib) >> (4 * h)) & 15u)) * 0.25f;
    o.b0 = 0.0f;
    o.b1 = 0.0f;
    return o;
}

// packed 84 bytes (A=5 C=1): qs[64] (32 grid low bytes, 32 sign bytes) qh[8] scales[8] d
inline DL dec_iq2_s(const uint *w, uint u, Tab t) {
    const uint ib = u >> 1, h = u & 1, qs = D32(4 * ib), sg = D32(32 + 4 * ib), qh = B8(64 + ib), l0 = 2 * h;
    const uint i0 = ((qs >> (8 * l0)) & 255u) | ((qh << (8 - 2 * l0)) & 0x300u);
    const uint i1 = ((qs >> (8 * l0 + 8)) & 255u) | ((qh << (6 - 2 * l0)) & 0x300u);
    const ulong g0 = t.g8[i0], g1 = t.g8[i1];
    const uint s0 = (sg >> (8 * l0)) & 255u, s1 = (sg >> (8 * l0 + 8)) & 255u;
    DL o;
    o.q0 = neg4(lo32(g0), t.m16[s0 & 15u]);
    o.q1 = neg4(hi32(g0), t.m16[s0 >> 4]);
    o.q2 = neg4(lo32(g1), t.m16[s1 & 15u]);
    o.q3 = neg4(hi32(g1), t.m16[s1 >> 4]);
    o.a = h2f(B16(80)) * (0.5f + (float)((B8(72 + ib) >> (4 * h)) & 15u)) * 0.25f;
    o.b0 = 0.0f;
    o.b1 = 0.0f;
    return o;
}

// packed 100 bytes (A=6 C=1): qs[96] (64 grid bytes, then a u32 per 32 weights: 4 x 7 sign bits + 4-bit scale) d
inline DL dec_iq3_xxs(const uint *w, uint u, Tab t) {
    const uint ib = u >> 1, h = u & 1, q = D32(8 * ib + 4 * h), aux = D32(64 + 4 * ib), l0 = 2 * h;
    const ulong m0 = t.k6[(aux >> (7 * l0)) & 127u], m1 = t.k6[(aux >> (7 * l0 + 7)) & 127u];
    DL o;
    o.q0 = neg4(t.g4[q & 255u], lo32(m0));
    o.q1 = neg4(t.g4[(q >> 8) & 255u], hi32(m0));
    o.q2 = neg4(t.g4[(q >> 16) & 255u], lo32(m1));
    o.q3 = neg4(t.g4[q >> 24], hi32(m1));
    o.a = h2f(B16(96)) * (0.5f + (float)(aux >> 28)) * 0.5f;
    o.b0 = 0.0f;
    o.b1 = 0.0f;
    return o;
}

// packed 112 bytes (A=7 C=0): qs[64] qh[8] signs[32] scales[4] d
// Table lookups use byte offsets built directly from the index bits (no separate index * 4 shift).
#define L4(p, off) (*(__local const uint *)((__local const uchar *)(p) + (off)))
inline DL dec_iq3_s(const uint *w, uint u, Tab t) {
    const uint ib = u >> 1, h = u & 1, q = D32(8 * ib + 4 * h), qh = B8(64 + ib), sg = D32(72 + 4 * ib), l0 = 2 * h;
    const uint o0 = ((q << 2) & 0x3FCu) | ((qh << (10 - 2 * l0)) & 0x400u), o1 = ((q >> 6) & 0x3FCu) | ((qh << (9 - 2 * l0)) & 0x400u);
    const uint o2 = ((q >> 14) & 0x3FCu) | ((qh << (8 - 2 * l0)) & 0x400u), o3 = ((q >> 22) & 0x3FCu) | ((qh << (7 - 2 * l0)) & 0x400u);
    const uint s0 = sg >> (8 * l0), s1 = sg >> (8 * l0 + 8);  // sign bytes in the low byte
    DL o;
    o.q0 = neg4(L4(t.g4, o0), L4(t.m16, (s0 << 2) & 0x3Cu));
    o.q1 = neg4(L4(t.g4, o1), L4(t.m16, (s0 >> 2) & 0x3Cu));
    o.q2 = neg4(L4(t.g4, o2), L4(t.m16, (s1 << 2) & 0x3Cu));
    o.q3 = neg4(L4(t.g4, o3), L4(t.m16, (s1 >> 2) & 0x3Cu));
    o.a = h2f(B16(108)) * (float)(1 + 2 * ((B8(104 + (ib >> 1)) >> (4 * (ib & 1))) & 15u));
    o.b0 = 0.0f;
    o.b1 = 0.0f;
    return o;
}

// 210 bytes, packed 212 (A=13 C=1; the file layout plus 2 bytes of padding): ql[128] qh[64] scales[16] (int8) d; w = d * scale * (q6 - 32), 16 weights share a scale
inline DL dec_q6_k(const uint *w, uint u, Tab t) {
    const uint n = u >> 3, j = u & 7, qd = j >> 1, hh = j & 1;
    const uint qli = (n * 64 + (qd & 1) * 32 + hh * 16) >> 2, qhi = (128 + n * 32 + hh * 16) >> 2, lsh = (qd >> 1) * 4, hsh = qd * 2;
    uint q[4];
    for (int k = 0; k < 4; k++) {
        const uint x = ((w[qli + k] >> lsh) & 0x0F0F0F0Fu) | (((w[qhi + k] >> hsh) & 0x03030303u) << 4);
        q[k] = ((x | 0x80808080u) - 0x20202020u) ^ 0x80808080u;  // per byte x - 32 (x < 64: no borrow across bytes)
    }
    DL o;
    o.q0 = q[0]; o.q1 = q[1]; o.q2 = q[2]; o.q3 = q[3];
    o.a = h2f(B16(208)) * (float)(int)(char)B8(192 + n * 8 + hh + 2 * qd);
    o.b0 = 0.0f;
    o.b1 = 0.0f;
    return o;
}

// 272 bytes = 8 blocks of 32 weights (A=17 C=0): qs[256] (int8, the 8 blocks one after the other) d[8] (fp16); w = d * q, 32 weights share a scale (the repack moves the
// scales of the file blocks behind the quants)
inline DL dec_q8_0(const uint *w, uint u, Tab t) {
    DL o;
    o.q0 = w[4 * u];
    o.q1 = w[4 * u + 1];
    o.q2 = w[4 * u + 2];
    o.q3 = w[4 * u + 3];
    o.a = h2f(B16(256 + 2 * (u >> 1)));
    o.b0 = 0.0f;
    o.b1 = 0.0f;
    return o;
}

// 56 bytes (A=3 C=2): qs[32] qh[16] scales[8]; the fp16 super-block scale is spread over the top nibbles of the four scale words
inline DL dec_iq1_m(const uint *w, uint u, Tab t) {
    const uint ib = u >> 1, h = u & 1;
    const uint s0 = B16(48), s1 = B16(50), s2 = B16(52), s3 = B16(54);
    const uint su = (s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u);
    const uint sc = B16(48 + 2 * (ib >> 1));
    const uint qh = B8(32 + 2 * ib + h), qs0 = B8(4 * ib + 2 * h), qs1 = B8(4 * ib + 2 * h + 1);
    const ulong g0 = t.g8[qs0 | ((qh << 8) & 0x700u)], g1 = t.g8[qs1 | ((qh << 4) & 0x700u)];
    DL o;
    o.q0 = lo32(g0);
    o.q1 = hi32(g0);
    o.q2 = lo32(g1);
    o.q3 = hi32(g1);
    o.a = h2f(su) * (float)(2 * ((sc >> (6 * (ib & 1) + 3 * h)) & 7u) + 1);
    o.b0 = (qh & 0x08u) ? -0.125f : 0.125f;
    o.b1 = (qh & 0x80u) ? -0.125f : 0.125f;
    return o;
}

// ---- consumers of a decoded unit ----

inline float4 xw(uint v) { return (float4)((float)(char)(v & 255u), (float)(char)((v >> 8) & 255u), (float)(char)((v >> 16) & 255u), (float)(char)(v >> 24)); }

// Weights as fp32 in ggml-quants.c order (scheme S: a * q; U: a * q - b0; I: a * (q + delta)); contraction off so the roundings match the C code.
#pragma OPENCL FP_CONTRACT OFF
inline void deq_s(DL d, float *w) {
    const uint q[4] = {d.q0, d.q1, d.q2, d.q3};
    for (int k = 0; k < 4; k++) {
        const float4 v = xw(q[k]);
        w[4 * k] = d.a * v.x;
        w[4 * k + 1] = d.a * v.y;
        w[4 * k + 2] = d.a * v.z;
        w[4 * k + 3] = d.a * v.w;
    }
}
inline void deq_u(DL d, float *w) {
    const uint q[4] = {d.q0, d.q1, d.q2, d.q3};
    for (int k = 0; k < 4; k++) {
        w[4 * k] = d.a * (float)(q[k] & 255u) - d.b0;
        w[4 * k + 1] = d.a * (float)((q[k] >> 8) & 255u) - d.b0;
        w[4 * k + 2] = d.a * (float)((q[k] >> 16) & 255u) - d.b0;
        w[4 * k + 3] = d.a * (float)(q[k] >> 24) - d.b0;
    }
}
inline void deq_i(DL d, float *w) {
    const uint q[4] = {d.q0, d.q1, d.q2, d.q3};
    for (int k = 0; k < 4; k++) {
        const float4 v = xw(q[k]);
        const float dl = k < 2 ? d.b0 : d.b1;
        w[4 * k] = d.a * (v.x + dl);
        w[4 * k + 1] = d.a * (v.y + dl);
        w[4 * k + 2] = d.a * (v.z + dl);
        w[4 * k + 3] = d.a * (v.w + dl);
    }
}
#pragma OPENCL FP_CONTRACT ON

// Accumulation of one unit against the quantized activation: acc' = fma(dx, a * dp4a-sum, acc) (Q4_K/Q2_K: minus min * sum(x)). Explicit fmas with contraction
// off, so every kernel (one row or many, any window) rounds a row the same way.
#pragma OPENCL FP_CONTRACT OFF
inline float acc_s(float acc, DL d, uint4 x, float dx) {
    int s = dp_ss(d.q0, x.x, 0);
    s = dp_ss(d.q1, x.y, s);
    s = dp_ss(d.q2, x.z, s);
    s = dp_ss(d.q3, x.w, s);
    return fma(dx, d.a * (float)s, acc);
}
inline float acc_u(float acc, DL d, uint4 x, float dx) {
    int s = dp_us(d.q0, x.x, 0);
    s = dp_us(d.q1, x.y, s);
    s = dp_us(d.q2, x.z, s);
    s = dp_us(d.q3, x.w, s);
    int sx = dp_us(0x01010101u, x.x, 0);
    sx = dp_us(0x01010101u, x.y, sx);
    sx = dp_us(0x01010101u, x.z, sx);
    sx = dp_us(0x01010101u, x.w, sx);
    return fma(dx, fma(d.a, (float)s, -(d.b0 * (float)sx)), acc);
}
inline float acc_i(float acc, DL d, uint4 x, float dx) {
    int s = dp_ss(d.q0, x.x, 0);
    s = dp_ss(d.q1, x.y, s);
    int s2 = dp_ss(d.q2, x.z, 0);
    s2 = dp_ss(d.q3, x.w, s2);
    int sx = dp_us(0x01010101u, x.x, 0);
    sx = dp_us(0x01010101u, x.y, sx);
    int sx2 = dp_us(0x01010101u, x.z, 0);
    sx2 = dp_us(0x01010101u, x.w, sx2);
    const float t = fma(d.b0, (float)sx, (float)s) + fma(d.b1, (float)sx2, (float)s2);
    return fma(dx, d.a * t, acc);
}
#pragma OPENCL FP_CONTRACT ON

// ---- work-group machinery ----

#define WGT (NSGW * 16)
#define STAGEP(dst, src, n) for (uint i_ = get_local_id(0); i_ < (n); i_ += WGT) dst[i_] = src[i_];
#define TABLES(N8, G8, N4, G4, NK, NM, NT) \
    __local ulong tg8[N8]; __local uint tg4[N4]; __local ulong tk6[NK]; __local uint tm16[NM]; __local ushort ttk[NT]; \
    STAGEP(tg8, G8, N8) STAGEP(tg4, G4, N4) STAGEP(tk6, ksigns64, NK) STAGEP(tm16, kmask16, NM) \
    if (NT > 1) for (uint i_ = get_local_id(0); i_ < NT; i_ += WGT) ttk[i_] = (ushort)(((uint)kvalues_iq4nl[i_ & 15u] & 255u) | (((uint)kvalues_iq4nl[(i_ >> 4) & 15u] & 255u) << 8)); \
    Tab tb; tb.g8 = tg8; tb.g4 = tg4; tb.k6 = tk6; tb.m16 = tm16; tb.tk = ttk;

// x (bf16) block of 32 -> int8 + scale, one work-item a block: d = amax * fl(1/127), q = rne(x * (127 / amax)) (0 for an all-zero block).
// The divide is the native reciprocal-based one (not IEEE-exact; a quotient one ulp off can move a tie by one level).
inline float quant32c(uint4 a, uint4 b, uint4 c, uint4 d, uint *q) {
    const uint u[16] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
    float v[32];
    float amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        v[2 * i] = as_float(u[i] << 16);
        v[2 * i + 1] = as_float(u[i] & 0xffff0000u);
        amax = fmax(amax, fmax(fabs(v[2 * i]), fabs(v[2 * i + 1])));
    }
    const float id = amax > 0.0f ? native_divide(127.0f, amax) : 0.0f;
    for (int j = 0; j < 8; j++)
        q[j] = ((uint)convert_int_rte(v[4 * j] * id) & 255u) | (((uint)convert_int_rte(v[4 * j + 1] * id) & 255u) << 8) |
               (((uint)convert_int_rte(v[4 * j + 2] * id) & 255u) << 16) | ((uint)convert_int_rte(v[4 * j + 3] * id) << 24);
    return amax * 0.007874015718698502f;  // fl(1/127), a multiply so the result is IEEE-exact
}
inline void quant32(__global const ushort *x, uint blk, __local char *xq, __local float *xd) {
    const __global uint4 *p = (const __global uint4 *)(x + blk * 32);
    uint q[8];
    const float sc = quant32c(p[0], p[1], p[2], p[3], q);
    __local uint4 *o = (__local uint4 *)(xq + blk * 32);
    o[0] = (uint4)(q[0], q[1], q[2], q[3]);
    o[1] = (uint4)(q[4], q[5], q[6], q[7]);
    xd[blk] = sc;
}

// Block of one row from the interleaved layout: base = group base + block * 16 * PB, lane = row within the group.
#define LOADBLK(wv, base, lane, A, C) \
    { \
        _Pragma("unroll") for (int p_ = 0; p_ < (A); p_++) { \
            const uint4 v_ = *(__global const uint4 *)((base) + (p_ * 16 + (lane)) * 16); \
            wv[4 * p_] = v_.x; wv[4 * p_ + 1] = v_.y; wv[4 * p_ + 2] = v_.z; wv[4 * p_ + 3] = v_.w; \
        } \
        _Pragma("unroll") for (int t_ = 0; t_ < (C); t_++) wv[4 * (A) + t_] = *(__global const uint *)((base) + (A) * 256 + (t_ * 16 + (lane)) * 4); \
    }

#ifdef NODEC
#undef BODY_GEN_X
#endif
#ifdef NOQUANT
#define QUANT_LOOP
#else
#define QUANT_LOOP for (uint blk = get_local_id(0); blk < k / 32; blk += WGT) quant32(x, blk, xq, xd);
#endif
#define XV(b, u) (((__local const uint4 *)xq)[(b) * 16 + (u)])

#define XV(b, u) (((__local const uint4 *)xq)[(b) * 16 + (u)])

// BODY(T, SCHEME, WV, BI): acc += the 256 weights of block register array WV (block index BI) dotted with the quantized x.
#define BODY_GEN(T, SCHEME, WV, BI) \
    _Pragma("unroll") for (uint u = 0; u < 16; u++) { \
        const DL d = dec_##T(WV, u, tb); \
        acc = acc_##SCHEME(acc, d, XV(BI, u), xd[(BI) * 8 + (u >> 1)]); \
    }
#define BODY_IQ4(T, SCHEME, WV, BI) \
    _Pragma("unroll") for (uint ib = 0; ib < 8; ib++) { \
        DL lo, hi; \
        dec2_##T(WV, ib, tb, &lo, &hi); \
        acc = acc_s(acc, lo, XV(BI, 2 * ib), xd[(BI) * 8 + ib]); \
        acc = acc_s(acc, hi, XV(BI, 2 * ib + 1), xd[(BI) * 8 + ib]); \
    }
#ifdef NODEC
#undef BODY_GEN
#undef BODY_IQ4
#define BODY_GEN(T, SCHEME, WV, BI) { uint xr = 0; _Pragma("unroll") for (uint u = 0; u < sizeof(WV) / 4; u++) xr ^= WV[u]; acc += (float)xr; }
#define BODY_IQ4(T, SCHEME, WV, BI) BODY_GEN(T, SCHEME, WV, BI)
#endif
// DEQ(T, SCHEME, WV, OUT): the 256 weights as fp32.
#define DEQ_GEN(T, SCHEME, WV, OUT) \
    _Pragma("unroll") for (uint u = 0; u < 16; u++) deq_##SCHEME(dec_##T(WV, u, tb), (OUT) + 16 * u);
#define DEQ_IQ4(T, SCHEME, WV, OUT) \
    _Pragma("unroll") for (uint ib = 0; ib < 8; ib++) { \
        DL lo, hi; \
        dec2_##T(WV, ib, tb, &lo, &hi); \
        deq_s(lo, (OUT) + 32 * ib); \
        deq_s(hi, (OUT) + 32 * ib + 16); \
    }

// One sub-group's share of a row group: the blocks b0..b1 of group `grp` of weights `w`, double buffered in registers (the next block's loads are
// issued before the current one is decoded). Needs lane, tb, xq, xd in scope; adds into acc.
#define ROWLOOP(T, PB, A, C, BODY, SCHEME, w, grp, b0, b1, nb) \
    { \
        P gb = (w) + (ulong)(grp) * (nb) * (16 * PB); \
        uint wa[PB / 4], wb[PB / 4]; \
        if ((b0) < (b1)) LOADBLK(wa, gb + (ulong)(b0) * (16 * PB), lane, A, C) \
        for (uint b = (b0); b < (b1); b++) { \
            if (b + 1 < (b1)) LOADBLK(wb, gb + (ulong)(b + 1) * (16 * PB), lane, A, C) \
            BODY(T, SCHEME, wa, b) \
            _Pragma("unroll") for (uint i_ = 0; i_ < PB / 4; i_++) wa[i_] = wb[i_]; \
        } \
    }

// Starts the first block's memory traffic before the work-group stages tables and quantizes x (the loads then overlap that prologue).
#define PREFETCH_BLOCK(w, grp, b0, nb, PB) \
    if ((b0) < (b1)) { \
        __global const uint4 *pp_ = (__global const uint4 *)((w) + ((ulong)(grp) * (nb) + (b0)) * (16 * PB)); \
        for (uint i_ = lane; i_ < PB; i_ += 16) prefetch(pp_ + i_, 1); \
    }

#define DEFINE_MV(NAME, OUTT, STORE, T, PB, A, C, BODY, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME(P w, __global const ushort *x, __global OUTT *y, uint k, uint y_off, uint rows, uint ksplit, __local char *xq, __local float *xd) { \
    const uint sg = get_sub_group_id(), lane = get_sub_group_local_id(); \
    const uint nwg = get_num_groups(0), wgi = get_group_id(0), gtot = rows / 16, gend = (wgi + 1) * gtot / nwg;  /* even share of the row groups a work-group */ \
    const uint grp = wgi * gtot / nwg + sg / ksplit, ks = sg % ksplit, nb = k / 256; \
    const uint b0 = nb * ks / ksplit, b1 = nb * (ks + 1) / ksplit; \
    if (grp < gend) PREFETCH_BLOCK(w, grp, b0, nb, PB) \
    TABLES(N8, G8, N4, G4, NK, NM, NT) \
    __local float part[WGT]; \
    QUANT_LOOP \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float acc = 0.0f; \
    if (grp < gend) ROWLOOP(T, PB, A, C, BODY, SCHEME, w, grp, b0, b1, nb) \
    if (ksplit > 1) { \
        part[sg * 16 + lane] = acc; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (ks == 0) for (uint j = 1; j < ksplit; j++) acc += part[(sg + j) * 16 + lane]; \
    } \
    if (ks == 0 && grp < gend) { STORE } \
}

// embed_<T>: row ids[0] as bf16 (one work-item a block); dq_<T>: `rows` rows as fp32 (one work-item a row and block, tests).
#define DEFINE_DQ(T, PB, A, C, DEQ, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) \
__kernel void embed_##T(P w, __global const uint *ids, __global ushort *y, uint nb) { \
    TABLES(N8, G8, N4, G4, NK, NM, NT) \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const uint b = get_global_id(0); \
    if (b >= nb) return; \
    const uint row = ids[get_group_id(1)]; \
    uint wv[PB / 4]; \
    float out[256]; \
    LOADBLK(wv, w + ((ulong)(row >> 4) * nb + b) * (16 * PB), row & 15u, A, C) \
    DEQ(T, SCHEME, wv, out) \
    for (int i = 0; i < 256; i++) y[(ulong)get_group_id(1) * nb * 256 + b * 256 + i] = to_bf(out[i]); \
} \
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) \
__kernel void dq_##T(P w, __global float *y, uint rows, uint nb) { \
    TABLES(N8, G8, N4, G4, NK, NM, NT) \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const uint g = get_global_id(0); \
    if (g >= rows * nb) return; \
    const uint row = g / nb, b = g % nb; \
    uint wv[PB / 4]; \
    float out[256]; \
    LOADBLK(wv, w + ((ulong)(row >> 4) * nb + b) * (16 * PB), row & 15u, A, C) \
    DEQ(T, SCHEME, wv, out) \
    for (int i = 0; i < 256; i++) y[(ulong)row * nb * 256 + b * 256 + i] = out[i]; \
}

#define STORE_BF y[y_off + grp * 16 + lane] = to_bf(acc);
#define STORE_F y[y_off + grp * 16 + lane] = acc;

// (T, PB, A, C, BODY, DEQ, SCHEME, tables: N8 G8 N4 G4 NK(ksigns64) NM(kmask16) NT(iq4 pair table))
// ---- multi-row matvec (RM activation rows against the same weights; a row's bits do not depend on RM, its slot or m) ----
// The activation rows are quantized beforehand (quant_rows: [row][k bytes of int8 then k/32 fp32 scales], stride XSTR(k)); the weights are read once per block and
// decoded once per unit, every row accumulates in its own register in the same order as the single-row kernels. Rows r >= m read row 0 and are not stored.
#define XSTR(k) ((k) + (k) / 8)
// A block of a row is 16 units of 16 int8 + 8 scales: lane L loads unit L (and scale L & 7) once per block, a unit is then read from its lane with a constant-index
// shuffle (a register region read, no memory instruction): the per-unit loads of every row were the limit of the 8-row kernels.
#define XG(r, b, u) ((uint4)(intel_sub_group_shuffle(xr_[r].x, (u)), intel_sub_group_shuffle(xr_[r].y, (u)), intel_sub_group_shuffle(xr_[r].z, (u)), intel_sub_group_shuffle(xr_[r].w, (u))))
#define XDG(r, b, u) intel_sub_group_shuffle(sc_[r], (u) >> 1)
#define XLOADR(RM, b) \
    uint4 xr_[RM]; float sc_[RM]; \
    _Pragma("unroll") for (uint r = 0; r < RM; r++) { \
        xr_[r] = ((__global const uint4 *)xp[r])[(b) * 16 + lane]; \
        sc_[r] = ((__global const float *)(xp[r] + k))[(b) * 8 + (lane & 7u)]; \
    }
#define BODYR_GEN(T, SCHEME, WV, BI, RM) \
    _Pragma("unroll") for (uint u = 0; u < 16; u++) { \
        const DL d = dec_##T(WV, u, tb); \
        _Pragma("unroll") for (uint r = 0; r < RM; r++) accr[r] = acc_##SCHEME(accr[r], d, XG(r, BI, u), XDG(r, BI, u)); \
    }
#define BODYR_IQ4(T, SCHEME, WV, BI, RM) \
    _Pragma("unroll") for (uint ib = 0; ib < 8; ib++) { \
        DL lo, hi; \
        dec2_##T(WV, ib, tb, &lo, &hi); \
        _Pragma("unroll") for (uint r = 0; r < RM; r++) { \
            accr[r] = acc_s(accr[r], lo, XG(r, BI, 2 * ib), XDG(r, BI, 2 * ib)); \
            accr[r] = acc_s(accr[r], hi, XG(r, BI, 2 * ib + 1), XDG(r, BI, 2 * ib)); \
        } \
    }
#define ROWLOOPR(T, PB, A, C, BODYR, SCHEME, RM, w, grp, b0, b1, nb) \
    { \
        P gb = (w) + (ulong)(grp) * (nb) * (16 * PB); \
        uint wa[PB / 4], wb[PB / 4]; \
        if ((b0) < (b1)) LOADBLK(wa, gb + (ulong)(b0) * (16 * PB), lane, A, C) \
        for (uint b = (b0); b < (b1); b++) { \
            XLOADR(RM, b) \
            if (b + 1 < (b1)) LOADBLK(wb, gb + (ulong)(b + 1) * (16 * PB), lane, A, C) \
            BODYR(T, SCHEME, wa, b, RM) \
            _Pragma("unroll") for (uint i_ = 0; i_ < PB / 4; i_++) wa[i_] = wb[i_]; \
        } \
    }

#define DEFINE_MVR_GEN(NAME, OUTT, CONV, RM, T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) \
__kernel void NAME(P w, __global const uchar *xqg, __global OUTT *y, uint k, uint y_off, uint rows, uint ksplit, uint m) { \
    const uint sg = get_sub_group_id(), lane = get_sub_group_local_id(); \
    const uint nwg = get_num_groups(0), wgi = get_group_id(0), gtot = rows / 16, gend = (wgi + 1) * gtot / nwg; \
    const uint grp = wgi * gtot / nwg + sg / ksplit, ks = sg % ksplit, nb = k / 256; \
    const uint b0 = nb * ks / ksplit, b1 = nb * (ks + 1) / ksplit; \
    if (grp < gend) PREFETCH_BLOCK(w, grp, b0, nb, PB) \
    TABLES(N8, G8, N4, G4, NK, NM, NT) \
    __local float part[RM * WGT]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float accr[RM]; \
    __global const uchar *xp[RM]; \
    _Pragma("unroll") for (uint r = 0; r < RM; r++) { accr[r] = 0.0f; xp[r] = xqg + (ulong)(r < m ? r : 0) * XSTR(k); } \
    if (grp < gend) ROWLOOPR(T, PB, A, C, BODYR, SCHEME, RM, w, grp, b0, b1, nb) \
    if (ksplit > 1) { \
        _Pragma("unroll") for (uint r = 0; r < RM; r++) part[r * WGT + sg * 16 + lane] = accr[r]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (ks == 0) for (uint j = 1; j < ksplit; j++) { _Pragma("unroll") for (uint r = 0; r < RM; r++) accr[r] += part[r * WGT + (sg + j) * 16 + lane]; } \
    } \
    if (ks == 0 && grp < gend) { _Pragma("unroll") for (uint r = 0; r < RM; r++) if (r < m) y[(ulong)r * rows + y_off + grp * 16 + lane] = CONV(accr[r]); } \
}

#define F32ID(x) (x)
#define DEFINE_MVR(RM, T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
    DEFINE_MVR_GEN(mvr##RM##_##T, ushort, to_bf, RM, T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT)
#define DEFINE_MVRF(RM, T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
    DEFINE_MVR_GEN(mvrf##RM##_##T, float, F32ID, RM, T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT)

#define DEFINE_TYPE_R(T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
    DEFINE_MVR(2, T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
    DEFINE_MVR(4, T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
    DEFINE_MVR(8, T, PB, A, C, BODYR, SCHEME, N8, G8, N4, G4, NK, NM, NT)

DEFINE_TYPE_R(q2_k, 84, 5, 1, BODYR_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE_R(q4_k, 144, 9, 0, BODYR_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE_R(iq4_xs, 136, 8, 2, BODYR_IQ4, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)
DEFINE_TYPE_R(iq2_xxs, 68, 4, 1, BODYR_GEN, s, 256, iq2xxs_grid, 1, iq3s_grid, 128, 1, 1)
DEFINE_TYPE_R(iq2_xs, 76, 4, 3, BODYR_GEN, s, 512, iq2xs_grid, 1, iq3s_grid, 128, 1, 1)
DEFINE_TYPE_R(iq2_s, 84, 5, 1, BODYR_GEN, s, 1024, iq2s_grid, 1, iq3s_grid, 1, 16, 1)
DEFINE_TYPE_R(iq3_xxs, 100, 6, 1, BODYR_GEN, s, 1, iq2xxs_grid, 256, iq3xxs_grid, 128, 1, 1)
DEFINE_TYPE_R(iq3_s, 112, 7, 0, BODYR_GEN, s, 1, iq2xxs_grid, 512, iq3s_grid, 1, 16, 1)
DEFINE_TYPE_R(iq1_m, 56, 3, 2, BODYR_GEN, i, 2048, iq1s_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE_R(q6_k, 212, 13, 1, BODYR_GEN, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE_R(q5_k, 176, 11, 0, BODYR_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE_R(iq4_nl, 144, 9, 0, BODYR_IQ4, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)
DEFINE_TYPE_R(q8_0, 272, 17, 0, BODYR_GEN, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_MVRF(2, q4_k, 144, 9, 0, BODYR_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_MVRF(4, q4_k, 144, 9, 0, BODYR_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_MVRF(8, q4_k, 144, 9, 0, BODYR_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_MVRF(2, iq4_xs, 136, 8, 2, BODYR_IQ4, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)
DEFINE_MVRF(4, iq4_xs, 136, 8, 2, BODYR_IQ4, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)
DEFINE_MVRF(8, iq4_xs, 136, 8, 2, BODYR_IQ4, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)

// Quantizes m activation rows (bf16 [m][k]) to the multi-row kernels' layout: one work-item a 32-block.
__kernel void quant_rows(__global const ushort *x, __global uchar *xqg, uint k, uint m) {
    const uint g = get_global_id(0), nb32 = k / 32;
    if (g >= m * nb32) return;
    const uint r = g / nb32, blk = g % nb32;
    const __global uint4 *p = (const __global uint4 *)(x + (ulong)r * k + blk * 32);
    uint q[8];
    const float sc = quant32c(p[0], p[1], p[2], p[3], q);
    __global uchar *o = xqg + (ulong)r * XSTR(k);
    ((__global uint4 *)(o + blk * 32))[0] = (uint4)(q[0], q[1], q[2], q[3]);
    ((__global uint4 *)(o + blk * 32))[1] = (uint4)(q[4], q[5], q[6], q[7]);
    ((__global float *)(o + k))[blk] = sc;
}

#define DEFINE_TYPE(T, PB, A, C, BODY, DEQ, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
    DEFINE_MV(mv_##T, ushort, STORE_BF, T, PB, A, C, BODY, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
    DEFINE_MV(mvf_##T, float, STORE_F, T, PB, A, C, BODY, SCHEME, N8, G8, N4, G4, NK, NM, NT) \
    DEFINE_DQ(T, PB, A, C, DEQ, SCHEME, N8, G8, N4, G4, NK, NM, NT)

DEFINE_TYPE(q2_k, 84, 5, 1, BODY_GEN, DEQ_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE(q4_k, 144, 9, 0, BODY_GEN, DEQ_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE(iq4_xs, 136, 8, 2, BODY_IQ4, DEQ_IQ4, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)
DEFINE_TYPE(iq2_xxs, 68, 4, 1, BODY_GEN, DEQ_GEN, s, 256, iq2xxs_grid, 1, iq3s_grid, 128, 1, 1)
DEFINE_TYPE(iq2_xs, 76, 4, 3, BODY_GEN, DEQ_GEN, s, 512, iq2xs_grid, 1, iq3s_grid, 128, 1, 1)
DEFINE_TYPE(iq2_s, 84, 5, 1, BODY_GEN, DEQ_GEN, s, 1024, iq2s_grid, 1, iq3s_grid, 1, 16, 1)
DEFINE_TYPE(iq3_xxs, 100, 6, 1, BODY_GEN, DEQ_GEN, s, 1, iq2xxs_grid, 256, iq3xxs_grid, 128, 1, 1)
DEFINE_TYPE(iq3_s, 112, 7, 0, BODY_GEN, DEQ_GEN, s, 1, iq2xxs_grid, 512, iq3s_grid, 1, 16, 1)
DEFINE_TYPE(iq1_m, 56, 3, 2, BODY_GEN, DEQ_GEN, i, 2048, iq1s_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE(q6_k, 212, 13, 1, BODY_GEN, DEQ_GEN, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE(q5_k, 176, 11, 0, BODY_GEN, DEQ_GEN, u, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_TYPE(iq4_nl, 144, 9, 0, BODY_IQ4, DEQ_IQ4, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)
DEFINE_TYPE(q8_0, 272, 17, 0, BODY_GEN, DEQ_GEN, s, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)

// The row loop of the type with id `ty` (see mvm); needs tb (tables pointing at the staged ones), lane, xq, xd, acc in scope.
#define ROWSWITCH(ty, w, grp, b0, b1, nb) \
    switch (ty) { \
    case 0: ROWLOOP(q2_k, 84, 5, 1, BODY_GEN, u, w, grp, b0, b1, nb) break; \
    case 1: ROWLOOP(q4_k, 144, 9, 0, BODY_GEN, u, w, grp, b0, b1, nb) break; \
    case 2: ROWLOOP(iq4_xs, 136, 8, 2, BODY_IQ4, s, w, grp, b0, b1, nb) break; \
    case 3: tb.g8 = g8xxs; ROWLOOP(iq2_xxs, 68, 4, 1, BODY_GEN, s, w, grp, b0, b1, nb) break; \
    case 4: tb.g8 = g8xs; ROWLOOP(iq2_xs, 76, 4, 3, BODY_GEN, s, w, grp, b0, b1, nb) break; \
    case 5: tb.g8 = g8s; ROWLOOP(iq2_s, 84, 5, 1, BODY_GEN, s, w, grp, b0, b1, nb) break; \
    case 6: tb.g4 = g4xxs; ROWLOOP(iq3_xxs, 100, 6, 1, BODY_GEN, s, w, grp, b0, b1, nb) break; \
    default: tb.g4 = g4s; ROWLOOP(iq3_s, 112, 7, 0, BODY_GEN, s, w, grp, b0, b1, nb) break; \
    }

// ---- fused multi-projection matvec: up to 3 segments (weights, output, rows, offset, type each) over one quantized activation, one launch ----
// type ids: 0 q2_k, 1 q4_k, 2 iq4_xs, 3 iq2_xxs, 4 iq2_xs, 5 iq2_s, 6 iq3_xxs, 7 iq3_s. tmask: bit t set when some segment has type t (its tables get staged).
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void mvm(P w0, P w1, P w2, __global const ushort *x, __global ushort *y0, __global ushort *y1, __global ushort *y2, uint k, uint ksplit,
                  uint rows0, uint rows1, uint rows2, uint off0, uint off1, uint off2, uint types, uint tmask, __local char *xq, __local float *xd) {
    __local ulong g8xxs[256], g8xs[512], g8s[1024], k6[128];
    __local uint g4xxs[256], g4s[512], m16[16];
    __local ushort ttk[256];
    __local float part[WGT];
    if (tmask & 0x08u) { STAGEP(g8xxs, iq2xxs_grid, 256) }
    if (tmask & 0x10u) { STAGEP(g8xs, iq2xs_grid, 512) }
    if (tmask & 0x20u) { STAGEP(g8s, iq2s_grid, 1024) }
    if (tmask & 0x58u) { STAGEP(k6, ksigns64, 128) }  // iq2_xxs, iq2_xs, iq3_xxs
    if (tmask & 0x40u) { STAGEP(g4xxs, iq3xxs_grid, 256) }
    if (tmask & 0x80u) { STAGEP(g4s, iq3s_grid, 512) }
    if (tmask & 0xa0u) { STAGEP(m16, kmask16, 16) }  // iq2_s, iq3_s
    if (tmask & 0x04u) for (uint i_ = get_local_id(0); i_ < 256; i_ += WGT) ttk[i_] = (ushort)(((uint)kvalues_iq4nl[i_ & 15u] & 255u) | (((uint)kvalues_iq4nl[(i_ >> 4) & 15u] & 255u) << 8));
    const uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    QUANT_LOOP
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint nb = k / 256, gr0 = rows0 / 16, gr1 = rows1 / 16, gr2 = rows2 / 16;
    const uint nwg = get_num_groups(0), wgi = get_group_id(0), gtot = gr0 + gr1 + gr2, gend = (wgi + 1) * gtot / nwg;  // even share of the row groups
    const uint gg = wgi * gtot / nwg + sg / ksplit, ks = sg % ksplit;
    const uint b0 = nb * ks / ksplit, b1 = nb * (ks + 1) / ksplit;
    uint seg = 0, grp = gg;
    if (grp >= gr0) { seg = 1; grp -= gr0; }
    if (seg == 1 && grp >= gr1) { seg = 2; grp -= gr1; }
    const bool live = gg < gend;
    const uint ty = (types >> (8 * seg)) & 255u;
    P w = seg == 0 ? w0 : (seg == 1 ? w1 : w2);
    Tab tb;
    tb.g8 = g8xxs; tb.g4 = g4xxs; tb.k6 = k6; tb.m16 = m16; tb.tk = ttk;
    float acc = 0.0f;
    if (live) {
        ROWSWITCH(ty, w, grp, b0, b1, nb)
    }
    if (ksplit > 1) {
        part[sg * 16 + lane] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) for (uint j = 1; j < ksplit; j++) acc += part[(sg + j) * 16 + lane];
    }
    if (ks == 0 && live) {
        __global ushort *y = seg == 0 ? y0 : (seg == 1 ? y1 : y2);
        const uint off = seg == 0 ? off0 : (seg == 1 ? off1 : off2);
        y[off + grp * 16 + lane] = to_bf(acc);
    }
}

// MLP gate, up and SwiGLU in one launch: act[row] = bf16(silu(g) * u) with g = bf16(Wg x), u = bf16(Wu x) (the two kernels' rounding). Each sub-group owns 16 rows of
// both matrices (types tg | tu << 8 in `types`), rows a multiple of 16, one K range a row group. A row group streams its gate blocks, then its up blocks.
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void mvgu(P wg, P wu, __global const ushort *x, __global ushort *act, uint k, uint rows, uint types, uint tmask, __local char *xq, __local float *xd) {
    __local ulong g8xxs[256], g8xs[512], g8s[1024], k6[128];
    __local uint g4xxs[256], g4s[512], m16[16];
    __local ushort ttk[256];
    if (tmask & 0x08u) { STAGEP(g8xxs, iq2xxs_grid, 256) }
    if (tmask & 0x10u) { STAGEP(g8xs, iq2xs_grid, 512) }
    if (tmask & 0x20u) { STAGEP(g8s, iq2s_grid, 1024) }
    if (tmask & 0x58u) { STAGEP(k6, ksigns64, 128) }
    if (tmask & 0x40u) { STAGEP(g4xxs, iq3xxs_grid, 256) }
    if (tmask & 0x80u) { STAGEP(g4s, iq3s_grid, 512) }
    if (tmask & 0xa0u) { STAGEP(m16, kmask16, 16) }
    if (tmask & 0x04u) for (uint i_ = get_local_id(0); i_ < 256; i_ += WGT) ttk[i_] = (ushort)(((uint)kvalues_iq4nl[i_ & 15u] & 255u) | (((uint)kvalues_iq4nl[(i_ >> 4) & 15u] & 255u) << 8));
    const uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    QUANT_LOOP
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint nb = k / 256, nwg = get_num_groups(0), wgi = get_group_id(0), gtot = rows / 16, gend = (wgi + 1) * gtot / nwg;
    const uint grp = wgi * gtot / nwg + sg;
    if (grp < gend) {
        Tab tb;
        tb.g8 = g8xxs; tb.g4 = g4xxs; tb.k6 = k6; tb.m16 = m16; tb.tk = ttk;
        float accg = 0.0f, accu = 0.0f;
        { float acc = 0.0f; ROWSWITCH(types & 255u, wg, grp, 0u, nb, nb) accg = acc; }
        { float acc = 0.0f; ROWSWITCH((types >> 8) & 255u, wu, grp, 0u, nb, nb) accu = acc; }
        const float a = bf(to_bf(accg)), u = bf(to_bf(accu));
        act[grp * 16 + lane] = to_bf(a / (1.0f + exp(-a)) * u);
    }
}

// ---- multi-row variants of the fused kernels (RM = 2 or 4 activation rows; row-invariant like mvr) ----
#define ROWSWITCHR(RM, ty, w, grp, b0, b1, nb) \
    switch (ty) { \
    case 0: ROWLOOPR(q2_k, 84, 5, 1, BODYR_GEN, u, RM, w, grp, b0, b1, nb) break; \
    case 1: ROWLOOPR(q4_k, 144, 9, 0, BODYR_GEN, u, RM, w, grp, b0, b1, nb) break; \
    case 2: ROWLOOPR(iq4_xs, 136, 8, 2, BODYR_IQ4, s, RM, w, grp, b0, b1, nb) break; \
    case 3: tb.g8 = g8xxs; ROWLOOPR(iq2_xxs, 68, 4, 1, BODYR_GEN, s, RM, w, grp, b0, b1, nb) break; \
    case 4: tb.g8 = g8xs; ROWLOOPR(iq2_xs, 76, 4, 3, BODYR_GEN, s, RM, w, grp, b0, b1, nb) break; \
    case 5: tb.g8 = g8s; ROWLOOPR(iq2_s, 84, 5, 1, BODYR_GEN, s, RM, w, grp, b0, b1, nb) break; \
    case 6: tb.g4 = g4xxs; ROWLOOPR(iq3_xxs, 100, 6, 1, BODYR_GEN, s, RM, w, grp, b0, b1, nb) break; \
    default: tb.g4 = g4s; ROWLOOPR(iq3_s, 112, 7, 0, BODYR_GEN, s, RM, w, grp, b0, b1, nb) break; \
    }

#define STAGE_FUSED \
    __local ulong g8xxs[256], g8xs[512], g8s[1024], k6[128]; \
    __local uint g4xxs[256], g4s[512], m16[16]; \
    __local ushort ttk[256]; \
    if (tmask & 0x08u) { STAGEP(g8xxs, iq2xxs_grid, 256) } \
    if (tmask & 0x10u) { STAGEP(g8xs, iq2xs_grid, 512) } \
    if (tmask & 0x20u) { STAGEP(g8s, iq2s_grid, 1024) } \
    if (tmask & 0x58u) { STAGEP(k6, ksigns64, 128) } \
    if (tmask & 0x40u) { STAGEP(g4xxs, iq3xxs_grid, 256) } \
    if (tmask & 0x80u) { STAGEP(g4s, iq3s_grid, 512) } \
    if (tmask & 0xa0u) { STAGEP(m16, kmask16, 16) } \
    if (tmask & 0x04u) for (uint i_ = get_local_id(0); i_ < 256; i_ += WGT) ttk[i_] = (ushort)(((uint)kvalues_iq4nl[i_ & 15u] & 255u) | (((uint)kvalues_iq4nl[(i_ >> 4) & 15u] & 255u) << 8)); \
    barrier(CLK_LOCAL_MEM_FENCE);

// mvm for m rows: y_seg[r * rows_seg + off_seg + row].
#define DEFINE_MVMR(RM) \
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) \
__kernel void mvmr##RM(P w0, P w1, P w2, __global const uchar *xqg, __global ushort *y0, __global ushort *y1, __global ushort *y2, uint k, uint ksplit, \
                       uint rows0, uint rows1, uint rows2, uint off0, uint off1, uint off2, uint types, uint tmask, uint m) { \
    STAGE_FUSED \
    __local float part[RM * WGT]; \
    const uint sg = get_sub_group_id(), lane = get_sub_group_local_id(); \
    const uint nb = k / 256, gr0 = rows0 / 16, gr1 = rows1 / 16, gr2 = rows2 / 16; \
    const uint nwg = get_num_groups(0), wgi = get_group_id(0), gtot = gr0 + gr1 + gr2, gend = (wgi + 1) * gtot / nwg; \
    const uint gg = wgi * gtot / nwg + sg / ksplit, ks = sg % ksplit; \
    const uint b0 = nb * ks / ksplit, b1 = nb * (ks + 1) / ksplit; \
    uint seg = 0, grp = gg; \
    if (grp >= gr0) { seg = 1; grp -= gr0; } \
    if (seg == 1 && grp >= gr1) { seg = 2; grp -= gr1; } \
    const bool live = gg < gend; \
    const uint ty = (types >> (8 * seg)) & 255u; \
    P w = seg == 0 ? w0 : (seg == 1 ? w1 : w2); \
    Tab tb; \
    tb.g8 = g8xxs; tb.g4 = g4xxs; tb.k6 = k6; tb.m16 = m16; tb.tk = ttk; \
    float accr[RM]; \
    __global const uchar *xp[RM]; \
    _Pragma("unroll") for (uint r = 0; r < RM; r++) { accr[r] = 0.0f; xp[r] = xqg + (ulong)(r < m ? r : 0) * XSTR(k); } \
    if (live) { ROWSWITCHR(RM, ty, w, grp, b0, b1, nb) } \
    if (ksplit > 1) { \
        _Pragma("unroll") for (uint r = 0; r < RM; r++) part[r * WGT + sg * 16 + lane] = accr[r]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (ks == 0) for (uint j = 1; j < ksplit; j++) { _Pragma("unroll") for (uint r = 0; r < RM; r++) accr[r] += part[r * WGT + (sg + j) * 16 + lane]; } \
    } \
    if (ks == 0 && live) { \
        __global ushort *y = seg == 0 ? y0 : (seg == 1 ? y1 : y2); \
        const uint off = seg == 0 ? off0 : (seg == 1 ? off1 : off2), rs = seg == 0 ? rows0 : (seg == 1 ? rows1 : rows2); \
        _Pragma("unroll") for (uint r = 0; r < RM; r++) if (r < m) y[(ulong)r * rs + off + grp * 16 + lane] = to_bf(accr[r]); \
    } \
}
DEFINE_MVMR(2)
DEFINE_MVMR(4)
DEFINE_MVMR(8)

// mvgu for m rows: act[r * rows + row].
#define DEFINE_MVGUR(RM) \
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) \
__kernel void mvgur##RM(P wg, P wu, __global const uchar *xqg, __global ushort *act, uint k, uint rows, uint types, uint tmask, uint m) { \
    STAGE_FUSED \
    const uint sg = get_sub_group_id(), lane = get_sub_group_local_id(); \
    const uint nb = k / 256, nwg = get_num_groups(0), wgi = get_group_id(0), gtot = rows / 16, gend = (wgi + 1) * gtot / nwg; \
    const uint grp = wgi * gtot / nwg + sg; \
    if (grp < gend) { \
        Tab tb; \
        tb.g8 = g8xxs; tb.g4 = g4xxs; tb.k6 = k6; tb.m16 = m16; tb.tk = ttk; \
        float accg[RM], accu[RM]; \
        __global const uchar *xp[RM]; \
        _Pragma("unroll") for (uint r = 0; r < RM; r++) { xp[r] = xqg + (ulong)(r < m ? r : 0) * XSTR(k); } \
        { float accr[RM]; _Pragma("unroll") for (uint r = 0; r < RM; r++) accr[r] = 0.0f; ROWSWITCHR(RM, types & 255u, wg, grp, 0u, nb, nb) _Pragma("unroll") for (uint r = 0; r < RM; r++) accg[r] = accr[r]; } \
        { float accr[RM]; _Pragma("unroll") for (uint r = 0; r < RM; r++) accr[r] = 0.0f; ROWSWITCHR(RM, (types >> 8) & 255u, wu, grp, 0u, nb, nb) _Pragma("unroll") for (uint r = 0; r < RM; r++) accu[r] = accr[r]; } \
        _Pragma("unroll") for (uint r = 0; r < RM; r++) { \
            const float a = bf(to_bf(accg[r])), u = bf(to_bf(accu[r])); \
            if (r < m) act[(ulong)r * rows + grp * 16 + lane] = to_bf(a / (1.0f + exp(-a)) * u); \
        } \
    } \
}
DEFINE_MVGUR(2)
DEFINE_MVGUR(4)
DEFINE_MVGUR(8)

// ---- prefill GEMM on the matrix engine (DPAS), for many activation rows: Y[R][N] = X[R][K] W^T ----
// The weights of the whole matrix are decoded once into fp16 DPAS B fragments, already scaled (w = a * q [- b]; the fp16 rounding of a weight is 2^-12 relative, far below the
// quantization), laid out Wv [kt][tile = 16 columns][8 pairs][16 lanes] (kt = a 16-wide k tile). The activations become fp16 (exact for the bf16 values in range; clamped at the
// fp16 limit). The GEMM is a plain DPAS chain over the k tiles in order with fp32 accumulation, so a row's value depends only on its own activations and the column's weights,
// not on the row panel, R or how a prompt is chunked. IQ1_M has no prefill kernel (a single small tensor; the engine falls back to the multi-row matvec windows).

inline uint half_pair(float lo, float hi) { return (uint)as_ushort(convert_half_rte(lo)) | ((uint)as_ushort(convert_half_rte(hi)) << 16); }
inline float sx8(uint q, int j) { return (float)(char)((q >> (8 * j)) & 255u); }
inline float zx8(uint q, int j) { return (float)((q >> (8 * j)) & 255u); }

// One unit of 16 weights of output channel n = g * 16 + lane -> its 8 fp16 pair words at Wv[n][kt * 8 ..]: row-major W [N][K] (K = nb * 256), k contiguous, the layout the
// 2D-block-load GEMM reads. SU: unsigned bytes with a minimum (w = a q - b0), else signed (w = a q). `nb` is the kernel's block count a row.
#define PF_WRITE_UNIT(d, SU, kt, g, lane, ntiles) \
    { \
        const uint q_[4] = {d.q0, d.q1, d.q2, d.q3}; \
        uint8 w8_; \
        _Pragma("unroll") for (int k_ = 0; k_ < 4; k_++) { \
            float v_[4]; \
            _Pragma("unroll") for (int j_ = 0; j_ < 4; j_++) v_[j_] = (SU) ? fma(d.a, zx8(q_[k_], j_), -d.b0) : d.a * sx8(q_[k_], j_); \
            w8_[2 * k_] = half_pair(v_[0], v_[1]); \
            w8_[2 * k_ + 1] = half_pair(v_[2], v_[3]); \
        } \
        if (((kt) & 1) == 0) wprev = w8_; \
        else { /* two units = 16 words a lane: transpose through local memory so that each message writes one 64 B line of one row */ \
            __local uint *st_ = pfs + ((ulong)get_sub_group_id() * 16 + (lane)) * 17; \
            _Pragma("unroll") for (int e_ = 0; e_ < 8; e_++) { st_[e_] = wprev[e_]; st_[8 + e_] = w8_[e_]; } \
            sub_group_barrier(CLK_LOCAL_MEM_FENCE); \
            _Pragma("unroll") for (int r_ = 0; r_ < 16; r_++) \
                Wv[(ulong)((g) * 16 + r_) * (nb * 128) + (ulong)((kt) - 1) * 8 + (lane)] = pfs[((ulong)get_sub_group_id() * 16 + r_) * 17 + (lane)]; \
            sub_group_barrier(CLK_LOCAL_MEM_FENCE); \
        } \
    }

#define PFD_GEN(T, SU, WV, g, b, lane, ntiles) \
    _Pragma("unroll") for (uint u = 0; u < 16; u++) { \
        const DL d = dec_##T(WV, u, tb); \
        PF_WRITE_UNIT(d, SU, (b) * 16 + u, g, lane, ntiles) \
    }
#define PFD_IQ4(T, SU, WV, g, b, lane, ntiles) \
    _Pragma("unroll") for (uint ib = 0; ib < 8; ib++) { \
        DL lo, hi; \
        dec2_##T(WV, ib, tb, &lo, &hi); \
        PF_WRITE_UNIT(lo, SU, (b) * 16 + 2 * ib, g, lane, ntiles) \
        PF_WRITE_UNIT(hi, SU, (b) * 16 + 2 * ib + 1, g, lane, ntiles) \
    }

// Decode kernel: one sub-group a (row group, block); grid ceil(groups * nb / NSGW) work-groups.
#define DEFINE_PFDEC(T, PB, A, C, PFD, SU, N8, G8, N4, G4, NK, NM, NT) \
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) \
__kernel void pfdec_##T(P w, __global uint *Wv, uint nb, uint ngroups) { \
    TABLES(N8, G8, N4, G4, NK, NM, NT) \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const uint lane = get_sub_group_local_id(), i = get_group_id(0) * NSGW + get_sub_group_id(); \
    if (i >= ngroups * nb) return; \
    const uint g = i / nb, b = i % nb; \
    uint8 wprev; \
    __local uint pfs[NSGW * 16 * 17]; \
    uint wv[PB / 4]; \
    LOADBLK(wv, w + ((ulong)g * nb + b) * (16 * PB), lane, A, C) \
    PFD(T, SU, wv, g, b, lane, ngroups) \
}

DEFINE_PFDEC(q2_k, 84, 5, 1, PFD_GEN, 1, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_PFDEC(q4_k, 144, 9, 0, PFD_GEN, 1, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_PFDEC(iq4_xs, 136, 8, 2, PFD_IQ4, 0, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)
DEFINE_PFDEC(iq2_xxs, 68, 4, 1, PFD_GEN, 0, 256, iq2xxs_grid, 1, iq3s_grid, 128, 1, 1)
DEFINE_PFDEC(iq2_xs, 76, 4, 3, PFD_GEN, 0, 512, iq2xs_grid, 1, iq3s_grid, 128, 1, 1)
DEFINE_PFDEC(iq2_s, 84, 5, 1, PFD_GEN, 0, 1024, iq2s_grid, 1, iq3s_grid, 1, 16, 1)
DEFINE_PFDEC(iq3_xxs, 100, 6, 1, PFD_GEN, 0, 1, iq2xxs_grid, 256, iq3xxs_grid, 128, 1, 1)
DEFINE_PFDEC(iq3_s, 112, 7, 0, PFD_GEN, 0, 1, iq2xxs_grid, 512, iq3s_grid, 1, 16, 1)
DEFINE_PFDEC(q6_k, 212, 13, 1, PFD_GEN, 0, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_PFDEC(q5_k, 176, 11, 0, PFD_GEN, 1, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)
DEFINE_PFDEC(iq4_nl, 144, 9, 0, PFD_IQ4, 0, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 256)
DEFINE_PFDEC(q8_0, 272, 17, 0, PFD_GEN, 0, 1, iq2xxs_grid, 1, iq3s_grid, 1, 1, 1)

// Activation prep: X [R][K] bf16 -> Xt [K][Rp] fp16 (k-major, the 2D-block-load GEMM's B operand; tokens R .. Rp zero). One work-item a (token, 8 k): 16 B read, 8 two-byte writes
// that are contiguous across the tokens of a sub-group. Grid (Rp / 64, K / 8), local 64.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void pf_prep(__global const ushort *x, __global ushort *xt, uint R, uint K, uint Rp) {
    const uint row = get_global_id(0), kb = get_global_id(1);
    ushort8 v = (ushort8)(0);
    if (row < R) v = vload8(0, x + (ulong)row * K + kb * 8);
    for (uint j = 0; j < 8; j++) xt[(ulong)(kb * 8 + j) * Rp + row] = as_ushort(convert_half_rte(clamp(bf(v[j]), -65504.0f, 65504.0f)));
}

// Test kernel for the activation quantization (the same quant32 the matvec runs): xq[K] bytes, then xd[K/32] fp32 at byte offset K.
__kernel void quant_x(__global const ushort *x, __global char *xq, __global float *xd, uint k) {
    __local char q[MAXK];
    __local float d[MAXK / 32];
    const uint blk = get_global_id(0);
    if (blk * 32 < k) quant32(x, blk, q, d);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (blk * 32 < k) {
        for (uint i = 0; i < 32; i++) xq[blk * 32 + i] = q[blk * 32 + i];
        xd[blk] = d[blk];
    }
}

// bf16-weight matvec for the small gate projections (in_proj_a/in_proj_b: 48 rows x 5120): y[y_off + row] = bf16(sum x[i] * w[row][i]). One work-group of
// 4 sub-groups a row (one sub-group a row left it latency bound), each a quarter of the inputs with 16-byte loads, fp32 accumulation, the quarters
// added in order. in_dim a multiple of 512. mv_bf16x2 runs two such projections of the same input (rows each) in one launch: group g < rows is
// row g of w0 -> y0, else row g - rows of w1 -> y1.
inline float bf16_row_part(__global const ushort *wr, __global const ushort *x, uint in_dim, uint sg, uint lane) {
    const uint chunks = in_dim / 8, per = chunks / 4;  // 8 columns a chunk
    float a0 = 0.0f, a1 = 0.0f;
    for (uint c = sg * per + lane; c < (sg + 1) * per; c += 32) {
        const float8 w0 = as_float8(convert_uint8(vload8(c, wr)) << 16);
        const float8 x0 = as_float8(convert_uint8(vload8(c, x)) << 16);
        a0 += dot(w0.s0123, x0.s0123) + dot(w0.s4567, x0.s4567);
        if (c + 16 < (sg + 1) * per) {
            const float8 w1 = as_float8(convert_uint8(vload8(c + 16, wr)) << 16);
            const float8 x1 = as_float8(convert_uint8(vload8(c + 16, x)) << 16);
            a1 += dot(w1.s0123, x1.s0123) + dot(w1.s4567, x1.s4567);
        }
    }
    return sub_group_reduce_add(a0 + a1);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void mv_bf16(__global const ushort *w, __global const ushort *x, __global ushort *y, uint in_dim, uint y_off, uint rows) {
    __local float part[4];
    const uint row = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const float s = bf16_row_part(w + (ulong)row * in_dim, x, in_dim, sg, lane);
    if (lane == 0) part[sg] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0 && lane == 0) y[y_off + row] = to_bf(part[0] + part[1] + part[2] + part[3]);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void mv_bf16x2(__global const ushort *w0, __global const ushort *w1, __global const ushort *x, __global ushort *y0, __global ushort *y1, uint in_dim, uint rows) {
    __local float part[4];
    const uint g = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const uint row = g < rows ? g : g - rows;
    const float s = bf16_row_part((g < rows ? w0 : w1) + (ulong)row * in_dim, x, in_dim, sg, lane);
    if (lane == 0) part[sg] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0 && lane == 0) (g < rows ? y0 : y1)[row] = to_bf(part[0] + part[1] + part[2] + part[3]);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnorm_f32w(__global const ushort *x, __global const float *w, __global ushort *y, uint n, float eps) {
    __local float part[4];
    const uint lid = get_local_id(0);
    float ss = 0.0f;
    for (uint i = lid; i < n; i += 64) {
        const float v = bf(x[i]);
        ss += v * v;
    }
    ss = sub_group_reduce_add(ss);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float scale = rsqrt((part[0] + part[1] + part[2] + part[3]) / (float)n + eps);
    for (uint i = lid; i < n; i += 64) y[i] = to_bf(bf(x[i]) * scale * w[i]);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_prep_f32w(__global const ushort *qg, __global const ushort *kraw, __global const float *qw, __global const float *kw,
                             __global const float *rope, __global ushort *qo, __global ushort *kc, uint k_off, float eps) {
    __local float part[16];
    __local float nv[256];
    const uint head = get_group_id(0), i = get_local_id(0);
    const bool is_q = head < 24;
    const float x = is_q ? bf(qg[head * 512 + i]) : bf(kraw[(head - 24) * 256 + i]);
    const float ss = sub_group_reduce_add(x * x);
    if (get_sub_group_local_id() == 0) part[get_sub_group_id()] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    float total = 0.0f;
    for (int j = 0; j < 16; j++) total += part[j];
    const float r = rsqrt(total / 256.0f + eps);
    nv[i] = bf(to_bf(x * r * (is_q ? qw[i] : kw[i])));
    barrier(CLK_LOCAL_MEM_FENCE);
    float o = nv[i];
    if (i < 32) o = nv[i] * rope[i] - nv[i + 32] * rope[32 + i];
    else if (i < 64) o = nv[i] * rope[i - 32] + nv[i - 32] * rope[32 + i - 32];
    if (is_q) qo[head * 256 + i] = to_bf(o);
    else kc[k_off + (head - 24) * 256 + i] = to_bf(o);
}

// Bandwidth probe: every sub-group streams `n16` consecutive 16-byte units per iteration step, 4 units in flight a lane; sums them (so nothing is dead code).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void bw_probe(__global const uint4 *w, __global uint *out, uint units_per_sg) {
    const uint sg = get_group_id(0) * 4 + get_sub_group_id(), lane = get_sub_group_local_id();
    __global const uint4 *p = w + (ulong)sg * units_per_sg;
    uint acc = 0;
    for (uint i = 0; i < units_per_sg; i += 64) {
        const uint4 a = p[i + lane], b = p[i + 16 + lane], c = p[i + 32 + lane], d = p[i + 48 + lane];
        acc += a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}

// Probe 2: the matvec's exact load pattern and balanced work-group layout, no staging, no quantization, no decode: sub-group g streams `nb` blocks of PB=112 B x 16 lanes
// (7 uint4 pieces [piece][lane]) from its group's region, double buffered like ROWLOOP.
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void bw_probe2(__global const uchar *w, __global uint *out, uint nb, uint gtot) {
    const uint sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const uint nwg = get_num_groups(0), wgi = get_group_id(0), gend = (wgi + 1) * gtot / nwg;
    const uint grp = wgi * gtot / nwg + sg;
    uint acc = 0;
    if (grp < gend) {
        __global const uchar *gb = w + (ulong)grp * nb * 1792;
        uint4 pa[7], pb[7];
        for (int p = 0; p < 7; p++) pa[p] = *(__global const uint4 *)(gb + (p * 16 + lane) * 16);
        for (uint b = 0; b < nb; b += 2) {
            if (b + 1 < nb) for (int p = 0; p < 7; p++) pb[p] = *(__global const uint4 *)(gb + (ulong)(b + 1) * 1792 + (p * 16 + lane) * 16);
            for (int p = 0; p < 7; p++) acc += pa[p].x ^ pa[p].y ^ pa[p].z ^ pa[p].w;
            if (b + 1 < nb) {
                if (b + 2 < nb) for (int p = 0; p < 7; p++) pa[p] = *(__global const uint4 *)(gb + (ulong)(b + 2) * 1792 + (p * 16 + lane) * 16);
                for (int p = 0; p < 7; p++) acc += pb[p].x ^ pb[p].y ^ pb[p].z ^ pb[p].w;
            }
        }
    }
    if (acc == 0x12345678u) out[0] = acc;
}

// Occupancy probe: each work-group registers itself, spins ~spin iterations, deregisters; maxv records the peak number of resident work-groups.
__attribute__((reqd_work_group_size(WGT, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void occ_probe(__global volatile uint *cnt, __global volatile uint *maxv, uint spin, uint slm_use, __local uchar *dyn) {
    __local uint x[1];
    if (get_local_id(0) == 0) {
        const uint now = atomic_add(cnt, 1u) + 1u;
        atomic_max(maxv, now);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float a = (float)get_local_id(0);
    for (uint i = 0; i < spin; i++) a = fma(a, 1.0001f, 0.5f);
    if (slm_use && a == 1234.5f) dyn[get_local_id(0)] = 1;
    if (a == 1234.5f) x[0] = 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0) atomic_sub(cnt, 1u);
}

// bf16-weight gate projection for m tokens: y[t * rows + y_off + row] (grid {rows, m}); the same arithmetic per token as mv_bf16.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void mv_bf16r(__global const ushort *w, __global const ushort *x, __global ushort *y, uint in_dim, uint y_off, uint rows) {
    __local float part[4];
    const uint row = get_group_id(0), t = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
    const float s = bf16_row_part(w + (ulong)row * in_dim, x + (ulong)t * in_dim, in_dim, sg, lane);
    if (lane == 0) part[sg] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0 && lane == 0) y[(ulong)t * rows + y_off + row] = to_bf(part[0] + part[1] + part[2] + part[3]);
}

