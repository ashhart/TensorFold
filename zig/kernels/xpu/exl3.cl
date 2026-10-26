// SPDX-License-Identifier: Apache-2.0

// EXL3 (ExLlamaV3 trellis) projections on Xe2: port of zig/kernels/metal/core/exl3_projection.metal.

// y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh; H is 128x128 Sylvester Hadamard / sqrt(128).

// Model-agnostic launches use a descriptor table; codebook mul1; 2 to 8 bits (docs/recipes/exl3.md).

// Sub-group 16 lanes (Xe2): same Hadamard butterfly as Metal 32-lane mapping; matvec decodes tiles in registers.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG 16
#ifndef ROWS
#define ROWS 1
#endif
#define INV_SQRT128 0.08838834764831845f

typedef struct {
    ulong trellis, suh, svh, x, rot, sums, out;
    uint k, n, halves, rows;
    uint x_stride, out_stride, out_type, x_f16;
} exl3_desc;

#ifdef EXL3_ABS
#define AT(T, off) ((__global T *)(off))
#else
#define AT(T, off) ((__global T *)(base + (off)))
#endif

inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    if (isnan(f)) return (ushort)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

inline float load_x(__global const ushort *x, ulong i, uint f16) { return f16 ? vload_half(i, (__global const half *)x) : bf(x[i]); }

inline void store_y(__global uchar *out, ulong at, uint type, float y) {
    if (type == 1) ((__global float *)out)[at] = y;
    else if (type == 2) vstore_half_rte(y, at, (__global half *)out);
    else ((__global ushort *)out)[at] = to_bf(y);
}

inline void hadamard128(float v[8], uint lane) {
    for (uint d = 1; d < SG; d *= 2) {
        for (uint j = 0; j < 8; j++) {
            const float peer = sub_group_shuffle_xor(v[j], d);
            v[j] = (lane & d) ? peer - v[j] : v[j] + peer;
        }
    }
    for (uint d = 1; d < 8; d *= 2) {
        for (uint j = 0; j < 8; j++) {
            if (j & d) continue;
            const float a = v[j], b = v[j + d];
            v[j] = a + b;
            v[j + d] = a - b;
        }
    }
}

inline uint stream_word(__global const ushort *tile, uint i) { return (uint)tile[2 * i] | ((uint)tile[2 * i + 1] << 16); }

inline uint state_map(uint halves, uint p) {
    const uint ring = 128 * halves;
    const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
    const uint start = (end + ring - 16) % ring;
    const uint wi = start / 32, off = start % 32, next = wi + 1 == ring / 32 ? 0 : wi + 1;
    return wi | (off << 6) | (next << 11);
}

inline uint mapped_state(__global const ushort *tile, uint map) {
    const uint wi = map & 63u, off = (map >> 6) & 31u;
    const uint lo = stream_word(tile, wi);
    const uint hi = stream_word(tile, map >> 11);
    const uint carry = off ? hi >> (32 - off) : 0u;
    return ((lo << off) | carry) >> 16;
}

inline half mul1_half(uint s) {
    const uint h = dot_acc_sat_4x8packed_uu_uint(s * 0x83DCD12Du, 0x01010101u, 0x6400u);
    return fma(as_half((ushort)h), as_half((ushort)0x1EEE), as_half((ushort)0xC931));
}
inline float mul1_value(uint s) { return (float)mul1_half(s); }

inline uint lane_value(uint c, uint v) { return 32 * (c % 8) + 8 * (v / 4) + 4 * (c / 8) + (v % 4); }
inline uint lane_row(uint v) { return 2 * (v / 4) + (v & 1) + 8 * ((v >> 1) & 1); }

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_decode_tiles(__global const ushort *trellis, __global uint *states, __global half *wq, uint halves) {
    const uint tn = get_group_id(0), tk = get_group_id(1), c = get_sub_group_local_id();
    const uint tiles_n = get_num_groups(0), ti = tk * tiles_n + tn;
    __global const ushort *tile = trellis + (ulong)ti * 8 * halves;
    for (uint v = 0; v < 16; v++) {
        const uint p = lane_value(c, v);
        const uint s = mapped_state(tile, state_map(halves, p));
        states[(ulong)ti * 256 + p] = s;
        vstore_half(mul1_value(s), ((ulong)tk * 16 + lane_row(v)) * tiles_n * 16 + tn * 16 + c, wq);
    }
}

__kernel void exl3_mul1_check(__global float *out) {
    const uint i = get_global_id(0);
    out[i] = mul1_value(i);
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_rotate_in(__global const uchar *base, __global const exl3_desc *desc) {
    const exl3_desc d = desc[get_group_id(2)];
    const uint blk = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
    if (blk * 128 >= d.k || row >= d.rows) return;
    __global const ushort *x = AT(const ushort, d.x) + (ulong)row * d.x_stride;
    __global const half *suh = AT(const half, d.suh);
    float v[8];
    for (uint j = 0; j < 8; j++) {
        const uint i = blk * 128 + lane + SG * j;
        v[j] = load_x(x, i, d.x_f16) * vload_half(i, suh);
    }
    hadamard128(v, lane);
    __global float *rot = AT(float, d.rot) + (ulong)row * d.k;
    for (uint j = 0; j < 8; j++) rot[blk * 128 + lane + SG * j] = v[j] * INV_SQRT128;
}

#ifdef BITS
#define RINGW (8 * BITS)                              // uint32 stream words a tile
#define OFF0 (BITS + 16)                              // S(32 g) mod 32, for whole bits below 16
#define SHW ((4 * BITS) / 32)                         // the h = 1 shift: whole words
#define SHB ((4 * BITS) % 32)                         //                 and bits
#define USEW ((OFF0 + 27 * BITS + 16 + 31) / 32 + 1)  // aligned words the windows touch
#define LOADW (USEW + SHW + 1)
#ifndef NS
#define NS 1                                  // column strips a sub-group (N / 16 must divide by NS)
#endif

#define RI (ROWS <= 4)
#if RI
inline void rotate_block(float vr[ROWS][8], __global const uchar *base, exl3_desc d, uint blk, uint lane) {
    __global const half *suh = AT(const half, d.suh);
    for (uint r = 0; r < ROWS; r++) {
        __global const ushort *x = AT(const ushort, d.x) + (ulong)r * d.x_stride;
        for (uint j = 0; j < 8; j++) {
            const uint i = blk + lane + SG * j;
            vr[r][j] = r < d.rows ? load_x(x, i, d.x_f16) * vload_half(i, suh) : 0.0f;
        }
        hadamard128(vr[r], lane);
        for (uint j = 0; j < 8; j++) vr[r][j] *= INV_SQRT128;
    }
}
#define LOAD_XR(xr, kb, c)                                                                    \
    if (RI && fuse && ((kb) & 127) == 0) rotate_block(vr, base, d, (kb), (c));                   \
    for (uint r = 0; r < ROWS; r++)                                                           \
        xr[r] = r < d.rows ? (RI && fuse ? vr[r][((kb) / 16) & 7] : rot[(ulong)r * d.k + (kb) + (c)]) : 0.0f;
#else
#define LOAD_XR(xr, kb, c) \
    for (uint r = 0; r < ROWS; r++) xr[r] = r < d.rows ? rot[(ulong)r * d.k + (kb) + (c)] : 0.0f;
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_matvec(__global const uchar *base, __global const exl3_desc *desc, uint ks, uint fuse) {
    const exl3_desc d = desc[get_group_id(1)];
    const uint tn0 = get_global_id(0) / SG * NS, c = get_sub_group_local_id();
    if (tn0 * 16 >= d.n || d.halves != 2 * BITS) return;
    const uint g = c % 8, h = c / 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
    __global const uint *tile = (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn0 * 16 * BITS);
    __global const float *rot = AT(const float, d.rot);
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;  // uint32 words between k tiles
    const uint split = get_group_id(2), kspan = d.k / ks, k0 = split * kspan;
    tile += (ulong)(k0 / 16) * tile_k;
#if RI
    float vr[ROWS][8];  // fuse: the rotated block, K span a multiple of 128 (the host checks)
#endif
    float acc[NS][ROWS];
    for (uint s = 0; s < NS; s++)
        for (uint r = 0; r < ROWS; r++) acc[s][r] = 0.0f;
    for (uint kb = k0; kb < k0 + kspan; kb += 16, tile += tile_k) {
        uint raw[NS][LOADW], w[NS][USEW];
        for (uint s = 0; s < NS; s++)
            for (uint j = 0; j < LOADW; j++) raw[s][j] = tile[s * RINGW + (wbase + j) % RINGW];  // = stream_word
        for (uint s = 0; s < NS; s++)
            for (uint j = 0; j < USEW; j++)
                w[s][j] = h ? (SHB ? (raw[s][j + SHW] << SHB) | (raw[s][j + SHW + 1] >> (32 - SHB)) : raw[s][j + SHW]) : raw[s][j];
        float xr[ROWS];
        LOAD_XR(xr, kb, c)
        for (uint v = 0; v < 16; v++) {
            const uint bit = OFF0 + (8 * (v / 4) + (v % 4)) * BITS, wi = bit / 32, off = bit % 32;
            const uint kr = lane_row(v);
            float wv[NS];
            for (uint s = 0; s < NS; s++)
                wv[s] = mul1_value(off ? ((w[s][wi] << off) | (w[s][wi + 1] >> (32 - off))) >> 16 : w[s][wi] >> 16);
            for (uint r = 0; r < ROWS; r++) {
                const float xv = sub_group_broadcast(xr[r], kr);  // shared by the NS strips
                for (uint s = 0; s < NS; s++) acc[s][r] = fma(xv, wv[s], acc[s][r]);
            }
        }
    }
    __global float *sums = AT(float, d.sums) + (ulong)split * d.rows * d.n;
    for (uint s = 0; s < NS; s++)
        for (uint r = 0; r < ROWS; r++)
            if (r < d.rows) sums[(ulong)r * d.n + (tn0 + s) * 16 + c] = acc[s][r];
}
#endif

#define RO_PARTS 8
__attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(SG, RO_PARTS, 1)))
__kernel void exl3_rotate_out(__global uchar *base, __global const exl3_desc *desc, uint ks) {
    __local float part[RO_PARTS - 1][128];
    const exl3_desc d = desc[get_group_id(2)];
    const uint blk = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id(), p = get_local_id(1);
    if (blk * 128 >= d.n || row >= d.rows) return;  // the whole work-group
    __global const float *sums = AT(const float, d.sums) + (ulong)row * d.n + blk * 128 + lane;
    __global const half *svh = AT(const half, d.svh);
    float v[8];
    for (uint j = 0; j < 8; j++) {
        v[j] = 0.0f;
        for (uint z = p; z < ks; z += RO_PARTS) v[j] += sums[(ulong)z * d.rows * d.n + SG * j];
    }
    if (p) for (uint j = 0; j < 8; j++) part[p - 1][lane + SG * j] = v[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p) return;
    for (uint q = 0; q < RO_PARTS - 1; q++)
        for (uint j = 0; j < 8; j++) v[j] += part[q][lane + SG * j];
    hadamard128(v, lane);
    for (uint j = 0; j < 8; j++) {
        const uint i = blk * 128 + lane + SG * j;
        const float y = v[j] * INV_SQRT128 * vload_half(i, svh);
        store_y(AT(uchar, d.out), (ulong)row * d.out_stride + i, d.out_type, y);
    }
}

__kernel void exl3_route(__global const exl3_desc *experts, __global const uint *ids, __global exl3_desc *desc,
                         uint topk, ulong x, uint x_stride, uint x_per_slot, ulong rot, ulong sums, ulong out, uint out_stride, uint out_type, uint ks,
                         uint x_f16) {
    const uint s = get_global_id(0), token = s / topk;
    exl3_desc d = experts[ids[s]];
    d.rows = 1;
    d.x = x + (ulong)(x_per_slot ? s : token) * x_stride * 2;
    d.x_stride = x_stride;
    d.x_f16 = x_f16;
    d.rot = rot + (ulong)s * d.k * 4;
    d.sums = sums + (ulong)s * d.n * 4 * ks;  // the slot's ks partial rows
    d.out_stride = out_stride;
    d.out_type = out_type;
    d.out = out + (ulong)s * out_stride * (out_type == 1 ? 4 : 2);
    desc[s] = d;
}

#if defined(MOE_P) && defined(BITS)
#ifndef MOE_KMAX
#define MOE_KMAX 2048                     // largest gate/up K (hidden) and down K (intermediate)
#endif
#ifndef MOE_PD
#define MOE_PD MOE_P
#endif
#ifndef MOE_DS
#define MOE_DS 1                          // down strips a sub-group: 1, 2 or 4
#endif

inline uint moe_expert(__global const uint *ids, uint s, uint pairs, uint shared_id) { return s < pairs ? ids[s] : shared_id; }
inline uint moe_token(uint s, uint pairs, uint topk) { return s < pairs ? s / topk : s - pairs; }

#define STRIP_DOT(MS)                                                                                                  \
    inline void strip_dot##MS(__global const uint *tile[MS], ulong tile_k, uint kt, __local const float *xr[MS],    \
                              float acc[MS]) {                                                                     \
        const uint c = get_sub_group_local_id(), g = c % 8, h = c / 8;                                             \
        const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;                                \
        for (uint i = 0; i < MS; i++) acc[i] = 0.0f;                                                               \
        for (uint t = 0; t < kt; t++) {                                                                            \
            uint raw[MS][LOADW], w[MS][USEW];                                                                      \
            for (uint i = 0; i < MS; i++)                                                                          \
                for (uint j = 0; j < LOADW; j++) raw[i][j] = tile[i][(ulong)t * tile_k + (wbase + j) % RINGW];     \
            for (uint i = 0; i < MS; i++)                                                                          \
                for (uint j = 0; j < USEW; j++)                                                                    \
                    w[i][j] = h ? (SHB ? (raw[i][j + SHW] << SHB) | (raw[i][j + SHW + 1] >> (32 - SHB)) : raw[i][j + SHW]) \
                                : raw[i][j];                                                                       \
            float x[MS];                                                                                           \
            for (uint i = 0; i < MS; i++) x[i] = xr[i][t * 16 + c];                                                \
            for (uint v = 0; v < 16; v++) {                                                                        \
                const uint bit = OFF0 + (8 * (v / 4) + (v % 4)) * BITS, wi = bit / 32, off = bit % 32;             \
                for (uint i = 0; i < MS; i++) {                                                                    \
                    const float wv = mul1_value(off ? ((w[i][wi] << off) | (w[i][wi + 1] >> (32 - off))) >> 16 : w[i][wi] >> 16); \
                    acc[i] = fma(sub_group_broadcast(x[i], lane_row(v)), wv, acc[i]);                              \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
    }
STRIP_DOT(1)
STRIP_DOT(2)
STRIP_DOT(4)
#define STRIP_DOT_N(MS) strip_dot##MS
#define STRIP_DOT_X(MS) STRIP_DOT_N(MS)

inline __global const uint *strip_tiles(__global const uchar *base, exl3_desc d, uint tn, uint kt0) {
    return (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn * 16 * BITS) + (ulong)kt0 * (d.n / 16) * RINGW;
}

inline void sums_out(float v[8], __global const float *sums, uint splits, ulong stride, __global const half *svh, uint lane) {
    for (uint j = 0; j < 8; j++) {
        v[j] = sums[lane + SG * j];
        for (uint z = 1; z < splits; z++) v[j] += sums[z * stride + lane + SG * j];
    }
    hadamard128(v, lane);
    for (uint j = 0; j < 8; j++) v[j] = v[j] * INV_SQRT128 * vload_half(lane + SG * j, svh);
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_moe_gate_up(__global const uchar *base, __global const exl3_desc *gate, __global const exl3_desc *up,
                               __global const uint *ids, uint topk, uint pairs, uint shared_id, ulong x, uint x_stride,
                               ulong gu, uint x_f16) {
    __local float xr[2][MOE_KMAX / MOE_P];
    const uint nb = get_group_id(0) / MOE_P, p = get_group_id(0) % MOE_P, s = get_group_id(1);
    const uint q = get_sub_group_id(), c = get_sub_group_local_id();
    const uint e = moe_expert(ids, s, pairs, shared_id);
    const exl3_desc dg = gate[e], du = up[e];
    const uint kt = dg.k / 16 / MOE_P, k0 = p * kt * 16, ff = dg.n;
    __global const ushort *xs = AT(const ushort, x) + (ulong)moe_token(s, pairs, topk) * x_stride + k0;
    for (uint i = q; i < 2 * (kt / 8); i += 8) {  // rotate_in of this split's blocks, for gate and up (their suh differ)
        const uint proj = i & 1, blk = i >> 1;
        __global const half *suh = AT(const half, proj ? du.suh : dg.suh) + k0;
        float v[8];
        for (uint j = 0; j < 8; j++) {
            const uint at = blk * 128 + c + SG * j;
            v[j] = load_x(xs, at, x_f16) * vload_half(at, suh);
        }
        hadamard128(v, c);
        for (uint j = 0; j < 8; j++) xr[proj][blk * 128 + c + SG * j] = v[j] * INV_SQRT128;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global const uint *tile[2] = {strip_tiles(base, dg, nb * 8 + q, p * kt), strip_tiles(base, du, nb * 8 + q, p * kt)};
    __local const float *xp[2] = {xr[0], xr[1]};
    float acc[2];
    strip_dot2(tile, (ulong)(ff / 16) * RINGW, kt, xp, acc);
    __global float *sums = AT(float, gu) + ((ulong)s * MOE_P + p) * 2 * ff + (nb * 8 + q) * 16 + c;
    sums[0] = acc[0];
    sums[ff] = acc[1];
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_moe_act(__global const uchar *base, __global const exl3_desc *gate, __global const exl3_desc *up,
                           __global const exl3_desc *down, __global const uint *ids, uint pairs, uint shared_id, ulong gu,
                           ulong act, uint splits) {
    __local float gv[2][128];
    const uint nb = get_group_id(0), s = get_group_id(1), q = get_sub_group_id(), c = get_sub_group_local_id();
    const uint e = moe_expert(ids, s, pairs, shared_id);
    const exl3_desc g = q ? up[e] : gate[e];
    const uint ff = g.n;
    float v[8];
    sums_out(v, AT(const float, gu) + (ulong)s * splits * 2 * ff + q * ff + nb * 128, splits, 2 * ff, AT(const half, g.svh) + nb * 128, c);
    for (uint j = 0; j < 8; j++) gv[q][c + SG * j] = v[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (q) return;
    __global const half *suh = AT(const half, down[e].suh) + nb * 128;
    for (uint j = 0; j < 8; j++) {
        const float a = gv[0][c + SG * j];
        v[j] = a / (1.0f + exp(-a)) * gv[1][c + SG * j] * vload_half(c + SG * j, suh);
    }
    hadamard128(v, c);
    __global float *o = AT(float, act) + (ulong)s * ff + nb * 128;
    for (uint j = 0; j < 8; j++) o[c + SG * j] = v[j] * INV_SQRT128;
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_moe_down(__global const uchar *base, __global const exl3_desc *down, __global const uint *ids,
                            uint pairs, uint shared_id, ulong act, ulong y, uint y_stride, uint out_type) {
    __local float xr[MOE_KMAX];
    __local float part[MOE_PD][128];
    const uint nb = get_group_id(0), s = get_group_id(1), sg = get_sub_group_id(), c = get_sub_group_local_id();
    const uint q = sg % (8 / MOE_DS), p = sg / (8 / MOE_DS);
    const exl3_desc d = down[moe_expert(ids, s, pairs, shared_id)];
    const uint kt = d.k / 16 / MOE_PD;
    __global const float *a = AT(const float, act) + (ulong)s * d.k;
    for (uint i = get_local_id(0); i < d.k; i += get_local_size(0)) xr[i] = a[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    __global const uint *tile[MOE_DS];
    __local const float *xp[MOE_DS];
    for (uint i = 0; i < MOE_DS; i++) {
        tile[i] = strip_tiles(base, d, nb * 8 + q * MOE_DS + i, p * kt);
        xp[i] = xr + p * kt * 16;
    }
    float acc[MOE_DS];
    STRIP_DOT_X(MOE_DS)(tile, (ulong)(d.n / 16) * RINGW, kt, xp, acc);
    for (uint i = 0; i < MOE_DS; i++) part[p][(q * MOE_DS + i) * 16 + c] = acc[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg) return;
    __global const half *svh = AT(const half, d.svh) + nb * 128;
    float v[8];
    for (uint j = 0; j < 8; j++) {
        v[j] = part[0][c + SG * j];
        for (uint z = 1; z < MOE_PD; z++) v[j] += part[z][c + SG * j];
    }
    hadamard128(v, c);
    for (uint j = 0; j < 8; j++)
        store_y(AT(uchar, y), (ulong)s * y_stride + nb * 128 + c + SG * j, out_type, v[j] * INV_SQRT128 * vload_half(c + SG * j, svh));
}

#include "exl3_prompt.cl"

#endif
