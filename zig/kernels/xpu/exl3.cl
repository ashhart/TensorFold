// SPDX-License-Identifier: Apache-2.0
// EXL3 (ExLlamaV3 trellis) projections on Xe2: a port of zig/kernels/metal/core/exl3_projection.metal.
// y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh, H the 128x128 Sylvester Hadamard / sqrt(128).
// Model-agnostic: a launch covers many groups through a descriptor table. Codebook mul1 only; 2 to 8 bits.
// The format is docs/recipes/exl3.md; zig/src/core/exl3_format.zig is the scalar reference.
//
// Metal runs SIMD-groups of 32. Here a sub-group is 16 lanes (Xe2 native). The mapping:
// - Hadamard: Metal lane l holds indices l + 32 j (j < 4), five shuffle stages and two in the lane.
//   Here lane l holds l + 16 j (j < 8), four shuffle stages and three in the lane. The butterfly is the same.
// - Matvec: Metal decodes a 16x16 tile into threadgroup memory (8 values a lane) and then multiplies.
//   Here lane c decodes the 16 values of tile column c straight into registers. No local memory, no barrier.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define SG 16
#ifndef ROWS
#define ROWS 1
#endif
#define INV_SQRT128 0.08838834764831845f

// One group: offsets in bytes from the base pointer of the launch. Built with -DEXL3_ABS (the Level Zero host,
// zig/src/xpu/exl3.zig) they are device addresses and base is not read.
// x is bf16 or (x_f16) fp16 [rows][x_stride]; rot fp32 [rows][k] (fp16 lost 1e-3 on Sage down_proj);
// sums fp32 [rows][n]; out [rows][out_stride] in bf16 (out_type 0), fp32 (1) or fp16 (2).
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
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

inline float load_x(__global const ushort *x, ulong i, uint f16) { return f16 ? vload_half(i, (__global const half *)x) : bf(x[i]); }

// out[at] = y as out_type (0 bf16, 1 fp32, 2 fp16, rounded to nearest even).
inline void store_y(__global uchar *out, ulong at, uint type, float y) {
    if (type == 1) ((__global float *)out)[at] = y;
    else if (type == 2) vstore_half_rte(y, at, (__global half *)out);
    else ((__global ushort *)out)[at] = to_bf(y);
}

// Lane l holds indices l + 16 j. Four cross-lane stages, then three in the lane; fixed FP32 order.
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

// Two little-endian int16 words make one stream word, read from its most significant bit.
inline uint stream_word(__global const ushort *tile, uint i) { return (uint)tile[2 * i] | ((uint)tile[2 * i + 1] << 16); }

// The window of value p as (word, bit offset, next word): the 16 stream bits that end at E(p), wrapping.
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

// mul1: x = s * 0x83DCD12D; fp16((1024 + the bytes of x) * fp16(0x1EEE) + fp16(0xC931)), one rounding.
// The byte sum is one dp4a; 1024 + sum <= 2044 is exact in fp16, so one fp16 fma gives the format's rounding
// (method from exl3xpu, MIT, 0xSero). The fp16 type matters: IGC on driver 26.05 folds
// (float)convert_half_rte(f) to f, so an FP32 fma followed by a conversion loses the rounding.
// exl3_mul1_check proves all 65536 states.
inline half mul1_half(uint s) {
    // 0x6400 is fp16 1024 and the sum is below 1024, so the dp4a result is already the fp16 bits of 1024 + sum.
    const uint h = dot_acc_sat_4x8packed_uu_uint(s * 0x83DCD12Du, 0x01010101u, 0x6400u);
    return fma(as_half((ushort)h), as_half((ushort)0x1EEE), as_half((ushort)0xC931));
}
inline float mul1_value(uint s) { return (float)mul1_half(s); }

// Lane c owns tile column c. Its 16 stream values are p = 32 (c % 8) + 8 m + 4 (c / 8) + i (m, i < 4),
// at tile row 2 m + (i & 1) + 8 (i >> 1). This is the inverse of exl3_format.tilePlacement.
inline uint lane_value(uint c, uint v) { return 32 * (c % 8) + 8 * (v / 4) + 4 * (c / 8) + (v % 4); }
inline uint lane_row(uint v) { return 2 * (v / 4) + (v & 1) + 8 * ((v >> 1) & 1); }

// Qualification only: the 256 states of each tile in stream order, and W_q fp16 [K, N] by placement.
// Grid (tiles_n, tiles_k), one sub-group a work-group.
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

// Qualification only: every mul1 state's value as the matvec uses it (fp32), to compare with the float64 host table.
__kernel void exl3_mul1_check(__global float *out) {
    const uint i = get_global_id(0);
    out[i] = mul1_value(i);
}

// rot[row][k] = H_K (x * suh) for one 128 block. Grid (K_max / 128, ROWS_max, groups).
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

// sums[row][n] = rot[row] @ W_q[:, n] for one 16-column strip and up to ROWS rows (fuse: rot from x, see RI). Built per (BITS, ROWS);
// a launch holds groups of one width only (the host sorts them). Grid (strips_max * SG / NS rounded to the
// work-group, groups, ks), work-group 8 sub-groups, one sub-group NS strips.
// K split: split z of ks (K / 16 divides by ks) sums its K / ks rows of W into sums + z * rows * n. A small
// launch (one layer's experts) has too few strips to fill the GPU; the splits multiply the sub-groups.
//
// Lane c (column c) reads only the stream words its 16 values use. With g = c % 8 and h = c / 8, its values are
// p = 32 g + q, q = 8 m + i + 4 h, and value p's window starts at bit S(p) = p * BITS + BITS - 16 (mod the ring).
// The lane loads LOADW words from word floor(S(32 g) / 32); lanes with h = 1 shift them left by 4 * BITS bits.
// After that every window's word and bit offset is a compile-time constant.
#ifdef BITS
#define RINGW (8 * BITS)                              // uint32 stream words a tile
#define OFF0 (BITS + 16)                              // S(32 g) mod 32, for whole bits below 16
#define SHW ((4 * BITS) / 32)                         // the h = 1 shift: whole words
#define SHB ((4 * BITS) % 32)                         //                 and bits
#define USEW ((OFF0 + 27 * BITS + 16 + 31) / 32 + 1)  // aligned words the windows touch
#define LOADW (USEW + SHW + 1)
#ifndef HS
#define HS 0                                  // see exl3_matvec (needs NS >= 2)
#endif
#ifndef NS
#define NS 1                                  // column strips a sub-group (N / 16 must divide by NS)
#endif
// Fused rotate_in (ROWS <= 4): the sub-group rotates the 128 block of x that holds kb in registers, as
// exl3_rotate_in does. After hadamard128 lane c holds k = blk + c + 16 j in v[j]: the xr of k tile j.
#define DPH(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), c)

inline int8 tile_b(__global const uint *t, uint c) {
    const uint g = c % 8, h = c / 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
    uint raw[LOADW], w[USEW];
    for (uint j = 0; j < LOADW; j++) raw[j] = t[(wbase + j) % RINGW];
    for (uint j = 0; j < USEW; j++) w[j] = h ? (SHB ? (raw[j + SHW] << SHB) | (raw[j + SHW + 1] >> (32 - SHB)) : raw[j + SHW]) : raw[j];
    half kv[16];
    for (uint v = 0; v < 16; v++) {
        const uint bit = OFF0 + (8 * (v / 4) + (v % 4)) * BITS, wi = bit / 32, off = bit % 32;
        kv[lane_row(v)] = mul1_half(off ? ((w[wi] << off) | (w[wi + 1] >> (32 - off))) >> 16 : w[wi] >> 16);
    }
    int8 b;
    for (uint j = 0; j < 8; j++) ((int *)&b)[j] = as_int((half2)(kv[2 * j], kv[2 * j + 1]));
    return b;
}

#ifndef DP
#define DP (ROWS <= 8)                                // decode on XMX: tiles decoded in registers as DPAS B operands, rows padded to 8
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
#if DP
    float8 acc[NS];
    for (uint s = 0; s < NS; s++) acc[s] = 0.0f;
    for (uint kb = k0; kb < k0 + kspan; kb += 16, tile += tile_k) {
        float xr[ROWS];
        LOAD_XR(xr, kb, c)
        short8 ah, al;
        for (uint r = 0; r < 8; r++) {
            const float f = r < ROWS ? xr[r] : 0.0f;
            const half hv = convert_half_rte(f);
            ah[r] = as_short(hv);
            al[r] = as_short(convert_half_rte(f - (float)hv));
        }
        for (uint s = 0; s < NS; s++) {
            const int8 b = tile_b(tile + s * RINGW, c);
            acc[s] = DPH(ah, b, acc[s]);
            acc[s] = DPH(al, b, acc[s]);
        }
    }
    __global float *sums = AT(float, d.sums) + (ulong)split * d.rows * d.n;
    for (uint s = 0; s < NS; s++)
        for (uint r = 0; r < ROWS; r++)
            if (r < d.rows) sums[(ulong)r * d.n + (tn0 + s) * 16 + c] = acc[s][r];
#elif HS && NS >= 2
    // HS: the half h = c / 8 takes strips h * NS / 2 and up, and each lane decodes both column halves of its
    // strips from the same words (g + 0 and g + 8). The h offset is then a compile-time constant: no shift.
    const uint sl = h * (NS / 2);
    float acc[NS / 2][2][ROWS];
    for (uint s = 0; s < NS / 2; s++)
        for (uint e = 0; e < 2; e++)
            for (uint r = 0; r < ROWS; r++) acc[s][e][r] = 0.0f;
    for (uint kb = k0; kb < k0 + kspan; kb += 16, tile += tile_k) {
        uint raw[NS / 2][LOADW];
        for (uint s = 0; s < NS / 2; s++)
            for (uint j = 0; j < LOADW; j++) raw[s][j] = tile[(sl + s) * RINGW + (wbase + j) % RINGW];
        float xr[ROWS];
        LOAD_XR(xr, kb, c)
        for (uint v = 0; v < 16; v++) {
            const uint kr = lane_row(v);
            float wv[NS / 2][2];
            for (uint e = 0; e < 2; e++) {
                const uint bit = OFF0 + e * 4 * BITS + (8 * (v / 4) + (v % 4)) * BITS, wi = bit / 32, off = bit % 32;
                for (uint s = 0; s < NS / 2; s++)
                    wv[s][e] = mul1_value(off ? ((raw[s][wi] << off) | (raw[s][wi + 1] >> (32 - off))) >> 16 : raw[s][wi] >> 16);
            }
            for (uint r = 0; r < ROWS; r++) {
                const float xv = sub_group_broadcast(xr[r], kr);
                for (uint s = 0; s < NS / 2; s++)
                    for (uint e = 0; e < 2; e++) acc[s][e][r] = fma(xv, wv[s][e], acc[s][e][r]);
            }
        }
    }
    __global float *sums = AT(float, d.sums) + (ulong)split * d.rows * d.n;
    for (uint s = 0; s < NS / 2; s++)
        for (uint e = 0; e < 2; e++)
            for (uint r = 0; r < ROWS; r++)
                if (r < d.rows) sums[(ulong)r * d.n + (tn0 + sl + s) * 16 + g + 8 * e] = acc[s][e][r];
#else
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
#endif
}
#endif

// out[row][n] = (H_N sums) * svh in out_type, for one 128 block. Sub-group p of RO_PARTS adds splits p, p + RO_PARTS,
// ... in order; sub-group 0 then adds the parts in order (deterministic). One sub-group for all 16 splits of a narrow
// launch (N 2048) took longer than its matvec. Work-groups (N_max / 128, ROWS_max, groups) of SG x RO_PARTS.
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

// P3.2 routed experts. One descriptor a (token, pick) slot, built on the device from the router's picks, so the
// three kernels above run with no host round trip. experts holds the layer's per-expert descriptors (trellis,
// suh, svh, k, n, halves; the rest ignored). Slot s = token * topk + j reads x row token (x_per_slot = 0, gate and up)
// or row s (x_per_slot = 1, down), and writes out row s.
// x, rot, sums, out are base offsets; rot, sums and out give each slot its own row. Grid (tokens * topk).
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

// Fused routed MLP for one MoE layer: y_s = down(silu(gate x) * up x) for every (token, pick) slot s in four
// launches, with no slot table and no host round trip. Built per BITS with MOE_P, MOE_PD (K splits of gate|up and of
// down) and MOE_DS (down strips a sub-group). Slot s < pairs is pick s % topk of token s / topk, expert ids[s]; slots
// from pairs on are token s - pairs on the shared expert (shared_id). gate, up and down are expert tables as for
// exl3_route (trellis, suh, svh, k, n). The steps and the fp32 order are those of rotate_in, matvec (ks = MOE_P or
// MOE_PD) and rotate_out. Small work-groups (one K split of 8 strips each) spread one layer over all Xe-cores; the
// split sums go through memory and are added in split order, so the result does not depend on the schedule.
// The gate|up epilogue (SiLU and the down input Hadamard per 128 block) follows exl3_moe.a8.sycl (MIT, 0xSero).
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

// MS 16-column strips over kt k tiles: the exl3_matvec loop at ROWS = 1, strip i reading its input from xr[i].
// The strips' tile loads are independent, so a sub-group keeps MS loads in flight.
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

// The tile words of strip tn of d from k tile kt0 on.
inline __global const uint *strip_tiles(__global const uchar *base, exl3_desc d, uint tn, uint kt0) {
    return (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn * 16 * BITS) + (ulong)kt0 * (d.n / 16) * RINGW;
}

// v = (H_128 of the split sums) / sqrt(128) * svh for one 128 block; sums [splits][stride], added in split order.
inline void sums_out(float v[8], __global const float *sums, uint splits, ulong stride, __global const half *svh, uint lane) {
    for (uint j = 0; j < 8; j++) {
        v[j] = sums[lane + SG * j];
        for (uint z = 1; z < splits; z++) v[j] += sums[z * stride + lane + SG * j];
    }
    hadamard128(v, lane);
    for (uint j = 0; j < 8; j++) v[j] = v[j] * INV_SQRT128 * vload_half(lane + SG * j, svh);
}

// gu[s][p][proj][I]: split p of gate (proj 0) and up (1) for the 8 strips of one 128 block nb of I.
// Grid (I / 128 * MOE_P * 128, slots), work-group 8 sub-groups, each the gate and the up strip of one column.
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

// act[s][I] = H_I((silu(gate x) * up x) * suh_down) / sqrt(128), the down projection's rotated input: rotate_out of
// gate and up from gu (`splits` splits in order: MOE_P in decode, 1 for prompt tiles), SiLU, then the down rotate_in, one 128 block a work-group.
// Grid (I / 128 * 32, slots), work-group 2 sub-groups (gate, up).
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

// y[s] for one 128 block nb of N in out_type: MOE_PD K parts of down over act[s] (8 / MOE_DS sub-groups a part,
// MOE_DS strips a sub-group), added in order in local memory, then the output Hadamard and svh. The parts went
// through global memory to a fourth launch before; the sums are the same. Grid (N / 128 * 128 / MOE_DS * MOE_PD,
// slots), work-group 128 / MOE_DS * MOE_PD.
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

// ---------------------------------------------------------------- prompt rows (P3.4): trellis tiles on XMX
// A k tile of 16 rows and a 16-column strip is one DPAS B operand: lane c decodes column c (as exl3_matvec) and puts
// its 16 weights in k order, k pairs 2j and 2j + 1 a dword. A is 8 rows of the rotated input, k = lane, in fp16 (the
// mul1 weights are fp16 values, so only the input rounds). The tiling is that of the W4A16 grouped GEMM for Xe2 in
// vllm-xpu-kernels: weights decoded in registers straight into the B operand, fp32 accumulators.
#ifndef GEMM_M
#define GEMM_M 16                         // rows a sub-group: GEMM_M / 8 DPAS a k tile
#endif
#ifndef GEMM_LO
#define GEMM_LO 1                         // 1: x = hi + lo in two fp16 DPAS (fp16 x alone gave 1.35e-3 on down_proj)
#endif


// The rotated input of prompt rows as two fp16 planes, x = hi + lo: rot holds hi [rows][k], then lo. Grid (K_max / 128,
// rows_max, groups) of one sub-group, as exl3_rotate_in.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_rotate_in_h(__global const uchar *base, __global const exl3_desc *desc, uint rows) {
    exl3_desc d = desc[get_group_id(2)];
    d.rows = min(d.rows, rows);
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
    __global half *hi = AT(half, d.rot) + (ulong)row * d.k;
    __global half *lo = AT(half, d.rot) + (ulong)desc[get_group_id(2)].rows * d.k + (ulong)row * d.k;
    for (uint j = 0; j < 8; j++) {
        const float f = v[j] * INV_SQRT128;
        const half h = convert_half_rte(f);
        hi[blk * 128 + lane + SG * j] = h;
        lo[blk * 128 + lane + SG * j] = convert_half_rte(f - (float)h);
    }
}

#ifndef GEMM_S
#define GEMM_S 4                          // 16-column strips a work-group (N divides by 16 * GEMM_S)
#endif
#ifndef GEMM_KC
#define GEMM_KC 8                         // k tiles decoded into local memory between two barriers (K divides by 16 * KC)
#endif
#define GEMM_WM (8 * GEMM_M)              // rows a work-group: 8 sub-groups of GEMM_M rows

// sums[row][n] = rot[row] @ W_q for GEMM_S strips and GEMM_WM rows of a group (any row count; exl3_rotate_in_h before,
// exl3_rotate_out_rows after). The work-group decodes each weight tile once into local memory as DPAS B operands, and
// its 8 sub-groups (GEMM_M rows each) all use it: the decode is shared by GEMM_WM rows. Grid (N_max / 16 / GEMM_S * 128,
// rows_max / GEMM_WM, groups), work-group 128.
__attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void exl3_gemm(__global const uchar *base, __global const exl3_desc *desc, uint rows) {
    __local int8 bs[GEMM_KC][GEMM_S][SG];
    exl3_desc d = desc[get_group_id(2)];
    d.rows = min(d.rows, rows);
    const uint q = get_sub_group_id(), c = get_sub_group_local_id();
    const uint tn0 = get_group_id(0) * GEMM_S, w0 = get_group_id(1) * GEMM_WM, m0 = w0 + q * GEMM_M;
    if (tn0 * 16 >= d.n || w0 >= d.rows || d.halves != 2 * BITS) return;  // the whole work-group
    __global const uint *tiles = (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn0 * 16 * BITS);
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;
    const uint last = d.rows - 1;
    __global const ushort *hi = AT(const ushort, d.rot);
    __global const ushort *lo = hi + (ulong)desc[get_group_id(2)].rows * d.k;
    float8 acc[GEMM_S][GEMM_M / 8];
    for (uint s = 0; s < GEMM_S; s++)
        for (uint rb = 0; rb < GEMM_M / 8; rb++) acc[s][rb] = 0.0f;
    for (uint kc = 0; kc < d.k; kc += 16 * GEMM_KC) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint job = q; job < GEMM_KC * GEMM_S; job += 8) {
            const uint t = job / GEMM_S, s = job % GEMM_S;
            bs[t][s][c] = tile_b(tiles + (ulong)(kc / 16 + t) * tile_k + s * RINGW, c);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint t = 0; t < GEMM_KC; t++) {
            const uint kb = kc + 16 * t;
            short8 ah[GEMM_M / 8], al[GEMM_M / 8];
            for (uint rb = 0; rb < GEMM_M / 8; rb++)
                for (uint r = 0; r < 8; r++) {
                    const ulong at = (ulong)min(m0 + rb * 8 + r, last) * d.k + kb;  // rows past the end are not stored
                    ah[rb][r] = as_short(intel_sub_group_block_read_us(hi + at));
                    if (GEMM_LO) al[rb][r] = as_short(intel_sub_group_block_read_us(lo + at));
                }
            for (uint s = 0; s < GEMM_S; s++) {
                const int8 b = bs[t][s][c];
                for (uint rb = 0; rb < GEMM_M / 8; rb++) {
                    acc[s][rb] = DPH(ah[rb], b, acc[s][rb]);
                    if (GEMM_LO) acc[s][rb] = DPH(al[rb], b, acc[s][rb]);
                }
            }
        }
    }
    __global float *sums = AT(float, d.sums) + tn0 * 16 + c;
    for (uint s = 0; s < GEMM_S; s++)
        for (uint rb = 0; rb < GEMM_M / 8; rb++)
            for (uint r = 0; r < 8; r++) {
                const uint row = m0 + rb * 8 + r;
                if (row < d.rows) sums[(ulong)row * d.n + s * 16] = acc[s][rb][r];
            }
}

// out[row][n] = (H_N sums[row]) * svh in out_type for one 128 block, one split. Grid (N_max / 128, rows_max, groups) of
// one sub-group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_rotate_out_rows(__global uchar *base, __global const exl3_desc *desc, uint rows) {
    exl3_desc d = desc[get_group_id(2)];
    d.rows = min(d.rows, rows);
    const uint blk = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
    if (blk * 128 >= d.n || row >= d.rows) return;
    __global const float *sums = AT(const float, d.sums) + (ulong)row * d.n + blk * 128;
    __global const half *svh = AT(const half, d.svh) + blk * 128;
    float v[8];
    for (uint j = 0; j < 8; j++) v[j] = sums[lane + SG * j];
    hadamard128(v, lane);
    for (uint j = 0; j < 8; j++)
        store_y(AT(uchar, d.out), (ulong)row * d.out_stride + blk * 128 + lane + SG * j, d.out_type, v[j] * INV_SQRT128 * vload_half(lane + SG * j, svh));
}


// ---------------------------------------------------------------- prompt MoE: expert tiles on XMX
// A chunk's slots (pair p < pairs: token p / topk on expert ids[p]; slot pairs + t: token t on the shared expert) are
// sorted by expert into tiles of at most 16 slots (the runner's sort: uint4 expert, first, count; order holds slots).
// A tile reads its expert's weights once for all its slots.

// rot[proj][slot][K] = H_K (x[token] * suh) / sqrt(128) for gate (proj 0) and up (1): their suh differ. Grid
// (K / 128, slots, 2) of one sub-group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_moe_rot(__global const uchar *base, __global const exl3_desc *gate, __global const exl3_desc *up,
                           __global const uint *ids, uint topk, uint pairs, uint shared_id, ulong x, uint x_stride,
                           uint x_f16, ulong rot, uint slots) {
    const uint blk = get_group_id(0), s = get_group_id(1), proj = get_group_id(2), lane = get_sub_group_local_id();
    const exl3_desc d = (proj ? up : gate)[moe_expert(ids, s, pairs, shared_id)];
    __global const ushort *xs = AT(const ushort, x) + (ulong)moe_token(s, pairs, topk) * x_stride;
    __global const half *suh = AT(const half, d.suh);
    float v[8];
    for (uint j = 0; j < 8; j++) {
        const uint i = blk * 128 + lane + SG * j;
        v[j] = load_x(xs, i, x_f16) * vload_half(i, suh);
    }
    hadamard128(v, lane);
    __global float *o = AT(float, rot) + ((ulong)proj * slots + s) * d.k + blk * 128;
    for (uint j = 0; j < 8; j++) o[lane + SG * j] = v[j] * INV_SQRT128;
}

#ifndef TILE_NS
#define TILE_NS 2                         // strips a sub-group in exl3_moe_tile
#endif
#ifndef TILE_M
#define TILE_M 16                         // slots a tile (exl3_moe_sort)
#endif
#define SORT_MAX 1024                     // most experts (with the shared one) exl3_moe_sort takes

// The slots of a chunk grouped by expert (shared expert last, index n_experts): order holds slot numbers; tiles are
// (expert, first entry of order, count, 0) of at most TILE_M slots; ntiles[0] is their count. One work-group of 256.
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void exl3_moe_sort(__global const uint *ids, __global uint *order, __global uint4 *tiles, __global uint *ntiles,
                   uint n_experts, uint pairs, uint tokens) {
    __local uint cnt[SORT_MAX], start[SORT_MAX], tbase[SORT_MAX], fill[SORT_MAX];
    const uint lid = get_local_id(0);
    for (uint e = lid; e <= n_experts; e += 256) cnt[e] = e == n_experts ? tokens : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint p = lid; p < pairs; p += 256) atomic_inc(&cnt[ids[p]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        uint s = 0, t = 0;
        for (uint e = 0; e <= n_experts; e++) {
            start[e] = s;
            tbase[e] = t;
            fill[e] = s;
            s += cnt[e];
            t += (cnt[e] + TILE_M - 1) / TILE_M;
        }
        ntiles[0] = t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint e = lid; e <= n_experts; e += 256)
        for (uint i = 0; i * TILE_M < cnt[e]; i++) tiles[tbase[e] + i] = (uint4)(e, start[e] + i * TILE_M, min((uint)TILE_M, cnt[e] - i * TILE_M), 0);
    for (uint p = lid; p < pairs; p += 256) order[atomic_inc(&fill[ids[p]])] = p;
    for (uint t = lid; t < tokens; t += 256) order[start[n_experts] + t] = pairs + t;
}

// sums[slot][proj][n] = a[proj][slot] @ W_q (expert of the tile, table proj: 0 gate or down, 1 up) for TILE_NS strips
// and one tile of at most TILE_M slots: x = hi + lo in two fp16 DPAS, as exl3_gemm. a is fp32 [proj][slots][K] (a_proj
// floats between the projections); sums rows are s_stride floats, s_proj floats between the projections.
// Grid (N / 16 / TILE_NS * SG rounded to 128, tile bound, projections), work-group 128.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_moe_tile(__global const uchar *base, __global const exl3_desc *t0, __global const exl3_desc *t1,
                            __global const uint *order, __global const uint4 *tiles, __global const uint *ntiles,
                            ulong a, ulong a_proj, ulong sums, uint s_stride, uint s_proj) {
    if (get_group_id(1) >= ntiles[0]) return;
    const uint4 tl = tiles[get_group_id(1)];
    const uint proj = get_group_id(2), c = get_sub_group_local_id();
    const exl3_desc d = (proj ? t1 : t0)[tl.x];
    const uint tn0 = get_global_id(0) / SG * TILE_NS;
    if (tn0 * 16 >= d.n || d.halves != 2 * BITS) return;
    __global const uint *tile = (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn0 * 16 * BITS);
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;
    __global const float *ap = AT(const float, a) + proj * a_proj + c;
    const uint rbs = (tl.z + 7) / 8;  // 8-row blocks that hold slots
    uint slot[TILE_M];
    for (uint i = 0; i < TILE_M; i++) slot[i] = order[tl.y + min(i, tl.z - 1)];  // rows past the tile repeat its last slot
    float8 acc[TILE_NS][TILE_M / 8];
    for (uint s = 0; s < TILE_NS; s++)
        for (uint rb = 0; rb < TILE_M / 8; rb++) acc[s][rb] = 0.0f;
    for (uint kb = 0; kb < d.k; kb += 16, tile += tile_k) {
        short8 ah[TILE_M / 8], al[TILE_M / 8];
        for (uint rb = 0; rb < TILE_M / 8; rb++)
            if (rb < rbs)
                for (uint r = 0; r < 8; r++) {
                    const float f = ap[(ulong)slot[rb * 8 + r] * d.k + kb];
                    const half h = convert_half_rte(f);
                    ah[rb][r] = as_short(h);
                    al[rb][r] = as_short(convert_half_rte(f - (float)h));
                }
        for (uint s = 0; s < TILE_NS; s++) {
            const int8 b = tile_b(tile + s * RINGW, c);
            for (uint rb = 0; rb < TILE_M / 8; rb++)
                if (rb < rbs) {
                    acc[s][rb] = DPH(ah[rb], b, acc[s][rb]);
                    acc[s][rb] = DPH(al[rb], b, acc[s][rb]);
                }
        }
    }
    __global float *o = AT(float, sums) + proj * s_proj + (tn0 * 16 + c);
    for (uint i = 0; i < tl.z; i++)
        for (uint s = 0; s < TILE_NS; s++) o[(ulong)slot[i] * s_stride + s * 16] = acc[s][i / 8][i % 8];
}

// y[slot] for one 128 block of N in out_type: (H_N sums[slot]) * svh of the slot's down. Grid (N / 128, slots) of one
// sub-group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_moe_out_rows(__global const uchar *base, __global const exl3_desc *down, __global const uint *ids,
                                uint pairs, uint shared_id, ulong sums, ulong y, uint y_stride, uint out_type) {
    const uint blk = get_group_id(0), s = get_group_id(1), lane = get_sub_group_local_id();
    const exl3_desc d = down[moe_expert(ids, s, pairs, shared_id)];
    __global const float *v0 = AT(const float, sums) + (ulong)s * d.n + blk * 128;
    __global const half *svh = AT(const half, d.svh) + blk * 128;
    float v[8];
    for (uint j = 0; j < 8; j++) v[j] = v0[lane + SG * j];
    hadamard128(v, lane);
    for (uint j = 0; j < 8; j++)
        store_y(AT(uchar, y), (ulong)s * y_stride + blk * 128 + lane + SG * j, out_type, v[j] * INV_SQRT128 * vload_half(lane + SG * j, svh));
}

#endif
