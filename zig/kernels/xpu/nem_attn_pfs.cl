#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
// Nemotron-H attention for prompt windows (many query rows) on the matrix engine, after qwen_attn_pfs.cl (bf16 build, one key range, no gate): the scores are computed transposed,
// S^T = K Q^T, so a lane owns one column (head, row) and the softmax maximum, rescale and denominator are lane-local; the P tile comes out in the layout of the B operand of
// O^T += V^T P^T without a transposition. Work-group: 8 sub-groups, 8 rows x HPW = 2 NT query heads of one kv head = 16 NT columns (NT tiles of 16 columns: two heads x 8 rows),
// blocks of 128 keys = 8 key tiles of 16. Sub-group s takes key tile s of a block for the scores (8 DPAS a 16-dim step and N tile) and the output dims 16 s .. 16 s + 15 for P V.
// The key tiles are absolute and rows are independent: a row's bits depend only on its own position (any window width or chunking gives the same result).
// Head 128 dims, 32 query heads, 2 kv heads (16 query heads each), no RoPE here (applied by the caller if at all), scale 1/sqrt(128); out[row][head][128] = bf16(o / l).
#ifndef NT
#define NT 4
#endif
#define HD 128
#define NH 32
#define NKV 2
#define GQ 16
#define HPW (2 * NT)
#define NHG (GQ / HPW)
#define MAD intel_sub_group_bf16_bf16_matrix_mad_k16
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// Queries q [row][32][128] bf16 -> B-operand tiles qb [wg][N tile][8 16-dim steps][8 dwords][lane = column], wg = row tile * (NKV * NHG) + kv head * NHG + head group; dword e of step t =
// dims 16 t + 2 e, + 1 of the column's query (column n of N tile j = head hg * HPW + 2 j + (n >> 3) of the kv head's group, row n & 7); rows past `rows` are zero.
// Global size ceil(rows / 8) * NKV * NHG * NT * 1024.
__kernel void nem_attn_pfs_prep(__global const ushort *q, __global uint *qb, uint rows) {
    const ulong g = get_global_id(0);
    const uint n = g & 15, e = (g >> 4) & 7, t = (g >> 7) & 7, j = (g >> 10) % NT;
    const ulong wgi = (g >> 10) / NT; // row tile * (NKV * NHG) + kv head * NHG + head group
    const uint hgrp = wgi % NHG, hk = (wgi / NHG) % NKV;
    const uint row = (wgi / (NKV * NHG)) * 8 + (n & 7), head = hk * GQ + hgrp * HPW + 2 * j + (n >> 3);
    uint v = 0;
    if (row < rows) {
        const ushort2 x = vload2(0, q + ((ulong)row * NH + head) * HD + t * 16 + 2 * e);
        v = (uint)x.x | ((uint)x.y << 16);
    }
    qb[g] = v;
}

// Grid (ceil(rows / 8), NKV * NHG); row z is at position pos0 + z. kx / vx: the bf16 caches [pos][2][128] (all keys 0 .. pos0 + rows - 1).
// The row groups run last-first (the late, long ones start first).
__attribute__((reqd_work_group_size(128, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void nem_attn_prefill_s(__global const uint *qb, __global const ushort *kx, __global const ushort *vx, __global ushort *out, uint pos0, uint rows) {
    __local float mxs[8][NT][16]; // [sub-group][N tile][column]: tile maxima, later denominators
    __local uint pt[8 * NT * 16 * 8]; // P^T: [key tile][N tile][column][8 dwords of key pairs]
    const uint hk = get_group_id(1) / NHG;
    const uint hgrp = get_group_id(1) % NHG;
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint rt = get_num_groups(0) - 1 - get_group_id(0);
    const uint r0 = rt * 8;
    const uint kmax = pos0 + min(r0 + 8, rows); // keys any row of the work-group sees
    const uint lim = pos0 + r0 + (lane & 7);    // the last key of this lane's row
    const ulong wgi = (ulong)rt * (NKV * NHG) + get_group_id(1);
    const __global uint *qt = qb + wgi * (NT * 1024);
    const uint ntiles = (kmax + 15) / 16;
    const int height = (int)kmax;
    float m[NT], ll[NT];
    float8 acc[NT][2]; // [N tile][8-dim group of the sub-group's 16 dims]: lane = column, element = dim
#pragma unroll
    for (int j = 0; j < NT; j++) {
        m[j] = -INFINITY;
        ll[j] = 0.0f;
        acc[j][0] = (float8)(0.0f);
        acc[j][1] = (float8)(0.0f);
    }
    for (uint kb = 0; kb < ntiles; kb += 8) {
        const uint tile = kb + sg;
        float8 c[NT][2];
#pragma unroll
        for (int j = 0; j < NT; j++) c[j][0] = c[j][1] = (float8)(0.0f);
#pragma unroll
        for (int t = 0; t < 8; t++) {
            ushort16 ka;
            intel_sub_group_2d_block_read_16b_16r16x1c((__global void *)kx, NKV * HD * 2, height, NKV * HD * 2, (int2)(hk * HD + t * 16, (int)(tile * 16)), (__private ushort *)&ka);
            const short8 a0 = as_short8(ka.lo), a1 = as_short8(ka.hi);
#pragma unroll
            for (int j = 0; j < NT; j++) {
                const int8 b = as_int8(intel_sub_group_block_read8(qt + (j * 8 + t) * 128));
                c[j][0] = MAD(a0, b, c[j][0]);
                c[j][1] = MAD(a1, b, c[j][1]);
            }
        }
#pragma unroll
        for (int j = 0; j < NT; j++) {
            float mx = -INFINITY;
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float s = (tile * 16 + h * 8 + i <= lim) ? c[j][h][i] * (0.08838834764831845f * 1.4426950408889634f) : -INFINITY;
                    c[j][h][i] = s;
                    mx = fmax(mx, s);
                }
            mxs[sg][j][lane] = mx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float al[NT];
#pragma unroll
        for (int j = 0; j < NT; j++) {
            float mn = m[j];
#pragma unroll
            for (int s = 0; s < 8; s++) mn = fmax(mn, mxs[s][j][lane]);
            al[j] = (m[j] == -INFINITY) ? 0.0f : exp2(m[j] - mn);
            m[j] = mn;
            uint pk[8];
            float sum = 0.0f;
#pragma unroll
            for (int d = 0; d < 8; d++) {
                ushort pb2[2];
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    const float s = c[j][d >> 2][2 * (d & 3) + b];
                    pb2[b] = (s == -INFINITY) ? (ushort)0 : to_bf(exp2(s - mn));
                    sum += bf(pb2[b]);
                }
                pk[d] = (uint)pb2[0] | ((uint)pb2[1] << 16);
            }
            ll[j] = ll[j] * al[j] + sum;
            vstore8((uint8)(pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6], pk[7]), 0, pt + ((sg * NT + j) * 16 + lane) * 8);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[j][0] *= al[j];
            acc[j][1] *= al[j];
        }
#pragma unroll
        for (int uu = 0; uu < 8; uu++) {
            if (kb + uu >= ntiles) break;
            int8 bp[NT];
#pragma unroll
            for (int j = 0; j < NT; j++) bp[j] = as_int8(vload8(0, pt + ((uu * NT + j) * 16 + lane) * 8));
            uint8 va;
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)vx, NKV * HD * 2, height, NKV * HD * 2, (int2)(hk * (HD / 2) + sg * 8, (int)((kb + uu) * 16)), (__private uint *)&va);
#pragma unroll
            for (int gg = 0; gg < 2; gg++) {
                const short8 a = as_short8(gg == 0 ? va.lo : va.hi);
#pragma unroll
                for (int j = 0; j < NT; j++) acc[j][gg] = MAD(a, bp[j], acc[j][gg]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NT; j++) mxs[sg][j][lane] = ll[j]; // the maxima are dead
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int j = 0; j < NT; j++) {
        float l = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; s++) l += mxs[s][j][lane];
        const float inv = 1.0f / l;
        const uint row = r0 + (lane & 7), head = hk * GQ + hgrp * HPW + 2 * j + (lane >> 3);
        if (row >= rows) continue;
#pragma unroll
        for (int g = 0; g < 2; g++) {
            ushort8 o;
#pragma unroll
            for (int i = 0; i < 8; i++) o[i] = to_bf(acc[j][g][i] * inv);
            vstore8(o, 0, out + ((ulong)row * NH + head) * HD + sg * 16 + g * 8);
        }
    }
}
