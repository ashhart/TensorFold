#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Qwen3.8 full-attention for prompt windows on the matrix engine, all six query heads of a kv head in one work-group. Builds (compiler options):
//   (none)    bf16 KV cache read directly, bf16 DPAS (as the earlier qwen_attn_pf.cl: Q, K, V, P in bf16, fp32 accumulation);
//   -DFP16    fp16 K' / V' (the quantized cache expanded by kvq_expand), fp16 DPAS, P x 1024 in fp16 (the factor cancels against the denominator and keeps small P out of
//             the fp16 subnormals); -DKVQ=8 / 4 (with -DFP16) add the kernel kvq_expand for the q8 / q4 records of qwen_kvq.cl.
// Work-group: 8 sub-groups, 8 rows x 6 heads = 48 columns (3 matrix-engine tiles of 16 columns: two heads x 8 rows), one kv head, blocks of 128 keys = 8 key tiles of 16.
// The scores are computed transposed, S^T = K Q^T: A = a K tile (8 keys x 16 dims, a 2D block read), B = the queries of 16 columns, lane = column. So a lane holds the
// scores of ITS column (head, row) for the keys of the tile: the softmax maximum is a lane-local reduction, the maximum / rescale / denominator are one scalar a column,
// and the P tile comes out in the layout of the B operand of the second product (key pairs in dwords) with no transposition.
//  - sub-group s takes key tile s of the block for all 48 columns (6 DPAS a 16-dim step), tile maxima go through local memory, P = exp2(s - m) is rounded to 16 bits
//    (the denominator sums the rounded P) and stored to local memory (the other sub-groups need it);
//  - O^T += V^T P^T: A = V tile (8 dims x 16 keys, a transposed 2D block read gives lane = key and 8 consecutive dims), B = P^T. Sub-group s owns the output dims
//    32 s .. 32 s + 31 of all columns.
// The key tiles are absolute and rows are independent: a row's bits depend only on its own position (any window width or chunking gives the same result). The keys
// come in ranges (the expanded scratch holds one); the online-softmax state of a work-group is carried through global memory from one range to the next, which does
// not change a bit either.
#define HD 256
#define NH 24
#define NKV 4
#define GQ 6
#ifdef FP16
#define MAD intel_sub_group_f16_f16_matrix_mad_k16
#define PSC 1024.0f
inline ushort pcvt(float f) { return as_ushort(convert_half_rte(f)); }
inline float pback(ushort h) { return convert_float(as_half(h)); }
#else
#define MAD intel_sub_group_bf16_bf16_matrix_mad_k16
#define PSC 1.0f
inline ushort pcvt(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float pback(ushort v) { return as_float((uint)v << 16); }
#endif
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float bfr(float f) { return bf(to_bf(f)); }

// Queries q [row][24][256] bf16 -> B-operand tiles qb [row tile of 8][kv head][N tile][16-dim step][dword][lane = column] (dword e of step t = dims 16 t + 2 e, + 1 of
// the column's query, in the matrix engine's 16-bit format; column n of N tile j = head 2 j + (n >> 3) of the group, row n & 7; rows past `rows` are zero).
// Global size (ceil(rows / 8) * 4 * 3 * 2048).
__kernel void attn_pfs_prep(__global const ushort *q, __global uint *qb, uint rows) {
    const ulong g = get_global_id(0);
    const uint n = g & 15, e = (g >> 4) & 7, t = (g >> 7) & 15, j = (g >> 11) % 3;
    const ulong wgi = (g >> 11) / 3; // row tile * 4 + kv head
    const uint hk = wgi & 3, row = (wgi >> 2) * 8 + (n & 7), head = hk * GQ + 2 * j + (n >> 3);
    uint v = 0;
    if (row < rows) {
        const ushort2 x = vload2(0, q + ((ulong)row * NH + head) * HD + t * 16 + 2 * e);
#ifdef FP16
        v = as_uint(convert_half2_rte((float2)(bf(x.x), bf(x.y))));
#else
        v = (uint)x.x | ((uint)x.y << 16);
#endif
    }
    qb[g] = v;
}

#ifdef KVQ
#if KVQ == 8
#define REC 272
#define DATA 256
#else
#define REC 144
#define DATA 128
#endif
// Quantized records of the keys k0 .. k0 + n - 1 -> fp16 [key - k0][4 kv heads][256] (q * block scale, one rounding). Global size n * 4 * 32: one item an 8-dim group.
__kernel void kvq_expand(__global const uchar *src, __global ushort *dst, uint k0) {
    const ulong g = get_global_id(0);
    const uint d8 = g & 31, h = (g >> 5) & 3;
    const ulong key = g >> 7;
    const __global uchar *rec = src + ((key + k0) * NKV + h) * REC;
    const float sc = vload_half(d8 >> 2, (const __global half *)(rec + DATA));
    ushort8 o;
#if KVQ == 8
    const uint2 w = vload2(0, (const __global uint *)(rec + d8 * 8));
#pragma unroll
    for (int i = 0; i < 4; i++) {
        o[i] = as_ushort(convert_half_rte((float)((int)(w.x << (24 - 8 * i)) >> 24) * sc));
        o[4 + i] = as_ushort(convert_half_rte((float)((int)(w.y << (24 - 8 * i)) >> 24) * sc));
    }
#else
    const uint w = *(const __global uint *)(rec + d8 * 4);
#pragma unroll
    for (int p = 0; p < 4; p++) {
        o[2 * p] = as_ushort(convert_half_rte((float)((int)((w >> (4 * p)) & 0xFu) - 8) * sc));
        o[2 * p + 1] = as_ushort(convert_half_rte((float)((int)((w >> (4 * p + 16)) & 0xFu) - 8) * sc));
    }
#endif
    vstore8(o, 0, dst + (key * NKV + h) * HD + d8 * 8);
}
#endif

// The state of a work-group between key ranges: [work-group][sub-group][STSZ floats]: O (12 vectors of 8 a lane), then m (3) and l (3) a lane.
#define STSZ 1632
inline __global float *stbase(__global float *st, uint wgi, uint sg) { return st + ((ulong)wgi * 8 + sg) * STSZ; }

// work-group: 8 sub-groups; grid (ceil(rows / 8), 4 kv heads).
// qb: the query tiles from attn_pfs_prep, kx / vx: the K / V rows (16-bit) of the keys koff .. [key - koff][4][256], qg [row][24][512], out [row][24][256]; row z is at
// position pos0 + z. This launch covers the keys kstart .. kend (kstart a multiple of 128, inside koff ..); flags: bit 0 first range (state starts empty), bit 1 last range.
__attribute__((reqd_work_group_size(128, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_prefill_s(__global const uint *qb, __global const ushort *kx, __global const ushort *vx, __global const ushort *qg, __global ushort *out,
                             __global float *st, uint pos0, uint rows, uint koff, uint kstart, uint kend, uint flags) {
    __local float mxs[8][3][16]; // [sub-group][N tile][column]: tile maxima, later denominators
    __local uint pt[8 * 3 * 16 * 8]; // P^T: [key tile][N tile][column][8 dwords of key pairs]
    const uint hk = get_group_id(1);
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();
    const uint r0 = get_group_id(0) * 8;
    const uint kmax = pos0 + min(r0 + 8, rows); // keys any row of the work-group sees
    if (kstart >= kmax) return;                 // uniform: finished in an earlier range
    const uint lim = pos0 + r0 + (lane & 7);    // the last key of this lane's row
    const uint wgi = get_group_id(0) * 4 + hk;
    const __global uint *qt = qb + (ulong)wgi * (3 * 2048); // this work-group's query tiles [N tile][step][dword][lane]
    const uint kfin = min(kend, kmax);
    const uint ntiles = (kmax + 15) / 16; // tiles with a valid key
    const uint thi = (kfin + 15) / 16;    // end of this range
    const int height = (int)(kfin - koff);
    float m[3], ll[3];
    float8 acc[3][4]; // [N tile][8-dim group of the sub-group's 32 dims]: lane = column, element = dim
    if (flags & 1) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            m[j] = -INFINITY;
            ll[j] = 0.0f;
#pragma unroll
            for (int g = 0; g < 4; g++) acc[j][g] = (float8)(0.0f);
        }
    } else {
        const __global float *sb = stbase(st, wgi, sg);
#pragma unroll
        for (int j = 0; j < 3; j++) {
            m[j] = sb[1536 + j * 16 + lane];
            ll[j] = sb[1584 + j * 16 + lane];
#pragma unroll
            for (int g = 0; g < 4; g++) acc[j][g] = vload8(lane, sb + (j * 4 + g) * 128);
        }
    }
    for (uint kb = kstart / 16; kb < thi; kb += 8) {
        const uint tile = kb + sg;
        float8 c[3][2];
#pragma unroll
        for (int j = 0; j < 3; j++) c[j][0] = c[j][1] = (float8)(0.0f);
#pragma unroll
        for (int t = 0; t < 16; t++) {
            ushort16 ka;
            intel_sub_group_2d_block_read_16b_16r16x1c((__global void *)kx, 2048, height, 2048, (int2)(hk * 256 + t * 16, (int)(tile * 16 - koff)), (__private ushort *)&ka);
            const short8 a0 = as_short8(ka.lo), a1 = as_short8(ka.hi);
#pragma unroll
            for (int j = 0; j < 3; j++) {
                const int8 b = as_int8(intel_sub_group_block_read8(qt + (j * 16 + t) * 128));
                c[j][0] = MAD(a0, b, c[j][0]);
                c[j][1] = MAD(a1, b, c[j][1]);
            }
        }
#pragma unroll
        for (int j = 0; j < 3; j++) {
            float mx = -INFINITY;
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float s = (tile * 16 + h * 8 + i <= lim) ? c[j][h][i] * (0.0625f * 1.4426950408889634f) : -INFINITY;
                    c[j][h][i] = s;
                    mx = fmax(mx, s);
                }
            mxs[sg][j][lane] = mx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float al[3];
#pragma unroll
        for (int j = 0; j < 3; j++) {
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
                    pb2[b] = (s == -INFINITY) ? (ushort)0 : pcvt(exp2(s - mn) * PSC);
                    sum += pback(pb2[b]);
                }
                pk[d] = (uint)pb2[0] | ((uint)pb2[1] << 16);
            }
            ll[j] = ll[j] * al[j] + sum;
            vstore8((uint8)(pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6], pk[7]), 0, pt + ((sg * 3 + j) * 16 + lane) * 8);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int j = 0; j < 3; j++)
#pragma unroll
            for (int g = 0; g < 4; g++) acc[j][g] *= al[j];
#pragma unroll
        for (int uu = 0; uu < 8; uu++) {
            if (kb + uu >= thi) break;
            int8 bp[3];
#pragma unroll
            for (int j = 0; j < 3; j++) bp[j] = as_int8(vload8(0, pt + ((uu * 3 + j) * 16 + lane) * 8));
#pragma unroll
            for (int g2 = 0; g2 < 2; g2++) {
                uint8 va;
                intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)vx, 2048, height, 2048, (int2)(hk * 128 + sg * 16 + g2 * 8, (int)((kb + uu) * 16 - koff)), (__private uint *)&va);
#pragma unroll
                for (int gg = 0; gg < 2; gg++) {
                    const int g = g2 * 2 + gg;
                    const short8 a = as_short8(gg == 0 ? va.lo : va.hi);
#pragma unroll
                    for (int j = 0; j < 3; j++) acc[j][g] = MAD(a, bp[j], acc[j][g]);
                }
            }
        }
    }
    if (!(flags & 2) && kend < kmax) { // more ranges follow for this work-group: park the state
        __global float *sb = stbase(st, wgi, sg);
#pragma unroll
        for (int j = 0; j < 3; j++) {
            sb[1536 + j * 16 + lane] = m[j];
            sb[1584 + j * 16 + lane] = ll[j];
#pragma unroll
            for (int g = 0; g < 4; g++) vstore8(acc[j][g], lane, sb + (j * 4 + g) * 128);
        }
        return;
    }
#pragma unroll
    for (int j = 0; j < 3; j++) mxs[sg][j][lane] = ll[j]; // the maxima are dead
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int j = 0; j < 3; j++) {
        float l = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; s++) l += mxs[s][j][lane];
        const float inv = 1.0f / l;
        const uint row = r0 + (lane & 7), head = hk * GQ + 2 * j + (lane >> 3);
        if (row >= rows) continue;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const uint d0 = sg * 32 + g * 8;
            const ushort8 gv = vload8(0, qg + ((ulong)row * NH + head) * 2 * HD + HD + d0);
            ushort8 o;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float gate = bf(gv[i]);
                o[i] = to_bf(bfr(acc[j][g][i] * inv) * (1.0f / (1.0f + exp(-gate))));
            }
            vstore8(o, 0, out + ((ulong)row * NH + head) * HD + d0);
        }
    }
}
