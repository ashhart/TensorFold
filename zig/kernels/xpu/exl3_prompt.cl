// SPDX-License-Identifier: Apache-2.0

// Prompt-row kernels included from exl3.cl when MOE_P and BITS are set.

// fp32 rot plane, exl3_gemm, exl3_moe_tile: same fp32 FMA order as exl3_matvec.

#ifndef GEMM_M
#define GEMM_M 16                         // rows a sub-group
#endif

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
    __global float *rot = AT(float, d.rot) + (ulong)row * d.k;
    for (uint j = 0; j < 8; j++) rot[blk * 128 + lane + SG * j] = v[j] * INV_SQRT128;
}

#ifndef GEMM_S
#define GEMM_S 4                          // 16-column strips a work-group (N divides by 16 * GEMM_S)
#endif
#ifndef GEMM_KC
#define GEMM_KC 8                         // k tiles decoded into local memory between two barriers (K divides by 16 * KC)
#endif
#define GEMM_WM (8 * GEMM_M)              // rows a work-group: 8 sub-groups of GEMM_M rows

__attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void exl3_gemm(__global const uchar *base, __global const exl3_desc *desc, uint rows) {
    exl3_desc d = desc[get_group_id(2)];
    d.rows = min(d.rows, rows);
    const uint q = get_sub_group_id(), c = get_sub_group_local_id();
    const uint g = c % 8, h = c / 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
    const uint tn0 = get_group_id(0) * GEMM_S, w0 = get_group_id(1) * GEMM_WM, m0 = w0 + q * GEMM_M;
    if (tn0 * 16 >= d.n || w0 >= d.rows || d.halves != 2 * BITS) return;
    __global const uint *tile = (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn0 * 16 * BITS);
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;
    __global const float *rot = AT(const float, d.rot);
    float acc[GEMM_S][GEMM_M];
    for (uint s = 0; s < GEMM_S; s++)
        for (uint r = 0; r < GEMM_M; r++) acc[s][r] = 0.0f;
    for (uint kb = 0; kb < d.k; kb += 16, tile += tile_k) {
        uint raw[GEMM_S][LOADW], w[GEMM_S][USEW];
        for (uint s = 0; s < GEMM_S; s++)
            for (uint j = 0; j < LOADW; j++) raw[s][j] = tile[s * RINGW + (wbase + j) % RINGW];
        for (uint s = 0; s < GEMM_S; s++)
            for (uint j = 0; j < USEW; j++)
                w[s][j] = h ? (SHB ? (raw[s][j + SHW] << SHB) | (raw[s][j + SHW + 1] >> (32 - SHB)) : raw[s][j + SHW]) : raw[s][j];
        float xr[GEMM_M];
        for (uint r = 0; r < GEMM_M; r++) {
            const uint row = m0 + r;
            xr[r] = row < d.rows ? rot[(ulong)row * d.k + kb + c] : 0.0f;
        }
        for (uint v = 0; v < 16; v++) {
            const uint bit = OFF0 + (8 * (v / 4) + (v % 4)) * BITS, wi = bit / 32, off = bit % 32;
            const uint kr = lane_row(v);
            float wv[GEMM_S];
            for (uint s = 0; s < GEMM_S; s++)
                wv[s] = mul1_value(off ? ((w[s][wi] << off) | (w[s][wi + 1] >> (32 - off))) >> 16 : w[s][wi] >> 16);
            for (uint r = 0; r < GEMM_M; r++) {
                const float xv = sub_group_broadcast(xr[r], kr);
                for (uint s = 0; s < GEMM_S; s++) acc[s][r] = fma(xv, wv[s], acc[s][r]);
            }
        }
    }
    __global float *sums = AT(float, d.sums) + tn0 * 16 + c;
    for (uint s = 0; s < GEMM_S; s++)
        for (uint r = 0; r < GEMM_M; r++) {
            const uint row = m0 + r;
            if (row < d.rows) sums[(ulong)row * d.n + s * 16] = acc[s][r];
        }
}

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

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_moe_tile(__global const uchar *base, __global const exl3_desc *t0, __global const exl3_desc *t1,
                            __global const uint *order, __global const uint4 *tiles, __global const uint *ntiles,
                            ulong a, ulong a_proj, ulong sums, uint s_stride, uint s_proj) {
    if (get_group_id(1) >= ntiles[0]) return;
    const uint4 tl = tiles[get_group_id(1)];
    const uint proj = get_group_id(2), c = get_sub_group_local_id(), g = c % 8, h = c / 8;
    const uint wbase = ((32 * g * BITS + BITS + 32 * RINGW - 16) / 32) % RINGW;
    const exl3_desc d = (proj ? t1 : t0)[tl.x];
    const uint tn0 = get_global_id(0) / SG * TILE_NS;
    if (tn0 * 16 >= d.n || d.halves != 2 * BITS) return;
    __global const uint *tile = (__global const uint *)(AT(const ushort, d.trellis) + (ulong)tn0 * 16 * BITS);
    const ulong tile_k = (ulong)(d.n / 16) * RINGW;
    __global const float *ap = AT(const float, a) + proj * a_proj;
    uint slot[TILE_M];
    for (uint i = 0; i < TILE_M; i++) slot[i] = order[tl.y + min(i, tl.z - 1)];
    float acc[TILE_NS][TILE_M];
    for (uint s = 0; s < TILE_NS; s++)
        for (uint i = 0; i < TILE_M; i++) acc[s][i] = 0.0f;
    for (uint kb = 0; kb < d.k; kb += 16, tile += tile_k) {
        uint raw[TILE_NS][LOADW], w[TILE_NS][USEW];
        for (uint s = 0; s < TILE_NS; s++)
            for (uint j = 0; j < LOADW; j++) raw[s][j] = tile[s * RINGW + (wbase + j) % RINGW];
        for (uint s = 0; s < TILE_NS; s++)
            for (uint j = 0; j < USEW; j++)
                w[s][j] = h ? (SHB ? (raw[s][j + SHW] << SHB) | (raw[s][j + SHW + 1] >> (32 - SHB)) : raw[s][j + SHW]) : raw[s][j];
        float xr[TILE_M];
        for (uint i = 0; i < TILE_M; i++)
            xr[i] = i < tl.z ? ap[(ulong)slot[i] * d.k + kb + c] : 0.0f;
        for (uint v = 0; v < 16; v++) {
            const uint bit = OFF0 + (8 * (v / 4) + (v % 4)) * BITS, wi = bit / 32, off = bit % 32;
            const uint kr = lane_row(v);
            float wv[TILE_NS];
            for (uint s = 0; s < TILE_NS; s++)
                wv[s] = mul1_value(off ? ((w[s][wi] << off) | (w[s][wi + 1] >> (32 - off))) >> 16 : w[s][wi] >> 16);
            for (uint i = 0; i < TILE_M; i++) {
                if (i >= tl.z) continue;
                const float xv = sub_group_broadcast(xr[i], kr);
                for (uint s = 0; s < TILE_NS; s++) acc[s][i] = fma(xv, wv[s], acc[s][i]);
            }
        }
    }
    __global float *o = AT(float, sums) + proj * s_proj + (tn0 * 16 + c);
    for (uint i = 0; i < tl.z; i++)
        for (uint s = 0; s < TILE_NS; s++) o[(ulong)slot[i] * s_stride + s * 16] = acc[s][i];
}

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
