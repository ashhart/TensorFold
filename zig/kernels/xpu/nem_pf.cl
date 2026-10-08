// Nemotron-H prefill support kernels (bf16 DPAS, weights decoded inside the GEMM, nothing materialised): the products are exact (bf16 x bf16 -> fp32) and the quantized weights enter
// as the integers 128 + q (bf16 0x4300 | q, two nibbles a dword with one shift and one mask), the group's scale and bias applied on the fp32 accumulators as in the decode path:
//   sum_k x_k (q_k s + b) = s * C1 + (b - 128 s) * Cs,   C1 = sum x_k (128 + q_k) over a group of 64 inputs, Cs = sum x_k (a ones-DPAS, shared by all columns).
// A row's value depends on its own activations and the weights only (fixed k order), so any chunking of a prompt gives the same bits.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
#define PG 2
#define PN 4
#define SORT_MAXSLOTS 8192 // 32 KB of local memory

// B fragment of k tile t (16 inputs) of a group's 8 words: dword d holds inputs 2d, 2d + 1 of the tile as bf16 (128 + q).
#define BFRAG(ws, t) ((int8)( \
    (int)(((((t) < 2 ? ((t) == 0 ? (ws).s0 : (ws).s2) : ((t) == 2 ? (ws).s4 : (ws).s6)) >> 0) & 0x000F000Fu) | 0x43004300u), \
    (int)(((((t) < 2 ? ((t) == 0 ? (ws).s0 : (ws).s2) : ((t) == 2 ? (ws).s4 : (ws).s6)) >> 4) & 0x000F000Fu) | 0x43004300u), \
    (int)(((((t) < 2 ? ((t) == 0 ? (ws).s0 : (ws).s2) : ((t) == 2 ? (ws).s4 : (ws).s6)) >> 8) & 0x000F000Fu) | 0x43004300u), \
    (int)(((((t) < 2 ? ((t) == 0 ? (ws).s0 : (ws).s2) : ((t) == 2 ? (ws).s4 : (ws).s6)) >> 12) & 0x000F000Fu) | 0x43004300u), \
    (int)(((((t) < 2 ? ((t) == 0 ? (ws).s1 : (ws).s3) : ((t) == 2 ? (ws).s5 : (ws).s7)) >> 0) & 0x000F000Fu) | 0x43004300u), \
    (int)(((((t) < 2 ? ((t) == 0 ? (ws).s1 : (ws).s3) : ((t) == 2 ? (ws).s5 : (ws).s7)) >> 4) & 0x000F000Fu) | 0x43004300u), \
    (int)(((((t) < 2 ? ((t) == 0 ? (ws).s1 : (ws).s3) : ((t) == 2 ? (ws).s5 : (ws).s7)) >> 8) & 0x000F000Fu) | 0x43004300u), \
    (int)(((((t) < 2 ? ((t) == 0 ? (ws).s1 : (ws).s3) : ((t) == 2 ? (ws).s5 : (ws).s7)) >> 12) & 0x000F000Fu) | 0x43004300u)))

// One work-group of E * parts items (item t: expert e = t % E, slice p = t / E of the slots): counts, padded segment offsets (multiples of pad rows), the list of experts with rows, the slot
// of every padded row (0xffffffff: padding), and meta = {total padded rows, largest count}. ids [slots] are the experts the slots chose. Slice p of an expert's rows follows the counts of the
// slices before it, so row_slot lists the slots of an expert in ascending order whatever `parts` is.
inline uint sid(__global const uint *ids, __local const uint *lids, bool staged, uint s) { return staged ? lids[s] : ids[s]; }

__kernel void moe_sort(__global const uint *ids, __global uint *cnt, __global uint *segoff, __global uint *row_slot, __global uint *meta, __global uint *elist, uint slots, uint E, uint pad) {
    __local uint lcp[1024]; // counts of slice p of expert e at [p * E + e]
    __local uint lo[257];
    __local uint lids[SORT_MAXSLOTS]; // the slots' experts, staged once
    const uint t = get_local_id(0), e = t % E, p = t / E, parts = get_local_size(0) / E;
    const bool staged = slots <= SORT_MAXSLOTS;
    if (staged)
        for (uint s = t; s < slots; s += get_local_size(0)) lids[s] = ids[s];
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint s0 = (uint)((ulong)p * slots / parts), s1 = (uint)((ulong)(p + 1) * slots / parts);
    uint c = 0;
    for (uint s = s0; s < s1; s++) c += sid(ids, lids, staged, s) == e;
    lcp[p * E + e] = c;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
        uint off = 0, mx = 0, na = 0;
        for (uint j = 0; j < E; j++) {
            uint tot = 0;
            for (uint q = 0; q < parts; q++) tot += lcp[q * E + j];
            lo[j] = off;
            off += (tot + pad - 1) / pad * pad;
            mx = max(mx, tot);
            if (tot) elist[1 + na++] = j;
        }
        lo[E] = off;
        elist[0] = na;
        meta[0] = off;
        meta[1] = mx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint base = lo[e];
    uint n = 0, tot = 0;
    for (uint q = 0; q < parts; q++) {
        if (q < p) n += lcp[q * E + e];
        tot += lcp[q * E + e];
    }
    if (p == 0) {
        cnt[e] = tot;
        segoff[e] = base;
    }
    for (uint s = s0; s < s1; s++)
        if (sid(ids, lids, staged, s) == e) row_slot[base + n++] = s;
    if (p == parts - 1)
        for (uint j = n; j < (tot + pad - 1) / pad * pad; j++) row_slot[base + j] = 0xffffffffu;
}

// The nibble pairs of a word are (j, j + 4), so k = 2j holds input j and k = 2j + 1 input j + 4 (and 8 more for the second word): the activation tile is read in that order.
inline uint perm16(uint k) { return (k & 8) + ((k & 1) ? ((k & 7) >> 1) + 4 : ((k & 7) >> 1)); }

// Activation prep (bf16 kept as is): X [R][K] -> xt [row group of 8][K][8 rows] (DPAS A operand: lane = k, element = row; rows >= R zero). One work-item a (k tile, row group): the 8 rows'
// 16 values are read as 16 B vectors and written as 16 vectors of the 8 rows' values of one k (16 B stores).
__kernel void pf_prep_b(__global const ushort *x, __global ushort *xt, uint R, uint K) {
    const uint kt = get_global_id(0), rg = get_global_id(1);
    if (kt >= K / 16) return;
    ushort rv[8][16];
#pragma unroll
    for (uint r = 0; r < 8; r++) {
        const uint row = rg * 8 + r;
        ushort8 lo = (ushort8)(0), hi = (ushort8)(0);
        if (row < R) {
            lo = vload8(0, x + (ulong)row * K + kt * 16);
            hi = vload8(1, x + (ulong)row * K + kt * 16);
        }
#pragma unroll
        for (uint j = 0; j < 8; j++) {
            rv[r][j] = lo[j];
            rv[r][8 + j] = hi[j];
        }
    }
#pragma unroll
    for (uint j = 0; j < 16; j++) {
        ushort8 o;
#pragma unroll
        for (uint r = 0; r < 8; r++) o[r] = rv[r][perm16(j)];
        vstore8(o, 0, xt + ((ulong)rg * K + kt * 16 + j) * 8);
    }
}

// The same for gathered rows: padded row pr takes row (slot / div) of x (zero for padding).
__kernel void pf_prep_gb(__global const ushort *x, __global ushort *xt, __global const uint *row_slot, uint K, uint div) {
    const uint kt = get_global_id(0), rg = get_global_id(1);
    if (kt >= K / 16) return;
    ushort rv[8][16];
#pragma unroll
    for (uint r = 0; r < 8; r++) {
        const uint slot = row_slot[rg * 8 + r];
        ushort8 lo = (ushort8)(0), hi = (ushort8)(0);
        if (slot != 0xffffffffu) {
            lo = vload8(0, x + (ulong)(slot / div) * K + kt * 16);
            hi = vload8(1, x + (ulong)(slot / div) * K + kt * 16);
        }
#pragma unroll
        for (uint j = 0; j < 8; j++) {
            rv[r][j] = lo[j];
            rv[r][8 + j] = hi[j];
        }
    }
#pragma unroll
    for (uint j = 0; j < 16; j++) {
        ushort8 o;
#pragma unroll
        for (uint r = 0; r < 8; r++) o[r] = rv[r][perm16(j)];
        vstore8(o, 0, xt + ((ulong)rg * K + kt * 16 + j) * 8);
    }
}

#define ONES(lane) ((int8)((int)(0x3F803F80u | ((lane) & (0x3F803F80u >> 31)))))

// The inner loop shared by the dense and grouped GEMMs: one sub-group, PG row groups (8 rows each) x PN column tiles (16 outputs each) over K inputs. crow0 = the first weight row
// (lane = row within the tile); a[g] is read from xt at row group rg0 + g.
#define GEMM_BODY_R(rg0, crow0, gi0, gi1) \
    float8 acc[PG][PN]; \
    _Pragma("unroll") for (int g = 0; g < PG; g++) \
        _Pragma("unroll") for (int u = 0; u < PN; u++) acc[g][u] = (float8)(0.0f); \
    const int8 ones = ONES(lane); \
    for (uint gi = (gi0); gi < (gi1); gi++) { \
        uint8 ws[PN]; \
        float s[PN], bb[PN]; \
        _Pragma("unroll") for (int u = 0; u < PN; u++) { \
            const ulong crow = (crow0) + u * 16; \
            ws[u] = vload8(0, w + crow * (K / 8) + gi * 8); \
            s[u] = bf(sc[crow * groups + gi]); \
            bb[u] = bf(bi[crow * groups + gi]) - 128.0f * s[u]; \
        } \
        float8 c1[PG][PN], cs[PG]; \
        _Pragma("unroll") for (int g = 0; g < PG; g++) { \
            cs[g] = (float8)(0.0f); \
            _Pragma("unroll") for (int u = 0; u < PN; u++) c1[g][u] = (float8)(0.0f); \
        } \
        _Pragma("unroll") for (int t = 0; t < 4; t++) { \
            const uint kt = gi * 4 + t; \
            short8 a[PG]; \
            _Pragma("unroll") for (int g = 0; g < PG; g++) { \
                a[g] = as_short8(vload8(0, xt + ((ulong)((rg0) + g) * K + kt * 16 + lane) * 8)); \
                cs[g] = intel_sub_group_bf16_bf16_matrix_mad_k16(a[g], ones, cs[g]); \
            } \
            _Pragma("unroll") for (int u = 0; u < PN; u++) { \
                const int8 bv = BFRAG(ws[u], t); \
                _Pragma("unroll") for (int g = 0; g < PG; g++) c1[g][u] = intel_sub_group_bf16_bf16_matrix_mad_k16(a[g], bv, c1[g][u]); \
            } \
        } \
        _Pragma("unroll") for (int g = 0; g < PG; g++) \
            _Pragma("unroll") for (int u = 0; u < PN; u++) acc[g][u] = fma(c1[g][u], (float8)(s[u]), fma(cs[g], (float8)(bb[u]), acc[g][u])); \
    }

#define GEMM_BODY(rg0, crow0) GEMM_BODY_R(rg0, crow0, 0u, groups)

// Dense: grid (ceil(R / 16), N / 64); y [R][N] at y_off, bf16 or fp32.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pfgemm_b(__global const ushort *xt, __global const uint *w, __global const ushort *sc, __global const ushort *bi, __global void *y, uint R, uint K, uint N,
                       uint y_off, uint f32out) {
    const uint rb = get_group_id(0), cb = get_group_id(1), lane = get_sub_group_local_id();
    const uint groups = K / 64;
    GEMM_BODY(PG * rb, (ulong)cb * (PN * 16) + lane)
#pragma unroll
    for (int g = 0; g < PG; g++)
#pragma unroll
        for (int u = 0; u < PN; u++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const uint row = (PG * rb + g) * 8 + r;
                if (row < R) {
                    const ulong at = (ulong)row * N + y_off + (cb * PN + u) * 16 + lane;
                    if (f32out) ((__global float *)y)[at] = acc[g][u][r];
                    else ((__global ushort *)y)[at] = to_bf(acc[g][u][r]);
                }
            }
}

// Grouped (stacked experts [E][N][K/8]): work-group (row panel rp of the expert's segment, column block cb, expert e). Output of row i (slot s) column col: mode 0:
// bf16(relu(bf16(acc))^2) at out16[s * N + col]; mode 1: fp32 at out32[s * N + col].
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pfgemm_moe_b(__global const ushort *xt, __global const uint *w, __global const ushort *sc, __global const ushort *bi, __global ushort *out16, __global float *out32,
                           __global const uint *cnt, __global const uint *segoff, __global const uint *row_slot, uint K, uint N, uint mode) {
    const uint rp = get_group_id(0), cb = get_group_id(1), e = get_group_id(2), lane = get_sub_group_local_id();
    const uint ce = cnt[e];
    if (rp * 16 >= ce) return;
    const uint seg = segoff[e];
    const uint groups = K / 64;
    const uint rbase = seg / 8 + rp * 2; // row group of the panel's first 8 rows
    GEMM_BODY(rbase, (ulong)e * N + (ulong)cb * (PN * 16) + lane)
#pragma unroll
    for (int g = 0; g < PG; g++)
#pragma unroll
        for (int r = 0; r < 8; r++) {
            const uint row = rp * 16 + g * 8 + r;
            if (row < ce) {
                const ulong slot = row_slot[seg + row];
#pragma unroll
                for (int u = 0; u < PN; u++) {
                    const uint col = (cb * PN + u) * 16 + lane;
                    const float v = acc[g][u][r];
                    if (mode == 0) {
                        const float h = fmax(bf(to_bf(v)), 0.0f);
                        out16[slot * N + col] = to_bf(h * h);
                    } else out32[slot * N + col] = v;
                }
            }
        }
}

// relu2 in place over bf16 values: x = bf16(relu(x)^2), one work-item a value.
__kernel void relu2_bf16(__global ushort *x, uint n) {
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float h = fmax(bf(x[i]), 0.0f);
    x[i] = to_bf(h * h);
}

// ---- split-K forms for windows of up to 16 rows (decode is a window of one row): the k groups are cut in S ranges by the shape only, partial fp32 results z [S][16][N]
// (padding rows come out as zero), summed in range order by a finish kernel, so a row's bits do not depend on the window.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pfgemm_bs(__global const ushort *xt, __global const uint *w, __global const ushort *sc, __global const ushort *bi, __global float *z, uint K, uint N, uint S) {
    const uint cb = get_group_id(0), sp = get_group_id(1), lane = get_sub_group_local_id();
    const uint groups = K / 64, gper = groups / S;
    GEMM_BODY_R(0, (ulong)cb * (PN * 16) + lane, sp * gper, sp * gper + gper)
#pragma unroll
    for (int g = 0; g < PG; g++)
#pragma unroll
        for (int u = 0; u < PN; u++)
#pragma unroll
            for (int r = 0; r < 8; r++) z[((ulong)sp * 16 + g * 8 + r) * N + (cb * PN + u) * 16 + lane] = acc[g][u][r];
}

// y[r][y_off + col] = bf16 (or fp32 for f32out 1; bf16(relu(bf16(sum))^2) for f32out 2) of the S partials summed in order, r < n. One work-item an output; grid (ceil(N / 64), n), 64 items.
__kernel void fin_bs(__global const float *z, __global void *y, uint n, uint N, uint S, uint y_off, uint f32out) {
    const uint col = get_global_id(0), r = get_group_id(1);
    if (col >= N || r >= n) return;
    float s = 0.0f;
    for (uint q = 0; q < S; q++) s += z[((ulong)q * 16 + r) * N + col];
    const ulong at = (ulong)r * N + y_off + col;
    if (f32out == 1) ((__global float *)y)[at] = s;
    else if (f32out == 2) {
        const float h = fmax(bf(to_bf(s)), 0.0f);
        ((__global ushort *)y)[at] = to_bf(h * h);
    } else ((__global ushort *)y)[at] = to_bf(s);
}

// Grouped, split-K: grid (1, N / 64, min(E, slots) * S) (an expert has at most 16 rows in a window of 16): work-group z = j * S + s computes the group range s of the j-th chosen expert for the slots
// routed to it; partials zm [S][slots][N] fp32 (slot = the gathered row's slot).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pfgemm_moe_bs(__global const ushort *xt, __global const uint *w, __global const ushort *sc, __global const ushort *bi, __global float *zm, __global const uint *cnt,
                            __global const uint *segoff, __global const uint *row_slot, __global const uint *elist, uint K, uint N, uint S, uint slots) {
    const uint cb = get_group_id(1), zi = get_group_id(2), lane = get_sub_group_local_id();
    if (zi / S >= elist[0]) return;
    const uint e = elist[1 + zi / S], sp = zi % S;
    const uint ce = cnt[e];
    const uint seg = segoff[e];
    const uint groups = K / 64, gper = groups / S;
    const uint rbase = seg / 8;
    GEMM_BODY_R(rbase, (ulong)e * N + (ulong)cb * (PN * 16) + lane, sp * gper, sp * gper + gper)
#pragma unroll
    for (int g = 0; g < PG; g++)
#pragma unroll
        for (int r = 0; r < 8; r++) {
            const uint row = g * 8 + r;
            if (row < ce) {
                const ulong slot = row_slot[seg + row];
#pragma unroll
                for (int u = 0; u < PN; u++) zm[((ulong)sp * slots + slot) * N + (cb * PN + u) * 16 + lane] = acc[g][u][r];
            }
        }
}

// Sum of the S partials of every slot and column, in order: mode 0: act[slot][col] = bf16(relu(bf16(sum))^2); mode 1: fp32 ey[slot][col]. Grid (ceil(N / 64), slots), 64 items.
__kernel void fin_moe(__global const float *zm, __global ushort *out16, __global float *out32, uint N, uint S, uint slots, uint mode) {
    const uint col = get_global_id(0), slot = get_group_id(1);
    if (col >= N) return;
    float s = 0.0f;
    for (uint q = 0; q < S; q++) s += zm[((ulong)q * slots + slot) * N + col];
    if (mode == 0) {
        const float h = fmax(bf(to_bf(s)), 0.0f);
        out16[(ulong)slot * N + col] = to_bf(h * h);
    } else out32[(ulong)slot * N + col] = s;
}
