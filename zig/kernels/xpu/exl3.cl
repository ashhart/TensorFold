// EXL3 (ExLlamaV3 trellis) quantized linear: tile decode, input/output Hadamard rotations, row-invariant DPAS matmul for any row count.
// Format and arithmetic follow ref/tensorfold-py exl3 (format.py, decode.cuh, linear.cu): y = ((((x*suh) H) Wq) H) * svh + bias.
// K2 = 2 * bits (2,3,4,5,6,7,8,10,12,14,16), CB = codebook (0 3inst, 1 mcg, 2 mul1). Kernels run in 16-lane sub-groups; exl3_mx work-groups
// are one sub-group (no shared memory, partial sums go to Z). Dtype codes: 0 fp16, 1 bf16, 2 fp32. Build: ocloc -options -cl-std=CL3.0 (dot product).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#define INL inline __attribute__((always_inline))
#define HAD_SCALE 0.08838834764831845f
// E(p): one past the last stream bit of value p's 16-bit window.
#define SE(K2, p) ((((K2) & 1) ? (((p) + 1) * (K2) - (((p) + 1) & 1)) / 2 : ((p) + 1) * ((K2) / 2)))

INL float bf(ushort v) { return as_float((uint)v << 16); }
INL ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

INL float ld(__global const void *p, uint dt, ulong i) {
    if (dt == 0) return (float)((__global const half *)p)[i];
    if (dt == 1) return bf(((__global const ushort *)p)[i]);
    return ((__global const float *)p)[i];
}

INL void st(__global void *p, uint dt, ulong i, float v) {
    if (dt == 0) ((__global half *)p)[i] = convert_half_rte(v);
    else if (dt == 1) ((__global ushort *)p)[i] = to_bf(v);
    else ((__global float *)p)[i] = v;
}

// The fp16 value (as float) of a 16-bit state; sums and products are exact in fp32, so the one rounding is the fp16 one.
INL float cbval(uint s, const int cb) { // s: a 16-bit state, upper bits zero or ignored by the callers' casts
    float r;
    if (cb == 2) {
        const uint x = s * 0x83DCD12Du;
        // The float 2^23 + byte sum straight from dp4a (no int->float convert): (2^23 + b) * S - 56771.453125 = (1024 + b) * S - 10.3828125 exactly,
        // with S = fp16(0x1EEE) = 1774 / 2^18 (2^23 S = 56768) and -10.3828125 = fp16(0xC931); every step fits in 24 bits.
        r = fma(as_float(dot_acc_sat_4x8packed_uu_uint(x, 0x01010101u, 0x4B000000u)), 0.00676727294921875f, -56771.453125f);
    } else {
        uint x = cb == 1 ? s * 0xCBAC1FEDu : s * 89226354u + 64248484u;
        x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
        r = (float)as_half((ushort)(x & 0xFFFFu)) + (float)as_half((ushort)(x >> 16));
    }
    return (float)convert_half_rte(r);
}

// (decode kernel) Lane l's 16 tile values (stream positions 16l..16l+15), from the tile's TW = 4*K2 words (circular bitstream, MSB first).
INL void tile_vals(__global const uint *tile, const uint l, const int K2, const int cb, float v[16]) {
    const int TW = 4 * K2;
    const uint first = SE(K2, 16 * l) - 16 + 128 * K2; // the first window's first bit, made non-negative
    const uint w0 = (first >> 5) % TW, off = first & 31;
    const int nw = ((SE(K2, 15) - SE(K2, 0)) >> 5) + 3;
    uint w[8], u[7];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint idx = w0 + i;
        idx = idx >= TW ? idx - TW : idx;
        w[i] = i < nw ? tile[idx] : 0u;
    }
#pragma unroll
    for (int i = 0; i < 7; i++) u[i] = (w[i] << off) | ((w[i + 1] >> 1) >> (31 - off)); // the stream from the lane's first window
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int d = SE(K2, j) - SE(K2, 0); // even stream positions only: the offset does not depend on the lane
        const uint win = ((u[d >> 5] << (d & 31)) | ((u[(d >> 5) + 1] >> 1) >> (31 - (d & 31)))) >> 16;
        v[j] = cbval(win, cb);
    }
}

// Unscaled Walsh-Hadamard transform of 128 values, index = lane + 16 * i.
INL void fwht8(float v[8], const uint lane) {
#pragma unroll
    for (int m = 1; m < 8; m <<= 1)
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (!(i & m)) {
                const float a = v[i], b = v[i | m];
                v[i] = a + b;
                v[i | m] = a - b;
            }
#pragma unroll
    for (int m = 1; m < 16; m <<= 1)
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float o = intel_sub_group_shuffle_xor(v[i], (uint)m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
}

// Output rotation of one row's 128 sums: H / sqrt(128), * svh, + bias, stored.
INL void finish(float s[8], const uint lane, __global const half *svh, __global const half *bias, uint has_bias, uint col0,
                __global void *y, uint ydt, ulong row_off) {
    fwht8(s, lane);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint col = col0 + lane + 16 * i;
        float r = s[i] * HAD_SCALE * (float)svh[col];
        if (has_bias) r += (float)bias[col];
        st(y, ydt, row_off + col, r);
    }
}

// Finish: sums the SK partials in order, then the output rotation. Grid (N/128, M), one sub-group each.
__attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_finish(__global const float *Z, __global const half *svh,
                                                                         __global const half *bias, uint has_bias, __global void *y,
                                                                         uint ydt, uint M, uint N, uint SK) {
    const uint lane = get_sub_group_local_id(), nb = get_group_id(0), row = get_group_id(1);
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = 0.0f;
    // the partials are summed in order; the usual split counts take a static trip count (all the loads of a batch issued together: the plain loop waits for every one)
    if (SK == 16 || SK == 8 || SK == 5 || SK == 4) {
        const uint sk = SK;
        for (uint q0 = 0; q0 < sk; q0 += (sk == 5 ? 5 : 4)) {
            float z[5][8];
#pragma unroll
            for (int j = 0; j < 5; j++)
#pragma unroll
                for (int i = 0; i < 8; i++) z[j][i] = (j < (sk == 5 ? 5 : 4)) ? Z[((ulong)(q0 + j) * M + row) * N + nb * 128 + lane + 16 * i] : 0.0f;
#pragma unroll
            for (int j = 0; j < 5; j++)
                if (j < (sk == 5 ? 5 : 4))
#pragma unroll
                    for (int i = 0; i < 8; i++) s[i] += z[j][i];
        }
    } else {
        for (uint q = 0; q < SK; q++)
#pragma unroll
            for (int i = 0; i < 8; i++) s[i] += Z[((ulong)q * M + row) * N + nb * 128 + lane + 16 * i];
    }
    finish(s, lane, svh, bias, has_bias, nb * 128, y, ydt, (ulong)row * N);
}

// Tile (kt, nt) starts at T + kt * sk + (nt / 8) * snb + (nt % 8) * snt (words).
// Decode to W [K, N] row-major, fp16 (odt 0) or bf16 (odt 1). Grid (N/16, K/16), one sub-group a tile.
INL void dec_body(const int K2, const int cb, __global const uint *T, __global ushort *W, uint N, ulong sk, ulong snt, ulong snb, uint odt) {
    const uint nt = get_group_id(0), kt = get_group_id(1), lane = get_sub_group_local_id();
    float v[16];
    tile_vals(T + kt * sk + (nt >> 3) * snb + (nt & 7) * snt, lane, K2, cb, v);
#pragma unroll
    for (int jj = 0; jj < 16; jj++) {
        const int h = jj >> 3, j = jj & 7;
        const uint l = 2 * lane + h; // CUDA lane of the value: row 2 (l % 4) + (j & 1) + 8 ((j >> 1) & 1), column l / 4 + 8 (j >> 2)
        const uint row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1), col = (l >> 2) + 8 * (j >> 2);
        W[(ulong)(kt * 16 + row) * N + nt * 16 + col] = odt ? to_bf(v[jj]) : as_ushort(convert_half_rte(v[jj]));
    }
}

// Column n of a tile (n < 8: values j < 4, n >= 8: j >= 4) as the 8 packed fp16 pairs of a DPAS B operand: pair i = rows 2i, 2i+1.
// Its 16 values sit at stream positions 32 (n % 8) + 4 (n / 8) + 8q + t (q, t < 4; row 2q + (t & 1) + 8 (t >> 1)), all within 28 positions.
INL uint col_first(const uint n, const int K2) { return SE(K2, 32 * (n & 7) + 4 * (n >> 3)) - 16 + 128 * K2; }

// The lane's words of a tile (the circular stream words its 28 positions touch).
INL void col_load(__global const uint *tile, const uint n, const int K2, uint w[10]) {
    const int TW = 4 * K2;
    const uint w0 = (col_first(n, K2) >> 5) % TW;
    const int nw = ((SE(K2, 27) - SE(K2, 0)) >> 5) + 3;
#pragma unroll
    for (int i = 0; i < 10; i++) {
        uint idx = w0 + i;
        idx = idx >= TW ? idx - TW : idx;
        w[i] = i < nw ? tile[idx] : 0u;
    }
}

INL int8 col_decode(const uint w[10], const uint n, const int K2, const int cb) {
    int8 b;
    const uint off = col_first(n, K2) & 31;
    const uint mk = (1u << off) - 1u;
    uint R[10], u[9], c[8];
#pragma unroll
    for (int i = 0; i < 10; i++) R[i] = rotate(w[i], off);
#pragma unroll
    for (int i = 0; i < 9; i++) u[i] = bitselect(R[i], R[i + 1], mk); // the stream from the lane's first window, word i
#pragma unroll
    for (int i = 0; i < 8; i++) c[i] = bitselect(rotate(u[i], 16u), rotate(u[i + 1], 16u), 0xFFFFu); // the same stream 16 bits on: every window sits inside one word
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
        for (int jh = 0; jh < 2; jh++) {
            float2 v;
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int d = SE(K2, 8 * q + 2 * jh + e) - SE(K2, 0), a = d >> 5, bt = d & 31;
                const uint s = bt <= 16 ? (u[a] >> (16 - bt)) : (c[a] >> (32 - bt));
                if (e == 0) v.x = cbval((ushort)s, cb); else v.y = cbval((ushort)s, cb);
            }
            b[q + 4 * jh] = as_int(convert_half2_rte(v));
        }
    return b;
}

// DPAS matmul for 8 G rows a pass, tiles read nt-major (a 16-column tile column's k tiles contiguous):
// tile (kt, nt) at T + nb * snb + (nt % 8) * snt + kt * sk. The input is xt [row group][K][8] halves (exl3_rot_in_t), zero rows padded.
// One DPAS a tile and 8 rows; a row never sees the other rows, so its bits do not depend on M. NTU tiles of different nt are in flight; G row groups (8 rows each) share every decoded tile.
INL void mx_body(const int K2, const int cb, const int G, const int NTU, __global const short *xt, __global const uint *T, ulong sk, ulong snt, ulong snb,
                 __global float *Z, uint M, uint K, uint N, uint SK) {
    const uint nb = get_group_id(0), split = get_group_id(1), lane = get_sub_group_local_id();
    const uint per = (K >> 4) / SK, kt0 = split * per; // one sub-group a work-group: no shared memory, so every sub-group slot of the core is usable
    const int RR = 8 * G;
    const uint groups = (M + 7) >> 3;
    for (uint m0 = 0; m0 < M; m0 += RR) {
        {
            const uint nt0 = get_group_id(2) * NTU; // grid z: the two halves of a 128-column block are separate sub-groups
            float8 c[2][4]; // [row group][tile column]
#pragma unroll
            for (int g = 0; g < G; g++)
#pragma unroll
                for (int u = 0; u < NTU; u++) c[g][u] = (float8)(0.0f);
            __global const uint *base = T + nb * snb + nt0 * snt;
            for (uint i = 0; i < per; i++) {
                const uint kt = kt0 + i;
                short8 a[2];
#pragma unroll
                for (int g = 0; g < G; g++) a[g] = (m0 >> 3) + g < groups ? vload8(0, xt + ((ulong)((m0 >> 3) + g) * K + kt * 16 + lane) * 8) : (short8)(0);
                uint w[4][10];
#pragma unroll
                for (int u = 0; u < NTU; u++) col_load(base + u * snt + kt * sk, lane, K2, w[u]);
#pragma unroll
                for (int u = 0; u < NTU; u++) {
                    const int8 bv = col_decode(w[u], lane, K2, cb);
#pragma unroll
                    for (int g = 0; g < G; g++) c[g][u] = intel_sub_group_f16_f16_matrix_mad_k16(a[g], bv, c[g][u]);
                }
            }
            // the partial sums of this split: Z [split][row][N], lane = column of the tile (exl3_finish adds the splits in order)
#pragma unroll
            for (int g = 0; g < G; g++)
#pragma unroll
                for (int u = 0; u < NTU; u++)
#pragma unroll
                    for (int r = 0; r < 8; r++)
                        if (m0 + 8 * g + r < M) Z[((ulong)split * M + m0 + 8 * g + r) * N + nb * 128 + (nt0 + u) * 16 + lane] = c[g][u][r];
        }
    }
}

// Rotate the input for exl3_mx: xt [row group][K][8] = fp16(H (x * suh) / sqrt(128)), rows >= M zero. Grid (K/128, M rounded up to 8).
__attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_rot_in_t(__global const void *x, uint xdt, __global const half *suh,
                                                                           __global half *xt, uint K, uint M) {
    const uint lane = get_sub_group_local_id(), row = get_group_id(1), k0 = get_group_id(0) * 128 + lane;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = row < M ? ld(x, xdt, (ulong)row * K + k0 + 16 * i) * (float)suh[k0 + 16 * i] : 0.0f;
    fwht8(v, lane);
#pragma unroll
    for (int i = 0; i < 8; i++) xt[((ulong)(row >> 3) * K + k0 + 16 * i) * 8 + (row & 7)] = convert_half_rte(v[i] * HAD_SCALE);
}

// Rotated input for the 2D-load prefill GEMM: xt [K][Rp] (k-major), rows M .. Rp zero. The 2-byte stores of one row to 128 different k rows were scattered; a work-group of 32 sub-groups
// (32 rows) rotates its rows into local memory and writes 128 k x 32 rows as 64 B runs. Grid (K/128, Rp/32), local 512; Rp a multiple of 32.
__attribute__((reqd_work_group_size(512, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_rot_in_k(__global const void *x, uint xdt, __global const half *suh,
                                                                           __global half *xt, uint K, uint M, uint Rp) {
    __local ushort tile[128 * 40]; // [k][32 rows + 8 padding]
    const uint lane = get_sub_group_local_id(), sg = get_sub_group_id(), row = get_group_id(1) * 32 + sg, k0 = get_group_id(0) * 128 + lane;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = row < M ? ld(x, xdt, (ulong)row * K + k0 + 16 * i) * (float)suh[k0 + 16 * i] : 0.0f;
    fwht8(v, lane);
#pragma unroll
    for (int i = 0; i < 8; i++) tile[(lane + 16 * i) * 40 + sg] = as_ushort(convert_half_rte(v[i] * HAD_SCALE));
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint t = get_local_id(0); // 512 items: 128 k x 4 vectors of 8 rows
    const uint k = t >> 2, part = t & 3;
    const ushort8 val = vload8(0, tile + k * 40 + part * 8);
    vstore8(val, 0, (__global ushort *)xt + (ulong)(get_group_id(0) * 128 + k) * Rp + get_group_id(1) * 32 + part * 8);
}

#define MX(K2, CB, G, NT) \
    __attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_mx_##K2##_##CB##_##G( \
        __global const short *xt, __global const uint *T, ulong sk, ulong snt, ulong snb, __global float *Z, uint M, uint K, uint N, uint SK) { \
        mx_body(K2, CB, G, NT, xt, T, sk, snt, snb, Z, M, K, N, SK); \
    }

#define DEC(K2, CB) \
    __attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_dec_##K2##_##CB( \
        __global const uint *T, __global ushort *W, uint N, ulong sk, ulong snt, ulong snb, uint odt) { \
        dec_body(K2, CB, T, W, N, sk, snt, snb, odt); \
    }
// Prefill path (large row counts): W_q decoded once into fp16 DPAS B fragments, Wv [kt][tile col in chunk][8 pairs][16 lanes] (int, pair = rows 2i, 2i+1
// of column lane), then one GEMM over all rows. Tile columns nt_first .. nt_first + ntc of the layer; grid (ntc, K/64).
INL void decv_body(const int K2, const int cb, __global const uint *T, __global uint *Wv, ulong sk, ulong snt, ulong snb, uint ntc, uint nt_first) {
    const uint c = get_group_id(0), lane = get_sub_group_local_id(), nt = nt_first + c;
    uint w[4][10];
    const uint kt0 = get_group_id(1) * 4; // four k tiles a sub-group, loaded together
#pragma unroll
    for (int j = 0; j < 4; j++) col_load(T + (kt0 + j) * sk + (nt >> 3) * snb + (nt & 7) * snt, lane, K2, w[j]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int8 b = col_decode(w[j], lane, K2, cb);
#pragma unroll
        for (int i = 0; i < 8; i++) Wv[((ulong)((kt0 + j) * ntc + c) * 8 + i) * 16 + lane] = (uint)b[i];
    }
}
// The same decode for the 2D-load GEMM: Wr [ntc * 16][K] fp16 row-major (row = column of the chunk), lane l of tile (kt, c) writes the 16 k values of column l as 32 contiguous bytes; the
// four k tiles of a sub-group are contiguous (128 B).
INL void decr_body(const int K2, const int cb, __global const uint *T, __global uint *Wr, ulong sk, ulong snt, ulong snb, uint ntc, uint nt_first, uint K) {
    const uint c = get_group_id(0), lane = get_sub_group_local_id(), nt = nt_first + c;
    uint w[4][10];
    const uint kt0 = get_group_id(1) * 4;
#pragma unroll
    for (int j = 0; j < 4; j++) col_load(T + (kt0 + j) * sk + (nt >> 3) * snb + (nt & 7) * snt, lane, K2, w[j]);
    int8 b[4];
#pragma unroll
    for (int j = 0; j < 4; j++) b[j] = col_decode(w[j], lane, K2, cb);
    __global uint *dst = Wr + (ulong)(c * 16 + lane) * (K / 2) + kt0 * 8; // 128 contiguous bytes: two 64 B stores
    vstore16((uint16)(as_uint8(b[0]), as_uint8(b[1])), 0, dst);
    vstore16((uint16)(as_uint8(b[2]), as_uint8(b[3])), 1, dst);
}
#define DECR(K2, CB) \
    __attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_decr_##K2##_##CB( \
        __global const uint *T, __global uint *Wr, ulong sk, ulong snt, ulong snb, uint ntc, uint nt_first, uint K) { \
        decr_body(K2, CB, T, Wr, sk, snt, snb, ntc, nt_first, K); \
    }
// decr_body with the stores through local memory: a work-group of 8 sub-groups takes one column tile (16 columns) and 512 k (8 x 4 k tiles); the decoded 16 x 512 block is assembled in
// local memory and written as 16 columns x 1 KB, 16 B a work-item, so one message writes 256 contiguous bytes (the stores of decr_body are 64 B a lane to 16 different rows). K % 512 == 0.
INL void decw_body(const int K2, const int cb, __global const uint *T, __global uint *Wr, ulong sk, ulong snt, ulong snb, uint ntc, uint nt_first, uint K, __local uint *tile) {
    const uint c = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id(), nt = nt_first + c;
    uint w[4][10];
    const uint kt0 = get_group_id(1) * 32 + sg * 4;
#pragma unroll
    for (int j = 0; j < 4; j++) col_load(T + (kt0 + j) * sk + (nt >> 3) * snb + (nt & 7) * snt, lane, K2, w[j]);
    int8 b[4];
#pragma unroll
    for (int j = 0; j < 4; j++) b[j] = col_decode(w[j], lane, K2, cb);
    // tile [16 columns][256 + 4 words]: column `lane`, this sub-group's 64 k at word sg * 32
    __local uint *dst = tile + lane * 260 + sg * 32;
#pragma unroll
    for (int j = 0; j < 4; j++) vstore8(as_uint8(b[j]), j, dst);
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint t = get_local_id(0); // 128 items: 16 columns x 64 vectors of 16 B
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint v = t + 128 * i; // 0 .. 1023
        const uint col = v >> 6, part = v & 63;
        const uint4 val = vload4(0, tile + col * 260 + part * 4);
        vstore4(val, 0, Wr + (ulong)(c * 16 + col) * (K / 2) + get_group_id(1) * 256 + part * 4);
    }
}
#define DECW(K2, CB) \
    __attribute__((reqd_work_group_size(128, 1, 1))) __attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_decw_##K2##_##CB( \
        __global const uint *T, __global uint *Wr, ulong sk, ulong snt, ulong snb, uint ntc, uint nt_first, uint K) { \
        __local uint tile[16 * 260]; \
        decw_body(K2, CB, T, Wr, sk, snt, snb, ntc, nt_first, K, tile); \
    }
#define DECV(K2, CB) \
    __attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_decv_##K2##_##CB( \
        __global const uint *T, __global uint *Wv, ulong sk, ulong snt, ulong snb, uint ntc, uint nt_first) { \
        decv_body(K2, CB, T, Wv, sk, snt, snb, ntc, nt_first); \
    }

// C [M][N] fp32 columns of the chunk = rotated rows x Wv; one sub-group a block of 16 rows x 16 GN columns (2 row groups x GN tile columns).
// Grid (ntc / GN, ceil(M / 16)); k tiles summed in order, so a row's bits depend only on the row and the chunk layout.
#ifndef GEMM_GN
#define GEMM_GN 2
#endif
__attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_gemm(__global const short *xt, __global const uint *Wv, __global float *C, uint M,
                                                                       uint K, uint N, uint ntc, uint nt_first) {
    const uint cb = get_group_id(0), rb = get_group_id(1), lane = get_sub_group_local_id();
    const uint groups = (M + 7) >> 3;
    float8 c[2][GEMM_GN];
#pragma unroll
    for (int g = 0; g < 2; g++)
#pragma unroll
        for (int u = 0; u < GEMM_GN; u++) c[g][u] = (float8)(0.0f);
#pragma unroll 2
    for (uint kt = 0; kt < (K >> 4); kt++) {
        short8 a[2];
#pragma unroll
        for (int g = 0; g < 2; g++) a[g] = 2 * rb + g < groups ? vload8(0, xt + ((ulong)(2 * rb + g) * K + kt * 16 + lane) * 8) : (short8)(0);
#pragma unroll
        for (int u = 0; u < GEMM_GN; u++) {
            const int8 bv = as_int8(intel_sub_group_block_read8(Wv + (ulong)(kt * ntc + cb * GEMM_GN + u) * 128));
#pragma unroll
            for (int g = 0; g < 2; g++) c[g][u] = intel_sub_group_f16_f16_matrix_mad_k16(a[g], bv, c[g][u]);
        }
    }
#pragma unroll
    for (int g = 0; g < 2; g++)
#pragma unroll
        for (int u = 0; u < GEMM_GN; u++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const uint row = rb * 16 + 8 * g + r;
                if (row < M) C[(ulong)row * N + (nt_first + cb * GEMM_GN + u) * 16 + lane] = c[g][u][r];
            }
}

// exl3_gemm in the tiling of the ggml prefill GEMM (ggml_quant.cl pfgemm): a sub-group takes 16 rows x 64 columns, row panels are the fastest grid dimension (a column block's
// weight tiles are read from DRAM once, then hit in the cache for the other row panels). xt must be padded to a whole number of 64-row panels (no bounds test on the loads).
// Grid (ceil(M / 16), ntc / 4); ntc a multiple of 4. The k tiles are summed in order: a row's bits depend on the row and the chunk layout only.
#define PF_PG 2
#define PF_PN 4
__attribute__((intel_reqd_sub_group_size(16))) __kernel void exl3_pfgemm(__global const short *xt, __global const uint *Wv, __global float *C, uint M,
                                                                         uint K, uint N, uint ntc, uint nt_first) {
    const uint rb = get_group_id(0), cb = get_group_id(1), lane = get_sub_group_local_id();
    float8 c[PF_PG][PF_PN];
#pragma unroll
    for (int g = 0; g < PF_PG; g++)
#pragma unroll
        for (int u = 0; u < PF_PN; u++) c[g][u] = (float8)(0.0f);
#pragma unroll 2
    for (uint kt = 0; kt < (K >> 4); kt++) {
        short8 a[PF_PG];
#pragma unroll
        for (int g = 0; g < PF_PG; g++) a[g] = vload8(0, xt + ((ulong)(PF_PG * rb + g) * K + kt * 16 + lane) * 8);
#pragma unroll
        for (int u = 0; u < PF_PN; u++) {
            const int8 bv = as_int8(intel_sub_group_block_read8(Wv + (ulong)(kt * ntc + cb * PF_PN + u) * 128));
#pragma unroll
            for (int g = 0; g < PF_PG; g++) c[g][u] = intel_sub_group_f16_f16_matrix_mad_k16(a[g], bv, c[g][u]);
        }
    }
#pragma unroll
    for (int g = 0; g < PF_PG; g++)
#pragma unroll
        for (int u = 0; u < PF_PN; u++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const uint row = (PF_PG * rb + g) * 8 + r;
                if (row < M) C[(ulong)row * N + (nt_first + cb * PF_PN + u) * 16 + lane] = c[g][u][r];
            }
}

#define VARIANT(K2, CB) DEC(K2, CB) DECV(K2, CB) DECR(K2, CB) MX(K2, CB, 1, 4) MX(K2, CB, 2, 4)

#ifdef EXL3_MUL1_ONLY
#define ALL_CB(K2) VARIANT(K2, 2)
#else
#define ALL_CB(K2) VARIANT(K2, 0) VARIANT(K2, 1) VARIANT(K2, 2)
#endif
ALL_CB(2) ALL_CB(4) ALL_CB(6) ALL_CB(8) ALL_CB(10) ALL_CB(12) ALL_CB(14) ALL_CB(16)
VARIANT(3, 2) VARIANT(5, 2) VARIANT(7, 2)
DECW(6, 2)
