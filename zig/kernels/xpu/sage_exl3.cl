// EXL3 (ExLlamaV3 trellis) projections on Xe2, first form: 1..16 rows, the stored tile order, the mul1 codebook.
// y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias, H the 128x128 Sylvester Hadamard / sqrt(128).
// The format is docs/recipes/exl3.md; zig/src/core/exl3_format.zig is the scalar reference these kernels follow.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
inline float bf(ushort v) { return as_float((uint)v << 16); }
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    if (isnan(f)) return (ushort)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

#define SG 16
#define MAXR 16
#define INV_SQRT128 0.08838834764831845f

// The 128-wide Hadamard of v, lane l holding indices l + 16 j (j = 0..7): four shuffle stages, then three in the lane.
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

// rot[row][k] = fp16(H_K (x * suh)) for one 128 block. Grid (K / 128, rows), one sub-group a work-group.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_rotate_in(__global const ushort *x, __global const half *suh, __global half *rot, uint k, uint x_stride) {
    const uint blk = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
    const uint base = blk * 128;
    float v[8];
    for (uint j = 0; j < 8; j++) {
        const uint i = base + lane + SG * j;
        v[j] = bf(x[(ulong)row * x_stride + i]) * vload_half(i, suh);
    }
    hadamard128(v, lane);
    for (uint j = 0; j < 8; j++) vstore_half(v[j] * INV_SQRT128, (ulong)row * k + base + lane + SG * j, rot);
}

// Two little-endian int16 words make one stream word, read from its most significant bit.
inline uint stream_word(__global const ushort *tile, uint i) { return (uint)tile[2 * i] | ((uint)tile[2 * i + 1] << 16); }

// The 16-bit state of value p: the 16 stream bits that end at E(p), wrapping past the tile's last bit.
inline uint tile_state(__global const ushort *tile, uint halves, uint p) {
    const uint ring = 128 * halves;
    const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
    const uint start = (end + ring - 16) % ring;
    const uint wi = start / 32, off = start % 32;
    const uint lo = stream_word(tile, wi);
    if (off <= 16) return (lo >> (16 - off)) & 0xffffu;
    const uint hi = stream_word(tile, (wi + 1) % (ring / 32));
    return ((lo << (off - 16)) | (hi >> (48 - off))) & 0xffffu;
}

// mul1: x = s * 0x83DCD12D; value = fp16((1024 + the sum of x's bytes) * fp16(0x1EEE) + fp16(0xC931)), one rounding.
inline float mul1_value(uint s) {
    const uint x = s * 0x83DCD12Du;
    const uint h = 1024u + (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
    const float scale = as_float(0x3BDDC000u); // fp16 0x1EEE as fp32
    const float bias = as_float(0xC1262000u);  // fp16 0xC931 as fp32
    return convert_float(convert_half_rte(fma((float)h, scale, bias)));
}

// sums[row][n] = rot[row] @ W_q[:, n] for one 16-column tile strip. Grid (N / 16, 1), one sub-group a work-group.
// Lane c owns column c of the strip; of a tile's 256 stream values, the 16 that land in column c are
// p = 32 (c % 8) + 8 m + 4 (c / 8) + i (m, i = 0..3), at tile row 2 m + (i & 1) + 8 (i >> 1).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_matvec(__global const ushort *trellis, __global const half *rot, __global float *sums,
                          uint k, uint n, uint rows, uint halves) {
    const uint tn = get_group_id(0), c = get_sub_group_local_id();
    const uint tiles_n = n / 16, words = 8 * halves;
    float acc[MAXR];
    for (uint r = 0; r < MAXR; r++) acc[r] = 0.0f;
    for (uint tk = 0; tk < k / 16; tk++) {
        __global const ushort *tile = trellis + ((ulong)tk * tiles_n + tn) * words;
        for (uint m = 0; m < 4; m++) {
            for (uint i = 0; i < 4; i++) {
                const uint p = 32 * (c % 8) + 8 * m + 4 * (c / 8) + i;
                const float w = mul1_value(tile_state(tile, halves, p));
                const uint kr = tk * 16 + 2 * m + (i & 1) + 8 * (i >> 1);
                for (uint r = 0; r < rows; r++) acc[r] = fma(vload_half((ulong)r * k + kr, rot), w, acc[r]);
            }
        }
    }
    for (uint r = 0; r < rows; r++) sums[(ulong)r * n + tn * 16 + c] = acc[r];
}

// out[row][n] = bf16((H_N sums) * svh + bias) for one 128 block. Grid (N / 128, rows); bias may be 0 (has_bias = 0).
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void exl3_rotate_out(__global const float *sums, __global const half *svh, __global const half *bias,
                              __global ushort *out, uint n, uint out_stride, uint has_bias) {
    const uint blk = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
    const uint base = blk * 128;
    float v[8];
    for (uint j = 0; j < 8; j++) v[j] = sums[(ulong)row * n + base + lane + SG * j];
    hadamard128(v, lane);
    for (uint j = 0; j < 8; j++) {
        const uint i = base + lane + SG * j;
        float y = v[j] * INV_SQRT128 * vload_half(i, svh);
        if (has_bias) y += vload_half(i, bias);
        out[(ulong)row * out_stride + i] = to_bf(y);
    }
}
