#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
// EXL3 prefill GEMM on 2D block loads, C^T = W X^T (the tiling of ggml_pfgemm.cl; built with -cl-intel-256-GRF-per-thread, see exl3.zig).
//   W  [Nc][K] fp16 row-major: the decoded weights of a column chunk (exl3_decr_*), row n = output column, k contiguous. One 2D read gives 32 columns x 32 k.
//   Xt [K][Rp] fp16 k-major: the rotated activations (exl3_rot_in_k), tokens padded with zeros to a multiple of 64.
//   C  [token][ldc] fp32, C[token * ldc + c_off + n]: lane = token, element = column, so a lane stores 8 consecutive columns as two 16 B vectors.
// A sub-group computes 32 columns x 64 tokens (16 accumulators of 8 x 16). Every output accumulates its k tiles in ascending order in one DPAS chain with fp32 accumulation
// (the order of exl3_pfgemm: the bits are those of the fragment-layout GEMM).
// Grid (ceil(R / 64), Nc / 32), token blocks the fast axis, local 16.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void exl3_pfgemm2d(__global const ushort *W, __global const ushort *Xt, __global float *C, uint R, uint Rp, uint K, uint Nc, uint ldc, uint c_off) {
    const uint tb = get_group_id(0), nb = get_group_id(1), lane = get_sub_group_local_id();
    const int n0 = nb * 32, t0 = tb * 64;
    float8 acc[4][4];
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
        for (int u = 0; u < 4; u++) acc[g][u] = (float8)(0.0f);
    for (int k0 = 0; k0 < (int)K; k0 += 32) {
        ushort ab[64] __attribute__((aligned(16)));
        intel_sub_group_2d_block_read_16b_32r16x2c((__global void *)W, K * 2, Nc, K * 2, (int2)(k0, n0), ab);
#pragma unroll
        for (int kh = 0; kh < 2; kh++) {
#pragma unroll
            for (int u2 = 0; u2 < 2; u2++) {
                uint bb[16] __attribute__((aligned(16)));
                intel_sub_group_2d_block_read_transform_16b_16r16x2c((__global void *)Xt, Rp * 2, K, Rp * 2, (int2)(t0 + u2 * 32, k0 + kh * 16), bb);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int8 b = *(__private int8 *)(bb + 8 * h);
#pragma unroll
                    for (int g = 0; g < 4; g++) {
                        const short8 a = *(__private short8 *)(ab + kh * 32 + g * 8);
                        acc[g][u2 * 2 + h] = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc[g][u2 * 2 + h]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const uint tok = t0 + u * 16 + lane;
            if (tok < R) {
                __global float *o = C + (ulong)tok * ldc + c_off + n0 + g * 8;
                vstore4(acc[g][u].s0123, 0, o);
                vstore4(acc[g][u].s4567, 1, o);
            }
        }
}
