#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
// ggml prefill GEMM on 2D block loads, C^T = W X^T (built with -cl-intel-256-GRF-per-thread, see ggml.zig).
//   W  [N][K] fp16 row-major: the decoded weights (pfdec_*), row n = output channel, k contiguous. One 2D read gives 32 channels x 32 k (two DPAS A operands of 8 channels x 16 k per
//      channel group): lane = k, element = channel.
//   Xt [K][Rp] fp16 k-major: the activations (pf_prep), tokens padded with zeros to a multiple of 64. The VNNI transform read gives the B operand of 16 k x 16 tokens (lane = token).
//   y  [token][N] bf16, y[token * N + y_off + n]: lane = token, element = channel, so a lane stores 8 consecutive channels as one 16 B vector.
// A sub-group computes 32 channels x 64 tokens (16 accumulators of 8 x 16, 128 GRFs). Every output accumulates its k tiles in ascending order in one DPAS chain with fp32
// accumulation, so a token's bits depend only on its own activations and the weights: not on R, on the token block or on how a prompt is chunked.
// N a multiple of 16: the last 32-channel block may be half empty (the 2D read returns zeros past row N, the stores are guarded). Grid (ceil(R / 64), ceil(N / 32)), token blocks the fast axis (the 32 channel rows of W are then shared by the sub-groups that run together), local 16.
inline ushort to_bf(float f) {
    uint u = as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pfgemm(__global const ushort *W, __global const ushort *Xt, __global ushort *y, uint R, uint Rp, uint K, uint N, uint y_off) {
    const uint tb = get_group_id(0), nb = get_group_id(1), lane = get_sub_group_local_id();
    const int n0 = nb * 32, t0 = tb * 64;
    float8 acc[4][4];
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
        for (int u = 0; u < 4; u++) acc[g][u] = (float8)(0.0f);
    for (int k0 = 0; k0 < (int)K; k0 += 32) {
        ushort ab[64] __attribute__((aligned(16)));
        intel_sub_group_2d_block_read_16b_32r16x2c((__global void *)W, K * 2, N, K * 2, (int2)(k0, n0), ab);
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
            ushort8 o;
#pragma unroll
            for (int r = 0; r < 8; r++) o[r] = to_bf(acc[g][u][r]);
            if (tok < R && n0 + g * 8 < N) *(__global ushort8 *)(y + (ulong)tok * N + y_off + n0 + g * 8) = o;
        }
}
