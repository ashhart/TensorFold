#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
// DPAS ceiling micro-benchmarks. dpas_loop: 8 independent accumulators, operands in registers (pure matrix-engine rate).
// dpas_tile: a 32 x 64 output micro-tile per sub-group (4 x 4 DPAS tiles of 8 x 16), operands loaded from a small (cache resident) buffer each k step of 16:
// 16 DPAS per 4 A loads and 4 B loads, the register/load pattern of a GEMM inner loop without the global traffic.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_loop(__global float *out, uint iters) {
    const uint lane = get_sub_group_local_id();
    short8 a = (short8)((short)(lane + 0x3f00));
    int8 b = (int8)((int)(0x3f803f80 + lane));
    float8 acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = (float8)(0.0f);
    for (uint i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc[j]);
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) s += acc[j][0] + acc[j][7];
    out[get_global_id(0)] = s;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_tile(__global const ushort *A, __global const ushort *B, __global float *out, uint iters) {
    const uint lane = get_sub_group_local_id();
    const uint wg = get_group_id(0);
    float8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = (float8)(0.0f);
    __global const uint *Bp = (__global const uint *)B;
    for (uint k = 0; k < iters; k++) {
        const uint o = ((k + wg) & 63u);
        short8 a[4];
        int8 b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = vload8(((o * 4 + i) * 16 + lane), (__global const short *)A);
#pragma unroll
        for (int j = 0; j < 4; j++) b[j] = as_int8(vload8(0, Bp + (o * 4 + j) * 128 + lane * 8));
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = intel_sub_group_bf16_bf16_matrix_mad_k16(a[i], b[j], acc[i][j]);
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) s += acc[i][j][0] + acc[i][j][7];
    out[get_global_id(0)] = s;
}
