// Fixed four-stream HC mix: two adjacent dimensions per thread, with the original scalar rounding points.
#pragma once

extern "C" __global__ void __launch_bounds__(256) fn_hc_dense_mix_vec(
        const __nv_bfloat16* __restrict__ UP, const __nv_bfloat16* __restrict__ NORMED,
        __nv_bfloat16* __restrict__ MIXED, float* __restrict__ XSM) {
    constexpr int D = 2560, SS = 4;
    __shared__ float row[8][64];
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.y * 8 + warp, d = group * 64 + lane * 2;
    float total0 = 0.0f, total1 = 0.0f;
#pragma unroll
    for (int stream = 0; stream < SS; ++stream) {
        const long long at = (static_cast<long long>(r) * SS + stream) * D + d;
        const float2 u = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(UP + at));
        const float2 n = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(NORMED + at));
        const float v0 = rbf(__fmul_rn(bsig(u.x), n.x));
        const float v1 = rbf(__fmul_rn(bsig(u.y), n.y));
        total0 = __fadd_rn(total0, v0);
        total1 = __fadd_rn(total1, v1);
    }
    const float m0 = rbf(__fdiv_rn(total0, 4.0f)), m1 = rbf(__fdiv_rn(total1, 4.0f));
    *reinterpret_cast<__nv_bfloat162*>(MIXED + static_cast<long long>(r) * D + d) = __floats2bfloat162_rn(m0, m1);
    row[warp][lane * 2] = m0;
    row[warp][lane * 2 + 1] = m1;
    __syncwarp();
    if (lane < 2) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) sum = __fadd_rn(sum, row[warp][lane * 32 + i]);
        XSM[static_cast<long long>(r) * (D / 32) + group * 2 + lane] = sum;
    }
}
