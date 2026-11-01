// BF16 prompt HC writeback and normalization, retaining the separate kernels' reductions and rounding points.
#pragma once

namespace fn_hcfused {

constexpr int D = 2560, SS = 4;

template <int MODE>
__device__ __forceinline__ void writeback_norm(
        __nv_bfloat16* H, float* PSS, const void* BR, const __nv_bfloat16* INJ,
        const float* Y, const float* WTS, const float* SCALE, __nv_bfloat16* NORM,
        long long RS, int SLICES) {
    const int r = blockIdx.x / SS, st = blockIdx.x % SS, t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    __shared__ float red[D / 256][8];
    __shared__ float rinv;
    float hv[D / 256];
    const float gate = MODE == 0 ? 0.0f : bf(INJ[r * SS + st]);
#pragma unroll
    for (int c = 0; c < D / 256; ++c) {
        const int d = c * 256 + t;
        const long long at = (static_cast<long long>(r) * SS + st) * D + d;
        float branch = 0.0f;
        if constexpr (MODE == 4) {
            const float* p = static_cast<const float*>(BR) + static_cast<long long>(r) * D + d;
            float acc = p[0];
            for (int k = 1; k < SLICES; ++k) acc = __fadd_rn(acc, p[k * RS]);
            branch = rbf(acc);
        } else if constexpr (MODE == 2) {
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < 11; ++k) {
                const long long slot = static_cast<long long>(r) * 11 + k;
                acc = __fmaf_rn(Y[slot * D + d], WTS[slot], acc);
            }
            branch = rbf(acc);
        }
        float v = bf(H[at]);
        if constexpr (MODE != 0) {
            v = rbf(__fmaf_rn(branch, gate, v));
            H[at] = __float2bfloat16_rn(v);
        }
        hv[c] = v;
        const float ss = tf_glue::warp_sum(__fmul_rn(v, v));
        if (lane == 0) red[c][warp] = ss;
    }
    __syncthreads();
    if (t == 0) {
        float total = 0.0f;
#pragma unroll
        for (int c = 0; c < D / 256; ++c) {
            float part = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) part = __fadd_rn(part, red[c][w]);
            PSS[(static_cast<long long>(r) * (D / 256) + c) * SS + st] = part;
            total = __fadd_rn(total, part);
        }
        rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, float(D)), 1e-6f)));
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < D / 256; ++c) {
        const int col = st * D + c * 256 + t;
        NORM[static_cast<long long>(r) * SS * D + col] =
            __float2bfloat16_rn(__fmul_rn(__fmul_rn(hv[c], rinv), SCALE[col]));
    }
}

}  // namespace fn_hcfused

extern "C" __global__ void __launch_bounds__(256) fn_hc_writeback_norm(
        __nv_bfloat16* H, float* PSS, const void* BR, const __nv_bfloat16* INJ,
        const float* Y, const float* WTS, const float* SCALE, __nv_bfloat16* NORM,
        long long RS, int SLICES, int MODE) {
    using namespace fn_hcfused;
    switch (MODE) {
        case 0: writeback_norm<0>(H, PSS, BR, INJ, Y, WTS, SCALE, NORM, RS, SLICES); break;
        case 2: writeback_norm<2>(H, PSS, BR, INJ, Y, WTS, SCALE, NORM, RS, SLICES); break;
        case 4: writeback_norm<4>(H, PSS, BR, INJ, Y, WTS, SCALE, NORM, RS, SLICES); break;
    }
}
