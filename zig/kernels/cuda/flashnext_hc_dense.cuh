// Flash Next's dense hyper-connection normalize and mix, included in place by flashnext_hc.cu (one image).
#pragma once

// Dense HC uses the same stored stream sums and FP32 gamma before one BF16 normalization rounding.
extern "C" __global__ void fn_hc_normalize(const __nv_bfloat16* H, const float* PSS, const float* SCALE,
                                            __nv_bfloat16* OUT, float eps, int D, int SS) {
    __shared__ float rinv;
    const int r = blockIdx.x, stream = blockIdx.y, nc = D / 256;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int c = 0; c < nc; ++c) total = __fadd_rn(total, PSS[(static_cast<long long>(r) * nc + c) * SS + stream]);
        rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
    }
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const int col = stream * D + d;
        const long long at = static_cast<long long>(r) * SS * D + col;
        OUT[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(H[at]), rinv), SCALE[col]));
    }
}

// fn_hc_normalize a block a (row, stream, 256-column slice): each block sums the stream the same way, so the same bits.
extern "C" __global__ void __launch_bounds__(256) fn_hc_normalize_cols(const __nv_bfloat16* H, const float* PSS, const float* SCALE,
                                                                       __nv_bfloat16* OUT, float eps, int D, int SS) {
    __shared__ float rinv;
    const int r = blockIdx.x, stream = blockIdx.y, nc = D / 256;
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int c = 0; c < nc; ++c) total = __fadd_rn(total, PSS[(static_cast<long long>(r) * nc + c) * SS + stream]);
        rinv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
    }
    __syncthreads();
    const int d = blockIdx.z * 256 + threadIdx.x;
    const int col = stream * D + d;
    const long long at = static_cast<long long>(r) * SS * D + col;
    OUT[at] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(H[at]), rinv), SCALE[col]));
}

// Dense up-projection values are already BF16: sigmoid, product and stream mean keep their original rounding seams.
extern "C" __global__ void fn_hc_dense_mix(const __nv_bfloat16* UP, const __nv_bfloat16* NORMED,
                                            __nv_bfloat16* MIXED, float* XSM, int D, int SS) {
    __shared__ float row[32];
    const int r = blockIdx.x, group = blockIdx.y, d = group * 32 + threadIdx.x;
    float total = 0.0f;
    for (int stream = 0; stream < SS; ++stream) {
        const long long at = (static_cast<long long>(r) * SS + stream) * D + d;
        const float value = rbf(__fmul_rn(bsig(bf(UP[at])), bf(NORMED[at])));
        total = __fadd_rn(total, value);
    }
    const float mixed = rbf(__fdiv_rn(total, static_cast<float>(SS)));
    MIXED[static_cast<long long>(r) * D + d] = __float2bfloat16_rn(mixed);
    row[threadIdx.x] = mixed;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) sum = __fadd_rn(sum, row[i]);
        XSM[static_cast<long long>(r) * (D / 32) + group] = sum;
    }
}

// fn_hc_dense_mix, a warp a 32-column group and eight a block: the same ops, lane 0 summing its group in order.
extern "C" __global__ void __launch_bounds__(256) fn_hc_dense_mix8(const __nv_bfloat16* UP, const __nv_bfloat16* NORMED,
                                                                   __nv_bfloat16* MIXED, float* XSM, int D, int SS) {
    __shared__ float row[8][32];
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.y * 8 + warp, d = group * 32 + lane;
    if (group >= D / 32) return;
    float total = 0.0f;
    for (int stream = 0; stream < SS; ++stream) {
        const long long at = (static_cast<long long>(r) * SS + stream) * D + d;
        const float value = rbf(__fmul_rn(bsig(bf(UP[at])), bf(NORMED[at])));
        total = __fadd_rn(total, value);
    }
    const float mixed = rbf(__fdiv_rn(total, static_cast<float>(SS)));
    MIXED[static_cast<long long>(r) * D + d] = __float2bfloat16_rn(mixed);
    row[warp][lane] = mixed;
    __syncwarp();
    if (lane == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) sum = __fadd_rn(sum, row[warp][i]);
        XSM[static_cast<long long>(r) * (D / 32) + group] = sum;
    }
}
