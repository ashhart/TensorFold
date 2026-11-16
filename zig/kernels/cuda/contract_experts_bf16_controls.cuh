// Gate 4's control for experts_bf16.cu: its fp32 down with each pair's sum rounded to bf16 at every input.
#pragma once

extern "C" __global__ void tf_ctl_bf16_experts_bf16_sums(const __nv_bfloat16* X, int x_stride, const uint4* W, int KG, int NB,
                                                         const int* items, const int* counts, const int* members, float* out,
                                                         int N) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    const int cb = col / 32, j = (col % 32) / 8;
    for (int it = blockIdx.y; it < counts[0]; it += gridDim.y) {
        const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
        const __nv_bfloat16* blk = reinterpret_cast<const __nv_bfloat16*>(W + (static_cast<int64_t>(e) * NB + cb) * KG * 128);
        for (int i = 0; i < cnt; ++i) {
            const int pr = members[first + i];
            float s = 0.0f;
            for (int k = 0; k < KG * 32; ++k) {
                const int q = k % 16, lane = 4 * (col % 8) + q / 4;
                const float w = __bfloat162float(blk[(static_cast<int64_t>(k / 32) * 128 + 32 * j + lane) * 8 + 4 * (k % 32 / 16) + q % 4]);
                s = __bfloat162float(__float2bfloat16_rn(s + tf_ctl_x(X, static_cast<int64_t>(pr) * x_stride + k) * w));
            }
            out[static_cast<int64_t>(pr) * N + col] = s;
        }
    }
}
