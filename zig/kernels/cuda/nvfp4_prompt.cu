// Native NVFP4 prefill over qmmf-packed weights, with calibrated E2M1 activation rows and FP32 sums.
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "common/qmm_frag.cuh"
#include "common/native_fp4.cuh"

namespace tf_nvfp4 {

__device__ __forceinline__ void quantize(const __nv_bfloat16* x, uint8_t* codes, uint8_t* scales,
                         int m, int k, int ldx, float inverse) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= m * (k / 16)) return;
    float values[16], maximum = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int row = group / (k / 16), col = (group % (k / 16)) * 16;
        values[i] = __bfloat162float(x[static_cast<size_t>(row) * ldx + col + i]);
        maximum = (isnan(values[i]) || isnan(maximum)) ? nanf("") : fmaxf(maximum, fabsf(values[i]));
    }
    uint8_t scale = isfinite(maximum) ?
        __nv_cvt_float_to_fp8(__fdiv_rn(__fmul_rn(maximum, inverse), 6.0f), __NV_SATFINITE, __NV_E4M3) : 0x7f;
    scales[group] = scale;
    const float divisor = tf_common::e4m3f(scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float2 pair = divisor == 0 ? make_float2(0, 0) :
            make_float2(__fdiv_rn(__fmul_rn(values[2 * i], inverse), divisor),
                        __fdiv_rn(__fmul_rn(values[2 * i + 1], inverse), divisor));
        codes[group * 8 + i] = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
    }
}

template<bool F32, int BM = 64, int BN = 128, int WN = 8>
__device__ __forceinline__ void gemm(
        const uint8_t* ax, const uint8_t* as, const uint8_t* packed_w, const uint8_t* packed_s,
        void* output, int m, int n, int k, int npad, float alpha) {
    constexpr int STAGES = 2, THREADS = WN * 32, MT = BM / 16, NT = BN / WN / 8;
    struct Shared {
        alignas(16) uint8_t a[STAGES][BM * 32];
        alignas(16) uint8_t b[STAGES][BN * 32];
        alignas(16) uint32_t sa[STAGES][BM];
        alignas(16) uint32_t sb[STAGES][BN];
    };
    __shared__ Shared shared;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, groups = k / 64;
    auto stage = [&](int s, int g) {
        for (int task = tid; task < BM * 2; task += THREADS) {
            const int row = task / 2, chunk = task & 1;
            const int safe_row = min(m0 + row, m - 1);
            qmm_frag::cp16z(shared.a[s] + row * 32 + (chunk ^ (row & 1)) * 16,
                ax + static_cast<size_t>(safe_row) * (k / 2) + g * 32 + chunk * 16, m0 + row < m);
        }
        for (int task = tid; task < BN * 2; task += THREADS) {
            const int tile = task / 128, chunk = task % 128;
            const int row = n0 + tile * 64;
            qmm_frag::cp16z(shared.b[s] + task * 16,
                packed_w + (static_cast<size_t>(min(row, npad - 64) / 64) * groups + g) * 2048 + chunk * 16,
                row < npad);
        }
        for (int row = tid; row < BM; row += THREADS)
            shared.sa[s][row] = m0 + row < m ?
                reinterpret_cast<const uint32_t*>(as)[static_cast<size_t>(m0 + row) * groups + g] : 0;
        for (int row = tid; row < BN; row += THREADS)
            shared.sb[s][row] = n0 + row < npad ?
                reinterpret_cast<const uint32_t*>(packed_s)[
                    (static_cast<size_t>((n0 + row) / 64) * groups + g) * 64 + row % 64] : 0;
    };
    float accum[MT][NT][4] = {};
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        if (s < groups) stage(s, s);
        qmm_frag::commit();
    }
    for (int g = 0; g < groups; ++g) {
        qmm_frag::wait<1>();
        __syncthreads();
        const int s = g % STAGES;
        uint32_t a[MT][4], sa[MT];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            const int row = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            const int chunk = (lane >> 4) ^ (row & 1);
            qmm_frag::ldmatrix4(a[i], shared.a[s] + row * 32 + chunk * 16);
            sa[i] = shared.sa[s][i * 16 + (lane >> 2) + (lane & 1) * 8];
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int jt = warp * NT + j;
            const uint2 old = reinterpret_cast<const uint2*>(shared.b[s])[jt * 32 + lane];
            const uint32_t b0 = contiguous8(old.x, lane), b1 = contiguous8(old.y, lane);
            const uint32_t sb = shared.sb[s][jt * 8 + (lane >> 2)];
#pragma unroll
            for (int i = 0; i < MT; ++i) mma(accum[i][j], a[i], b0, b1, sa[i], sb);
        }
        __syncthreads();
        if (g + STAGES < groups) stage(s, g + STAGES);
        qmm_frag::commit();
    }
    qmm_frag::wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + i * 16 + (lane >> 2) + h * 8;
                const int col = n0 + (warp * NT + j) * 8 + (lane & 3) * 2;
                if (row < m) for (int p = 0; p < 2; ++p) if (col + p < n) {
                    const size_t at = static_cast<size_t>(row) * n + col + p;
                    const float value = accum[i][j][h * 2 + p] * alpha;
                    if constexpr (F32) reinterpret_cast<float*>(output)[at] = value;
                    else reinterpret_cast<__nv_bfloat16*>(output)[at] = __float2bfloat16_rn(value);
                }
            }
}
}

extern "C" __global__ __launch_bounds__(256) void tf_nvfp4_quant(
    const __nv_bfloat16* x, uint8_t* codes, uint8_t* scales, int m, int k, int ldx, float inverse) {
    tf_nvfp4::quantize(x, codes, scales, m, k, ldx, inverse);
}

extern "C" __global__ __launch_bounds__(256) void tf_nvfp4_prompt(
    const uint8_t* ax, const uint8_t* as, const uint8_t* w, const uint8_t* ws,
    void* y, int m, int n, int k, int npad, float alpha) {
    tf_nvfp4::gemm<false>(ax, as, w, ws, y, m, n, k, npad, alpha);
}

extern "C" __global__ __launch_bounds__(256) void tf_nvfp4_prompt_f32(
    const uint8_t* ax, const uint8_t* as, const uint8_t* w, const uint8_t* ws,
    void* y, int m, int n, int k, int npad, float alpha) {
    tf_nvfp4::gemm<true>(ax, as, w, ws, y, m, n, k, npad, alpha);
}
