#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "launch_shape.h"
#include <limits.h>
#include <math.h>

namespace {
constexpr unsigned norm_columns = 5120;
constexpr unsigned norm_threads = 128;
constexpr int bf16_type = 0;
constexpr int f32_type = 1;

template <bool fuse_square>
__device__ void normalized_row(const __nv_bfloat16* source, const __nv_bfloat16* weight,
                               __nv_bfloat16* output, float* scales, float epsilon) {
    __shared__ float warp_totals[4];
    __shared__ float inverse_scale;
    const unsigned worker = threadIdx.x;
    const unsigned lane = worker & 31;
    const unsigned warp = worker >> 5;
    const uint64_t start = uint64_t(blockIdx.x) * norm_columns;
    float accumulator = 0.0f;
    // Four consecutive values are consumed before advancing to the next packet owned by this worker.
    for (unsigned first = worker * 4; first < norm_columns; first += norm_threads * 4) {
        for (unsigned part = 0; part != 4; ++part) {
            const float value = __bfloat162float(source[start + first + part]);
            if constexpr (fuse_square) accumulator = __fmaf_rn(value, value, accumulator);
            else accumulator = __fadd_rn(accumulator, __fmul_rn(value, value));
        }
    }
    for (unsigned pass = 0; pass != 5; ++pass) {
        const unsigned distance = 1u << (4 - pass);
        accumulator = __fadd_rn(accumulator, __shfl_down_sync(0xffffffffu, accumulator, distance));
    }
    if (lane == 0) warp_totals[warp] = accumulator;
    __syncthreads();
    if (worker == 0) {
        const float even = __fadd_rn(warp_totals[0], warp_totals[2]);
        const float odd = __fadd_rn(warp_totals[1], warp_totals[3]);
        const float mean_square = __fdiv_rn(__fadd_rn(even, odd), float(norm_columns));
        inverse_scale = ::rsqrtf(__fadd_rn(mean_square, epsilon));
        if (scales) scales[blockIdx.x] = inverse_scale;
    }
    __syncthreads();
    for (unsigned column = worker; column < norm_columns; column += norm_threads) {
        const float normalized = __fmul_rn(inverse_scale, __bfloat162float(source[start + column]));
        output[start + column] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(weight[column]), normalized));
    }
}

bool valid_norm(const void* source, const void* weight, void* output, float* scales, uint64_t rows,
                uint32_t columns, int source_type, int weight_type, float epsilon) {
    const uintptr_t combined = reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(weight)
        | reinterpret_cast<uintptr_t>(output);
    return source && weight && output && rows <= INT_MAX && columns == norm_columns
        && source_type == bf16_type && weight_type == bf16_type && !(combined & 7)
        && (!scales || !(reinterpret_cast<uintptr_t>(scales) & 3)) && isfinite(epsilon) && epsilon >= 0.0f;
}
}

// Device ABI requires block(128,1,1), grid(rows,1,1), BF16 contiguous width5120 and aligned weight/output.
extern "C" __global__ void tf_dflash_rms5120_bf16_fma_kernel(const __nv_bfloat16* source,
    const __nv_bfloat16* weight, __nv_bfloat16* output, float* scales, float epsilon) {
    normalized_row<true>(source, weight, output, scales, epsilon);
}

extern "C" __global__ void tf_dflash_rms5120_bf16_separate_kernel(const __nv_bfloat16* source,
    const __nv_bfloat16* weight, __nv_bfloat16* output, float* scales, float epsilon) {
    normalized_row<false>(source, weight, output, scales, epsilon);
}

// Development ABI uses BF16 dtype0 and FP32 dtype1; mixed/unaligned norms refuse instead of casting.
extern "C" cudaError_t tf_dflash_rms5120_bf16_fma(const void* source, const void* weight, void* output,
    float* scales, uint64_t rows, uint32_t columns, int source_type, int weight_type,
    float epsilon, cudaStream_t stream) {
    if (!valid_norm(source, weight, output, scales, rows, columns, source_type, weight_type, epsilon))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    tf_dflash_rms5120_bf16_fma_kernel<<<uint32_t(rows), norm_threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source), static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output), scales, epsilon);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_dflash_rms5120_bf16_separate(const void* source, const void* weight, void* output,
    float* scales, uint64_t rows, uint32_t columns, int source_type, int weight_type,
    float epsilon, cudaStream_t stream) {
    if (!valid_norm(source, weight, output, scales, rows, columns, source_type, weight_type, epsilon))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    tf_dflash_rms5120_bf16_separate_kernel<<<uint32_t(rows), norm_threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source), static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output), scales, epsilon);
    return cudaGetLastError();
}

// Positions and inverse frequencies are supplied as exact FP32 oracle bytes; inverse-frequency generation is separate.
extern "C" __global__ void tf_dflash_phase_trig_kernel(const float* positions, const float* inverse,
    float* phase_out, float* cosine, float* sine, uint64_t rows, uint32_t half) {
    const uint64_t count = rows * half;
    for (uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < count;
         at += uint64_t(gridDim.x) * blockDim.x) {
        const float phase = __fmul_rn(positions[at / half], inverse[at % half]);
        if (phase_out) phase_out[at] = phase;
        cosine[at] = ::cosf(phase);
        sine[at] = ::sinf(phase);
    }
}

extern "C" cudaError_t tf_dflash_phase_trig(const void* positions, const void* inverse, void* phase,
    void* cosine, void* sine, uint64_t rows, uint32_t half, int dtype, cudaStream_t stream) {
    const uintptr_t combined = reinterpret_cast<uintptr_t>(positions) | reinterpret_cast<uintptr_t>(inverse)
        | reinterpret_cast<uintptr_t>(phase) | reinterpret_cast<uintptr_t>(cosine) | reinterpret_cast<uintptr_t>(sine);
    if (!positions || !inverse || !cosine || !sine || dtype != f32_type || half != 64
        || (combined & 3) || rows > UINT64_MAX / (half * sizeof(float))) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    const uint64_t count = rows * half;
    const uint32_t blocks = tf_launch_blocks(count, 256);
    tf_dflash_phase_trig_kernel<<<blocks, 256, 0, stream>>>(static_cast<const float*>(positions),
        static_cast<const float*>(inverse), static_cast<float*>(phase), static_cast<float*>(cosine),
        static_cast<float*>(sine), rows, half);
    return cudaGetLastError();
}
