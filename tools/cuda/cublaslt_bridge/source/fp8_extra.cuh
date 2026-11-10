// FP8 row preparation over TensorFold's existing packed weights and BF16 inputs.
#pragma once
namespace tf_lt_gpu {
__global__ void fp8_unpack(const uint8_t* source, uint8_t* output, int n, int k) {
    const int tile = blockIdx.y, group = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = tile * 64 + warp * 8 + lane / 4;
    const size_t at = ((size_t(tile) * (k / 64) + group) * 8 + warp) * 32 + lane;
    const uint4 v = transpose(reinterpret_cast<const uint4*>(source)[at], lane & 3);
    const uint4 result = make_uint4((v.x & 65535) | (v.y << 16), (v.z & 65535) | (v.w << 16),
        (v.x >> 16) | (v.y & 0xffff0000u), (v.z >> 16) | (v.w & 0xffff0000u));
    if (row < n) *reinterpret_cast<uint4*>(output + size_t(row) * k + group * 64 + (lane & 3) * 16) = result;
}
__global__ void fp8_units(float* values) {
    if (threadIdx.x < 4) values[threadIdx.x] = 1.0f;
}
__global__ void fp8_rows(const __nv_bfloat16* input, uint8_t* output, float* scales,
    int m, int k, int stride, bool dynamic, float scale) {
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (row >= m) return;
    __shared__ float maxima[8];
    __shared__ uint32_t flags[8];
    float maximum = 0; uint32_t invalid = 0;
    if (dynamic) {
        for (int col = tid; col < k; col += 256) {
            const float value = __bfloat162float(input[size_t(row) * stride + col]);
            maximum = fmaxf(maximum, fabsf(value)); invalid |= !isfinite(value);
        }
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
            invalid |= __shfl_xor_sync(0xffffffffu, invalid, offset);
        }
        if (!lane) { maxima[warp] = maximum; flags[warp] = invalid; }
        __syncthreads();
        if (!warp) {
            maximum = lane < 8 ? maxima[lane] : 0; invalid = lane < 8 ? flags[lane] : 0;
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) {
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
                invalid |= __shfl_xor_sync(0xffffffffu, invalid, offset);
            }
            if (!lane) { maxima[0] = invalid ? nanf("") : __fdiv_rn(maximum, 448.0f); flags[0] = invalid; }
        }
        __syncthreads(); scale = maxima[0]; invalid = flags[0];
    }
    if (!tid) scales[row] = scale;
    const float inverse = __fdiv_rn(1.0f, scale);
    for (int col = tid; col < k; col += 256) {
        const float value = __bfloat162float(input[size_t(row) * stride + col]);
        const float q = scale == 0 ? value : dynamic ? __fdiv_rn(value, scale) : __fmul_rn(value, inverse);
        output[size_t(row) * k + col] = invalid ? 127 : __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    }
}
template<bool F32> __global__ void fp8_finish(const float* product, const float* rows, const float* weights,
    void* output, size_t count, int n, int k, bool dynamic, float scalar) {
    const size_t at = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (at >= count) return;
    const int row = at / n, col = at % n;
    const float weight = dynamic ? weights[size_t(col / 64) * (k / 64) * 64 + col % 64] : scalar;
    const float value = __fmul_rn(product[at], __fmul_rn(rows[row], weight));
    if constexpr (F32) reinterpret_cast<float*>(output)[at] = value;
    else reinterpret_cast<__nv_bfloat16*>(output)[at] = __float2bfloat16_rn(value);
}
}
