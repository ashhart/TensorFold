// Expand only filled FP8 KV rows, preserving the exact conversion performed by attention staging.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(256) void tf_qwen_expand_kv(const uint8_t* __restrict__ keys,
    const uint8_t* __restrict__ values, const float* __restrict__ key_scales,
    const float* __restrict__ value_scales, __nv_bfloat16* __restrict__ output,
    int length, int heads, int capacity, const int* __restrict__ pages, int64_t page_bytes, int page_tokens) {
    constexpr int dimension = 256, chunks = dimension / 8;
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(2) * heads * length * chunks) return;
    const int chunk = index % chunks;
    const int key = (index / chunks) % length;
    const int head = (index / (size_t(chunks) * length)) % heads;
    const bool value = index >= size_t(heads) * length * chunks;
    const int64_t row = pages ? int64_t(head) * page_tokens + key % page_tokens : int64_t(head) * capacity + key;
    const int64_t page = pages ? int64_t(pages[key / page_tokens]) * page_bytes : 0;
    const uint8_t* source = value ? values : keys;
    const float* scales = value ? value_scales : key_scales;
    const uint2 codes = __ldg(reinterpret_cast<const uint2*>(source + page + row * dimension + chunk * 8));
    const float scale = __ldg(reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(scales) + page) + row);
    uint4 expanded;
    const auto* pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&codes);
    auto* words = reinterpret_cast<uint32_t*>(&expanded);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 x = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pairs[pair], __NV_E4M3)));
        const __nv_bfloat162 result = __floats2bfloat162_rn(__fmul_rn(x.x, scale), __fmul_rn(x.y, scale));
        words[pair] = *reinterpret_cast<const uint32_t*>(&result);
    }
    reinterpret_cast<uint4*>(output)[index] = expanded;
}
