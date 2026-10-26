// Cosmos3-Edge's glue on CUDA: cosmos3.cuh's bodies with every rounding point kept and no fault planted.

#include "cosmos3.cuh"

using namespace tf_cos3;

extern "C" __global__ void __launch_bounds__(256) tf_cos3_embed(const int* __restrict__ ids, const uint8_t* __restrict__ table,
                                                               const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
                                                               __nv_bfloat16* __restrict__ out) {
    embed<ALL>(ids, table, scales, biases, out);
}

extern "C" __global__ void __launch_bounds__(256) tf_cos3_place(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ rows,
                                                               const int* __restrict__ map) {
    place<0>(h, rows, map);
}

extern "C" __global__ void __launch_bounds__(256) tf_cos3_norm(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ r,
                                                              const __nv_bfloat16* __restrict__ w, float eps, int add,
                                                              __nv_bfloat16* __restrict__ x, float* __restrict__ xs) {
    norm<ALL>(h, r, w, eps, add, x, xs);
}

extern "C" __global__ void __launch_bounds__(32) tf_cos3_qk(const __nv_bfloat16* __restrict__ qkv, const int* __restrict__ pos,
                                                           const int* __restrict__ slots, int cap, float log2_theta,
                                                           __nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ keys,
                                                           __nv_bfloat16* __restrict__ values) {
    qk<ALL>(qkv, pos, slots, cap, log2_theta, queries, keys, values);
}

extern "C" __global__ void __launch_bounds__(256) tf_cos3_relu2(__nv_bfloat16* __restrict__ x, int width, float* __restrict__ xs) {
    relu2<ALL>(x, width, xs);
}

extern "C" __global__ void __launch_bounds__(32 * TR) tf_cos3_attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ keys,
                                                                  const __nv_bfloat16* __restrict__ values, __nv_bfloat16* __restrict__ out,
                                                                  const int* __restrict__ slots, int rows, int cap, float scale, float* __restrict__ xs) {
    attn<0>(q, keys, values, out, slots, rows, cap, scale, xs);
}

extern "C" __global__ void __launch_bounds__(256) tf_cos3_attn1(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ keys,
                                                               const __nv_bfloat16* __restrict__ values, __nv_bfloat16* __restrict__ out,
                                                               const int* __restrict__ slots, int cap, float scale, float* __restrict__ xs) {
    attn1<0>(q, keys, values, out, slots, cap, scale, xs);
}

extern "C" __global__ void __launch_bounds__(256) tf_cos3_widen8(const uint8_t* __restrict__ codes, const __nv_bfloat16* __restrict__ scales,
                                                                const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ out,
                                                                int k, int64_t count) {
    widen8<ALL>(codes, scales, biases, out, k, count);
}
