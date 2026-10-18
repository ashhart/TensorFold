// MoE glue for every CUDA family (float64 references: core/moe_ref.zig): sigmoid top-k and the weighted combine.

// Every row's bits depend on that row alone; kinds 0 fp32, 1 fp16, 2 bf16 (as norms.cu).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

namespace tf_moe {

constexpr int NT = 256;
constexpr int MOST = 4096; // experts a row ranks

__device__ __forceinline__ float load(const void* p, int kind, int64_t i) {
    if (kind == 0) return static_cast<const float*>(p)[i];
    if (kind == 1) return __half2float(static_cast<const __half*>(p)[i]);
    return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

__device__ __forceinline__ void store(void* p, int kind, int64_t i, float v) {
    if (kind == 0) static_cast<float*>(p)[i] = v;
    else if (kind == 1) static_cast<__half*>(p)[i] = __float2half_rn(v);
    else static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
}

// (value, id) wins over (other value, other id): the larger value, the lower id among equals.
__device__ __forceinline__ bool beats(float v, int i, float w, int j) {
    return v > w || (v == w && i < j);
}

}  // namespace tf_moe

using namespace tf_moe;

// Top `topk` by logit + bias (ties: lower id), weights sigmoid(logit) [normalized] * scale; `shared`: slot topk = ne.
extern "C" __global__ void __launch_bounds__(NT) tf_moe_topk_sigmoid(const float* logits, const float* bias, int* pick,
                                                                     float* wts, int ne, int topk, int shared,
                                                                     int normalize, float scale) {
    __shared__ float v[MOST];
    __shared__ float best_v[NT / 32];
    __shared__ int best_i[NT / 32];
    __shared__ int picked[64];
    const int64_t r = blockIdx.x;
    const float* lg = logits + r * ne;
    for (int e = threadIdx.x; e < ne; e += NT) v[e] = __fadd_rn(lg[e], bias != nullptr ? bias[e] : 0.f);
    __syncthreads();
    for (int k = 0; k < topk; ++k) {
        float bv = -INFINITY;
        int bi = ne;
        for (int e = threadIdx.x; e < ne; e += NT)
            if (beats(v[e], e, bv, bi)) bv = v[e], bi = e;
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, m);
            const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, m);
            if (beats(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        if ((threadIdx.x & 31) == 0) best_v[threadIdx.x >> 5] = bv, best_i[threadIdx.x >> 5] = bi;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tv = best_v[0];
            int ti = best_i[0];
            for (int w = 1; w < NT / 32; ++w)
                if (beats(best_v[w], best_i[w], tv, ti)) tv = best_v[w], ti = best_i[w];
            picked[k] = ti;
            v[ti] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int slots = topk + (shared ? 1 : 0);
        float sum = 0.f;
        float w[64];
        for (int k = 0; k < topk; ++k) {
            w[k] = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-lg[picked[k]])));
            sum = __fadd_rn(sum, w[k]);
        }
        for (int k = 0; k < topk; ++k) {
            pick[r * slots + k] = picked[k];
            wts[r * slots + k] = __fmul_rn(normalize ? __fdiv_rn(w[k], sum) : w[k], scale);
        }
        if (shared) pick[r * slots + topk] = ne, wts[r * slots + topk] = 1.f;
    }
}

// A float's order as an unsigned (-0 as +0, so equal values stay equal).
__device__ __forceinline__ unsigned ordered(float v) {
    const unsigned u = __float_as_uint(__fadd_rn(v, 0.f));
    return u & 0x80000000u ? ~u : u | 0x80000000u;
}

// tf_moe_topk_sigmoid with a warp a row (up to 32 * PER experts): values in registers, no block barrier.
constexpr int PER = 16;

extern "C" __global__ void __launch_bounds__(256) tf_moe_topk_sigmoid_warp(const float* logits, const float* bias,
                                                                          int* pick, float* wts, int rows, int ne,
                                                                          int topk, int shared, int normalize,
                                                                          float scale) {
    const int lane = threadIdx.x & 31;
    const int64_t r = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    const float* lg = logits + r * ne;
    float v[PER], x[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + 32 * j;
        x[j] = e < ne ? lg[e] : 0.f;
        v[j] = e < ne ? __fadd_rn(x[j], bias != nullptr ? bias[e] : 0.f) : -INFINITY;
    }
    const int slots = topk + (shared ? 1 : 0);
    float xk = 0.f; // lane k's pick's logit
    int pk = 0;     // lane k's pick
    for (int k = 0; k < topk; ++k) {
        float bv = -INFINITY;
        int bi = ne;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = lane + 32 * j;
            if (e < ne && beats(v[j], e, bv, bi)) bv = v[j], bi = e;
        }
        { // the warp's best: the largest value (redux on an order-keeping key), then the lowest id among equals
            const unsigned key = ordered(bv);
            const unsigned top = __reduce_max_sync(0xFFFFFFFFu, key);
            bi = static_cast<int>(__reduce_min_sync(0xFFFFFFFFu, key == top ? static_cast<unsigned>(bi) : 0xFFFFFFFFu));
        }
        float xb = 0.f;
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (lane + 32 * j == bi) v[j] = -INFINITY, xb = x[j];
        xb = __shfl_sync(0xFFFFFFFFu, xb, bi & 31);
        if (lane == k) xk = xb, pk = bi;
    }
    // the weights after the picks, each in its lane; the sum in pick order
    const float wk = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-xk)));
    float sum = 0.f;
    for (int k = 0; k < topk; ++k) sum = __fadd_rn(sum, __shfl_sync(0xFFFFFFFFu, wk, k));
    if (lane < topk) {
        pick[r * slots + lane] = pk;
        wts[r * slots + lane] = __fmul_rn(normalize ? __fdiv_rn(wk, sum) : wk, scale);
    }
    if (shared && lane == 0) pick[r * slots + topk] = ne, wts[r * slots + topk] = 1.f;
}

// out[r, d] = sum over the row's slots, in slot order, of w[r, k] * y[r, k, d] (fp32 fma), rounded once to out's kind.
template <int YK, int OK>
__device__ __forceinline__ void combine_row(const void* y, const float* w, void* out, int slots, int dims, int64_t r, int d) {
    float acc = 0.f;
    int k = 0;
    for (; k + 8 <= slots; k += 8) { // eight slots' loads in flight, then folded in order
        float v[8], wk[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = load(y, YK, (r * slots + k + i) * dims + d), wk[i] = w[r * slots + k + i];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc = __fmaf_rn(wk[i], v[i], acc);
    }
    if (k < slots) {
        float v[8], wk[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = k + i < slots ? load(y, YK, (r * slots + k + i) * dims + d) : 0.f;
            wk[i] = k + i < slots ? w[r * slots + k + i] : 0.f;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (k + i < slots) acc = __fmaf_rn(wk[i], v[i], acc);
    }
    store(out, OK, r * dims + d, acc);
}

extern "C" __global__ void __launch_bounds__(NT) tf_moe_combine(const void* y, int y_kind, const float* w, void* out,
                                                                int out_kind, int slots, int dims) {
    const int64_t r = blockIdx.x;
    const int d = blockIdx.y * NT + threadIdx.x;
    if (d >= dims) return;
    // the kinds picked once a thread (a branch an element costs instruction fetch on small calls)
    if (y_kind == 0 && out_kind == 2) combine_row<0, 2>(y, w, out, slots, dims, r, d);
    else if (y_kind == 2 && out_kind == 2) combine_row<2, 2>(y, w, out, slots, dims, r, d);
    else if (y_kind == 0 && out_kind == 0) combine_row<0, 0>(y, w, out, slots, dims, r, d);
    else {
        float acc = 0.f;
        for (int k = 0; k < slots; ++k) acc = __fmaf_rn(w[r * slots + k], load(y, y_kind, (r * slots + k) * dims + d), acc);
        store(out, out_kind, r * dims + d, acc);
    }
}
