// Muse Spark's DFlash2 drafter ops past dflash.cu's: dense rows, causal convs, SiLU-mul, top 16, selector walk.

#include <cstdint>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float w16(const uint16_t* p, int64_t i, int bf16) {
    const uint16_t b = p[i];
    return bf16 ? __uint_as_float(((uint32_t)b) << 16) : __half2float(__ushort_as_half(b));
}

constexpr int TOPK = 16;

// a beats b: the larger value, ties to the lower id
__device__ __forceinline__ bool beats(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

}  // namespace

// out[r, j] = sum_k x[r, k] * W[j, k] (W fp16 or bf16 [N, K]); grid (N / 8, n / 8), a warp an output.
extern "C" __global__ void dfl_dense(const __half* __restrict__ x, const uint16_t* __restrict__ W, int w_bf16,
                                     void* __restrict__ out, int out_f32, int n, int K, int N) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * 8 + warp, r0 = blockIdx.y * 8;
    if (j >= N) return;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    for (int k = lane; k < K; k += 32) {
        const float w = w16(W, (int64_t)j * K + k, w_bf16);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (r0 + i < n) acc[i] += w * __half2float(x[(int64_t)(r0 + i) * K + k]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float s = warp_sum(acc[i]);
        if (lane == 0 && r0 + i < n) {
            if (out_f32) ((float*)out)[(int64_t)(r0 + i) * N + j] = s;
            else ((__half*)out)[(int64_t)(r0 + i) * N + j] = __float2half_rn(s);
        }
    }
}

// GroupedDynamicCausalConv.prepare: out[r, c] = fp16(sum_t (base[t, c] + dyn[r, 0, t, c / gs]) * h[r - t, c]).
extern "C" __global__ void dfl_conv_prep(const __half* __restrict__ h, const __half* __restrict__ dyn,
                                         const float* __restrict__ base, __half* __restrict__ out, int B, int D, int KT,
                                         int gs) {
    const int r = blockIdx.x, b = r % B, G = D / gs;
    const __half* dr = dyn + (int64_t)r * 2 * KT * G;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < KT && t <= b; ++t) {
            const float w = base[t * D + c] + __half2float(dr[t * G + c / gs]);
            acc += w * __half2float(h[(int64_t)(r - t) * D + c]);
        }
        out[(int64_t)r * D + c] = __float2half_rn(acc);
    }
}

// GroupedDynamicCausalConv.finish into the residual: res += sum_t (base1[t, c] + dyn[r, 1, t, ...]) * y[r - t].
extern "C" __global__ void dfl_conv_finish(float* __restrict__ res, const float* __restrict__ y0,
                                           const float* __restrict__ y1, const __half* __restrict__ dyn,
                                           const float* __restrict__ base, int B, int D, int KT, int gs) {
    const int r = blockIdx.x, b = r % B, G = D / gs;
    const __half* dr = dyn + (int64_t)r * 2 * KT * G + KT * G;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < KT && t <= b; ++t) {
            const int64_t at = (int64_t)(r - t) * D + c;
            const float y = y1 ? y0[at] + y1[at] : y0[at];
            acc += (base[t * D + c] + __half2float(dr[t * G + c / gs])) * y;
        }
        res[(int64_t)r * D + c] += acc;
    }
}

// out = fp16(silu(g) * u), n elements (GatedMLP's interm in fp16).
extern "C" __global__ void dfl_silu_mul(const __half* __restrict__ g, const __half* __restrict__ u,
                                        __half* __restrict__ out, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const float x = __half2float(g[i]);
        out[i] = __float2half_rn(x / (1.f + expf(-x)) * __half2float(u[i]));
    }
}

// A rank's top 16 of each head-slice row: out [n, 32] = 16 values descending, then their global ids as bits.
extern "C" __global__ void __launch_bounds__(256) dfl_topk(const float* __restrict__ head, int w, int valid, int lo,
                                                           float* __restrict__ out) {
    __shared__ float hv[256];
    __shared__ int hi[256];
    __shared__ int ht[256];
    const int r = blockIdx.x, t = threadIdx.x;
    float v[TOPK];
    int id[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        v[k] = -CUDART_INF_F;
        id[k] = 0x7fffffff;
    }
    const float* row = head + (int64_t)r * w;
    for (int c = t; c < valid; c += 256) {
        const float x = row[c];
        const int g = lo + c;
        if (x != x) continue;  // a NaN logit is never a candidate
        if (!beats(x, g, v[TOPK - 1], id[TOPK - 1])) continue;
        int k = TOPK - 1;
        while (k > 0 && beats(x, g, v[k - 1], id[k - 1])) {
            v[k] = v[k - 1];
            id[k] = id[k - 1];
            --k;
        }
        v[k] = x;
        id[k] = g;
    }
    int ptr = 0;
    for (int pick = 0; pick < TOPK; ++pick) {
        hv[t] = ptr < TOPK ? v[ptr] : -CUDART_INF_F;
        hi[t] = ptr < TOPK ? id[ptr] : 0x7fffffff;
        ht[t] = t;
        __syncthreads();
        for (int o = 128; o > 0; o >>= 1) {
            if (t < o && beats(hv[t + o], hi[t + o], hv[t], hi[t])) {
                hv[t] = hv[t + o];
                hi[t] = hi[t + o];
                ht[t] = ht[t + o];
            }
            __syncthreads();
        }
        if (t == 0) {
            out[(int64_t)r * 2 * TOPK + pick] = hv[0];
            out[(int64_t)r * 2 * TOPK + TOPK + pick] = __int_as_float(hi[0]);
        }
        const int won = ht[0];
        __syncthreads();
        if (t == won) ++ptr;
    }
}

// CandidateSelector.select, greedy: a block a stream over the ranks' merged top 16s; the first maximum wins.
extern "C" __global__ void __launch_bounds__(512) dfl_walk(const float* __restrict__ recv, int W, int n1,
                                                           const __half* __restrict__ gate, int R,
                                                           const uint16_t* __restrict__ pcb,
                                                           const uint16_t* __restrict__ scb, int cb_bf16,
                                                           const int64_t* __restrict__ anchors, int P,
                                                           int32_t* __restrict__ out) {
    __shared__ float cv[TOPK];
    __shared__ int cid[TOPK];
    __shared__ float sc[TOPK];
    __shared__ int pred;
    const int s = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x == 0) pred = (int)anchors[s];
    for (int i = 0; i < P; ++i) {
        const int row = s * P + i;
        if (threadIdx.x == 0) {
            float av[64];
            int ai[64];
            const int m = W * TOPK;
            for (int q = 0; q < W; ++q)
                for (int j = 0; j < TOPK; ++j) {
                    const float* src = recv + ((int64_t)q * n1 + row) * 2 * TOPK;
                    av[q * TOPK + j] = src[j];
                    ai[q * TOPK + j] = __float_as_int(src[TOPK + j]);
                }
            for (int k = 0; k < TOPK; ++k) {
                int best = -1;
                for (int j = 0; j < m; ++j)
                    if (ai[j] >= 0 && ai[j] != 0x7fffffff && (best < 0 || beats(av[j], ai[j], av[best], ai[best])))
                        best = j;
                cv[k] = best < 0 ? -CUDART_INF_F : av[best];  // no valid id left (non-finite rows): no candidate
                cid[k] = best < 0 ? -1 : ai[best];
                if (best >= 0) ai[best] = -1;
            }
        }
        __syncthreads();
        if (warp < TOPK) {
            const int c = cid[warp], a = pred;
            const __half* g = gate + (int64_t)row * R;
            float acc = 0.f;
            for (int k = lane; c >= 0 && k < R; k += 32)
                acc += (w16(pcb, (int64_t)a * R + k, cb_bf16) * __half2float(g[k])) * w16(scb, (int64_t)c * R + k, cb_bf16);
            acc = warp_sum(acc);
            if (lane == 0) sc[warp] = c >= 0 ? cv[warp] + acc : -CUDART_INF_F;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int k = 1; k < TOPK; ++k)
                if (sc[k] > sc[best]) best = k;
            if (cid[best] >= 0) pred = cid[best];  // else the previous token again (the target rejects it)
            out[row] = pred;
        }
        __syncthreads();
    }
}
