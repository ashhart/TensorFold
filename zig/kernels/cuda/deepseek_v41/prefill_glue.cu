// Prefill segment glue: the torch ops between a segment's kernels, element-exact (integers and casts only).

// top_positions, cand_keys, counts, gm_picks, width_mask, f64_bf16_f32, widen_cat, ring_copy: see each kernel.

// Built with --fmad=false (nothing here multiplies and adds).

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace dsv41_pfglue {

constexpr long long KEY_NONE = (long long)(1ull << 63);    // -(2 ** 63)
constexpr int NT = 1024, CMAX = 2048;

__device__ __forceinline__ unsigned long long ukey(long long k) {   // signed order as unsigned
    return static_cast<unsigned long long>(k) ^ 0x8000000000000000ull;
}

// One CTA a row: 8-pass radix select of the c-th largest key (c = min(count, n)), bitonic sort ascending, -1 padded.
__global__ void __launch_bounds__(NT) top_positions_kernel(const long long* __restrict__ keys, long long ld, int n,
                                                           int count, int* __restrict__ out) {
    __shared__ int hist[256];
    __shared__ unsigned long long s_prefix, s_mask;
    __shared__ int s_need, s_n, s_eq;
    __shared__ int pos_sh[CMAX];
    const int tid = threadIdx.x;
    const long long* row = keys + (long long)blockIdx.x * ld;
    const int c = n < count ? n : count;
    if (tid == 0) { s_prefix = 0; s_mask = 0; s_need = c; }
    __syncthreads();
    for (int shift = 56; shift >= 0 && c > 0; shift -= 8) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const unsigned long long prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n; i += NT) {
            const unsigned long long u = ukey(row[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need, d = 255;
            for (; d > 0 && hist[d] < need; --d) need -= hist[d];
            s_need = need;                                   // how many keys equal to the c-th are taken
            s_prefix = prefix | ((unsigned long long)d << shift);
            s_mask = mask | (255ull << shift);
        }
        __syncthreads();
    }
    const unsigned long long kth = s_prefix;
    if (tid == 0) { s_n = 0; s_eq = 0; }
    __syncthreads();
    // winners: every key above the c-th, and `need` keys equal to it (equal keys are one value: any of them)
    for (int i = tid; i < n && c > 0; i += NT) {
        const long long k = row[i];
        const unsigned long long u = ukey(k);
        bool take = u > kth;
        if (!take && u == kth) take = atomicAdd(&s_eq, 1) < s_need;
        if (take) {
            long long p;
            if (k == KEY_NONE) p = 0x7FFFFFFF;                                   // -1 after the sort
            else {
                p = 0x7FFFFFFFll - (k & 0xFFFFFFFFll);
                if (p < 0) p = 0x7FFFFFFF;
            }
            pos_sh[atomicAdd(&s_n, 1)] = (int)p;
        }
    }
    __syncthreads();
    int m = 1;
    while (m < c) m <<= 1;
    for (int i = c + tid; i < m; i += NT) pos_sh[i] = 0x7FFFFFFF;
    __syncthreads();
    for (int size = 2; size <= m; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < m; i += NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool asc = (i & size) == 0;
                    const int a = pos_sh[i], b = pos_sh[j];
                    if (asc ? a > b : a < b) { pos_sh[i] = b; pos_sh[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    int* o = out + (long long)blockIdx.x * count;
    for (int i = tid; i < count; i += NT) {
        const int p = i < c ? pos_sh[i] : 0x7FFFFFFF;
        o[i] = p == 0x7FFFFFFF ? -1 : p;
    }
}

// blocks [R, nb] (row stride ld) -> out [R, nb * bs]
__global__ void __launch_bounds__(256) cand_keys_kernel(const int* __restrict__ blocks, long long ld, int R, int nb,
                                                        int bs, int* __restrict__ out) {
    const long long total = (long long)R * nb * bs;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const long long r = e / ((long long)nb * bs), q = e - r * nb * bs, b = q / bs, t = q - b * bs;
        const int blk = blocks[r * ld + b];
        out[e] = blk >= 0 ? (int)((long long)blk * bs + t) : -1;
    }
}

// one warp a row: sel [R, K] int32, pos int64 [R]
__global__ void __launch_bounds__(256) counts_kernel(const int* __restrict__ sel, int K, const long long* __restrict__ pos,
                                                     int ratio, int R, int* __restrict__ out) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= R) return;
    // positions in [0, 2^31 - 1) and ratio > 0: C's truncating division equals floor division
    const long long nvis = (long long)((int)(pos[r] + 1) / ratio);
    int c = 0;
    for (int j = lane; j < K; j += 32) {
        const int s = sel[(long long)r * K + j];
        c += (s >= 0 && (long long)s < nvis) ? 1 : 0;
    }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if (lane == 0) out[r] = c;
}

// pick / wts [n, ld] -> pk int32 [n, topk], w6 fp32 [n, topk] (the kit's fp16 weights)
__global__ void __launch_bounds__(256) gm_picks_kernel(const int* __restrict__ pick, const float* __restrict__ wts, int ld,
                                                       int topk, int n, int* __restrict__ pk, float* __restrict__ w6) {
    const long long total = (long long)n * topk;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const long long i = e / topk, j = e - i * topk;
        pk[e] = pick[i * ld + j];
        w6[e] = __half2float(__float2half(wts[i * ld + j]));
    }
}

__global__ void __launch_bounds__(256) width_mask_kernel(const int* __restrict__ pick, long long P,
                                                         const int* __restrict__ k2tab, int E, int k2,
                                                         int* __restrict__ out) {
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < P; e += (long long)gridDim.x * blockDim.x) {
        const int v = pick[e];
        const int w = (v >= 0 && v < E) ? k2tab[v] : 0;
        out[e] = w == k2 ? v : E;
    }
}

__global__ void __launch_bounds__(256) f64_bf16_f32_kernel(const double* __restrict__ x, float* __restrict__ y, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = __bfloat162float(__float2bfloat16(__double2float_rn(x[i])));
}

// a [n, hd] (and b [n, hd]) bf16 -> out [n, hd * (b ? 2 : 1)] fp32
__global__ void __launch_bounds__(256) widen_cat_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
                                                        int n, int hd, float* __restrict__ out) {
    const int w = b ? 2 * hd : hd;
    const long long total = (long long)n * w;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const long long i = e / w, j = e - i * w;
        const uint16_t v = j < hd ? a[i * hd + j] : b[i * hd + (j - hd)];
        out[e] = __uint_as_float(static_cast<uint32_t>(v) << 16);
    }
}

// rows p in [lo, hi) of a src_ring ring into a dst_ring ring, row_bytes each (4-byte units when all 4-byte aligned)
__global__ void __launch_bounds__(256) ring_copy_kernel(const unsigned char* __restrict__ src, int src_ring,
                                                        unsigned char* __restrict__ dst, int dst_ring, long long row_bytes,
                                                        long long lo, long long hi) {
    const bool words = (row_bytes % 4 == 0) && (((uintptr_t)src | (uintptr_t)dst) % 4 == 0);
    const long long unit = words ? 4 : 1, per = row_bytes / unit, total = (hi - lo) * per;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const long long p = lo + e / per, k = e - (e / per) * per;
        const long long s = p % src_ring, d = p % dst_ring;
        if (words)
            reinterpret_cast<uint32_t*>(dst + d * row_bytes)[k] = reinterpret_cast<const uint32_t*>(src + s * row_bytes)[k];
        else
            dst[d * row_bytes + k] = src[s * row_bytes + k];
    }
}

}  // namespace dsv41_pfglue
