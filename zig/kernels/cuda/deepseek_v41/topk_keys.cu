// Row top-k of bf16-valued fp32 logits on the device: value descending, ties by lower column (torch.topk of keys).

// keys = (mono(bits(x + 0.0)) & -65536) | (65535 - col): unique in a row (C <= 65,536), so any exact select matches.

// One CTA a row: 4-pass 8-bit radix select of the k-th key, gather k winners, bitonic sort; cols int64, vals fp32.

#include <cstdint>

namespace dsv41_topk_keys {

constexpr int NT = 1024, KMAX = 1024, COLS = 1 << 16;

__device__ __forceinline__ uint32_t ukey(float x, int col) {
    int b = __float_as_int(x + 0.0f);                  // -0.0 + 0.0 = +0.0
    const int mono = b >= 0 ? b : (b ^ 0x7FFFFFFF);
    const int key = (mono & -COLS) | (COLS - 1 - col);
    return static_cast<uint32_t>(key) ^ 0x80000000u;   // signed order as unsigned
}

__global__ void __launch_bounds__(NT) topk_kernel(const float* __restrict__ lg, long long ld, int C, int k,
                                                  float* __restrict__ vals, long long* __restrict__ cols) {
    __shared__ int hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need, s_n;
    __shared__ uint32_t key_sh[KMAX];
    __shared__ int col_sh[KMAX];
    const int tid = threadIdx.x;
    const float* row = lg + (long long)blockIdx.x * ld;

    // radix select: the k-th largest key, 8 bits a pass from the top
    if (tid == 0) { s_prefix = 0; s_mask = 0; s_need = k; }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        for (int c = tid; c < C; c += NT) {
            const uint32_t u = ukey(row[c], c);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {                                // the digit holding the need-th largest, from the top
            int need = s_need, d = 255;
            for (; d > 0 && hist[d] < need; --d) need -= hist[d];
            s_need = need;
            s_prefix = prefix | ((uint32_t)d << shift);
            s_mask = mask | (255u << shift);
        }
        __syncthreads();
    }
    const uint32_t kth = s_prefix;                     // the k-th largest key itself (keys are unique)

    // the winners (> kth, and kth): exactly k, in any order
    if (tid == 0) s_n = 0;
    __syncthreads();
    for (int c = tid; c < C; c += NT) {
        const uint32_t u = ukey(row[c], c);
        if (u >= kth) {
            const int at = atomicAdd(&s_n, 1);
            key_sh[at] = u;
            col_sh[at] = c;
        }
    }
    __syncthreads();
    int n = 1;
    while (n < k) n <<= 1;
    for (int i = k + tid; i < n; i += NT) { key_sh[i] = 0; col_sh[i] = -1; }   // pads sort last (keys >= 1)
    __syncthreads();

    // bitonic sort, descending
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < n; i += NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const uint32_t a = key_sh[i], b = key_sh[j];
                    if (desc ? a < b : a > b) {
                        key_sh[i] = b; key_sh[j] = a;
                        const int t = col_sh[i]; col_sh[i] = col_sh[j]; col_sh[j] = t;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < k; i += NT) {
        const int c = col_sh[i];
        cols[(long long)blockIdx.x * k + i] = c;
        vals[(long long)blockIdx.x * k + i] = row[c];
    }
}

}  // namespace dsv41_topk_keys
