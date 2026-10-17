// DeepSeek-V4.1 window glue: torch's CUDA arithmetic per element (fp32 opmath, bf16 / f16 rn casts, --fmad=false).

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace dsv41_glue {

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }
__device__ __forceinline__ uint16_t to_bf(float x) { return __bfloat16_as_ushort(__float2bfloat16(x)); }

// rows [n, D] = bf16_rn(f32 embed[ids - lo]) from the rank's slice [V, D], +0 off it (NaN -> 0x7FFF)
__global__ void __launch_bounds__(256) embed_send_kernel(const long long* __restrict__ ids, int n,
                                                         const uint16_t* __restrict__ embed, long long vocab_lo, int V,
                                                         int D, uint16_t* __restrict__ rows) {
    const long long total = (long long)n * D;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const long long i = e / D, d = e - i * D;
        const long long local = ids[i] - vocab_lo;
        rows[e] = (local >= 0 && local < V) ? to_bf(bf(embed[local * D + d])) : (uint16_t)0;
    }
}

// recv [W, n, D] bf16 summed in rank order (fp32 adds) -> out [n, 4 D] bf16, the sum repeated x4
__global__ void __launch_bounds__(256) embed_sum_kernel(const uint16_t* __restrict__ recv, int W, int n, int D,
                                                        uint16_t* __restrict__ out) {
    const long long total = (long long)n * D, plane = total;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        float s = bf(recv[e]);
        for (int w = 1; w < W; ++w) s = __fadd_rn(s, bf(recv[w * plane + e]));
        const uint16_t b = to_bf(s);
        const long long i = e / D, d = e - i * D;
        uint16_t* o = out + i * 4 * D + d;
#pragma unroll
        for (int r = 0; r < 4; ++r) o[r * D] = b;
    }
}

// y = bf16_rn(x) (the MoE partial)
__global__ void __launch_bounds__(256) cast_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y,
                                                        long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = to_bf(x[i]);
}

// y = f32(f16_rn(x))
__global__ void __launch_bounds__(256) kit_weights_kernel(const float* __restrict__ x, float* __restrict__ y,
                                                          long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = __half2float(__float2half(x[i]));
}

// x = f32(bf16_rn(x)) in place
__global__ void __launch_bounds__(256) kit_logits_kernel(float* __restrict__ x, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        x[i] = bf(to_bf(x[i]));
}

// pos[0] = start; pos64[i] = start + i; lo[i] = 0
__global__ void __launch_bounds__(256) positions_kernel(int start, int n, int* __restrict__ pos,
                                                        long long* __restrict__ pos64, int* __restrict__ lo) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) pos[0] = start;
    if (i < n) {
        pos64[i] = (long long)start + i;
        lo[i] = 0;
    }
}

// a graphed window's positions: `start` from the window's staged statics, so one captured graph serves every window
__global__ void __launch_bounds__(256) positions_dev_kernel(const int* __restrict__ start, int n, int* __restrict__ pos,
                                                            long long* __restrict__ pos64, int* __restrict__ lo) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = *start;
    if (i == 0) pos[0] = s;
    if (i < n) {
        pos64[i] = (long long)s + i;
        lo[i] = 0;
    }
}

// recv [W, n, c] -> out [n, W c] (bf16 words, any 2-byte type)
__global__ void __launch_bounds__(256) gather_cols_kernel(const uint16_t* __restrict__ recv, int W, int n, int c,
                                                          uint16_t* __restrict__ out) {
    const long long total = (long long)W * n * c;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (long long)gridDim.x * blockDim.x) {
        const long long i = e / ((long long)W * c), r = e - i * W * c, w = r / c, j = r - w * c;
        out[e] = recv[(w * n + i) * c + j];
    }
}

// compressor carry [cols] fp32 <- bf16 row `row` of src (row stride ld); the row from the device when `accepted` is set
__global__ void __launch_bounds__(256) carry_kernel(const uint16_t* __restrict__ src, long long ld,
                                                    const int* __restrict__ accepted, int row, int cols,
                                                    float* __restrict__ carry) {
    const int r = accepted ? *accepted : row;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cols; j += gridDim.x * blockDim.x)
        carry[j] = bf(src[(long long)r * ld + j]);
}

// bf16 -> fp32 (exact: the bits shifted into the high half), n elements
__global__ void __launch_bounds__(256) widen_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y,
                                                        long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = bf(x[i]);
}

// DSpark candidates: a CTA per fp32 logit row [rows, V] (stride ld); the best k by (value desc, column asc), exact.
constexpr int DS_NT = 1024, DS_KMAX = 128;

__device__ __forceinline__ unsigned long long ds_key(float x, int col) {   // unique in a row; -0.0 == +0.0
    const unsigned int b = __float_as_uint(x + 0.0f);                      // -0.0 + 0.0 = +0.0
    const unsigned int mono = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (static_cast<unsigned long long>(mono) << 32) | (0xFFFFFFFFull - static_cast<unsigned int>(col));
}

__global__ void __launch_bounds__(DS_NT) ds_cands_kernel(const float* __restrict__ lg, long long ld, int V, int k,
                                                         int id0, float* __restrict__ out) {
    __shared__ int hist[256];
    __shared__ unsigned long long s_prefix, s_mask;
    __shared__ int s_need, s_done, s_n;
    __shared__ unsigned long long key_sh[DS_KMAX];
    const int tid = threadIdx.x;
    const float* row = lg + (long long)blockIdx.x * ld;
    float* o = out + (long long)blockIdx.x * 2 * k;   // k values, then id0 + col as int bits; padding -inf, -1
    const int c = V < k ? V : k;
    if (tid == 0) { s_prefix = 0; s_mask = 0; s_need = c; s_done = 0; }
    __syncthreads();
    for (int shift = 56; shift >= 0 && c > 0; shift -= 8) {   // radix select of the k-th key, 8 bits a pass
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const unsigned long long prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < V; i += DS_NT) {
            const unsigned long long u = ds_key(row[i], i);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need, d = 255;
            for (; d > 0 && hist[d] < need; --d) need -= hist[d];
            s_need = need;
            s_prefix = prefix | ((unsigned long long)d << shift);
            s_mask = mask | (255ull << shift);
            s_done = hist[d] == need;                                       // the digit's keys: all winners
        }
        __syncthreads();
        if (s_done) break;
    }
    const unsigned long long prefix = s_prefix, mask = s_mask;
    if (tid == 0) s_n = 0;
    __syncthreads();
    for (int i = tid; i < V && c > 0; i += DS_NT) {
        const unsigned long long u = ds_key(row[i], i);
        if ((u & mask) >= prefix) {
            const int at = atomicAdd(&s_n, 1);
            if (at < DS_KMAX) key_sh[at] = u;
        }
    }
    __syncthreads();
    int m = 1;
    while (m < c) m <<= 1;
    for (int i = c + tid; i < m; i += DS_NT) key_sh[i] = 0;                // sorts last
    __syncthreads();
    for (int size = 2; size <= m; size <<= 1) {   // bitonic sort, descending
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < m; i += DS_NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const unsigned long long a = key_sh[i], b = key_sh[j];
                    if (desc ? a < b : a > b) { key_sh[i] = b; key_sh[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < k; i += DS_NT) {
        if (i < c) {
            const int col = (int)(0xFFFFFFFFull - (key_sh[i] & 0xFFFFFFFFull));
            o[i] = row[col];
            o[k + i] = __int_as_float(id0 + col);
        } else {
            o[i] = -__int_as_float(0x7F800000);
            o[k + i] = __int_as_float(-1);
        }
    }
}

// pairs[r] = (f64 vals[r], f64 (cols[r] + id0)), a rank's best; one thread a row
__global__ void __launch_bounds__(256) pick_pack_kernel(const float* __restrict__ vals, const long long* __restrict__ cols,
                                                        int n, long long id0, double* __restrict__ pairs) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    pairs[2 * r] = static_cast<double>(vals[r]);
    pairs[2 * r + 1] = static_cast<double>(cols[r] + id0);
}

// out[r]: best id over ranks (tie / NaN: lower rank), then accepted, bonus, next position; one block, n <= 1024
__global__ void __launch_bounds__(1024) pick_merge_kernel(const double* __restrict__ all, int W, int n,
                                                          const long long* __restrict__ ids, long long start,
                                                          long long* __restrict__ out) {
    const int r = threadIdx.x;
    if (r < n) {
        int bk = 0;
        for (int k = 1; k < W; k++)
            if (all[2 * (k * n + r)] > all[2 * (bk * n + r)]) bk = k;
        out[r] = static_cast<long long>(all[2 * (bk * n + r) + 1]);
    }
    __syncthreads();
    if (r == 0) {
        int acc = 0;
        if (ids != nullptr)
            while (acc + 1 < n && ids[acc + 1] == out[acc]) acc++;
        out[n] = acc;
        out[n + 1] = out[acc];
        out[n + 2] = start + acc + 1;
    }
}

// one block: DSpark's one-slot statics from the device pick, (anchor, start) = (bonus, next position); rest as staged
__global__ void __launch_bounds__(256) ds_stage_kernel(const long long* __restrict__ pick, int nw, long long* __restrict__ s64,
                                                      int* __restrict__ s32, int n, int W, int ring, long long valid,
                                                      int noise, int o_pos, int o_tok, int o_cnt, int o_lo, int o_hi,
                                                      int o_anchor) {
    const long long A = pick[nw + 1], P = pick[nw + 2];
    const long long lo_w = P > W ? P - W : 0;                       // p -| window
    const long long first = lo_w > valid ? lo_w : valid;
    const int count = P > first ? (int)(P - first) : 0;
    const int t = threadIdx.x;
    for (int i = t; i < n; i += blockDim.x) {
        s64[i] = i == 0 ? A : (long long)noise;                     // ids (offset 0)
        s64[o_pos + i] = P + i;
        s32[o_cnt + i] = count;
        s32[o_lo + i] = (int)P;
        s32[o_hi + i] = (int)(P + n - 1);
    }
    for (int e = t; e < n * W; e += blockDim.x) {
        const int j = e % W;
        s32[o_tok + e] = j < count ? (int)((first + j) % ring) : -1;
    }
    if (t == 0) {
        s32[0] = (int)P;                                            // start (offset 0)
        s32[o_anchor] = (int)A;
    }
}

// one thread a slot (segs [k][3]: first row, rows, start): pick_merge's chain walk -> acc [k][3]
__global__ void __launch_bounds__(32) ds_accept_rows_kernel(const long long* __restrict__ picks, const long long* __restrict__ ids,
                                                           const long long* __restrict__ segs, int k, long long* __restrict__ acc) {
    const int s = threadIdx.x;
    if (s >= k) return;
    const long long a = segs[3 * s], n = segs[3 * s + 1], start = segs[3 * s + 2];
    long long c = 0;
    while (c + 1 < n && ids[a + c + 1] == picks[a + c]) c++;
    acc[3 * s] = c;
    acc[3 * s + 1] = picks[a + c];
    acc[3 * s + 2] = start + c + 1;
}

// one block: a pass's members (mem [g][3]: acc index, slot, valid), each one's n rows from its (bonus, next position)
__global__ void __launch_bounds__(256) ds_stage_rows_kernel(const long long* __restrict__ acc, const long long* __restrict__ mem,
                                                           int g, long long* __restrict__ s64, int* __restrict__ s32, int n,
                                                           int W, int ring, int noise, int o_ids, int o_pos, int o_tok,
                                                           int o_cnt, int o_lo, int o_hi) {
    const int t = threadIdx.x;
    for (int m = 0; m < g; m++) {
        const long long j = mem[3 * m], slot = mem[3 * m + 1], valid = mem[3 * m + 2];
        const long long A = acc[3 * j + 1], P = acc[3 * j + 2];
        const long long lo_w = P > W ? P - W : 0;
        const long long first = lo_w > valid ? lo_w : valid;
        const int count = P > first ? (int)(P - first) : 0;
        const int a = m * n;
        for (int i = t; i < n; i += blockDim.x) {
            s64[o_ids + a + i] = i == 0 ? A : (long long)noise;
            s64[o_pos + a + i] = P + i;
            s32[o_cnt + a + i] = count;
            s32[o_lo + a + i] = (int)P;
            s32[o_hi + a + i] = (int)(P + n - 1);
        }
        for (int e = t; e < n * W; e += blockDim.x) {
            const int jj = e % W;
            s32[o_tok + a * W + e] = jj < count ? (int)(slot * ring + (first + jj) % ring) : -1;
        }
    }
}

__device__ __forceinline__ unsigned long long vs_mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ double vs_uniform(unsigned long long seed, unsigned long long pos, unsigned long long id) {
    unsigned long long x = vs_mix(seed + 0x9E3779B97F4A7C15ull);
    x = vs_mix(x ^ (pos * 0xD1B54A32D192ED03ull));
    x = vs_mix(x ^ id);
    return (double)(x >> 11) * 0x1p-53 + 0x1p-54;
}

__device__ __forceinline__ float vs_value(unsigned long long k) {
    const unsigned int mono = (unsigned int)(k >> 32);
    return __uint_as_float((mono & 0x80000000u) ? (mono & 0x7FFFFFFFu) : ~mono);
}

// numpy's float64 pairwise sum (leaves of 8..128, halves cut to x8); depth a template, so no runtime recursion
__device__ __forceinline__ double vs_leaf(const double* a, int n) {
    if (n < 8) {
        double r = -0.0;
        for (int i = 0; i < n; i++) r += a[i];
        return r;
    }
    double r[8];
    for (int j = 0; j < 8; j++) r[j] = a[j];
    int i = 8;
    for (; i < n - (n % 8); i += 8)
        for (int j = 0; j < 8; j++) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
}

template <int D> __device__ __forceinline__ double vs_pairwise(const double* a, int n) {
    if (n <= 128) return vs_leaf(a, n);
    int n2 = n / 2;
    n2 -= n2 % 8;
    return vs_pairwise<D - 1>(a, n2) + vs_pairwise<D - 1>(a + n2, n - n2);
}

template <> __device__ __forceinline__ double vs_pairwise<0>(const double* a, int n) { return vs_leaf(a, n); }

constexpr int VS_NT = 256, VS_MAX = 2048;

// one block a row: speculative keyed pick over g [W][n][2 k] (values, ids as int bits); the host's pick decides
__global__ void __launch_bounds__(VS_NT) vs_choose_kernel(const float* __restrict__ g, int W, int n, int k, int count,
                                                         unsigned long long seed, long long pos0, int top_k, double temp,
                                                         double top_p, double min_p, long long* __restrict__ out) {
    __shared__ unsigned long long keys[VS_MAX];
    __shared__ double work[1024];
    const int r = blockIdx.x, t = threadIdx.x, C = W * k;
    if (C > VS_MAX) { if (t == 0) out[r] = -1; return; }
    for (int i = t; i < C; i += VS_NT) {
        const int w = i / k, j = i - w * k;
        const float* row = g + ((long long)w * n + r) * 2 * k;
        const unsigned int bits = __float_as_uint(row[j] + 0.0f);
        const unsigned int mono = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        const unsigned int id = __float_as_uint(row[k + j]);
        keys[i] = ((unsigned long long)mono << 32) | (0xFFFFFFFFull - id);
    }
    int m = 1;
    while (m < C) m <<= 1;
    for (int i = C + t; i < m; i += VS_NT) keys[i] = 0;
    __syncthreads();
    for (int size = 2; size <= m; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = t; i < m; i += VS_NT) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const unsigned long long a = keys[i], b = keys[j];
                    if (desc ? a < b : a > b) { keys[i] = b; keys[j] = a; }
                }
            }
            __syncthreads();
        }
    if (t != 0) return;
    const long long id0 = (long long)(0xFFFFFFFFull - (keys[0] & 0xFFFFFFFFull));
    if (!(temp > 0.0)) { out[r] = id0; return; }                            // greedy: the first merged candidate
    if (top_k == 0 && top_p > 0.0 && top_p < 1.0) { out[r] = -1; return; }  // a nucleus row: the host's statistics
    const int width = (count > 0 && count < C) ? count : C;   // count 0: all W k
    int kk = top_k != 0 ? (top_k < width ? top_k : width) : width;
    if (kk < 1) kk = 1;
    if (kk > 1024) kk = 1024;
    const double tt = temp > 1e-6 ? temp : 1e-6;
    const double s0 = (double)vs_value(keys[0]) / tt;
    int keep = kk;
    if (top_p > 0.0 && top_p < 1.0) {
        for (int j = 0; j < kk; j++) work[j] = exp((double)vs_value(keys[j]) / tt - s0);
        const double total = vs_pairwise<3>(work, kk);   // kk <= 1024
        double cum = 0.0;
        int below = 0;
        for (int j = 0; j < kk; j++) {
            cum += work[j] / total;
            if (cum < top_p) below++;
        }
        keep = below + 1;
    }
    const double floor_ = s0 + (min_p > 0.0 ? log(min_p) : -INFINITY);
    const int lim = keep < kk ? keep : kk;
    int best = 0;
    double best_score = -INFINITY;
    for (int j = 0; j < lim; j++) {
        const double x = (double)vs_value(keys[j]) / tt;
        if (min_p > 0.0 && x < floor_) continue;
        const unsigned long long id = 0xFFFFFFFFull - (keys[j] & 0xFFFFFFFFull);
        const double score = x - log(-log(vs_uniform(seed, (unsigned long long)(pos0 + r), id)));
        if (j == 0 || score > best_score) { best = j; best_score = score; }
    }
    out[r] = (long long)(0xFFFFFFFFull - (keys[best] & 0xFFFFFFFFull));
}

}  // namespace dsv41_glue
