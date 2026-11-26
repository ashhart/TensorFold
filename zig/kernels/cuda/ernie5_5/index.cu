// ERNIE 5.5's DSA indexer: its keys (k's LayerNorm, then NeoX RoPE), and the sparse MLA's key lists (picks, window).

#include <cuda_bf16.h>

namespace ernie55i {

enum Fault { none, all_picks, short_window, gptj_pairs, no_bias };

// sincosf out of line: one copy of its large-angle path, not one a dim.
static __device__ __noinline__ void turn(float a, float* s, float* c) { sincosf(a, s, c); }

// Row blockIdx.x at pos[0] + row, a warp, 4 dims a lane: fp32 LayerNorm, dims below 64 NeoX-turned, bf16 once.
template <Fault F>
__device__ __forceinline__ void key(const __nv_bfloat16* __restrict__ k, long long ks,
                                    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b,
                                    __nv_bfloat16* __restrict__ out, long long os, const int* __restrict__ pos,
                                    float eps, const float* __restrict__ inv) {
    constexpr int D = 128, R = 64;
    constexpr unsigned all = 0xffffffffu;
    const long long row = blockIdx.x, at = static_cast<long long>(pos[0]) + row;
    const int d0 = threadIdx.x * 4;
    float x[4];
    for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(k[row * ks + d0 + j]);
    float s = __fadd_rn(__fadd_rn(__fadd_rn(x[0], x[1]), x[2]), x[3]);
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(all, s, o));
    const float mean = __fmul_rn(s, 1.0f / D);  // a power of two: exact
    float xc[4];
    for (int j = 0; j < 4; ++j) xc[j] = __fsub_rn(x[j], mean);
    float v = __fmul_rn(xc[0], xc[0]);
    for (int j = 1; j < 4; ++j) v = __fmaf_rn(xc[j], xc[j], v);
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(all, v, o));
    const float sd = __fsqrt_rn(__fadd_rn(__fmul_rn(v, 1.0f / D), eps));
    float y[4];
    for (int j = 0; j < 4; ++j) {
        const float h = F == no_bias ? 0.0f : __bfloat162float(b[d0 + j]);
        y[j] = __fmaf_rn(__fdiv_rn(xc[j], sd), __bfloat162float(w[d0 + j]), h);
    }
    // NeoX: lanes 0-7 hold dims 0-31 and lanes 8-15 dims 32-63, so lane ^ 8 holds each one's partner
    float p[4];
    for (int j = 0; j < 4; ++j) p[j] = __shfl_xor_sync(all, y[j], 8);
    if (d0 < R) {
        float t[4];
        for (int j = 0; j < 4; ++j) {
            const int d = d0 + j;
            // GPT-J's pairs (2i, 2i + 1), the control: the partner in this lane, pair d / 2
            const int i = F == gptj_pairs ? d / 2 : d % (R / 2);
            const float other = F == gptj_pairs ? y[j ^ 1] : p[j];
            const bool first = F == gptj_pairs ? d % 2 == 0 : d < R / 2;
            float sn, cs;
            turn(__fmul_rn(inv[i], static_cast<float>(at)), &sn, &cs);
            t[j] = first ? __fsub_rn(__fmul_rn(y[j], cs), __fmul_rn(other, sn))
                         : __fadd_rn(__fmul_rn(y[j], cs), __fmul_rn(other, sn));
        }
        for (int j = 0; j < 4; ++j) y[j] = t[j];
    }
    for (int j = 0; j < 4; ++j) out[row * os + d0 + j] = __float2bfloat16_rn(y[j]);
}

// Row blockIdx.x at pos[0] + row: picks to picked[row] in [0, k] and the first -1, the window to seq - 1, -1, a count.
template <Fault F>
__device__ __forceinline__ void merge(const int* __restrict__ picks, const int* __restrict__ picked,
                                      const int* __restrict__ pos, int* __restrict__ tok, int* __restrict__ cnt, int k,
                                      int window, long long w) {
    __shared__ int valid;
    const long long row = blockIdx.x, seq = static_cast<long long>(pos[0]) + row + 1;
    const int p = picked[row];
    const long long held = F == all_picks ? k : p < 0 ? 0 : p > k ? k : p;
    const long long sw = (seq < window ? seq : window) - (F == short_window ? 1 : 0), first = seq - sw;
    const int* in = picks + row * k;
    if (threadIdx.x == 0) valid = static_cast<int>(held);
    __syncthreads();
    // a negative pick inside the held count ends the picks (the control keeps every pick it is told)
    if (F != all_picks)
        for (long long c = threadIdx.x; c < held; c += blockDim.x)
            if (in[c] < 0) atomicMin(&valid, static_cast<int>(c));
    __syncthreads();
    const long long n = valid;
    int* out = tok + row * w;
    for (long long c = threadIdx.x; c < w; c += blockDim.x)
        out[c] = c < n ? in[c] : c < n + sw ? static_cast<int>(first + (c - n)) : -1;
    if (threadIdx.x == 0) cnt[row] = static_cast<int>(n + sw);
}

}  // namespace ernie55i

// One row a block of 256 threads; lists w wide (at least k + window).
extern "C" __global__ void __launch_bounds__(256)
    tf_ernie55_window_merge(const int* __restrict__ picks, const int* __restrict__ picked, const int* __restrict__ pos,
                            int* __restrict__ tok, int* __restrict__ cnt, int k, int window, long long w) {
    ernie55i::merge<ernie55i::none>(picks, picked, pos, tok, cnt, k, window, w);
}

// The contract's planted control: every row takes all k picks, its -1s past picked among them.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_window_merge_all_picks(const int* __restrict__ picks, const int* __restrict__ picked,
                                          const int* __restrict__ pos, int* __restrict__ tok, int* __restrict__ cnt,
                                          int k, int window, long long w) {
    ernie55i::merge<ernie55i::all_picks>(picks, picked, pos, tok, cnt, k, window, w);
}

// The contract's planted control: the window's oldest position dropped.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_window_merge_short_window(const int* __restrict__ picks, const int* __restrict__ picked,
                                             const int* __restrict__ pos, int* __restrict__ tok,
                                             int* __restrict__ cnt, int k, int window, long long w) {
    ernie55i::merge<ernie55i::short_window>(picks, picked, pos, tok, cnt, k, window, w);
}

// One key row a warp: k rows ks apart (bf16) into out rows os apart, row r at position pos[0] + r.
extern "C" __global__ void __launch_bounds__(32)
    tf_ernie55_index_key(const __nv_bfloat16* __restrict__ k, long long ks, const __nv_bfloat16* __restrict__ w,
                         const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, long long os,
                         const int* __restrict__ pos, float eps, const float* __restrict__ inv) {
    ernie55i::key<ernie55i::none>(k, ks, w, b, out, os, pos, eps, inv);
}

// The contract's planted control: GPT-J's pairs (2i, 2i + 1), as the MLA's own rope turns them, in place of NeoX's.
extern "C" __global__ void __launch_bounds__(32)
    tf_ctl_ernie55_index_key_gptj_pairs(const __nv_bfloat16* __restrict__ k, long long ks,
                                        const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b,
                                        __nv_bfloat16* __restrict__ out, long long os, const int* __restrict__ pos,
                                        float eps, const float* __restrict__ inv) {
    ernie55i::key<ernie55i::gptj_pairs>(k, ks, w, b, out, os, pos, eps, inv);
}

// The contract's planted control: the LayerNorm without k_norm's bias.
extern "C" __global__ void __launch_bounds__(32)
    tf_ctl_ernie55_index_key_no_bias(const __nv_bfloat16* __restrict__ k, long long ks,
                                     const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b,
                                     __nv_bfloat16* __restrict__ out, long long os, const int* __restrict__ pos,
                                     float eps, const float* __restrict__ inv) {
    ernie55i::key<ernie55i::no_bias>(k, ks, w, b, out, os, pos, eps, inv);
}
