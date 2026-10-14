// Flash Next port: device-side greedy candidates. One block per bf16 logits row: the row's first maximum (lowest
// column among equal maxima, as argmax picks it), mapped to a global token id, and the row's log-sum-exp in fp32.
// out[row] = {id, max (fp32 bits), lse (fp32 bits), 0}. Build: nvcc -cubin -arch=sm_121 -O3 -o fn_sample.cubin fn_sample.cu
#include <cuda_bf16.h>
#include <math_constants.h>

struct Top {
    float m;      // running maximum
    int i;        // first column holding it
    float s;      // sum of exp(x - m)
};

__device__ __forceinline__ Top merge(Top a, Top b) {
    if (b.m > a.m || (b.m == a.m && b.i < a.i)) { Top t = a; a = b; b = t; }
    // a holds the larger maximum (or the earlier column at a tie)
    float s = a.s + (b.m == -CUDART_INF_F ? 0.f : b.s * __expf(b.m - a.m));
    return {a.m, a.i, s};
}

extern "C" __global__ void __launch_bounds__(1024) fn_rows_top(const __nv_bfloat16* __restrict__ logits, int cols,
        int stride, const int* __restrict__ id_map, int offset, int* __restrict__ out) {
    const int row = blockIdx.x;
    const __nv_bfloat16* x = logits + (size_t)row * stride;
    Top t = {-CUDART_INF_F, 0x7fffffff, 0.f};
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[c]);
        if (v > t.m) { t.s = t.s * __expf(t.m - v) + 1.f; t.m = v; t.i = c; }
        else t.s += __expf(v - t.m);
    }
    for (int o = 16; o > 0; o >>= 1) {
        Top u = {__shfl_xor_sync(0xffffffff, t.m, o), __shfl_xor_sync(0xffffffff, t.i, o), __shfl_xor_sync(0xffffffff, t.s, o)};
        t = merge(t, u);
    }
    __shared__ Top warps[32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) warps[w] = t;
    __syncthreads();
    if (w == 0) {
        t = lane < (blockDim.x >> 5) ? warps[lane] : Top{-CUDART_INF_F, 0x7fffffff, 0.f};
        for (int o = 16; o > 0; o >>= 1) {
            Top u = {__shfl_xor_sync(0xffffffff, t.m, o), __shfl_xor_sync(0xffffffff, t.i, o), __shfl_xor_sync(0xffffffff, t.s, o)};
            t = merge(t, u);
        }
        if (lane == 0) {
            int* o4 = out + 4 * row;
            o4[0] = id_map ? id_map[t.i] : t.i + offset;
            o4[1] = __float_as_int(t.m);
            o4[2] = __float_as_int(t.m + __logf(t.s));
            o4[3] = 0;
        }
    }
}

// Sampling candidates: each row's 64 largest logits (by value, then the lower column at a tie), as global token ids
// and fp32 values, the row's log-sum-exp, and its log-sum-exp at the row's temperature (inv_t[row], x * inv_t); both
// ranks' sets are gathered and the keyed draw runs on the host over the merged 128 (whose first 64 are the vocabulary's
// top 64 exactly). Each thread keeps its columns' sorted top 64, then the block merges the threads' lists by repeated
// argmax over their heads (exact: no value of the row's top 64 can be missing from its thread's list).
// out[row * 132]: ids[64], values[64] (fp32 bits), lse, lse at the temperature (fp32 bits), 2 spare.
#define TOPK 64
#define TOPW (2 * TOPK + 4)
__device__ __forceinline__ bool above(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

__device__ __forceinline__ void lse_merge(float& m, float& s, float m2, float s2) {
    const float mm = fmaxf(m, m2);
    if (mm == -CUDART_INF_F) return;
    s = (m == -CUDART_INF_F ? 0.f : s * __expf(m - mm)) + (m2 == -CUDART_INF_F ? 0.f : s2 * __expf(m2 - mm));
    m = mm;
}

extern "C" __global__ void __launch_bounds__(256) fn_cand_topk(const __nv_bfloat16* __restrict__ logits, int cols,
        int stride, int offset, const float* __restrict__ inv_t, int* __restrict__ out) {
    const int row = blockIdx.x;
    const __nv_bfloat16* x = logits + (size_t)row * stride;
    const float it = inv_t ? inv_t[row] : 1.f;
    float lv[TOPK];
    int li[TOPK];
    int n = 0;
    float m = -CUDART_INF_F, s = 0.f, mt = -CUDART_INF_F, st = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = __bfloat162float(x[c]);
        if (v > m) { s = s * __expf(m - v) + 1.f; m = v; } else s += __expf(v - m);
        const float vt = v * it;
        if (vt > mt) { st = st * __expf(mt - vt) + 1.f; mt = vt; } else st += __expf(vt - mt);
        if (n < TOPK || above(v, c, lv[TOPK - 1], li[TOPK - 1])) {
            int j = n < TOPK ? n++ : TOPK - 1;
            while (j > 0 && above(v, c, lv[j - 1], li[j - 1])) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j; }
            lv[j] = v;
            li[j] = c;
        }
    }
    __shared__ float sm[8], ss[8], smt[8], sst[8];
    __shared__ float hv[256];
    __shared__ int hi[256], winner;
    for (int o = 16; o > 0; o >>= 1) {
        lse_merge(m, s, __shfl_xor_sync(0xffffffff, m, o), __shfl_xor_sync(0xffffffff, s, o));
        lse_merge(mt, st, __shfl_xor_sync(0xffffffff, mt, o), __shfl_xor_sync(0xffffffff, st, o));
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) { sm[w] = m; ss[w] = s; smt[w] = mt; sst[w] = st; }
    __syncthreads();
    int* o = out + (size_t)row * TOPW;
    if (threadIdx.x == 0) {
        float M = -CUDART_INF_F, S = 0.f, MT = -CUDART_INF_F, ST = 0.f;
        for (int i = 0; i < nw; ++i) { lse_merge(M, S, sm[i], ss[i]); lse_merge(MT, ST, smt[i], sst[i]); }
        o[2 * TOPK] = __float_as_int(M + __logf(S));
        o[2 * TOPK + 1] = __float_as_int(MT + __logf(ST));
    }
    int head = 0;
    for (int k = 0; k < TOPK; ++k) {
        hv[threadIdx.x] = head < n ? lv[head] : -CUDART_INF_F;
        hi[threadIdx.x] = head < n ? li[head] : 0x7fffffff;
        __syncthreads();
        if (threadIdx.x == 0) {
            int b = 0;
            for (int t = 1; t < (int)blockDim.x; ++t) if (above(hv[t], hi[t], hv[b], hi[b])) b = t;
            winner = b;
            o[k] = hi[b] == 0x7fffffff ? -1 : hi[b] + offset;
            o[TOPK + k] = __float_as_int(hv[b]);
        }
        __syncthreads();
        if (threadIdx.x == winner) ++head;
        __syncthreads();
    }
}

// Structured output: each listed row's tokens its grammar does not allow to -inf (rows[i]: the logits row, bits: the
// row's allowed bits over this rank's columns, token c: bit c % 32 of word c / 32), before the picks and draws.
extern "C" __global__ void __launch_bounds__(256) fn_grammar_rows(__nv_bfloat16* __restrict__ logits, int stride, int cols,
        const unsigned* __restrict__ bits, int words, const int* __restrict__ rows) {
    __nv_bfloat16* x = logits + (size_t)rows[blockIdx.x] * stride;
    const unsigned* w = bits + (size_t)blockIdx.x * words;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        if (!((w[c >> 5] >> (c & 31)) & 1u)) x[c] = __ushort_as_bfloat16((unsigned short)0xff80);
}

// fn_cand_topk's output, by radix selection over the bf16 bits instead of per-thread lists: a pass histograms the
// order-preserving keys' high byte (and takes both log-sum-exps), a second the low byte within the bin holding the
// 64th largest, which fixes the threshold key K; a third collects every column above K and, of the columns at K, the
// lowest ones (in column order), and the picks are ranked by value, then the lower column. The same 64 in the same
// order as fn_cand_topk (-0 counts as +0, as the float compare does); every pass reads the row once, coalesced.
__device__ __forceinline__ unsigned bf_key(unsigned short u) {
    if (u == 0x8000u) u = 0;   // -0: the same value as +0
    return (u & 0x8000u) ? (~u & 0xffffu) : (u | 0x8000u);
}

__device__ __forceinline__ void hist_add(unsigned* hist, int bin, bool valid) {
    const unsigned m = __match_any_sync(0xffffffffu, valid ? bin : -1);
    if (valid && (threadIdx.x & 31) == __ffs(m) - 1) atomicAdd(&hist[bin], __popc(m));
}

extern "C" __global__ void __launch_bounds__(1024) fn_cand_topk_sel(const __nv_bfloat16* __restrict__ logits, int cols,
        int stride, int offset, const float* __restrict__ inv_t, int* __restrict__ out) {
    const int row = blockIdx.x;
    const unsigned short* xu = reinterpret_cast<const unsigned short*>(logits + (size_t)row * stride);
    const float it = inv_t ? inv_t[row] : 1.f;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __shared__ unsigned hist[256];
    __shared__ float sm[32], ss[32], smt[32], sst[32];
    __shared__ int s_bin, s_gt, s_cnt, s_eq_seen;
    __shared__ int warp_eq[32];
    __shared__ float pv[TOPK];
    __shared__ int pi[TOPK];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
    if (threadIdx.x == 0) { s_cnt = 0; s_eq_seen = 0; }
    __syncthreads();
    float m = -CUDART_INF_F, s = 0.f, mt = -CUDART_INF_F, st = 0.f;
    for (int base = 0; base < cols; base += blockDim.x) {
        const int c = base + threadIdx.x;
        const bool valid = c < cols;
        unsigned key = 0;
        if (valid) {
            const unsigned short u = xu[c];
            key = bf_key(u);
            const float v = __uint_as_float((unsigned)u << 16);
            if (v > m) { s = s * __expf(m - v) + 1.f; m = v; } else s += __expf(v - m);
            const float vt = v * it;
            if (vt > mt) { st = st * __expf(mt - vt) + 1.f; mt = vt; } else st += __expf(vt - mt);
        }
        hist_add(hist, (int)(key >> 8), valid);
    }
    for (int o = 16; o > 0; o >>= 1) {
        lse_merge(m, s, __shfl_xor_sync(0xffffffff, m, o), __shfl_xor_sync(0xffffffff, s, o));
        lse_merge(mt, st, __shfl_xor_sync(0xffffffff, mt, o), __shfl_xor_sync(0xffffffff, st, o));
    }
    if (lane == 0) { sm[w] = m; ss[w] = s; smt[w] = mt; sst[w] = st; }
    __syncthreads();
    int* o = out + (size_t)row * TOPW;
    const int want = cols < TOPK ? cols : TOPK;
    if (threadIdx.x == 0) {
        float M = -CUDART_INF_F, S = 0.f, MT = -CUDART_INF_F, ST = 0.f;
        for (int i = 0; i < nw; ++i) { lse_merge(M, S, sm[i], ss[i]); lse_merge(MT, ST, smt[i], sst[i]); }
        o[2 * TOPK] = __float_as_int(M + __logf(S));
        o[2 * TOPK + 1] = __float_as_int(MT + __logf(ST));
        int cum = 0, b = 255;
        for (; b > 0; --b) { if (cum + (int)hist[b] >= want) break; cum += hist[b]; }
        s_bin = b;
        s_gt = cum;   // columns whose key's high byte is above the bin
    }
    __syncthreads();
    const unsigned hi = (unsigned)s_bin;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    for (int base = 0; base < cols; base += blockDim.x) {
        const int c = base + threadIdx.x;
        unsigned key = c < cols ? bf_key(xu[c]) : 0;
        const bool in = c < cols && (key >> 8) == hi;
        hist_add(hist, (int)(key & 0xffu), in);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int cum = s_gt, b = 255;
        for (; b > 0; --b) { if (cum + (int)hist[b] >= want) break; cum += hist[b]; }
        s_bin = (int)((hi << 8) | (unsigned)b);
        s_gt = cum;   // columns whose key is above K
    }
    __syncthreads();
    const unsigned K = (unsigned)s_bin;
    const int need_eq = want - s_gt;
    for (int base = 0; base < cols; base += blockDim.x) {
        const int c = base + threadIdx.x;
        unsigned key = c < cols ? bf_key(xu[c]) : 0;
        const bool gt = c < cols && key > K;
        const bool eq = c < cols && key == K;
        if (gt) { const int j = atomicAdd(&s_cnt, 1); pv[j] = __uint_as_float((unsigned)xu[c] << 16); pi[j] = c; }
        const unsigned bal = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) warp_eq[w] = __popc(bal);
        __syncthreads();
        if (eq) {
            int before = s_eq_seen + __popc(bal & ((1u << lane) - 1u));
            for (int i = 0; i < w; ++i) before += warp_eq[i];
            if (before < need_eq) { const int j = atomicAdd(&s_cnt, 1); pv[j] = __uint_as_float((unsigned)xu[c] << 16); pi[j] = c; }
        }
        __syncthreads();
        if (threadIdx.x == 0) { int t = 0; for (int i = 0; i < nw; ++i) t += warp_eq[i]; s_eq_seen += t; }
        __syncthreads();
        if (s_eq_seen >= need_eq && s_cnt >= want) break;   // uniform: shared values after a barrier
    }
    __syncthreads();
    if (threadIdx.x < TOPK) {
        const int k = threadIdx.x;
        if (k < want) {
            const float v = pv[k];
            const int c = pi[k];
            int r = 0;
            for (int j = 0; j < want; ++j) if (above(pv[j], pi[j], v, c)) ++r;
            o[r] = c + offset;
            o[TOPK + r] = __float_as_int(v);
        } else {
            o[k] = -1;
            o[TOPK + k] = __float_as_int(-CUDART_INF_F);
        }
    }
}
