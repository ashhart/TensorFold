// sample.cu's sliced draws for a window's sampled rows: each slice's top_k, then gpu_sample's tail on them.
#pragma once
namespace tf_sample {

// Sliced draws: each slice's top_k into CK / CI, then gpu_sample's tail (its two candidate paths share that prefix).
constexpr unsigned SW = 8192, ST = 256, ST_K = 32;

__device__ __forceinline__ bool sliced(const Rule& rl, unsigned V) {
    return rl.top_k != 0u && rl.top_k <= ST_K && (V + SW - 1) / SW * rl.top_k <= C;
}

__device__ __forceinline__ void slice_top(const __nv_bfloat16* L, unsigned V, const Rule* rule, unsigned* CK, unsigned* CI) {
    __shared__ unsigned sk[SW];
    __shared__ unsigned wk[ST / 32], wi[ST / 32];
    const unsigned row = blockIdx.y, lo = blockIdx.x * SW, t = threadIdx.x;
    const Rule rl = rule[row];
    if (!sliced(rl, V)) return;
    const __nv_bfloat16* x = L + static_cast<size_t>(row) * V;
    for (unsigned j = t; j < SW; j += ST) sk[j] = lo + j < V ? key(__bfloat162float(x[lo + j]) * rl.inv_t) : 0u;
    __syncthreads();
    unsigned* ok = CK + row * C + blockIdx.x * rl.top_k;
    unsigned* oi = CI + row * C + blockIdx.x * rl.top_k;
    for (unsigned r = 0; r < rl.top_k; r++) {
        unsigned bk = 0u, bi = 0xFFFFFFFFu;
        for (unsigned j = t; j < SW; j += ST)
            if (sk[j] > bk || (sk[j] == bk && j < bi)) {
                bk = sk[j];
                bi = j;
            }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned ok_ = __shfl_xor_sync(FULL, bk, off), oi_ = __shfl_xor_sync(FULL, bi, off);
            if (ok_ > bk || (ok_ == bk && oi_ < bi)) {
                bk = ok_;
                bi = oi_;
            }
        }
        if ((t & 31u) == 0) {
            wk[t >> 5] = bk;
            wi[t >> 5] = bi;
        }
        __syncthreads();
        if (t == 0) {
            for (unsigned w = 1; w < ST / 32; w++)
                if (wk[w] > bk || (wk[w] == bk && wi[w] < bi)) {
                    bk = wk[w];
                    bi = wi[w];
                }
            ok[r] = bk;
            oi[r] = lo + bi;
            sk[bi] = 0u;
        }
        __syncthreads();
    }
}

// Row r's draw: its slices' candidates sorted, the max their first, then gpu_sample's tail; else the one-block draw.
__device__ __forceinline__ void rows_finish(const __nv_bfloat16* L, unsigned V, const Rule* rule, const int* meta, const unsigned* CK,
                                            const unsigned* CI, unsigned* TOK) {
    __shared__ Shared s;
    const unsigned row = blockIdx.x, t = threadIdx.x;
    const Rule rl = rule[row];
    const unsigned position = static_cast<unsigned>(meta[row] + 1);
    if (!sliced(rl, V)) {
        const __nv_bfloat16* L_row = L + static_cast<size_t>(row) * V;
        if (rl.top_k == 0u || rl.top_k > C) sample_full<false>(L_row, V, rl, position, nullptr, TOK + row, nullptr, s);
        else gpu_sample<false>(L_row, V, rl, position, nullptr, TOK + row, nullptr, s);
        return;
    }
    const unsigned n = (V + SW - 1) / SW * rl.top_k;
    s.ck[t] = t < n ? CK[row * C + t] : 0u;
    s.ci[t] = t < n ? CI[row * C + t] : 0xFFFFFFFFu;
    unsigned sort_n = 32u;
    while (sort_n < n) sort_n <<= 1;
    __syncthreads();
    for (unsigned k = 2; k <= sort_n; k <<= 1) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            const unsigned p = t ^ j;
            if (p > t && p < sort_n) {
                const unsigned ka = s.ck[t], kb = s.ck[p], ia = s.ci[t], ib = s.ci[p];
                const bool a_first = ka > kb || (ka == kb && ia < ib);
                if (a_first != ((t & k) == 0)) {
                    s.ck[t] = kb;
                    s.ck[p] = ka;
                    s.ci[t] = ib;
                    s.ci[p] = ia;
                }
            }
            __syncthreads();
        }
    }
    pick<false>(rl, val(s.ck[0]), 0.0f, n, position, nullptr, TOK + row, nullptr, s);
}
}  // namespace tf_sample

// tf_draw_rows over SW-entry slices: grid (slices, rows) for each slice's top_k, then a block a row.
extern "C" __global__ void __launch_bounds__(256) tf_draw_slice_top(const __nv_bfloat16* L, unsigned V, const tf_sample::Rule* rule,
                                                                     unsigned* CK, unsigned* CI) {
    tf_sample::slice_top(L, V, rule, CK, CI);
}
extern "C" __global__ void __launch_bounds__(1024) tf_draw_rows_finish(const __nv_bfloat16* L, unsigned V, const tf_sample::Rule* rule,
                                                                        const int* meta, const unsigned* CK, const unsigned* CI,
                                                                        unsigned* TOK) {
    tf_sample::rows_finish(L, V, rule, meta, CK, CI, TOK);
}
