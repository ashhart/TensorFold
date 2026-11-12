// Planted faults for tf_exl3pf_rotate and the TP reduction's tf_exl3pf_tp_combine, _tp_finish and _tp_place.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_decode.cuh"

namespace tf_tppf_ctl {

enum Fault { FLAT = 0, NAN_DROPPED = 1, WRITES_INPUT = 2, BF16_ADDS = 3, WRONG = 4, F32_PARTIALS = 5 };

__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// One tree add in double; BF16_ADDS rounds the sum to bf16 (through float) the way a half-precision merge would.
__device__ __forceinline__ double add(double a, double b, bool bf16) {
    const double s = __dadd_rn(a, b);
    return bf16 ? static_cast<double>(rb(__double2float_rn(s))) : s;
}

// fwht128 of exl3_decode.cuh with every add and subtract rounded to bf16 (gate 4).
__device__ __forceinline__ void fwht128_bf16(float (&v)[4], int lane) {
    const float a = rb(v[0] + v[1]), b = rb(v[0] - v[1]), c = rb(v[2] + v[3]), d = rb(v[2] - v[3]);
    v[0] = rb(a + c);
    v[1] = rb(b + d);
    v[2] = rb(a - c);
    v[3] = rb(b - d);
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = rb((lane & m) ? o - v[j] : v[j] + o);
        }
    }
}

// tf_exl3pf_rotate, then the fault: NAN_DROPPED zeroes a non-finite x (gate 1), WRITES_INPUT (2), WRONG, BF16_ADDS (4).
template <int F>
__device__ __forceinline__ void rotate(__nv_bfloat16* x, int x_stride, const int* pick, const int* members, int slots,
                                       const half* suh_g, const half* suh_u, half* xg, half* xu, int D) {
    const int c = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int p = members[c];
    const int e = pick[p];
    const int row = p / slots;
    const int lane = threadIdx.x;
    const half* suh = ((mat && F != WRONG) ? suh_u : suh_g) + static_cast<size_t>(e) * D + blk * 128 + 4 * lane;
    __nv_bfloat16* xr = x + static_cast<size_t>(row) * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float xv = __bfloat162float(xr[j]);
        if (F == NAN_DROPPED && !isfinite(xv)) xv = 0.0f;
        v[j] = xv * __half2float(suh[j]);
    }
    if (F == BF16_ADDS) tf_tppf_ctl::fwht128_bf16(v, lane);
    else tf_exl3::fwht128(v, lane);
    half* o = (mat ? xu : xg) + static_cast<size_t>(c) * D + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * tf_exl3::HAD_SCALE);
    if (F == WRITES_INPUT && c == 0 && blk == 0 && mat == 0 && lane == 0) xr[0] = __float2bfloat16(__bfloat162float(xr[0]) + 1.0f);
}

// The maximal complete subtrees of the leaf run [lo, hi), left to right.
__device__ __forceinline__ int blocks(int lo, int hi, int* st, int* sz) {
    int n = 0;
    for (int at = lo; at < hi;) {
        int size = 1;
        while (at % (size * 2) == 0 && at + size * 2 <= hi) size *= 2;
        st[n] = at;
        sz[n] = size;
        ++n;
        at += size;
    }
    return n;
}

// A block's value as fn_exl3_leaf_value_f64 computes it (an fma chain a leaf in slot order from +0, the tree above it binary
// adds, an empty sibling +0), then the fault: FLAT one chain a block, NAN_DROPPED skips NaN products, BF16_ADDS rounds
// merges, F32_PARTIALS stores the block through float.
template <int F>
__device__ __forceinline__ double block_value(const float* y, const float* wts, const int* pick, int r, int d, int D, int slots,
                                              int start, int log) {
    const int end = start + (1 << log);
    int leaf[11];
    double val[11];
    int n = 0;
    double chain = 0.0;
    for (int k = 0; k < slots; ++k) {
        const int p = pick[r * slots + k];
        const int l = p >= 512 ? 63 : (p >> 3);
        if (l < start || l >= end) continue;
        const double yv = y[(static_cast<long long>(r) * slots + k) * D + d];
        const double wv = wts[r * slots + k];
        if (F == NAN_DROPPED) {
            const double product = __dmul_rn(yv, wv);
            if (product != product) continue;
        }
        chain = __fma_rn(yv, wv, chain);
        int j = 0;
        while (j < n && leaf[j] != l) ++j;
        if (j == n) {
            leaf[n] = l;
            val[n] = 0.0;
            ++n;
        }
        val[j] = __fma_rn(yv, wv, val[j]);
    }
    if (F == FLAT) return chain;
    for (int i = 1; i < n; ++i)
        for (int j = i; j > 0 && leaf[j - 1] > leaf[j]; --j) {
            const int tl = leaf[j - 1];
            leaf[j - 1] = leaf[j];
            leaf[j] = tl;
            const double tv = val[j - 1];
            val[j - 1] = val[j];
            val[j] = tv;
        }
    for (int i = 0; i < n; ++i) leaf[i] -= start;
    for (int lev = 0; lev < log; ++lev) {
        int m = 0;
        for (int q = 0; q < n;) {
            const int parent = leaf[q] >> 1;
            if (q + 1 < n && parent == (leaf[q + 1] >> 1)) {
                val[m] = add(val[q], val[q + 1], F == BF16_ADDS);
                q += 2;
            } else {
                val[m] = __dadd_rn(val[q], 0.0);
                q += 1;
            }
            leaf[m] = parent;
            ++m;
        }
        n = m;
    }
    const double value = n == 0 ? 0.0 : val[0];
    return F == F32_PARTIALS ? static_cast<double>(__double2float_rn(value)) : value;
}

// tf_exl3pf_tp_combine, then the fault: FLAT one chain a block (3), NAN_DROPPED (1), WRITES_INPUT (2), BF16_ADDS (3: the
// partials are double words, which gate 4 does not read), F32_PARTIALS (3).
template <int F>
__device__ __forceinline__ void combine(float* y, const float* wts, const int* pick, double* out, int slots, int D, int n,
                                        int sub, int world, int rank, int M) {
    const int r = blockIdx.x;
    const int llo = 64 * rank / world, lhi = 64 * (rank + 1) / world;
    int bst[16], bsz[16];
    const int nb = blocks(llo, lhi, bst, bsz);
    const int b = r / sub, rr = r - b * sub;
    const int mb = min(sub, n - b * sub);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        for (int j = 0; j < nb; ++j) {
            int log = 0;
            while ((2 << log) <= bsz[j]) ++log;
            out[(static_cast<size_t>(b) * sub * world * M + static_cast<size_t>(rank * M + j) * mb + rr) * D + d] =
                block_value<F>(y, wts, pick, r, d, D, slots, bst[j], log);
        }
    }
    __syncthreads();
    if (F == WRITES_INPUT && r == 0 && threadIdx.x == 0) y[0] = __fadd_rn(y[0], 1.0f);
}

// tf_exl3pf_tp_finish, then the fault: FLAT adds every piece in one chain (gate 3), WRITES_INPUT (2), BF16_ADDS (4),
// F32_PARTIALS reads every gathered piece through float, as an f32-typed gather would carry it (3).
template <int F>
__device__ __forceinline__ void finish(double* gath, float* sum, int D, int m, int world, int M) {
    __shared__ int bs_sh[96], bz_sh[96], bq_sh[96];
    __shared__ int nb_sh;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int q = 0; q < world; ++q) {
            int st[16], sz[16];
            const int c = blocks(64 * q / world, 64 * (q + 1) / world, st, sz);
            for (int j = 0; j < c; ++j) {
                bs_sh[n] = st[j];
                bz_sh[n] = sz[j];
                bq_sh[n] = q * M + j;
                ++n;
            }
        }
        nb_sh = n;
    }
    __syncthreads();
    const int rr = blockIdx.x;
    const int nb = nb_sh;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        double v[96];
        int s[96], z[96];
        double chain = 0.0;
        for (int i = 0; i < nb; ++i) {
            const double piece = gath[(static_cast<size_t>(bq_sh[i]) * m + rr) * D + d];
            v[i] = F == F32_PARTIALS ? static_cast<double>(__double2float_rn(piece)) : piece;
            s[i] = bs_sh[i];
            z[i] = bz_sh[i];
            chain = __dadd_rn(chain, v[i]);
        }
        int cnt = nb;
        while (cnt > 1) {
            int k = 0;
            for (int i = 0; i < cnt;) {
                if (i + 1 < cnt && z[i] == z[i + 1] && s[i] % (z[i] * 2) == 0) {
                    v[k] = add(v[i], v[i + 1], F == BF16_ADDS);
                    s[k] = s[i];
                    z[k] = z[i] * 2;
                    i += 2;
                } else {
                    v[k] = v[i];
                    s[k] = s[i];
                    z[k] = z[i];
                    ++i;
                }
                ++k;
            }
            cnt = k;
        }
        sum[static_cast<size_t>(rr) * D + d] = __double2float_rn(F == FLAT ? chain : v[0]);
    }
    __syncthreads();
    if (F == WRITES_INPUT && rr == 0 && threadIdx.x == 0) gath[0] = __dadd_rn(gath[0], 1.0);
}

// tf_exl3pf_tp_place, then the fault: NAN_DROPPED copies a NaN as 0 (gate 1), WRITES_INPUT (2), WRONG: weights 1 (3).
template <int F>
__device__ __forceinline__ void place(float* sum, float* y, float* wts, int slots, int D) {
    const int r = blockIdx.x;
    float* out = y + static_cast<size_t>(r) * slots * D;
    float* in = sum + static_cast<size_t>(r) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = in[d];
        out[d] = (F == NAN_DROPPED && v != v) ? 0.0f : v;
    }
    if (static_cast<int>(threadIdx.x) < slots) wts[r * slots + threadIdx.x] = (threadIdx.x == 0 || F == WRONG) ? 1.0f : 0.0f;
    __syncthreads();
    if (F == WRITES_INPUT && r == 0 && threadIdx.x == 0) sum[0] = __fadd_rn(sum[0], 1.0f);
}

}  // namespace tf_tppf_ctl

#define TF_CTL_TPPF_ROTATE(NAME, F)                                                                                  \
    extern "C" __global__ void NAME(__nv_bfloat16* x, int x_stride, const int* pick, const int* members, int slots,  \
                                    const half* suh_g, const half* suh_u, half* xg, half* xu, int D) {               \
        tf_tppf_ctl::rotate<F>(x, x_stride, pick, members, slots, suh_g, suh_u, xg, xu, D);                          \
    }
TF_CTL_TPPF_ROTATE(tf_ctl_tppf_rotate_nan_dropped, tf_tppf_ctl::NAN_DROPPED)
TF_CTL_TPPF_ROTATE(tf_ctl_tppf_rotate_writes_input, tf_tppf_ctl::WRITES_INPUT)
TF_CTL_TPPF_ROTATE(tf_ctl_tppf_rotate_wrong, tf_tppf_ctl::WRONG)
TF_CTL_TPPF_ROTATE(tf_ctl_tppf_rotate_bf16_adds, tf_tppf_ctl::BF16_ADDS)
#undef TF_CTL_TPPF_ROTATE

#define TF_CTL_TPPF_COMBINE(NAME, F)                                                                                 \
    extern "C" __global__ void __launch_bounds__(256) NAME(float* y, const float* wts, const int* pick, double* out, \
                                                           int slots, int D, int n, int sub, int world, int rank,    \
                                                           int M) {                                                  \
        tf_tppf_ctl::combine<F>(y, wts, pick, out, slots, D, n, sub, world, rank, M);                                \
    }
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_flat, tf_tppf_ctl::FLAT)
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_nan_dropped, tf_tppf_ctl::NAN_DROPPED)
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_writes_input, tf_tppf_ctl::WRITES_INPUT)
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_bf16_adds, tf_tppf_ctl::BF16_ADDS)
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_f32_partials, tf_tppf_ctl::F32_PARTIALS)
#undef TF_CTL_TPPF_COMBINE

#define TF_CTL_TPPF_FINISH(NAME, F)                                                                                  \
    extern "C" __global__ void __launch_bounds__(256) NAME(double* gath, float* sum, int D, int m, int world, int M) { \
        tf_tppf_ctl::finish<F>(gath, sum, D, m, world, M);                                                           \
    }
TF_CTL_TPPF_FINISH(tf_ctl_tppf_finish_flat, tf_tppf_ctl::FLAT)
TF_CTL_TPPF_FINISH(tf_ctl_tppf_finish_bf16_adds, tf_tppf_ctl::BF16_ADDS)
TF_CTL_TPPF_FINISH(tf_ctl_tppf_finish_writes_input, tf_tppf_ctl::WRITES_INPUT)
TF_CTL_TPPF_FINISH(tf_ctl_tppf_finish_f32_partials, tf_tppf_ctl::F32_PARTIALS)
#undef TF_CTL_TPPF_FINISH

#define TF_CTL_TPPF_PLACE(NAME, F)                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) NAME(float* sum, float* y, float* wts, int slots, int D) {     \
        tf_tppf_ctl::place<F>(sum, y, wts, slots, D);                                                                \
    }
TF_CTL_TPPF_PLACE(tf_ctl_tppf_place_nan_dropped, tf_tppf_ctl::NAN_DROPPED)
TF_CTL_TPPF_PLACE(tf_ctl_tppf_place_writes_input, tf_tppf_ctl::WRITES_INPUT)
TF_CTL_TPPF_PLACE(tf_ctl_tppf_place_wrong, tf_tppf_ctl::WRONG)
#undef TF_CTL_TPPF_PLACE
