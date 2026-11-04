// Planted faults for tf_exl3pf_rotate and the TP reduction's tf_exl3pf_tp_combine, _tp_finish and _tp_place.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_decode.cuh"

namespace tf_tppf_ctl {

enum Fault { FLAT = 0, NAN_DROPPED = 1, WRITES_INPUT = 2, BF16_ADDS = 3, WRONG = 4 };

__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float add(float a, float b, bool bf16) {
    const float s = __fadd_rn(a, b);
    return bf16 ? rb(s) : s;
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

// tf_exl3pf_tp_combine, then the fault: FLAT one chain a block (3), NAN_DROPPED (1), WRITES_INPUT (2), BF16_ADDS (4).
template <int F>
__device__ __forceinline__ void combine(float* y, const float* wts, const int* pick, float* out, int slots, int D, int n,
                                        int sub, int world, int rank, int M) {
    __shared__ float w_sh[16];
    __shared__ int nl_sh;
    __shared__ int leaf_sh[16];
    __shared__ unsigned mask_sh[16];
    const int r = blockIdx.x;
    const int llo = 64 * rank / world, lhi = 64 * (rank + 1) / world;
    if (static_cast<int>(threadIdx.x) < slots) w_sh[threadIdx.x] = wts[r * slots + threadIdx.x];
    if (threadIdx.x == 0) {
        int nl = 0;
        for (int k = 0; k < slots; ++k) {
            const int p = pick[r * slots + k];
            if (p < 0) continue;
            const int leaf = p >= 512 ? 63 : p / 8;
            if (leaf < llo || leaf >= lhi) continue;
            int at = 0;
            while (at < nl && leaf_sh[at] < leaf) ++at;
            if (at < nl && leaf_sh[at] == leaf) {
                mask_sh[at] |= 1u << k;
                continue;
            }
            for (int i = nl; i > at; --i) {
                leaf_sh[i] = leaf_sh[i - 1];
                mask_sh[i] = mask_sh[i - 1];
            }
            leaf_sh[at] = leaf;
            mask_sh[at] = 1u << k;
            ++nl;
        }
        nl_sh = nl;
    }
    __syncthreads();
    const int nl = nl_sh;
    int bst[16], bsz[16];
    const int nb = blocks(llo, lhi, bst, bsz);
    const int b = r / sub, rr = r - b * sub;
    const int mb = min(sub, n - b * sub);
    const float* yr = y + static_cast<size_t>(r) * slots * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val[16];
        int st[16];
        for (int i = 0; i < nl; ++i) {
            float acc = 0.0f;
            for (unsigned m = mask_sh[i]; m; m &= m - 1) {
                const int k = __ffs(m) - 1;
                const float yv = yr[static_cast<size_t>(k) * D + d];
                if (F == NAN_DROPPED && __fmul_rn(yv, w_sh[k]) != __fmul_rn(yv, w_sh[k])) continue;
                acc = fmaf(yv, w_sh[k], acc);
            }
            val[i] = acc;
            st[i] = leaf_sh[i];
        }
        for (int j = 0; j < nb; ++j) {
            float v[16];
            int s[16];
            int cnt = 0;
            float chain = 0.0f;
            for (int i = 0; i < nl; ++i)
                if (st[i] >= bst[j] && st[i] < bst[j] + bsz[j]) {
                    v[cnt] = val[i];
                    s[cnt] = st[i];
                    ++cnt;
                }
            if (F == FLAT) {
                for (int k = 0; k < slots; ++k) {
                    const int p = pick[r * slots + k];
                    const int leaf = p >= 512 ? 63 : p / 8;
                    if (p >= 0 && leaf >= bst[j] && leaf < bst[j] + bsz[j]) chain = fmaf(yr[static_cast<size_t>(k) * D + d], w_sh[k], chain);
                }
            }
            for (int size = 1; size < bsz[j]; size *= 2) {
                int m = 0;
                for (int i = 0; i < cnt;) {
                    const int start = s[i] & ~(2 * size - 1);
                    if (s[i] % (2 * size) == 0 && i + 1 < cnt && s[i + 1] == s[i] + size) {
                        v[m] = add(v[i], v[i + 1], F == BF16_ADDS);
                        i += 2;
                    } else {
                        v[m] = v[i];
                        ++i;
                    }
                    s[m] = start;
                    ++m;
                }
                cnt = m;
            }
            const float result = F == FLAT ? chain : (cnt ? v[0] : 0.0f);
            out[(static_cast<size_t>(b) * sub * world * M + static_cast<size_t>(rank * M + j) * mb + rr) * D + d] = result;
        }
    }
    __syncthreads();
    if (F == WRITES_INPUT && r == 0 && threadIdx.x == 0) y[0] = __fadd_rn(y[0], 1.0f);
}

// tf_exl3pf_tp_finish, then the fault: FLAT adds every piece in one chain (gate 3), WRITES_INPUT (2), BF16_ADDS (4).
template <int F>
__device__ __forceinline__ void finish(float* gath, float* sum, int D, int m, int world, int M) {
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
        float v[96];
        int s[96], z[96];
        float chain = 0.0f;
        for (int i = 0; i < nb; ++i) {
            v[i] = gath[(static_cast<size_t>(bq_sh[i]) * m + rr) * D + d];
            s[i] = bs_sh[i];
            z[i] = bz_sh[i];
            chain = __fadd_rn(chain, v[i]);
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
        sum[static_cast<size_t>(rr) * D + d] = F == FLAT ? chain : v[0];
    }
    __syncthreads();
    if (F == WRITES_INPUT && rr == 0 && threadIdx.x == 0) gath[0] = __fadd_rn(gath[0], 1.0f);
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
    extern "C" __global__ void __launch_bounds__(256) NAME(float* y, const float* wts, const int* pick, float* out,  \
                                                           int slots, int D, int n, int sub, int world, int rank,    \
                                                           int M) {                                                  \
        tf_tppf_ctl::combine<F>(y, wts, pick, out, slots, D, n, sub, world, rank, M);                                \
    }
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_flat, tf_tppf_ctl::FLAT)
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_nan_dropped, tf_tppf_ctl::NAN_DROPPED)
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_writes_input, tf_tppf_ctl::WRITES_INPUT)
TF_CTL_TPPF_COMBINE(tf_ctl_tppf_combine_bf16_adds, tf_tppf_ctl::BF16_ADDS)
#undef TF_CTL_TPPF_COMBINE

#define TF_CTL_TPPF_FINISH(NAME, F)                                                                                  \
    extern "C" __global__ void __launch_bounds__(256) NAME(float* gath, float* sum, int D, int m, int world, int M) { \
        tf_tppf_ctl::finish<F>(gath, sum, D, m, world, M);                                                           \
    }
TF_CTL_TPPF_FINISH(tf_ctl_tppf_finish_flat, tf_tppf_ctl::FLAT)
TF_CTL_TPPF_FINISH(tf_ctl_tppf_finish_bf16_adds, tf_tppf_ctl::BF16_ADDS)
TF_CTL_TPPF_FINISH(tf_ctl_tppf_finish_writes_input, tf_tppf_ctl::WRITES_INPUT)
#undef TF_CTL_TPPF_FINISH

#define TF_CTL_TPPF_PLACE(NAME, F)                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) NAME(float* sum, float* y, float* wts, int slots, int D) {     \
        tf_tppf_ctl::place<F>(sum, y, wts, slots, D);                                                                \
    }
TF_CTL_TPPF_PLACE(tf_ctl_tppf_place_nan_dropped, tf_tppf_ctl::NAN_DROPPED)
TF_CTL_TPPF_PLACE(tf_ctl_tppf_place_writes_input, tf_tppf_ctl::WRITES_INPUT)
TF_CTL_TPPF_PLACE(tf_ctl_tppf_place_wrong, tf_tppf_ctl::WRONG)
#undef TF_CTL_TPPF_PLACE
