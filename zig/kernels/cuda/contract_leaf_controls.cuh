// Planted faults for fn_exl3_leaf_tree, fn_exl3_leaf_partial and fn_exl3_leaf_finish: each control is the kernel with one change.
#pragma once
#include <cuda_bf16.h>
#include "flashnext_exl3_leaf_value.cuh"

namespace tf_leaf_ctl {

enum Fault { FLAT = 0, NAN_DROPPED = 1, WRITES_INPUT = 2, BF16_ADDS = 3, F32_PARTIALS = 4 };

__device__ __forceinline__ double add(double a, double b, bool bf16) {
    const double s = __dadd_rn(a, b);
    return bf16 ? static_cast<double>(__bfloat162float(__float2bfloat16_rn(__double2float_rn(s)))) : s;
}

// fn_exl3_leaf_tree, then the fault: FLAT one chain (gate 3), NAN_DROPPED (1), WRITES_INPUT (2), BF16_ADDS (4).
template <int F>
__device__ __forceinline__ void leaf_tree(float* y, const float* wts, const int* pick, float* out, int d, int slots,
                                          int start, int log) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    const int end = start + (1 << log);
    int leaf[11];
    double val[11];
    int n = 0;
    double chain = 0.0;
    for (int k = 0; k < slots; ++k) {
        const int p = pick[r * slots + k];
        const int l = p >= 512 ? 63 : (p >> 3);
        if (l < start || l >= end) continue;
        const double yv = y[(static_cast<long long>(r) * slots + k) * d + col];
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
    if (F == FLAT) {
        out[static_cast<long long>(r) * d + col] = __double2float_rn(chain);
        return;
    }
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
    out[static_cast<long long>(r) * d + col] = __double2float_rn(n == 0 ? 0.0 : val[0]);
    if (F == WRITES_INPUT && r == 0 && col == 0) y[0] = __fadd_rn(y[0], 1.0f);
}

// fn_exl3_leaf_finish, then the fault: FLAT adds every piece in one chain (gate 3), BF16_ADDS rounds merges (gate 4),
// F32_PARTIALS reads each partial through float (gate 3).
template <int F>
__device__ __forceinline__ void leaf_finish(const double* g, float* out, const int* meta, int pieces, int d, long long stride) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    int st[32], lg[32];
    double sv[32];
    int top = 0;
    double chain = 0.0;
    for (int i = 0; i < pieces; ++i) {
        st[top] = meta[3 * i];
        lg[top] = meta[3 * i + 1];
        sv[top] = g[static_cast<long long>(meta[3 * i + 2]) * stride + static_cast<long long>(r) * d + col];
        if (F == F32_PARTIALS) sv[top] = static_cast<double>(__double2float_rn(sv[top]));
        chain = __dadd_rn(chain, sv[top]);
        ++top;
        while (top >= 2 && lg[top - 1] == lg[top - 2] && st[top - 2] % (2 << lg[top - 1]) == 0) {
            sv[top - 2] = add(sv[top - 2], sv[top - 1], F == BF16_ADDS);
            lg[top - 2] += 1;
            --top;
        }
    }
    out[static_cast<long long>(r) * d + col] = __double2float_rn(F == FLAT ? chain : sv[0]);
}

}  // namespace tf_leaf_ctl

#define TF_CTL_LEAF_TREE(NAME, F)                                                                                   \
    extern "C" __global__ void NAME(float* y, const float* wts, const int* pick, float* out, int d, int slots,      \
                                    int start, int log) {                                                           \
        tf_leaf_ctl::leaf_tree<F>(y, wts, pick, out, d, slots, start, log);                                         \
    }
TF_CTL_LEAF_TREE(tf_ctl_leaf_tree_flat, tf_leaf_ctl::FLAT)
TF_CTL_LEAF_TREE(tf_ctl_leaf_tree_nan_dropped, tf_leaf_ctl::NAN_DROPPED)
TF_CTL_LEAF_TREE(tf_ctl_leaf_tree_writes_input, tf_leaf_ctl::WRITES_INPUT)
TF_CTL_LEAF_TREE(tf_ctl_leaf_tree_bf16_adds, tf_leaf_ctl::BF16_ADDS)
#undef TF_CTL_LEAF_TREE

#define TF_CTL_LEAF_FINISH(NAME, F)                                                                                 \
    extern "C" __global__ void NAME(const double* g, float* out, const int* meta, int pieces, int d, long long stride) { \
        tf_leaf_ctl::leaf_finish<F>(g, out, meta, pieces, d, stride);                                               \
    }
TF_CTL_LEAF_FINISH(tf_ctl_leaf_finish_flat, tf_leaf_ctl::FLAT)
TF_CTL_LEAF_FINISH(tf_ctl_leaf_finish_bf16_adds, tf_leaf_ctl::BF16_ADDS)
TF_CTL_LEAF_FINISH(tf_ctl_leaf_finish_f32, tf_leaf_ctl::F32_PARTIALS)
#undef TF_CTL_LEAF_FINISH

// fn_exl3_leaf_partial, then the fault: the partial stored through float (the rounding that moves the old f32 partials' error
// into the finish), gate 3.
extern "C" __global__ void tf_ctl_leaf_partial_f32(const float* y, const float* wts, const int* pick, double* out, int d, int slots,
                                                   int start, int log) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    out[static_cast<long long>(r) * d + col] =
        static_cast<double>(__double2float_rn(fn_exl3_leaf_value_f64(y, wts, pick, r, col, d, slots, start, log)));
}
