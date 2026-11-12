// EXL3 canonical expert reduction. The value of a leaf subtree is computed in double (flashnext_exl3_leaf_value.cuh);
// float appears once, in the root's output.
#include "flashnext_exl3_leaf_value.cuh"

// The whole-tree root (start 0, log 6) or any subtree: the double value rounded once to float, to nearest even.
extern "C" __global__ void fn_exl3_leaf_tree(const float* __restrict__ y, const float* __restrict__ wts, const int* __restrict__ pick,
                                             float* __restrict__ out, int d, int slots, int start, int log) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    out[static_cast<long long>(r) * d + col] = __double2float_rn(fn_exl3_leaf_value_f64(y, wts, pick, r, col, d, slots, start, log));
}

// A rank's partial subtree, unrounded: the double the finish (here or in another rank) merges.
extern "C" __global__ void fn_exl3_leaf_partial(const float* __restrict__ y, const float* __restrict__ wts, const int* __restrict__ pick,
                                                double* __restrict__ out, int d, int slots, int start, int log) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    out[static_cast<long long>(r) * d + col] = fn_exl3_leaf_value_f64(y, wts, pick, r, col, d, slots, start, log);
}

// The finish: `pieces` double partial subtrees (meta: start leaf, log, source piece; stride in double elements), in leaf
// order, merged left to right in double and rounded once to float.
extern "C" __global__ void fn_exl3_leaf_finish(const double* __restrict__ g, float* __restrict__ out, const int* __restrict__ meta,
                                               int pieces, int d, long long stride) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    int st[32], lg[32];
    double sv[32];
    int top = 0;
    for (int i = 0; i < pieces; ++i) {
        st[top] = meta[3 * i];
        lg[top] = meta[3 * i + 1];
        sv[top] = g[static_cast<long long>(meta[3 * i + 2]) * stride + static_cast<long long>(r) * d + col];
        ++top;
        while (top >= 2 && lg[top - 1] == lg[top - 2] && st[top - 2] % (2 << lg[top - 1]) == 0) {
            sv[top - 2] = __dadd_rn(sv[top - 2], sv[top - 1]);
            lg[top - 2] += 1;
            --top;
        }
    }
    out[static_cast<long long>(r) * d + col] = __double2float_rn(sv[0]);
}
