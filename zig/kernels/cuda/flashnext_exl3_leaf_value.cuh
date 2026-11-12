// Canonical F64 subtree with slot-order FMAs and empty +0 siblings; only the caller narrows the root.
#pragma once

__device__ __forceinline__ double fn_exl3_leaf_value_f64(const float* __restrict__ y, const float* __restrict__ wts,
                                                         const int* __restrict__ pick, int row, int col, int d, int slots,
                                                         int start, int log) {
    const int end = start + (1 << log);
    int leaf[11];
    double val[11];
    int n = 0;
    for (int k = 0; k < slots; ++k) {
        const int p = pick[row * slots + k];
        const int l = p >= 512 ? 63 : (p >> 3);
        if (l < start || l >= end) continue;
        int j = 0;
        while (j < n && leaf[j] != l) ++j;
        if (j == n) {
            leaf[n] = l;
            val[n] = 0.0;
            ++n;
        }
        val[j] = __fma_rn(static_cast<double>(y[(static_cast<long long>(row) * slots + k) * d + col]),
                          static_cast<double>(wts[row * slots + k]), val[j]);
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
                val[m] = __dadd_rn(val[q], val[q + 1]);
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
    return n == 0 ? 0.0 : val[0];
}
