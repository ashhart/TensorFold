// EXL3 canonical expert reduction: the value of the leaf subtree [start, start + 2^log).
extern "C" __global__ void fn_exl3_leaf_tree(const float* __restrict__ y, const float* __restrict__ wts, const int* __restrict__ pick,
                                             float* __restrict__ out, int d, int slots, int start, int log) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    const int end = start + (1 << log);
    int leaf[11];
    float val[11];
    int n = 0;
    for (int k = 0; k < slots; ++k) {
        const int p = pick[r * slots + k];
        const int l = p >= 512 ? 63 : (p >> 3);
        if (l < start || l >= end) continue;
        int j = 0;
        while (j < n && leaf[j] != l) ++j;
        if (j == n) {
            leaf[n] = l;
            val[n] = 0.0f;
            ++n;
        }
        val[j] = __fmaf_rn(y[(static_cast<long long>(r) * slots + k) * d + col], wts[r * slots + k], val[j]);
    }
    for (int i = 1; i < n; ++i)
        for (int j = i; j > 0 && leaf[j - 1] > leaf[j]; --j) {
            const int tl = leaf[j - 1];
            leaf[j - 1] = leaf[j];
            leaf[j] = tl;
            const float tv = val[j - 1];
            val[j - 1] = val[j];
            val[j] = tv;
        }
    for (int i = 0; i < n; ++i) leaf[i] -= start;
    for (int lev = 0; lev < log; ++lev) {
        int m = 0;
        for (int q = 0; q < n;) {
            const int parent = leaf[q] >> 1;
            if (q + 1 < n && parent == (leaf[q + 1] >> 1)) {
                val[m] = __fadd_rn(val[q], val[q + 1]);
                q += 2;
            } else {
                val[m] = __fadd_rn(val[q], 0.0f);
                q += 1;
            }
            leaf[m] = parent;
            ++m;
        }
        n = m;
    }
    out[static_cast<long long>(r) * d + col] = n == 0 ? 0.0f : val[0];
}

// The finish: `pieces` partial subtrees (meta: start leaf, log, source piece), in leaf order, merged left to right.
extern "C" __global__ void fn_exl3_leaf_finish(const float* __restrict__ g, float* __restrict__ out, const int* __restrict__ meta,
                                               int pieces, int d, long long stride) {
    const int r = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= d) return;
    int st[32], lg[32];
    float sv[32];
    int top = 0;
    for (int i = 0; i < pieces; ++i) {
        st[top] = meta[3 * i];
        lg[top] = meta[3 * i + 1];
        sv[top] = g[static_cast<long long>(meta[3 * i + 2]) * stride + static_cast<long long>(r) * d + col];
        ++top;
        while (top >= 2 && lg[top - 1] == lg[top - 2] && st[top - 2] % (2 << lg[top - 1]) == 0) {
            sv[top - 2] = __fadd_rn(sv[top - 2], sv[top - 1]);
            lg[top - 2] += 1;
            --top;
        }
    }
    out[static_cast<long long>(r) * d + col] = sv[0];
}
