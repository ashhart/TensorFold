// ERNIE 5.5's sparse-attention key lists: each row's indexer picks, then its sliding window, for the sparse MLA.

namespace ernie55i {

enum Fault { none, all_picks, short_window };

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
