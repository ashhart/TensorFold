// Every family's exact row top-k (common/topk_rows.cuh): row r's k best of its first counts[r] scores, in index order.
#include "common/topk_rows.cuh"

// out[r] gets min(k, n) indices ascending, -1 after them; picked[r] their count; n = counts[r] in [0, ld], else ld.
template <int T>
__device__ __forceinline__ void topk_rows(const float* scores, int64_t ld, const int* counts, int k, int* out,
                                          int* picked) {
  const int64_t r = blockIdx.x;
  const int64_t c = counts ? (int64_t)counts[r] : ld, n = c < 0 ? 0 : (c > ld ? ld : c);  // a count clamps to [0, ld]
  const int take = n < k ? (int)n : k;
  int* o = out + r * k;
  for (int i = take + threadIdx.x; i < k; i += T) o[i] = -1;
  if (threadIdx.x == 0) picked[r] = take;
  if (take > 0) tf_topk::select_row<T>(scores + r * ld, n, take, o);
}

// A row per 4-warp block: many rows (a prompt chunk's).
extern "C" __global__ void __launch_bounds__(128)
    tf_topk_rows(const float* scores, int64_t ld, const int* counts, int k, int* out, int* picked) {
  topk_rows<128>(scores, ld, counts, k, out, picked);
}

// A row per 1,024-thread block: a few rows over many scores (a decode step's); the same bytes.
extern "C" __global__ void __launch_bounds__(1024)
    tf_topk_rows_wide(const float* scores, int64_t ld, const int* counts, int k, int* out, int* picked) {
  topk_rows<1024>(scores, ld, counts, k, out, picked);
}
