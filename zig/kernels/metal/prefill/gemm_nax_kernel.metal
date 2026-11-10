// One tfp::gemm_nax kernel (bm64 bn128 bk256 wm2 wn4) by element type, layout and M/N edges; after gemm_nax.metal.
#define TF_GEMM_NAX_KERNEL(NAME, T, TA, TB, AM, AN)                                                                \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                \
      const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], const device int32_t* P [[buffer(2)]],     \
      device T* D [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],                                      \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                        \
    tfp::gemm_nax<T, TA, TB, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);                         \
  }
