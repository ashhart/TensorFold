// Spark2.5's prompt projections on the tensor units: bf16 x [M, K] by the release's W [N, K]; after gemm_nax.metal.

#define SP_GEMM(NAME, AM, AN)                                                                                       \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                  \
      const device bfloat16_t* A [[buffer(0)]], const device bfloat16_t* B [[buffer(1)]],                          \
      const device int32_t* P [[buffer(2)]], device bfloat16_t* D [[buffer(3)]],                                   \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                        \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                 \
    tfp::gemm_nax<bfloat16_t, false, true, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);           \
  }
SP_GEMM(sp_gemm_00, false, false)
SP_GEMM(sp_gemm_01, false, true)
SP_GEMM(sp_gemm_10, true, false)
SP_GEMM(sp_gemm_11, true, true)
