#ifdef TF_NAX_STRICT_FP32
#define QWV_EXACT_GEMM(NAME, AM, AN) \
  [[max_total_threads_per_threadgroup(256)]] kernel void NAME( \
      const device float* A, const device float* B, const device int* P, device float* D, \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
      uint3 tg [[threadgroup_position_in_grid]]) { \
    tfp::gemm_nax<float, false, true, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane); \
  }
QWV_EXACT_GEMM(qwv_gemm_f32_exact_nt_00, false, false)
QWV_EXACT_GEMM(qwv_gemm_f32_exact_nt_01, false, true)
QWV_EXACT_GEMM(qwv_gemm_f32_exact_nt_10, true, false)
QWV_EXACT_GEMM(qwv_gemm_f32_exact_nt_11, true, true)
#undef QWV_EXACT_GEMM
#endif
