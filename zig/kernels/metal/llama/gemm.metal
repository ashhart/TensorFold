// Llama's prompt projections on the tensor units: bf16 x [M, K] by the release's W [N, K]; after gemm_nax.metal.

TF_GEMM_NAX_KERNEL(ll_gemm_00, bfloat16_t, false, true, false, false)
TF_GEMM_NAX_KERNEL(ll_gemm_01, bfloat16_t, false, true, false, true)
TF_GEMM_NAX_KERNEL(ll_gemm_10, bfloat16_t, false, true, true, false)
TF_GEMM_NAX_KERNEL(ll_gemm_11, bfloat16_t, false, true, true, true)
