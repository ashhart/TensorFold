// Decode GEMM variants for Gemma 4 E4B: the Qwen GEMM file (group 64) with deeper weight prefetch. The prefetch distance changes
// no arithmetic: the bits are those of w4k1_d16 and the nGEN prompt kernels.
#include "gemm.cl"

W4K_KERNEL_SL(g4_d16_pf3, 16, 16, 4, 16, READ16, 3, 1)
W4K_KERNEL_SL(g4_d16_pf4, 16, 16, 4, 16, READ16, 4, 1)
W4K_KERNEL_SL(g4_d16_pf6, 16, 16, 4, 16, READ16, 6, 1)
W4K_KERNEL_SL(g4_d16_pf8, 16, 16, 4, 16, READ16, 8, 1)
W4K_KERNEL_SL(g4_d32_pf4, 16, 32, 4, 16, READ16, 4, 1)
W4K_KERNEL_SL(g4_d32_pf8, 16, 32, 4, 16, READ16, 8, 1)
