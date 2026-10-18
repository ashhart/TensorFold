// Copy of src/tensorfold/families/deepseek_v41/cuda/pfdense_k16.cu (blob b474eefeeb26): sass_check.sh



// pfdense (TF_DSV41_PF_DENSE=fused) at K2 = 16 (8 bits): strips only (the indexer's wk), every config
#include "pfdense.cuh"

#define DSV41_PFD_K2 16
#define DSV41_PFD_LN false
DSV41_PFD_CFGS(DSV41_PFD_INST)
template __global__ void dsv41_pfd::pfd_dequant_kernel<16, false>(const uint32_t*, long long, long long, half*, int);


DSV41_PFD_WIDTH_STRIPS(16)
