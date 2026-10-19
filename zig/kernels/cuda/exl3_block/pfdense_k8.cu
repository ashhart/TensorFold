// Copy of src/tensorfold/families/deepseek_v41/cuda/pfdense_k8.cu (blob f99f10d92688): sass_check.sh



// pfdense (TF_DSV41_PF_DENSE=fused) at K2 = 8 (4 bits): strips and lanes, every config of pfdense_args.h
#include "pfdense.cuh"

#define DSV41_PFD_K2 8
#define DSV41_PFD_LN false
DSV41_PFD_CFGS(DSV41_PFD_INST)
template __global__ void dsv41_pfd::pfd_dequant_kernel<8, false>(const uint32_t*, long long, long long, half*, int);
#undef DSV41_PFD_LN
#define DSV41_PFD_LN true
DSV41_PFD_CFGS(DSV41_PFD_INST)
template __global__ void dsv41_pfd::pfd_dequant_kernel<8, true>(const uint32_t*, long long, long long, half*, int);

DSV41_PFD_WIDTH(8)
