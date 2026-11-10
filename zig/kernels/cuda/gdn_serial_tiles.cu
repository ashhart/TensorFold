// Geometry-only instantiations of TensorFold's unchanged serial FP32 delta recurrence.
#include "gdn_prefill.cu"

#define TF_GDN_TILE(ROWS, STEPS) \
template __global__ void tf_gdn_prefill::chain_kernel<__nv_bfloat16, __nv_bfloat16, ROWS, STEPS>( \
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, \
    const float* __restrict__, const __nv_bfloat16* __restrict__, const float*, float*, \
    __nv_bfloat16* __restrict__, int, int, int);
TF_GDN_TILE(16, 16)
TF_GDN_TILE(16, 32)
TF_GDN_TILE(32, 16)
TF_GDN_TILE(32, 32)
TF_GDN_TILE(64, 16)
#undef TF_GDN_TILE
