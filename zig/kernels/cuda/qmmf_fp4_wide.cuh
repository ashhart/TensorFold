// Decode-sized FP4 columns share input staging; every column keeps the parent's K16 operations.
#define TF_QMMF_FP4_WIDE(F32, CLUSTER) \
template __global__ void qmmf_kernel<FP4, 16, 128, 1, 4, 4, F32, CLUSTER, false>( \
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, \
    float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
TF_QMMF_FP4_WIDE(false, false)
TF_QMMF_FP4_WIDE(false, true)
TF_QMMF_FP4_WIDE(true, false)
TF_QMMF_FP4_WIDE(true, true)
#undef TF_QMMF_FP4_WIDE
static_assert(Tile<FP4, 16, 128, 1, 4, 4>::SMEM == tf_qmmf_wide_layout::shared_bytes, "FP4 wide scratch");

// A real altered-order control: the same operands and slices, K16 groups accumulated in reverse.
extern "C" __global__ void __launch_bounds__(128) tf_qmmf_fp4_wide_order_control(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad,
        int ldx, int group) {
    constexpr int MODE = FP4, BM = 16, BN = 128, WM = 1, WN = 4, STAGES = 4;
    constexpr bool F32 = true, CLUSTER = false, FUSE = false;
    qmm_frag::grid_launch();
#define TF_QMMF_WFIRST
#define TF_QMMF_REVERSE_K16
#include "qmmf_body.cuh"
#undef TF_QMMF_REVERSE_K16
#undef TF_QMMF_WFIRST
}
