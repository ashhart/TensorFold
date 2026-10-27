// LFM2.5-24B-A2B's CUDA glue: Liquid's 3-tap short conv and commit (liquid_conv.cuh), the router (lfm2_moe.cuh).

#include "liquid_conv.cuh"
#include "lfm2_moe.cuh"

using namespace tf_lfm2m;

extern "C" __global__ void __launch_bounds__(tf_liquid::NT) tf_liquid_conv_3(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ Wc,
                                                                            const __nv_bfloat16* __restrict__ STATE, const int* __restrict__ SLOT,
                                                                            const int* __restrict__ OFF, __nv_bfloat16* __restrict__ Y,
                                                                            __nv_bfloat16* __restrict__ SNAP, int SNAP0, int D, long long SS) {
    tf_liquid::liquid_conv<3, 0>(P, Wc, STATE, SLOT, OFF, Y, SNAP, SNAP0, D, SS);
}

extern "C" __global__ void __launch_bounds__(tf_liquid::NT) tf_liquid_commit_3(__nv_bfloat16* __restrict__ STATE, const __nv_bfloat16* __restrict__ SNAP,
                                                                              const int* __restrict__ SLOT, const int* __restrict__ ROW, int D,
                                                                              long long state_stride, long long snap_stride, long long SS) {
    tf_liquid::liquid_commit<3, 0>(STATE, SNAP, SLOT, ROW, D, state_stride, snap_stride, SS);
}

extern "C" __global__ void __launch_bounds__(NT) tf_lfm2m_route(const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
                                                                 const float* __restrict__ BIAS, int* __restrict__ PICK,
                                                                 float* __restrict__ WTS, float* __restrict__ LOGITS, int rows,
                                                                 int D, float scale) {
    lfm2m_route<0>(X, W, BIAS, PICK, WTS, LOGITS, rows, D, scale);
}
