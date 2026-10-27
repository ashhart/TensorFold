// LFM3's CUDA glue: Liquid's 7-tap short conv and its commit (liquid_conv.cuh); everything else is shared.

#include "liquid_conv.cuh"

extern "C" __global__ void __launch_bounds__(tf_liquid::NT) tf_liquid_conv_7(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ Wc,
                                                                            const __nv_bfloat16* __restrict__ STATE, const int* __restrict__ SLOT,
                                                                            const int* __restrict__ OFF, __nv_bfloat16* __restrict__ Y,
                                                                            __nv_bfloat16* __restrict__ SNAP, int SNAP0, int D, long long SS) {
    tf_liquid::liquid_conv<7, 0>(P, Wc, STATE, SLOT, OFF, Y, SNAP, SNAP0, D, SS);
}

extern "C" __global__ void __launch_bounds__(tf_liquid::NT) tf_liquid_commit_7(__nv_bfloat16* __restrict__ STATE, const __nv_bfloat16* __restrict__ SNAP,
                                                                              const int* __restrict__ SLOT, const int* __restrict__ ROW, int D,
                                                                              long long state_stride, long long snap_stride, long long SS) {
    tf_liquid::liquid_commit<7, 0>(STATE, SNAP, SLOT, ROW, D, state_stride, snap_stride, SS);
}
