// Planted controls for LFM3's CUDA kernels: the 7-tap conv and commit, and the prompt rows at head dim 128.

// Included by contract_controls_set.cu after contract_controls.cu: it reuses liquid_conv.cuh and the PATTN_VLANE fault.

// The 7-tap conv and commit, the 3-tap pair's planted faults: conv bx unrounded, commits to stream 0's slot.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_liquid_conv_7(const __nv_bfloat16* P, const __nv_bfloat16* Wc, const __nv_bfloat16* STATE,
                                                                        const int* SLOT, const int* OFF, __nv_bfloat16* Y, __nv_bfloat16* SNAP,
                                                                        int SNAP0, int D, long long SS) {
    tf_liquid::liquid_conv<7, 1>(P, Wc, STATE, SLOT, OFF, Y, SNAP, SNAP0, D, SS);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_liquid_commit_7(__nv_bfloat16* STATE, const __nv_bfloat16* SNAP, const int* SLOT,
                                                                          const int* ROW, int D, long long state_stride, long long snap_stride,
                                                                          long long SS) {
    tf_liquid::liquid_commit<7, 1>(STATE, SNAP, SLOT, ROW, D, state_stride, snap_stride, SS);
}

namespace tf_prefill_attention {
// Paged prompt rows at head dim 128, 16 query heads over 4: values from the next 8 dims, as tf_ctl_pattn_64.
extern "C" __global__ void __launch_bounds__(256, 1) tf_ctl_pattn_128(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                                                      const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0,
                                                                      int W, int H, int HK, int G, float scale, const int* __restrict__ PT, long long PS) {
    constexpr int D = 128, WARPS = 8, HPC = 4, NS = 8;
    constexpr bool causal = true;
    constexpr int skip = 0;
    constexpr bool P256 = false, HPAGED = false, F8 = false, HM = false, IDS = false;
    constexpr int CAP = 0, PTOK = 1, PN = 0;
    const float *ks = nullptr, *vs = nullptr;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}
}  // namespace tf_prefill_attention
