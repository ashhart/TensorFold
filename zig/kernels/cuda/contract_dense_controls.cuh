// Planted controls for the dense decoders' prompt rows: pattn at two heads a block, values from the next lane's dims.

// Included by contract_controls_set.cu after contract_controls.cu, whose PATTN_VLANE fault and headers it reuses.

namespace tf_prefill_attention {
extern "C" __global__ void __launch_bounds__(256, 1) tf_ctl_pattn_128_hpc2(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                                                           const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0,
                                                                           int W, int H, int HK, int G, float scale, const int* __restrict__ PT, long long PS) {
    constexpr int D = 128, WARPS = 8, HPC = 2, NS = 8;
    constexpr bool causal = true;
    constexpr int skip = 0;
    constexpr bool P256 = false, HPAGED = false, F8 = false, HM = false, IDS = false;
    constexpr int CAP = 0, PTOK = 1, PN = 0;
    const float *ks = nullptr, *vs = nullptr;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

extern "C" __global__ void __launch_bounds__(256, 1) tf_ctl_pattn_64_hpc2(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                                                          const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0,
                                                                          int W, int H, int HK, int G, float scale, const int* __restrict__ PT, long long PS) {
    constexpr int D = 64, WARPS = 8, HPC = 2, NS = 8;
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

// Paged decode at head 128 (Nemotron's segs body), one planted fault: values from the neighbouring lane's dims.
namespace tf_ctl_paged {
#define SEG_HD 128
#define SEG_KW (128 / 2 + 2)
#define SEG_WIDE 1
#define SEG_VLANE 1
#define SEG_CHUNK tf_ctl_nemo_attn_chunk_segs
#define SEG_MERGE tf_ctl_nemo_attn_merge_segs
#include "paged_attn_segs.cuh"
}  // namespace tf_ctl_paged

// dots.mocr's q|k|v bias and RoPE with the rotation's two fp32 products summed and rounded once (qk_rows' order).
#include "dots_rope.cuh"
extern "C" __global__ void __launch_bounds__(64) tf_ctl_dots_qkv_rope(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                      const float* __restrict__ INV, const int* __restrict__ POS,
                                                                      __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KV, int H, int HK) {
    tf_dots::qkv_rope<false>(ACC, BIAS, INV, POS, Q, KV, H, HK);
}

// dots.mocr's vision rotary with cos and sin rounded to bf16 first (the text rotary's points, not HF's fp32 freqs).
#include "dots_vision.cuh"
extern "C" __global__ void __launch_bounds__(64)
    tf_ctl_dots_vision_rope(__nv_bfloat16* __restrict__ QKV, const int* __restrict__ HW, const float* __restrict__ INV,
                            int n) {
    if (static_cast<int>(blockIdx.x) < n) tf_dots_vision::rope<false>(QKV, HW, INV);
}

// dots.mocr's patch bias from the fp32 accumulator, rounded once (tf_vit_bias' order, not HIP's second rounding).
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_dots_patch_bias(const float* Y, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* O, int rows, int N) {
    if (static_cast<int>(blockIdx.x) < rows) tf_dots_vision::patch_bias(Y, B, O, N);
}
