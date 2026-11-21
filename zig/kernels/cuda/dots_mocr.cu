// dots.mocr's CUDA glue: the text q|k|v bias and Torch-order RoPE, the vision tower's 2-D rotary and patch bias.

#include "dots_rope.cuh"
#include "dots_vision.cuh"

// Grid (rows, H + 2 HK), 64 threads; KV feeds tf_nemo_kv_write_segs as its QKV with NQKV 2 HK x 128, QD 0.
extern "C" __global__ void __launch_bounds__(64) tf_dots_qkv_rope(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                  const float* __restrict__ INV, const int* __restrict__ POS,
                                                                  __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KV, int H, int HK) {
    tf_dots::qkv_rope<true>(ACC, BIAS, INV, POS, Q, KV, H, HK);
}

// Grid (n, 24), 64 threads: QKV [n][3][12][128]'s q and k heads rotated in place at (h, w) = HW[row], INV [32].
extern "C" __global__ void __launch_bounds__(64)
    tf_dots_vision_rope(__nv_bfloat16* __restrict__ QKV, const int* __restrict__ HW, const float* __restrict__ INV,
                        int n) {
    if (static_cast<int>(blockIdx.x) < n) tf_dots_vision::rope<true>(QKV, HW, INV);
}

// Grid (rows, ceil(N / 256)), 256 threads: O [rows][N] = bf16(Y + B[col]) from bf16 Y; O may be Y.
extern "C" __global__ void __launch_bounds__(256)
    tf_dots_patch_bias(const __nv_bfloat16* Y, const __nv_bfloat16* __restrict__ B, __nv_bfloat16* O, int rows, int N) {
    if (static_cast<int>(blockIdx.x) < rows) tf_dots_vision::patch_bias(Y, B, O, N);
}
