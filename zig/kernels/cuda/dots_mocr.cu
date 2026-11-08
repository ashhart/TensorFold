// dots.mocr's CUDA glue: the text decoder's q|k|v bias and Torch-order RoPE (hip/ops/dense_text.hpp's rounding).

#include "dots_rope.cuh"

// Grid (rows, H + 2 HK), 64 threads; KV feeds tf_nemo_kv_write_segs as its QKV with NQKV 2 HK x 128, QD 0.
extern "C" __global__ void __launch_bounds__(64) tf_dots_qkv_rope(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                  const float* __restrict__ INV, const int* __restrict__ POS,
                                                                  __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KV, int H, int HK) {
    tf_dots::qkv_rope<true>(ACC, BIAS, INV, POS, Q, KV, H, HK);
}
