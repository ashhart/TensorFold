// Idol's decoders' own CUDA steps beside idol.cu's: each its template's BREAK 0 (the controls are the other values).
#include "idol_decoder.cuh"
#include "idol_decoder_attention.cuh"

using tf_idol_dec::f16;

// h [rows, 1024] fp32 = in_proj(p) + pe(p) for points p [rows, 3] fp32.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dec_embed(const float* __restrict__ P, const float* __restrict__ W,
                                                                    const float* __restrict__ B, const float* __restrict__ F,
                                                                    float* __restrict__ Hs, int rows) {
  tf_idol_dec::embed<0>(P, W, B, F, Hs, rows);
}

// o [rows, width] fp16 = y + b by column.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dec_half(const float* Y, const float* __restrict__ B,
                                                                   f16* __restrict__ O, int width, int rows) {
  tf_idol_dec::to_half<0>(Y, B, O, width, rows);
}

// kv [rows, 2048] fp16 = to_kv of the fp16 latent rows [rows, 16].
extern "C" __global__ void __launch_bounds__(256) tf_idol_dec_kv(const f16* __restrict__ X, const f16* __restrict__ W,
                                                                 const f16* __restrict__ B, f16* __restrict__ KV, int rows) {
  tf_idol_dec::kv<0>(X, W, B, KV, rows);
}

// 16 heads of 64 at column `offset` of rows `stride` apart RMS-normed by gamma, in place when Y is X.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dec_heads(const f16* X, const float* __restrict__ G, f16* Y,
                                                                    int stride, int offset, int rows) {
  tf_idol_dec::heads<0>(X, G, Y, stride, offset, rows);
}

// o [lq, heads 64] = softmax(q k^T scale) v from query rows to key rows, P rounded to fp16 for P x V.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dec_attention(const f16* __restrict__ Q, int qs,
                                                                        const f16* __restrict__ KV, int ks,
                                                                        f16* __restrict__ O, int lq, int lk, int heads,
                                                                        float scale) {
  tf_idol_dec::rows<64, 8, 0>(Q, qs, KV, ks, O, lq, lk, heads, scale);
}

// The same for lq <= 8 query rows and lk <= 8192 keys, fp32 throughout.
extern "C" __global__ void __launch_bounds__(256) tf_idol_dec_few(const f16* __restrict__ Q, int qs,
                                                                  const f16* __restrict__ KV, int ks, f16* __restrict__ O,
                                                                  int lq, int lk, int heads, float scale) {
  tf_idol_dec::few<0>(Q, qs, KV, ks, O, lq, lk, heads, scale);
}
