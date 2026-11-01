// Idol's decoders' controls: idol_decoder.cu's templates at BREAK 1 to 3, each breaking a rounding point, sum or rule.
#pragma once

#include "idol_decoder.cuh"
#include "idol_decoder_attention.cuh"

#define TF_CTL_IDOL_DEC_EMBED(NAME, BREAK)                                                                    \
  extern "C" __global__ void __launch_bounds__(256)                                                          \
      NAME(const float* P, const float* W, const float* B, const float* F, float* Hs, int rows) {            \
    tf_idol_dec::embed<BREAK>(P, W, B, F, Hs, rows);                                                         \
  }
TF_CTL_IDOL_DEC_EMBED(tf_ctl_idol_dec_embed_pe16, 1)
TF_CTL_IDOL_DEC_EMBED(tf_ctl_idol_dec_embed_sum16, 2)
#undef TF_CTL_IDOL_DEC_EMBED

#define TF_CTL_IDOL_DEC_HALF(NAME, BREAK)                                                                     \
  extern "C" __global__ void __launch_bounds__(256)                                                          \
      NAME(const float* Y, const float* B, __half* O, int width, int rows) {                                 \
    tf_idol_dec::to_half<BREAK>(Y, B, O, width, rows);                                                       \
  }
TF_CTL_IDOL_DEC_HALF(tf_ctl_idol_dec_half_early, 1)
TF_CTL_IDOL_DEC_HALF(tf_ctl_idol_dec_half_writes, 2)
#undef TF_CTL_IDOL_DEC_HALF

#define TF_CTL_IDOL_DEC_KV(NAME, BREAK)                                                                       \
  extern "C" __global__ void __launch_bounds__(256)                                                          \
      NAME(const __half* X, const __half* W, const __half* B, __half* KV, int rows) {                        \
    tf_idol_dec::kv<BREAK>(X, W, B, KV, rows);                                                               \
  }
TF_CTL_IDOL_DEC_KV(tf_ctl_idol_dec_kv_early, 1)
TF_CTL_IDOL_DEC_KV(tf_ctl_idol_dec_kv_sum16, 2)
#undef TF_CTL_IDOL_DEC_KV

#define TF_CTL_IDOL_DEC_HEADS(NAME, BREAK)                                                                    \
  extern "C" __global__ void __launch_bounds__(256)                                                          \
      NAME(const __half* X, const float* G, __half* Y, int stride, int offset, int rows) {                   \
    tf_idol_dec::heads<BREAK>(X, G, Y, stride, offset, rows);                                                \
  }
TF_CTL_IDOL_DEC_HEADS(tf_ctl_idol_dec_heads_fmax, 1)
TF_CTL_IDOL_DEC_HEADS(tf_ctl_idol_dec_heads_div16, 2)
TF_CTL_IDOL_DEC_HEADS(tf_ctl_idol_dec_heads_sq16, 3)
#undef TF_CTL_IDOL_DEC_HEADS

#define TF_CTL_IDOL_DEC_ATTENTION(NAME, BODY)                                                                 \
  extern "C" __global__ void __launch_bounds__(256) NAME(const __half* Q, int qs, const __half* KV, int ks,    \
                                                         __half* O, int lq, int lk, int heads, float scale) { \
    BODY(Q, qs, KV, ks, O, lq, lk, heads, scale);                                                            \
  }
TF_CTL_IDOL_DEC_ATTENTION(tf_ctl_idol_dec_attention_unmasked, (tf_idol_dec::rows<64, 8, 1>))
TF_CTL_IDOL_DEC_ATTENTION(tf_ctl_idol_dec_attention_pv16, (tf_idol_dec::rows<64, 8, 2>))
TF_CTL_IDOL_DEC_ATTENTION(tf_ctl_idol_dec_few_p16, tf_idol_dec::few<1>)
TF_CTL_IDOL_DEC_ATTENTION(tf_ctl_idol_dec_few_pv16, tf_idol_dec::few<2>)
#undef TF_CTL_IDOL_DEC_ATTENTION
