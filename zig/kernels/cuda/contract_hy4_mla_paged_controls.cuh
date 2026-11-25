// Planted controls for hy4_mla.cu: each kernel instantiated with each tf_hy4::Fault, red at its corresponding gate.
#pragma once

#include "hy4_flash/mla_paged.cuh"

#define TF_HY4_CTL_ABSORB(tag, fault)                                                                                 \
  extern "C" __global__ void __launch_bounds__(128) tf_ctl_hy4_mla_absorb_##tag(                                      \
      const __nv_bfloat16* q, int64_t qs, const __nv_bfloat16* kvb, float* ql, int rows, int heads) {                 \
    tf_hy4::absorb<fault>(q, qs, kvb, ql, rows, heads);                                                               \
  }

#define TF_HY4_CTL_ATTN(tag, fault)                                                                                   \
  extern "C" __global__ void __launch_bounds__(256) tf_ctl_hy4_mla_attn_##tag(                                        \
      const float* ql, const __nv_bfloat16* qr, const __nv_bfloat16* cache, const int* pos, const int* idx,           \
      const float* sinks, float* ol, int heads, int window, int nidx, int ring, float scale) {                         \
    tf_hy4::attn<fault>(ql, qr, cache, pos, idx, sinks, ol, heads, window, nidx, ring, scale);                        \
  }

#define TF_HY4_CTL_ATTN_ROWS(tag, fault)                                                                              \
  extern "C" __global__ void __launch_bounds__(256) tf_ctl_hy4_mla_attn_rows_##tag(                                   \
      const float* ql, const __nv_bfloat16* qr, const unsigned long long* caches, const int* pos, const int* idx,     \
      const float* sinks, float* ol, int heads, int window, int nidx, int ring, float scale) {                         \
    tf_hy4::attn_rows<fault>(ql, qr, caches, pos, idx, sinks, ol, heads, window, nidx, ring, scale);                  \
  }

#define TF_HY4_CTL_VALUE(tag, fault)                                                                                  \
  extern "C" __global__ void __launch_bounds__(128) tf_ctl_hy4_mla_value_##tag(                                       \
      const float* ol, const __nv_bfloat16* kvb, const __nv_bfloat16* gate, int64_t gs, __nv_bfloat16* out,           \
      int64_t os, int rows, int heads) {                                                                              \
    tf_hy4::value<fault>(ol, kvb, gate, gs, out, os, rows, heads);                                                    \
  }

#define TF_HY4_CTL_POST(tag, fault)                                                                                   \
  extern "C" __global__ void __launch_bounds__(256) tf_ctl_hy4_mla_post_##tag(                                        \
      const float* parts, int64_t plane, int world, const __nv_bfloat16* w, __nv_bfloat16* out, int width,            \
      float eps) {                                                                                                    \
    tf_hy4::post<fault>(parts, plane, world, w, out, width, eps);                                                     \
  }

#define TF_HY4_CTL_ALL(K)              \
  K(bad_nan, tf_hy4::kBadNan)          \
  K(bad_repeat, tf_hy4::kBadRepeat)    \
  K(bad_tail, tf_hy4::kBadTail)        \
  K(bad_sums, tf_hy4::kBadSums)

TF_HY4_CTL_ALL(TF_HY4_CTL_ABSORB)
TF_HY4_CTL_ALL(TF_HY4_CTL_ATTN)
TF_HY4_CTL_ALL(TF_HY4_CTL_ATTN_ROWS)
TF_HY4_CTL_ALL(TF_HY4_CTL_VALUE)
TF_HY4_CTL_ALL(TF_HY4_CTL_POST)
