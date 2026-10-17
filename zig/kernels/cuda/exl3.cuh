// EXL3 on CUDA, shared by exl3.cu and exl3_experts.cu: codebooks, trellis states, the transform, a tile row's sums.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define TF_FULL 0xFFFFFFFFu

// The unnormalized Sylvester transform of 128 values held as lane + 32 q, in one fixed order of fp32 adds.
__device__ __forceinline__ void tf_exl3_fwht(float (&v)[4], unsigned lane) {
  for (unsigned d = 1; d < 32; d <<= 1) {
    for (int q = 0; q < 4; ++q) {
      const float other = __shfl_xor_sync(TF_FULL, v[q], d);
      v[q] = (lane & d) ? __fsub_rn(other, v[q]) : __fadd_rn(v[q], other);
    }
  }
  const float a0 = __fadd_rn(v[0], v[1]), a1 = __fsub_rn(v[0], v[1]);
  const float a2 = __fadd_rn(v[2], v[3]), a3 = __fsub_rn(v[2], v[3]);
  v[0] = __fadd_rn(a0, a2);
  v[1] = __fadd_rn(a1, a3);
  v[2] = __fsub_rn(a0, a2);
  v[3] = __fsub_rn(a1, a3);
}

// A state's fp16 weight, one rounding: mul1's sum is an integer times 2^-18 under 2^20, exact in fp32 before it.
template <int C>
__device__ __forceinline__ __half tf_exl3_value(unsigned state) {
  if (C == 2) {
    const float h = (float)__dp4a(state * 0x83DCD12Du, 0x01010101u, 1024u);
    return __float2half_rn(__fmaf_rn(h, __half2float(__ushort_as_half(0x1EEE)), __half2float(__ushort_as_half(0xC931))));
  }
  const unsigned x = C == 1 ? state * 0xCBAC1FEDu : state * 89226354u + 64248484u;
  const unsigned m = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  // the two halves' exponents differ by at most a few binades, so the fp32 sum is exact
  const float s = __fadd_rn(__half2float(__ushort_as_half((unsigned short)(m & 0xFFFFu))), __half2float(__ushort_as_half((unsigned short)(m >> 16))));
  return __float2half_rn(s);
}

// Lane l's stream words for its eight values 8 l .. 8 l + 7: the four u32 words its states span, MSB first.
template <int K>
__device__ __forceinline__ uint4 tf_exl3_words(const uint32_t* tile, unsigned lane) {
  constexpr unsigned ring = 256 * K, words = 8 * K;
  const unsigned i0 = ((8 * lane * K + K + ring - 16) % ring) / 32;
  return make_uint4(tile[i0], tile[(i0 + 1) % words], tile[(i0 + 2) % words], tile[(i0 + 3) % words]);
}

// The eight states out of those words: the 16 stream bits that end at each value.
template <int K>
__device__ __forceinline__ void tf_exl3_states(uint4 t, unsigned lane, unsigned (&st)[8]) {
  const unsigned off = (8 * lane * K + K + 256 * K - 16) % 32;
  // one variable shift lines the window up (a, b, c: its first 96 bits); every state after it is a constant shift
  const unsigned a = __funnelshift_l(t.y, t.x, off), b = __funnelshift_l(t.z, t.y, off), c = __funnelshift_l(t.w, t.z, off);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const unsigned p = n * K;
    st[n] = __funnelshift_l(p < 32 ? b : c, p < 32 ? a : b, p % 32) >> 16;
  }
}

template <int K, int C>
__device__ __forceinline__ void tf_exl3_unpack(uint4 t, unsigned lane, __half (&w)[8]) {
  unsigned st[8];
  tf_exl3_states<K>(t, lane, st);
  if constexpr (C == 2) {
    // mul1 two states at a time: the sums (<= 2044) are exact in fp16, and one fused fp16 multiply-add rounds once
    const __half2 scale = __half2half2(__ushort_as_half(0x1EEE)), bias = __half2half2(__ushort_as_half(0xC931));
#pragma unroll
    for (int n = 0; n < 8; n += 2) {
      const __half2 h = __halves2half2(__uint2half_rn(__dp4a(st[n] * 0x83DCD12Du, 0x01010101u, 1024u)),
                                       __uint2half_rn(__dp4a(st[n + 1] * 0x83DCD12Du, 0x01010101u, 1024u)));
      const __half2 v = __hfma2(h, scale, bias);
      w[n] = __low2half(v);
      w[n + 1] = __high2half(v);
    }
  } else {
#pragma unroll
    for (int n = 0; n < 8; ++n) w[n] = tf_exl3_value<C>(st[n]);
  }
}

template <int K, int C>
__device__ __forceinline__ void tf_exl3_decode(const uint32_t* tile, unsigned lane, __half (&w)[8]) {
  tf_exl3_unpack<K, C>(tf_exl3_words<K>(tile, lane), lane, w);
}

__device__ __forceinline__ void tf_exl3_store(uint8_t* y, uint64_t at, float v, unsigned f32) {
  if (f32) reinterpret_cast<float*>(y)[at] = v;
  else reinterpret_cast<__nv_bfloat16*>(y)[at] = __float2bfloat16_rn(v);
}

#define TF_EXL3_XS 136  // shared prompt rows 136 halves apart: a fragment's eight rows hit eight bank groups

// Prompt rows on the tensor units: fp16 A and B fragments, fp32 sums.
__device__ __forceinline__ void tf_exl3_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t tf_exl3_pair(__half lo, __half hi) {
  return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}
