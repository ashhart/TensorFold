#include <metal_stdlib>
using namespace metal;
namespace tfp {
template <typename T>
inline void tensor_attention_pv_scalar(thread frag<float>& lo, thread frag<float>& hi, thread const frag<float>& p,
                                       thread const frag<T>& v0, thread const frag<T>& v1, ushort lane) {
  TF_UNROLL
  for (ushort k = 0; k < 16; k++) {
    const ushort pl = (lane & 22) | ((k >> 2) & 1) | ((k >> 3) << 3);
    const ushort vl = ((k & 3) << 1) | ((k & 4) << 2) | (lane & 9);
    const float p0 = simd_shuffle(p[k & 3], pl), p1 = simd_shuffle(p[4 + (k & 3)], pl);
    TF_UNROLL
    for (ushort c = 0; c < 4; c++) {
      const float a = simd_shuffle(float(v0[4 * (k >> 3) + c]), vl);
      const float b = simd_shuffle(float(v1[4 * (k >> 3) + c]), vl);
      lo[c] = fma(p0, a, lo[c]); hi[c] = fma(p0, b, hi[c]);
      lo[4 + c] = fma(p1, a, lo[4 + c]); hi[4 + c] = fma(p1, b, hi[4 + c]);
    }
  }
}

// Split the significand into ten-bit tensor operands; the six leading products cover float precision.
inline float3 tensor_split(float x) {
  float high = as_type<float>(as_type<uint>(x) & 0xffffe000u);
  float rest = x - high;
  float middle = as_type<float>(as_type<uint>(rest) & 0xffffe000u);
  return float3(high, middle, rest - middle);
}

template <typename T>
inline void tensor_attention_pv(thread frag<float>& lo, thread frag<float>& hi, thread const frag<float>& p,
                                thread const frag<T>& v0, thread const frag<T>& v1, ushort lane) {
  bool exceptional = false;
  TF_UNROLL
  for (short e = 0; e < 8; e++) exceptional |= !isfinite(p[e]) || !isfinite(float(v0[e])) || !isfinite(float(v1[e]));
  if (simd_any(exceptional)) { tensor_attention_pv_scalar(lo, hi, p, v0, v1, lane); return; }
  frag<float> ph, pm, pt, vh0, vm0, vt0, vh1, vm1, vt1;
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    float3 ps = tensor_split(p[e]), a = tensor_split(float(v0[e])), b = tensor_split(float(v1[e]));
    ph[e] = ps.x; pm[e] = ps.y; pt[e] = ps.z;
    vh0[e] = a.x; vm0[e] = a.y; vt0[e] = a.z;
    vh1[e] = b.x; vm1[e] = b.y; vt1[e] = b.z;
  }
  mma_16x32<false, false>(lo, hi, ph, vh0, vh1);
  mma_16x32<false, false>(lo, hi, ph, vm0, vm1);
  mma_16x32<false, false>(lo, hi, pm, vh0, vh1);
  mma_16x32<false, false>(lo, hi, ph, vt0, vt1);
  mma_16x32<false, false>(lo, hi, pt, vh0, vh1);
  mma_16x32<false, false>(lo, hi, pm, vm0, vm1);
}
}
