#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
constant constexpr int S = 4, D = 5120, F = S * D, MIX = 2 * S + S * S, SLICES = F / 1024;
struct HcArgs { float eps; float hc_eps; int iters; };

[[kernel]] void tf_hc_coefficients(const device float* PART [[buffer(0)]],
                                  const device float* BASE [[buffer(1)]],
                                  const device float* SCALE [[buffer(2)]],
                                  constant HcArgs& a [[buffer(3)]],
                                  device float* PRE_OUT [[buffer(4)]],
                                  device float* POST_OUT [[buffer(5)]],
                                  device float* COMB_OUT [[buffer(6)]],
                                  uint r [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    float value = 0.0f;
    if (lane <= MIX)
      for (int j = 0; j < SLICES; j++)
        value += PART[(size_t(r) * SLICES + j) * (MIX + 1) + lane];
    const float sum_squares = simd_shuffle(value, ushort(MIX));
    const float pre_sum = value;
    const float post_sum = simd_shuffle(value, ushort(S + (lane & 3)));
    const float comb_sum = simd_shuffle(value, ushort(2 * S + (lane < S * S ? lane : 0)));

    const float rs = precise::rsqrt(sum_squares / float(F) + a.eps);
    if (lane < uint(S)) {
      PRE_OUT[r * S + lane] = 1.0f / (1.0f + precise::exp(-(pre_sum * rs * SCALE[0] + BASE[lane]))) + a.hc_eps;
      POST_OUT[r * S + lane] = 2.0f / (1.0f + precise::exp(-(post_sum * rs * SCALE[1] + BASE[S + lane])));
    }
    const bool act = lane < uint(S * S);
    const uint cell = act ? lane : 0;
    const float v = comb_sum * rs * SCALE[2] + BASE[2 * S + cell];
    float mx = max(v, simd_shuffle_xor(v, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
    const float e = precise::exp(v - mx);
    float row = e + simd_shuffle_xor(e, ushort(1));
    row = row + simd_shuffle_xor(row, ushort(2));
    float cm = act ? e / row + a.hc_eps : 0.0f;
    for (int it = 0; it < a.iters; it++) {
      if (it > 0) {
        float rsum = cm + simd_shuffle_xor(cm, ushort(1));
        rsum = rsum + simd_shuffle_xor(rsum, ushort(2));
        cm = act ? cm / (rsum + a.hc_eps) : 0.0f;
      }
      float csum = cm + simd_shuffle_xor(cm, ushort(4));
      csum = csum + simd_shuffle_xor(csum, ushort(8));
      cm = act ? cm / (csum + a.hc_eps) : 0.0f;
    }
    if (act) COMB_OUT[r * S * S + lane] = cm;
}
