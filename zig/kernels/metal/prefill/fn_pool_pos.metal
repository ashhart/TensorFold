// Flash Next's indexer pooling at an image stream's block positions (the standard frame), q4_idx_pool's arithmetic.
#include <metal_stdlib>
using namespace metal;
struct TF_RotaryScalar { float frequency, angle, cosine, sine, co, si; };
inline TF_RotaryScalar tf_rotary_scalar(int i, int position, const constant float* LOG2BASE) {
  const float freq = LOG2BASE[1] != 0.0f ? LOG2BASE[3 + i] : metal::precise::exp2(-(float(i) / 32.0f) * LOG2BASE[0]);
  const float angle = float(position) * freq;
  const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
  const float co = metal::precise::cos(angle), si = metal::precise::sin(angle);
  return {freq, angle, co, si, co * ys, si * ys};
}

// q4_idx_pool's arithmetic at block START[0] + j's positions BPOS[b * 3 ..] on fn_attn_prep_mrope's axes, out at j.
[[kernel]] void fz_idx_pool_pos(const device bfloat* RAW [[buffer(0)]], const constant int* START [[buffer(1)]],
    const device float* Wn [[buffer(2)]], const constant float* eps [[buffer(3)]], const constant float* LOG2BASE [[buffer(4)]],
    const device int* BPOS [[buffer(5)]], device bfloat* POOLED [[buffer(6)]], uint sgi [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int DI = 128;
  constexpr int RD = 64;
  const int j = int(tg.y);
  const int d = int(tpos.x);
  const int b = START[0] + j;
  threadgroup float part[DI / 32];
  threadgroup float normed[DI];
  const device bfloat* src = RAW + size_t(4 * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < 4; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m * 0.25f));
  float ss = simd_sum(x * x);
  if (lane == 0) part[sgi] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * Wn[d]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const int axis = (i % 3 == 1 && i < 33) ? 1 : ((i % 3 == 2 && i < 30) ? 2 : 0);
    const TF_RotaryScalar cs = tf_rotary_scalar(i, BPOS[b * 3 + axis], LOG2BASE);
    const float c = cs.co, s = cs.si;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  POOLED[size_t(j) * DI + d] = bfloat(out);
}
