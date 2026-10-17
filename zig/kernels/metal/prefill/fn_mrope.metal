#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

// The recorded q4_attn_prep at three positions a row (interleaved M-RoPE 11, 11, 10): a text row keeps its bits.
[[kernel]] void fn_attn_prep_mrope(const device bfloat16_t* P [[buffer(0)]], const device int32_t* POS3 [[buffer(1)]],
                                   const device float* QW [[buffer(2)]], const device float* KW [[buffer(3)]],
                                   const device float* IW [[buffer(4)]], const constant float* eps [[buffer(5)]],
                                   const constant float* LOG2BASE [[buffer(6)]], device bfloat16_t* Q [[buffer(7)]],
                                   device bfloat16_t* Kout [[buffer(8)]], device bfloat16_t* IQ [[buffer(9)]],
                                   uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                                   uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                                   uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                                   uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int NQ = 24;
  constexpr int NKV = 2;
  constexpr int HD = 256;
  constexpr int RD = 64;
  constexpr int PW = 13952;
  constexpr int NI = 4;
  constexpr int IHD = 128;
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  threadgroup float part[HD / 32];
  threadgroup float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = simd_sum(x * x);
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!live) return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const int axis = (i % 3 == 1 && i < 33) ? 1 : ((i % 3 == 2 && i < 30) ? 2 : 0);
    const float freq = metal::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(POS3[r * 3 + axis]) * freq;
    const float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(out);
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(out);
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(out);
}
