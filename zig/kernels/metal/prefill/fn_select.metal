// Flash Next's block selection in GPU-side rounds: the round's per-row metadata from the arena's position, then the indexer's block pooling (q4_idx_pool's arithmetic) at absolute blocks.
#include <metal_stdlib>
using namespace metal;

// One thread: rows t .. t + W - 1 (t = AR[1]) -> block ends, key counts, sparse flags and the pooling range; SEL: complete[16] ends[16] counts[16] sparse[16] start count pooled.
[[kernel]] void fz_sel_meta(const device int* AR [[buffer(0)]], const constant uint& W [[buffer(1)]],
    device int* SEL [[buffer(2)]], device int* POOLED_SHAPE [[buffer(3)]], device int* SC_SHAPE [[buffer(4)]],
    uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  constexpr int TOP = 512;
  const int tn = AR[1];
  const int pooled = min(SEL[66], tn / 4), last = (tn + int(W)) / 4, now = max(pooled, last);
  SEL[64] = pooled;
  SEL[65] = max(0, last - pooled);
  SEL[66] = now;
  for (int i = 0; i < int(W); i++) {
    const int e = tn + i + 1, c = e / 4;
    const bool sparse = c > TOP;
    SEL[i] = c;
    SEL[16 + i] = e;
    SEL[32 + i] = sparse ? 4 * TOP + e - 4 * c : e;
    SEL[48 + i] = sparse ? 1 : 0;
  }
  POOLED_SHAPE[0] = now;
  POOLED_SHAPE[1] = 128;
  SC_SHAPE[0] = int(W);
  SC_SHAPE[1] = now;
}

// Threadgroup j (128 threads) pools block SEL[64] + j: mean of its 4 raw keys (fp32 in order, bf16), RMSNorm (fp32, bf16), RoPE (64 dims, non-interleaved halves) at the block's first position.
[[kernel]] void fz_idx_pool_abs(const device bfloat* RAW [[buffer(0)]], const device int* SEL [[buffer(1)]],
    const device float* Wn [[buffer(2)]], const device float* eps [[buffer(3)]], const device float* LOG2BASE [[buffer(4)]],
    device bfloat* POOLED [[buffer(5)]], uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int DI = 128;
  constexpr int RD = 64;
  const int j = int(tg.y);
  if (j >= SEL[65]) return;
  const int d = int(tpos.x);
  const int b = SEL[64] + j;
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
    const float freq = metal::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(4 * b) * freq;
    const float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  POOLED[size_t(b) * DI + d] = bfloat(out);
}
