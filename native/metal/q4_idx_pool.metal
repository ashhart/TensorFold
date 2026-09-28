
  // Threadgroup j (DI threads): block START + j's pooled indexer key: the mean of its 4 raw keys (fp32 in order,
  // bf16), RMSNorm with (1 + w) (fp32, bf16), RoPE (RD dims, non-interleaved halves) at the block's first position.
  const int d = int(thread_position_in_threadgroup.x);
  const int j = int(threadgroup_position_in_grid.y);
  const int b = START[0] + j;
  threadgroup float part[DI / 32];
  threadgroup float normed[DI];
  const device bfloat* src = RAW + size_t(4 * b) * DI + d;
  float m = float(src[0]);
  for (int k = 1; k < 4; k++) m += float(src[k * DI]);
  const float x = float(bfloat(m * 0.25f));
  float ss = simd_sum(x * x);
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < DI / 32; k++) total += part[k];
  const float inv = metal::precise::rsqrt(total / float(DI) + eps[0]);
  normed[d] = float(bfloat((x * inv) * W[d]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const float freq = metal::precise::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(4 * b) * freq;
    const float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  OUT[j * DI + d] = bfloat(out);
