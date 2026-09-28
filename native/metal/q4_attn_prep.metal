
  // one threadgroup of HD threads per (row, head): heads [0, NQ) are queries (from the stacked projection's
  // [q | gate] pairs), [NQ, NQ + NKV) keys, then NI indexer queries (IHD dims each, after the values). RMSNorm
  // with (1 + w) in fp32, bf16 out, then RoPE on the first RD dims (non-interleaved halves), angles in fp32 at the
  // row's position.
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
  const float inv = metal::precise::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!live) return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    // as mx.fast.rope: inv_freq = exp2(-(i / half) * log2(base)), fast cos/sin of position * inv_freq
    const float freq = metal::precise::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(POS[r]) * freq;
    const float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(out);
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(out);
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(out);
