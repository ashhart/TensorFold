struct InkStageConvArgs { uint rows, width, commit, pairs; };
struct InkStageConvSegsArgs { uint width, n_segs, pairs; };
struct InkCommitHistoriesArgs { uint offset, slots, tap_rows; };
struct InkConvSeg { ulong hist0, raw0, staged0, out0, hist1, raw1, staged1, out1; uint rows, commit; };

inline void ink2_stage_pair(device bfloat* hist, const device bfloat* raw, device bfloat* staged,
                            uint rows, uint width, bool commit, uint c) {
  const size_t w = width;
  const bfloat h0 = hist[c];
  const bfloat h1 = hist[w + c];
  const bfloat h2 = hist[2 * w + c];
  staged[c] = h0;
  staged[w + c] = h1;
  staged[2 * w + c] = h2;
  for (uint r = 0; r < rows; ++r) staged[(size_t)(3 + r) * w + c] = raw[(size_t)r * w + c];
  if (commit) {
    if (rows == 1u) {
      hist[c] = h1;
      hist[w + c] = h2;
      hist[2 * w + c] = raw[c];
    } else if (rows == 2u) {
      hist[c] = h2;
      hist[w + c] = raw[c];
      hist[2 * w + c] = raw[w + c];
    } else {
      const size_t base = (size_t)(rows - 3u) * w + c;
      hist[c] = raw[base];
      hist[w + c] = raw[base + w];
      hist[2 * w + c] = raw[base + 2 * w];
    }
  }
}

[[max_total_threads_per_threadgroup(256)]]
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_stage_conv(
    device bfloat* hist0 [[buffer(0)]], const device bfloat* raw0 [[buffer(1)]], device bfloat* staged0 [[buffer(2)]],
    device bfloat* hist1 [[buffer(3)]], const device bfloat* raw1 [[buffer(4)]], device bfloat* staged1 [[buffer(5)]],
    constant InkStageConvArgs& a [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  const uint total = a.pairs * a.width;
  if (tid >= total) return;
  const uint p = tid / a.width;
  const uint c = tid % a.width;
  device bfloat* hist = (p == 0) ? hist0 : hist1;
  const device bfloat* raw = (p == 0) ? raw0 : raw1;
  device bfloat* staged = (p == 0) ? staged0 : staged1;
  ink2_stage_pair(hist, raw, staged, a.rows, a.width, a.commit != 0, c);
}

[[max_total_threads_per_threadgroup(256)]]
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_stage_conv_segs(
    const device InkConvSeg* segs [[buffer(0)]], constant InkStageConvSegsArgs& a [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]]) {
  const uint seg_idx = tid.y;
  if (tid.x >= a.pairs * a.width || seg_idx >= a.n_segs) return;
  const uint p = tid.x / a.width;
  const uint c = tid.x % a.width;
  const device InkConvSeg& seg = segs[seg_idx];
  device bfloat* hist = (device bfloat*)(p == 0 ? seg.hist0 : seg.hist1);
  const device bfloat* raw = (const device bfloat*)(p == 0 ? seg.raw0 : seg.raw1);
  device bfloat* staged = (device bfloat*)(p == 0 ? seg.staged0 : seg.staged1);
  ink2_stage_pair(hist, raw, staged, seg.rows, a.width, seg.commit != 0, c);
}

[[max_total_threads_per_threadgroup(256)]]
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_stage_conv_static(
    const device InkConvSeg* segs [[buffer(0)]], const device bfloat* weight0 [[buffer(1)]],
    const device bfloat* weight1 [[buffer(2)]], constant InkStageConvSegsArgs& a [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  if (tid.x >= a.pairs * a.width || tid.y >= a.n_segs) return;
  const uint p = tid.x / a.width;
  const uint c = tid.x % a.width;
  const device InkConvSeg& seg = segs[tid.y];
  device bfloat* hist = (device bfloat*)(p == 0 ? seg.hist0 : seg.hist1);
  const device bfloat* raw = (const device bfloat*)(p == 0 ? seg.raw0 : seg.raw1);
  device bfloat* staged = (device bfloat*)(p == 0 ? seg.staged0 : seg.staged1);
  device bfloat* out = (device bfloat*)(p == 0 ? seg.out0 : seg.out1);
  const device bfloat* weight = (p == 0) ? weight0 : weight1;
  const size_t w = a.width;
  bfloat t0 = hist[c];
  bfloat t1 = hist[w + c];
  bfloat t2 = hist[2 * w + c];
  staged[c] = t0;
  staged[w + c] = t1;
  staged[2 * w + c] = t2;
  const float w0 = float(weight[c * 4u + 0u]);
  const float w1 = float(weight[c * 4u + 1u]);
  const float w2 = float(weight[c * 4u + 2u]);
  const float w3 = float(weight[c * 4u + 3u]);
  const float norm = precise::sqrt(5.0f);
  for (uint r = 0; r < seg.rows; ++r) {
    const bfloat t3 = raw[(size_t)r * w + c];
    staged[(size_t)(3u + r) * w + c] = t3;
    float sum = float(t3);
    sum += w0 * float(t0);
    sum += w1 * float(t1);
    sum += w2 * float(t2);
    sum += w3 * float(t3);
    out[(size_t)r * w + c] = bfloat(precise::divide(sum, norm));
    t0 = t1;
    t1 = t2;
    t2 = t3;
  }
  if (seg.commit != 0) {
    hist[c] = t0;
    hist[w + c] = t1;
    hist[2 * w + c] = t2;
  }
}

[[max_total_threads_per_threadgroup(256)]]
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_ink2_commit_histories(
    const device bfloat* taps [[buffer(0)]], device bfloat* conv [[buffer(1)]],
    constant InkCommitHistoriesArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint slot_elems = 3u * 6656u;
  if (i >= a.slots * slot_elems) return;
  const uint slot = i / slot_elems;
  const uint rem = i % slot_elems;
  uint w;
  uint prefix;
  uint local;
  if (rem < 3072u) {
    w = 1024u; prefix = 0u; local = rem;
  } else if (rem < 6144u) {
    w = 1024u; prefix = 1024u; local = rem - 3072u;
  } else if (rem < 13056u) {
    w = 2304u; prefix = 2048u; local = rem - 6144u;
  } else {
    w = 2304u; prefix = 4352u; local = rem - 13056u;
  }
  const uint row = local / w;
  const uint c = local % w;
  const size_t src = (size_t)slot * a.tap_rows * 6656u + (size_t)a.tap_rows * prefix + (size_t)(a.offset + row) * w + c;
  conv[i] = taps[src];
}
