// DeepSeek-V4.1-Flash's sparse attention and window-key kernels, compiled after v41_index.metal.

// ---------------------------------------------------------------- sparse attention

struct AttnArgs {
  int first_pos;  // row r's position is first_pos + r
  int ring;       // the window ring's rows
  int window;
  int picks;      // compressed picks a row (0: window only)
  float scale;
  int window_end; // >= 0: every row's window ends at this position (a draft block's), else at the row's own
};

constant constexpr int AH = 8;     // heads a threadgroup
constant constexpr int AT = 8;     // keys a tile
constant constexpr int AMAX = 640; // keys a row at most: the window and the indexer's picks

// The row's j-th key: its window (positions pos - nwin + 1 .. pos, from the ring), then its compressed picks.
inline const device bfloat* attn_key(const device bfloat* RING, const device bfloat* COMP, const device int* picks,
                                     int pos, int nwin, int ring, int j) {
  if (j < nwin) return RING + size_t((pos - nwin + 1 + j) % ring) * HD;
  return COMP + size_t(max(picks[j - nwin], 0)) * HD;
}

// Prompt chunks: a threadgroup (512) a (row, 8 heads) over its window and picks, each 8-key tile read once.
[[kernel]] void ds_attend_rows(const device bfloat* Q [[buffer(0)]], const device bfloat* RING [[buffer(1)]],
                               const device bfloat* COMP [[buffer(2)]], const device int* PICKS [[buffer(3)]],
                               const device float* SINK [[buffer(4)]], device bfloat* O [[buffer(5)]],
                               constant AttnArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint t = tpos.x;
  const int h0 = int(tg.x) * AH, r = int(tg.y);
  const int pos = a.first_pos + r;
  const int nwin = min(a.window, pos + 1);
  const int nkeys = nwin + a.picks;
  const device int* picks = PICKS + size_t(r) * a.picks;
  const device bfloat* q = Q + (size_t(r) * 64 + h0) * HD;
  threadgroup bfloat ks[AT][HD];
  threadgroup float sc[AH][AMAX];
  threadgroup float den[AH];
  for (int k0 = 0; k0 < nkeys; k0 += AT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < AT * HD; i += 512) {
      const int kk = i / HD;
      ks[kk][i % HD] = (k0 + kk < nkeys) ? attn_key(RING, COMP, picks, pos, nwin, a.ring, k0 + kk)[i % HD] : bfloat(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int pr = int(sg); pr < AH * AT; pr += 16) {
      const int h = pr / AT, kk = pr % AT, j = k0 + kk;
      if (j >= nkeys) continue;
      float dot = 0.0f;
      for (int d = int(lane); d < HD; d += 32) dot += float(q[h * HD + d]) * float(ks[kk][d]);
      dot = simd_sum(dot);
      if (lane == 0) sc[h][j] = (j < nwin || picks[j - nwin] >= 0) ? dot * a.scale : -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(sg) < AH) {
    const int h = int(sg);
    float m = -1e30f;
    for (int j = int(lane); j < nkeys; j += 32) m = max(m, sc[h][j]);
    m = simd_max(m);
    float l = 0.0f;
    for (int j = int(lane); j < nkeys; j += 32) {
      const float p = precise::exp(sc[h][j] - m);
      sc[h][j] = p;
      l += p;
    }
    l = simd_sum(l);
    if (lane == 0) den[h] = l + precise::exp(SINK[h0 + h] - m);
  }
  float acc[AH];
  for (int h = 0; h < AH; h++) acc[h] = 0.0f;
  for (int k0 = 0; k0 < nkeys; k0 += AT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < AT * HD; i += 512) {
      const int kk = i / HD;
      ks[kk][i % HD] = (k0 + kk < nkeys) ? attn_key(RING, COMP, picks, pos, nwin, a.ring, k0 + kk)[i % HD] : bfloat(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int nt = min(AT, nkeys - k0);
    for (int kk = 0; kk < nt; kk++) {
      const float v = float(ks[kk][t]);
      for (int h = 0; h < AH; h++) acc[h] += sc[h][k0 + kk] * v;
    }
  }
  for (int h = 0; h < AH; h++) O[(size_t(r) * 64 + h0 + h) * HD + t] = bfloat(acc[h] / den[h]);
}

constant constexpr int SPLIT = 16;  // keys a split threadgroup takes (held in threadgroup memory: 16 KiB)
constant constexpr int HG = 16;     // heads a split threadgroup takes (a simdgroup each), sharing its key tiles

// A RoPE pair's (cos, sin) at row r's position and the i-th frequency: from the call's table (ds_rope_table, rows of

// 32, bound at the table of FREQ's variant) with TAB, else computed here. The same floats either way.
template <bool TAB>
inline float2 rope_cs(const device float* FREQ, const device int* POS, int r, int i) {
  if (TAB) return ((const device float2*)FREQ)[r * 32 + i];
  const float ang = float(POS[r]) * FREQ[i];
  return float2(precise::cos(ang), precise::sin(ang));
}

// Decode rows' attention split by keys: a threadgroup a (HG heads, chunk, row), RoPE as queries load; partials out.
template <bool TAB>
[[kernel]] void ds_attend_split(const device bfloat* Q [[buffer(0)]], const device bfloat* RING [[buffer(1)]],
                                const device bfloat* COMP [[buffer(2)]], const device int* PICKS [[buffer(3)]],
                                device float* PO [[buffer(4)]], device float2* PML [[buffer(5)]],
                                constant AttnArgs& a [[buffer(6)]], const device int* POS [[buffer(7)]],
                                const device float* FREQ [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
                                uint3 tgn [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint3 tpos [[thread_position_in_threadgroup]]) {
  const uint t = tpos.x;
  const int h = int(tg.x) * HG + int(sg), chunk = int(tg.y), r = int(tg.z), chunks = int(tgn.y);
  const int pos = a.window_end >= 0 ? a.window_end : a.first_pos + r;
  const int nwin = min(a.window, pos + 1);
  const int nkeys = nwin + a.picks;
  const int k0 = chunk * SPLIT, k1 = min(nkeys, k0 + SPLIT);
  const device int* picks = PICKS + size_t(r) * a.picks;
  threadgroup bfloat ks[SPLIT][HD];
  threadgroup float sc[HG][SPLIT];
  // the chunk's keys, once, 8 values a load
  for (int i = int(t); i < SPLIT * (HD / 8); i += HG * 32) {
    const int kk = i / (HD / 8), c8 = i % (HD / 8), j = k0 + kk;
    uint4 v = uint4(0);
    if (j < k1) v = ((const device uint4*)attn_key(RING, COMP, picks, pos, nwin, a.ring, j))[c8];
    ((threadgroup uint4*)ks[kk])[c8] = v;
  }
  float q[16];
  const device bfloat* qr = Q + (size_t(r) * 64 + h) * HD;
  for (int i = 0; i < 16; i++) q[i] = float(qr[int(lane) * 16 + i]);
  // the query's RoPE (its last 64 dims: lanes 28-31, pairs inside a lane), rounded to bf16 as ds_rope writes it
  if (lane >= uint((HD - 64) / 16)) {
    for (int p = 0; p < 8; p++) {
      const float2 csn = rope_cs<TAB>(FREQ, POS, r, (int(lane) - (HD - 64) / 16) * 8 + p);
      const float cs = csn.x, sn = csn.y;
      const float re = q[2 * p], im = q[2 * p + 1];
      q[2 * p] = float(bfloat(re * cs - im * sn));
      q[2 * p + 1] = float(bfloat(re * sn + im * cs));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // scores: simdgroup sg's head against each key of the chunk (lane: 16 dims)
  for (int kk = 0; kk < k1 - k0; kk++) {
    float d = 0.0f;
    for (int i = 0; i < 16; i++) d += q[i] * float(ks[kk][int(lane) * 16 + i]);
    d = simd_sum(d);
    const int j = k0 + kk;
    const bool valid = j < nwin || picks[j - nwin] >= 0;
    if (lane == 0) sc[sg][kk] = valid ? d * a.scale : -INFINITY;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float m = -1e30f;
  for (int j = int(lane); j < k1 - k0; j += 32) m = max(m, sc[sg][j]);
  m = simd_max(m);
  float l = 0.0f;
  for (int j = int(lane); j < k1 - k0; j += 32) {
    const float p = precise::exp(sc[sg][j] - m);
    sc[sg][j] = p;
    l += p;
  }
  l = simd_sum(l);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  // outputs: the same keys, still in threadgroup memory (lane: 16 dims of its head)
  float o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int kk = 0; kk < k1 - k0; kk++) {
    const float p = sc[sg][kk];
    for (int i = 0; i < 16; i++) o[i] += p * float(ks[kk][int(lane) * 16 + i]);
  }
  const size_t part = (size_t(r) * 64 + h) * chunks + chunk;
  for (int i = 0; i < 16; i++) PO[part * HD + int(lane) * 16 + i] = o[i];
  if (lane == 0) PML[part] = float2(m, l);
}

template [[host_name("ds_attend_split")]] [[kernel]] decltype(ds_attend_split<false>) ds_attend_split<false>;
template [[host_name("ds_attend_split_t")]] [[kernel]] decltype(ds_attend_split<true>) ds_attend_split<true>;

// A (row, head)'s partials and sink as one softmax, then (as ds_rope inverse, ds_fp8_qdq) RoPE taken back, fp8-rounded.
struct MergeArgs {
  int chunks;
};

// A head's 64 RoPE dims (the last of its 512) rotated, in a threadgroup's bf16 row, as ds_rope does (32 threads).
template <bool TAB>
static void rope_row(threadgroup bfloat* x, const device int* POS, int r, const device float* FREQ, bool inverse,
                     uint t) {
  if (t < 32) {
    threadgroup bfloat* p = x + HD - 64 + 2 * int(t);
    const float2 csn = rope_cs<TAB>(FREQ, POS, r, int(t));
    const float cs = csn.x;
    const float sn = inverse ? -csn.y : csn.y;
    const float re = float(p[0]), im = float(p[1]);
    p[0] = bfloat(re * cs - im * sn);
    p[1] = bfloat(re * sn + im * cs);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A threadgroup's bf16 row of 512 fp8-rounded in blocks of 32 as ds_fp8_qdq does: a block a simdgroup.
static void fp8_row(threadgroup bfloat* x, uint t, uint n) {
  const uint lane = t % 32;
  for (uint b = t / 32; b < uint(HD / 32); b += n / 32) {
    threadgroup bfloat* p = x + b * 32;
    const float v = float(p[lane]);
    const float amax = simd_max(fabs(v));
    const float s = pow2_ceil(max(amax, 1e-4f) / 448.0f);
    p[lane] = bfloat(e4m3_round(clamp(v / s, -448.0f, 448.0f)) * s);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Each thread's partials load before the softmax totals, so the loads overlap them; the sum keeps its chunk order.
template <bool TAB>
[[kernel]] void ds_attend_merge(const device float* PO [[buffer(0)]], const device float2* PML [[buffer(1)]],
                                const device float* SINK [[buffer(2)]], device bfloat* O [[buffer(3)]],
                                constant MergeArgs& ma [[buffer(4)]], const device int* POS [[buffer(5)]],
                                const device float* FREQ [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint2 tpos [[thread_position_in_threadgroup]]) {
  constexpr int MAXC = (128 + 512 + SPLIT - 1) / SPLIT;
  const uint t = tpos.x;
  const int h = int(tg.x), r = int(tg.y), chunks = ma.chunks;
  threadgroup bfloat row[HD];
  threadgroup float scale[MAXC];
  threadgroup float2 ml[MAXC];
  const size_t base = (size_t(r) * 64 + h) * chunks;
  float pv[MAXC];
  for (int c = 0; c < MAXC; c++) pv[c] = c < chunks ? PO[(base + c) * HD + t] : 0.0f;
  for (int c = int(t); c < chunks; c += HD) ml[c] = PML[base + c];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = SINK[h];
  for (int c = 0; c < chunks; c++) m = max(m, ml[c].x);
  for (int c = int(t); c < chunks; c += HD) scale[c] = precise::exp(ml[c].x - m);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float denom = precise::exp(SINK[h] - m);
  for (int c = 0; c < chunks; c++) denom += ml[c].y * scale[c];
  float acc = 0.0f;
  for (int c = 0; c < MAXC; c++)
    if (c < chunks) acc += pv[c] * scale[c];
  row[t] = bfloat(acc / denom);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rope_row<TAB>(row, POS, r, FREQ, true, t);
  fp8_row(row, t, HD);
  O[(size_t(r) * 64 + h) * HD + t] = row[t];
}

template [[host_name("ds_attend_merge")]] [[kernel]] decltype(ds_attend_merge<false>) ds_attend_merge<false>;
template [[host_name("ds_attend_merge_t")]] [[kernel]] decltype(ds_attend_merge<true>) ds_attend_merge<true>;

// A window key's tail a threadgroup a row: ds_rms, ds_rope, ds_fp8_qdq in place, then into the ring (ring > 0).
template <bool TAB>
[[kernel]] void ds_kv_finish(device bfloat* KV [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                             const device int* POS [[buffer(2)]], const device float* FREQ [[buffer(3)]],
                             device bfloat* RING [[buffer(4)]], constant float2& a [[buffer(5)]],
                             uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8];
  threadgroup bfloat row[HD];
  device bfloat* x = KV + size_t(r) * HD;
  float ss = 0.0f;
  for (int i = int(t); i < HD; i += 256) {
    const float v = float(x[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int k = 0; k < 8; k++) tot += red[k];
  const float inv = precise::rsqrt(tot / float(HD) + a.x);
  for (int i = int(t); i < HD; i += 256) row[i] = bfloat(float(W[i]) * (float(x[i]) * inv));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rope_row<TAB>(row, POS, int(r), FREQ, false, t);
  fp8_row(row, t, 256);
  const int ring = int(a.y);
  device bfloat* dst = RING + size_t(ring > 0 ? POS[r] % ring : 0) * HD;
  for (int i = int(t); i < HD; i += 256) {
    x[i] = row[i];
    if (ring > 0) dst[i] = row[i];
  }
}

template [[host_name("ds_kv_finish")]] [[kernel]] decltype(ds_kv_finish<false>) ds_kv_finish<false>;
template [[host_name("ds_kv_finish_t")]] [[kernel]] decltype(ds_kv_finish<true>) ds_kv_finish<true>;

struct ClippedMergeArgs {
  int written;
  int full;
  int first_pos;
  int window;
  int picks;
};

// The unstored suffix has the same partials as an all-masked chunk in the

// original split kernel. Keep its place in each ordered merge reduction.
template <bool TAB, bool TIGHT = false>
[[kernel]] void ds_attend_merge_clipped(
    const device float* PO [[buffer(0)]], const device float2* PML [[buffer(1)]],
    const device float* SINK [[buffer(2)]], device bfloat* O [[buffer(3)]],
    constant ClippedMergeArgs& a [[buffer(4)]], const device int* POS [[buffer(5)]],
    const device float* FREQ [[buffer(6)]], const device bfloat* COMP [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tpos [[thread_position_in_threadgroup]]) {
  constexpr int MAXC = (128 + 512 + SPLIT - 1) / SPLIT;
  constexpr float empty_max = -1e30f;
  const uint t = tpos.x;
  const int h = int(tg.x), r = int(tg.y);
  threadgroup bfloat row[HD];
  threadgroup float scale[MAXC];
  threadgroup float2 ml[MAXC];
  const size_t base = (size_t(r) * 64 + h) * a.written;
  float pv[MAXC];
  for (int c = 0; c < MAXC; c++) pv[c] = c < a.written ? PO[(base + c) * HD + t] : 0.0f;
  // The existing split uses key zero as its load for an invalid pick. A

  // non-finite value can still propagate through its zero-probability products.
  const bfloat fallback = COMP[t];
  const bool finite_fallback = (as_type<ushort>(fallback) & 0x7f80u) != 0x7f80u;
  if (!finite_fallback) {
    const int nkeys = min(a.window, a.first_pos + r + 1) + a.picks;
    const float probability = precise::exp(-INFINITY - empty_max);
    for (int c = a.written; c < a.full; c++) {
      float partial = 0.0f;
      for (int j = c * SPLIT; j < min(nkeys, (c + 1) * SPLIT); j++)
        partial += probability * float(fallback);
      pv[c] = partial;
    }
  }
  for (int c = int(t); c < a.full; c += HD)
    ml[c] = c < a.written ? PML[base + c] : float2(empty_max, 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = SINK[h];
  for (int c = 0; c < (TIGHT ? a.written : a.full); c++) m = max(m, ml[c].x);
  if (TIGHT) m = max(m, empty_max);
  for (int c = int(t); c < a.full; c += HD) scale[c] = precise::exp(ml[c].x - m);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool tight = TIGHT && finite_fallback && isfinite(m);
  const int terms = tight ? a.written : a.full;
  float denom = precise::exp(SINK[h] - m);
  for (int c = 0; c < terms; c++) denom += ml[c].y * scale[c];
  float acc = 0.0f;
  for (int c = 0; c < MAXC; c++)
    if (c < terms) acc += pv[c] * scale[c];
  if (tight) {
    // All suffix products are positive zero. Retain one addition for the

    // IEEE signed-zero case, and replay the suffix on non-finite accumulators.
    if (isfinite(denom) && isfinite(acc)) {
      denom += ml[a.written].y * scale[a.written];
      acc += pv[a.written] * scale[a.written];
    } else {
      for (int c = a.written; c < a.full; c++) {
        denom += ml[c].y * scale[c];
        acc += pv[c] * scale[c];
      }
    }
  }
  row[t] = bfloat(acc / denom);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rope_row<TAB>(row, POS, r, FREQ, true, t);
  fp8_row(row, t, HD);
  O[(size_t(r) * 64 + h) * HD + t] = row[t];
}

template [[host_name("ds_attend_merge_clipped")]] [[kernel]] decltype(ds_attend_merge_clipped<false>) ds_attend_merge_clipped<false>;
template [[host_name("ds_attend_merge_clipped_t")]] [[kernel]] decltype(ds_attend_merge_clipped<true>) ds_attend_merge_clipped<true>;

template [[host_name("ds_attend_merge_tight")]] [[kernel]] decltype(ds_attend_merge_clipped<false, true>) ds_attend_merge_clipped<false, true>;
template [[host_name("ds_attend_merge_tight_t")]] [[kernel]] decltype(ds_attend_merge_clipped<true, true>) ds_attend_merge_clipped<true, true>;

[[kernel]] void ds_attend_clipped_audit(const device ushort* actual [[buffer(0)]],
                                        const device ushort* expected [[buffer(1)]],
                                        device atomic_uint* mismatch [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
  if (actual[i] != expected[i]) atomic_fetch_or_explicit(mismatch, 1u, memory_order_relaxed);
}

[[kernel]] void tf_condition_ring_audit(const device ushort* ring [[buffer(0)]],
                                          const device int* pos [[buffer(1)]],
                                          device ushort* ref [[buffer(2)]],
                                          device atomic_uint* mismatch [[buffer(3)]],
                                          constant uint2& a [[buffer(4)]],
                                          uint2 tid [[thread_position_in_grid]]) {
  const size_t slot = size_t(uint(pos[tid.y]) % a.x) * HD + tid.x;
  const size_t i = size_t(tid.y) * HD + tid.x;
  if (a.y == 0) ref[i] = ring[slot];
  else if (ref[i] != ring[slot]) atomic_fetch_or_explicit(mismatch, 1u, memory_order_relaxed);
}

template <int HEADS, bool TAB>
[[kernel]] void tf_attend_heads(const device bfloat* Q [[buffer(0)]], const device bfloat* RING [[buffer(1)]],
                                const device bfloat* COMP [[buffer(2)]], const device int* PICKS [[buffer(3)]],
                                device float* PO [[buffer(4)]], device float2* PML [[buffer(5)]],
                                constant AttnArgs& a [[buffer(6)]], const device int* POS [[buffer(7)]],
                                const device float* FREQ [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
                                uint3 tgn [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint3 tpos [[thread_position_in_threadgroup]]) {
  const uint t = tpos.x;
  const int h = int(tg.x) * HEADS + int(sg), chunk = int(tg.y), r = int(tg.z), chunks = int(tgn.y);
  const int pos = a.window_end >= 0 ? a.window_end : a.first_pos + r;
  const int nwin = min(a.window, pos + 1);
  const int nkeys = nwin + a.picks;
  const int k0 = chunk * SPLIT, k1 = min(nkeys, k0 + SPLIT);
  const device int* picks = PICKS + size_t(r) * a.picks;
  threadgroup bfloat ks[SPLIT][HD];
  threadgroup float sc[HEADS][SPLIT];
  // the chunk's keys, once, 8 values a load
  for (int i = int(t); i < SPLIT * (HD / 8); i += HEADS * 32) {
    const int kk = i / (HD / 8), c8 = i % (HD / 8), j = k0 + kk;
    uint4 v = uint4(0);
    if (j < k1) v = ((const device uint4*)attn_key(RING, COMP, picks, pos, nwin, a.ring, j))[c8];
    ((threadgroup uint4*)ks[kk])[c8] = v;
  }
  float q[16];
  const device bfloat* qr = Q + (size_t(r) * 64 + h) * HD;
  for (int i = 0; i < 16; i++) q[i] = float(qr[int(lane) * 16 + i]);
  // the query's RoPE (its last 64 dims: lanes 28-31, pairs inside a lane), rounded to bf16 as ds_rope writes it
  if (lane >= uint((HD - 64) / 16)) {
    for (int p = 0; p < 8; p++) {
      const float2 csn = rope_cs<TAB>(FREQ, POS, r, (int(lane) - (HD - 64) / 16) * 8 + p);
      const float cs = csn.x, sn = csn.y;
      const float re = q[2 * p], im = q[2 * p + 1];
      q[2 * p] = float(bfloat(re * cs - im * sn));
      q[2 * p + 1] = float(bfloat(re * sn + im * cs));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // scores: simdgroup sg's head against each key of the chunk (lane: 16 dims)
  for (int kk = 0; kk < k1 - k0; kk++) {
    float d = 0.0f;
    for (int i = 0; i < 16; i++) d += q[i] * float(ks[kk][int(lane) * 16 + i]);
    d = simd_sum(d);
    const int j = k0 + kk;
    const bool valid = j < nwin || picks[j - nwin] >= 0;
    if (lane == 0) sc[sg][kk] = valid ? d * a.scale : -INFINITY;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float m = -1e30f;
  for (int j = int(lane); j < k1 - k0; j += 32) m = max(m, sc[sg][j]);
  m = simd_max(m);
  float l = 0.0f;
  for (int j = int(lane); j < k1 - k0; j += 32) {
    const float p = precise::exp(sc[sg][j] - m);
    sc[sg][j] = p;
    l += p;
  }
  l = simd_sum(l);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  // outputs: the same keys, still in threadgroup memory (lane: 16 dims of its head)
  float o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int kk = 0; kk < k1 - k0; kk++) {
    const float p = sc[sg][kk];
    for (int i = 0; i < 16; i++) o[i] += p * float(ks[kk][int(lane) * 16 + i]);
  }
  const size_t part = (size_t(r) * 64 + h) * chunks + chunk;
  for (int i = 0; i < 16; i++) PO[part * HD + int(lane) * 16 + i] = o[i];
  if (lane == 0) PML[part] = float2(m, l);
}


template [[host_name("tf_attend_heads8")]] [[kernel]] decltype(tf_attend_heads<8, false>) tf_attend_heads<8, false>;
template [[host_name("tf_attend_heads8_t")]] [[kernel]] decltype(tf_attend_heads<8, true>) tf_attend_heads<8, true>;

// Stream short vectors instead of keeping every chunk partial live in registers.
template <uint V, bool SHARED>
[[kernel]] void tf_stream_merge(
    const device float* PO [[buffer(0)]], const device float2* PML [[buffer(1)]],
    const device float* SINK [[buffer(2)]], device bfloat* O [[buffer(3)]],
    constant ClippedMergeArgs& a [[buffer(4)]], const device int* POS [[buffer(5)]],
    const device float* TRIG [[buffer(6)]], const device bfloat* COMP [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 tpos [[thread_position_in_threadgroup]]) {
  using Packed = vec<float, V>;
  constexpr int threads = HD / V;
  constexpr int max_chunks = AMAX / SPLIT;
  constexpr float empty_max = -1e30f;
  const uint t = tpos.x;
  const int h = int(group.x), r = int(group.y);
  const int first = int(t) * V;
  threadgroup bfloat row[HD];
  threadgroup float scale[max_chunks];
  threadgroup float2 ml[max_chunks];
  threadgroup float denominator;
  const size_t base = (size_t(r) * 64 + h) * a.written;
  if (int(t) < a.full)
    ml[t] = int(t) < a.written ? PML[base + t] : float2(empty_max, 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool clipped = a.written < a.full;
  float m = SINK[h];
  for (int c = 0; c < a.written; ++c) m = max(m, ml[c].x);
  if (clipped) m = max(m, empty_max);
  if (int(t) < a.full) scale[t] = precise::exp(ml[t].x - m);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float denom = precise::exp(SINK[h] - m);
  if (SHARED) {
    if (t == 0) {
      for (int c = 0; c < a.written; ++c) denom += ml[c].y * scale[c];
      denominator = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denom = denominator;
  } else {
    for (int c = 0; c < a.written; ++c) denom += ml[c].y * scale[c];
  }
  const device Packed* v = (const device Packed*)(PO + base * HD + first);
  Packed acc = Packed(0.0f);
  for (int c = 0; c < a.written; ++c)
    acc += v[size_t(c) * threads] * scale[c];

  if (clipped) {
    bool finite_fallback = true;
    Packed fallback;
    for (uint i = 0; i < V; ++i) {
      const bfloat f = COMP[first + i];
      finite_fallback = finite_fallback && (as_type<ushort>(f) & 0x7f80u) != 0x7f80u;
      fallback[i] = float(f);
    }
    if (finite_fallback && isfinite(m) && isfinite(denom) && all(isfinite(acc))) {
      denom += ml[a.written].y * scale[a.written];
      acc += Packed(0.0f) * scale[a.written];
    } else {
      const int nkeys = min(a.window, a.first_pos + r + 1) + a.picks;
      const float p = precise::exp(-INFINITY - empty_max);
      for (int c = a.written; c < a.full; ++c) {
        Packed partial = Packed(0.0f);
        if (!finite_fallback) {
          for (int j = c * SPLIT; j < min(nkeys, (c + 1) * SPLIT); ++j)
            partial += p * fallback;
        }
        denom += ml[c].y * scale[c];
        acc += partial * scale[c];
      }
    }
  }

  for (uint i = 0; i < V; ++i) row[first + i] = bfloat(acc[i] / denom);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rope_row<true>(row, POS, r, TRIG, true, t);
  fp8_row(row, t, threads);
  for (uint i = 0; i < V; ++i) O[(size_t(r) * 64 + h) * HD + first + i] = row[first + i];
}

template [[host_name("tf_stream_merge_2")]] [[kernel]] decltype(tf_stream_merge<2, false>) tf_stream_merge<2, false>;
template [[host_name("tf_stream_merge_4")]] [[kernel]] decltype(tf_stream_merge<4, false>) tf_stream_merge<4, false>;
template [[host_name("tf_stream_merge_4_shared")]] [[kernel]] decltype(tf_stream_merge<4, true>) tf_stream_merge<4, true>;
