// DeepSeek-V4.1-Flash's indexer and top-k, compiled after v41.metal (kernel_sources joins the files).

// ---------------------------------------------------------------- indexer

struct ScoreArgs {
  int rows;
  int groups;     // index keys in the cache
  int ratio;
  int first_pos;  // row r's position is first_pos + r
  int stride;     // score row stride
};

// score = sum_h bf16(relu(bf16(q_h . k_t)) * w_h) (bf16), -inf past visible keys: R-row by T-key tiles, a lane a head.
template <int R, int T>
[[kernel]] void ds_index_scores(const device bfloat* Q [[buffer(0)]], const device bfloat* WT [[buffer(1)]],
                                const device bfloat* K [[buffer(2)]], device bfloat* SCORE [[buffer(3)]],
                                constant ScoreArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  static_assert(R * T <= 32, "a lane sums a pair");
  threadgroup bfloat terms[8][R * T][IH + 2];  // a simdgroup's terms by pair, then head (bf16-exact)
  const int r0 = int(g.x) / 32 * R, t0 = int(g.y) * T, h = int(lane);
  if (r0 >= a.rows || t0 >= a.groups) return;
  const device bfloat4* q = (const device bfloat4*)(Q + size_t(r0) * IH * ID) + h * (ID / 4);
  const device bfloat4* k = (const device bfloat4*)(K + size_t(t0) * ID);
  const int rn = min(R, a.rows - r0) - 1, tn = min(T, a.groups - t0) - 1;  // the tile's last row and key in range
  float dot[R][T];
  for (int i = 0; i < R; i++)
    for (int j = 0; j < T; j++) dot[i][j] = 0.0f;
  for (int c = 0; c < ID / 4; c++) {
    float4 qv[R], kv[T];
    for (int i = 0; i < R; i++) qv[i] = float4(q[min(i, rn) * IH * (ID / 4) + c]);
    for (int j = 0; j < T; j++) kv[j] = float4(k[min(j, tn) * (ID / 4) + c]);
    for (int i = 0; i < R; i++)
      for (int j = 0; j < T; j++)
        for (int e = 0; e < 4; e++) dot[i][j] += qv[i][e] * kv[j][e];
  }
  for (int i = 0; i < R; i++) {
    const float w = float(WT[(r0 + min(i, rn)) * IH + h]);
    for (int j = 0; j < T; j++) terms[sg][i * T + j][h] = bfloat(max(bf_round(dot[i][j]), 0.0f) * w);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  // lane p adds pair p's head terms in head order
  const int i = h / T, j = h % T, r = r0 + i, t = t0 + j;
  if (h >= R * T || i > rn || j > tn) return;
  float total = 0.0f;
  for (int l = 0; l < IH; l++) total += float(terms[sg][h][l]);
  SCORE[size_t(r) * a.stride + t] = t < (a.first_pos + r + 1) / a.ratio ? bfloat(total) : bfloat(-INFINITY);
}

template [[host_name("ds_index_scores_1x8")]] [[kernel]] decltype(ds_index_scores<1, 8>) ds_index_scores<1, 8>;
template [[host_name("ds_index_scores_2x16")]] [[kernel]] decltype(ds_index_scores<2, 16>) ds_index_scores<2, 16>;

struct KeptArgs {
  int rows;
  int groups;
  int ratio;
  int first_pos;
  int block;
  int kept;  // KEEP's row width: the candidate blocks in ascending order, -1 after the last
};

// ds_index_scores' sums over a row's candidate blocks, a block a simdgroup: slot j is key keep[j / 8] * 8 + j % 8.
[[kernel]] void ds_index_scores_kept(const device bfloat* Q [[buffer(0)]], const device bfloat* WT [[buffer(1)]],
                                     const device bfloat* K [[buffer(2)]], device bfloat* SCORE [[buffer(3)]],
                                     const device int* KEEP [[buffer(4)]], constant KeptArgs& a [[buffer(5)]],
                                     uint2 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int T = 8;
  threadgroup bfloat terms[8][T][IH + 2];
  const int jb = int(g.x) / 32, r = int(g.y), h = int(lane);
  if (jb >= a.kept) return;
  const int b = KEEP[size_t(r) * a.kept + jb];
  const int live = b < 0 ? 0 : min(T, min(a.groups, (a.first_pos + r + 1) / a.ratio) - b * T);
  device bfloat* out = SCORE + (size_t(r) * a.kept + jb) * T;
  if (live <= 0) {
    if (h < T) out[h] = bfloat(-INFINITY);
    return;
  }
  const device bfloat4* q = (const device bfloat4*)(Q + (size_t(r) * IH + h) * ID);
  const device bfloat4* k = (const device bfloat4*)(K + size_t(b) * T * ID);
  float dot[T];
  for (int j = 0; j < T; j++) dot[j] = 0.0f;
  for (int c = 0; c < ID / 4; c++) {
    const float4 qv = float4(q[c]);
    for (int j = 0; j < T; j++) {
      const float4 kv = float4(k[min(j, live - 1) * (ID / 4) + c]);
      for (int e = 0; e < 4; e++) dot[j] += qv[e] * kv[e];
    }
  }
  const float w = float(WT[r * IH + h]);
  for (int j = 0; j < T; j++) terms[sg][j][h] = bfloat(max(bf_round(dot[j]), 0.0f) * w);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (h >= T) return;
  float total = 0.0f;
  for (int l = 0; l < IH; l++) total += float(terms[sg][h][l]);
  out[h] = h < live ? bfloat(total) : bfloat(-INFINITY);
}

// A row's picks over its candidate slots back to key positions (-1 stays).
[[kernel]] void ds_kept_keys(device int* PICKS [[buffer(0)]], const device int* KEEP [[buffer(1)]],
                             constant KeptArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int i = int(g.x), r = int(g.y);
  if (i >= IK || r >= a.rows) return;
  device int* p = PICKS + size_t(r) * IK + i;
  if (*p >= 0) *p = KEEP[size_t(r) * a.kept + *p / a.block] * a.block + *p % a.block;
}

struct BlockArgs {
  int rows;
  int groups;
  int block;
  int first_pos;
  int ratio;
  int stride;
};

// Block scores for the candidate pick (a block's max; the row's newest block pinned at +inf), a row's nb together.
[[kernel]] void ds_block_max(const device bfloat* SCORE [[buffer(0)]], device float* BLK [[buffer(1)]],
                             constant BlockArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int b = int(g.x), r = int(g.y);
  const int nb = (a.groups + a.block - 1) / a.block;
  if (b >= nb) return;
  const int visible = (a.first_pos + r + 1) / a.ratio;
  float m = -INFINITY;
  for (int i = 0; i < a.block; i++) {
    const int t = b * a.block + i;
    if (t < a.groups) m = max(m, float(SCORE[size_t(r) * a.stride + t]));
  }
  if (visible > 0 && b == (visible - 1) / a.block) m = FLT_MAX;  // pinned in (ds_topk takes finite scores only)
  BLK[size_t(r) * nb + b] = m;
}

struct TopkArgs {
  int n;       // candidates a row
  int k;       // picks a row
  int stride;  // score row stride
};

// A score's bits as an unsigned key in the score's order: f32 in 32 bits, bf16 in 16.
inline uint order_key(float v) {
  const uint b = as_type<uint>(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

inline uint order_key(bfloat v) {
  const uint b = as_type<ushort>(v);
  return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

// A 1,024-thread group's sum of v, every thread getting it (unsigned integers: exact in any order).
static uint tg_sum(uint v, uint t, uint lane, uint sg, threadgroup uint* sums) {
  const uint s = simd_sum(v);
  if (lane == 0) sums[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint total = simd_sum(sums[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// Each thread's sum of v over the threads before it (unsigned integers: exact in any order).
static uint tg_exclusive_sum(uint v, uint t, uint lane, uint sg, threadgroup uint* sums) {
  const uint inside = simd_prefix_exclusive_sum(v);
  if (lane == 31) sums[sg] = inside + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint lead = simd_prefix_exclusive_sum(sums[lane]);
  const uint before = simd_shuffle(lead, ushort(sg));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return before + inside;
}

// A row's k best finite scores (ties to the lower index) in index order, -1 after: a radix select, 8 key bits a pass.
template <typename T>
[[kernel]] void ds_topk(const device T* SCORE [[buffer(0)]], device int* OUT [[buffer(1)]],
                        constant TopkArgs& a [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix_bits, want;
  threadgroup uint sums[32];
  const device T* s = SCORE + size_t(r) * a.stride;
  device int* out = OUT + size_t(r) * a.k;
  // finite candidates only: -inf marks keys that are not visible or not candidates
  if (t == 0) {
    prefix_bits = 0;
    want = uint(a.k);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint finite = 0;
  for (int i = int(t); i < a.n; i += 1024) finite += isfinite(float(s[i])) ? 1 : 0;
  const uint total = tg_sum(finite, t, lane, sg, sums);
  if (t == 0) want = min(want, total);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint k = want;
  // the k-th largest key, 8 bits a pass from the top
  uint mask = 0;
  for (int shift = int(sizeof(T)) * 8 - 8; shift >= 0 && k > 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < a.n; i += 1024) {
      if (!isfinite(float(s[i]))) continue;
      const uint key = order_key(s[i]);
      if ((key & mask) == prefix_bits) atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFF], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the bin holding the want-th largest: bins from the top, each thread's count of those above its own by a scan
    {
      const uint b = 255 - min(t, 255u);
      const uint hb = t < 256 ? atomic_load_explicit(&hist[b], memory_order_relaxed) : 0;
      const uint above = tg_exclusive_sum(hb, t, lane, sg, sums);
      const uint w = want;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t < 256 && above < w && above + hb >= w) {
        prefix_bits |= b << shift;
        want = w - above;
      }
    }
    mask |= 0xFFu << shift;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint kth = prefix_bits;
  const uint ties = want;  // keys equal to kth to take, the lowest indices first
  // ordered compaction: each thread's span of indices, counted, then written in order
  const int span = (a.n + 1023) / 1024;
  const int lo = int(t) * span, hi = min(a.n, lo + span);
  uint above = 0, equal = 0;
  for (int i = lo; i < hi; i++) {
    if (!isfinite(float(s[i])) || k == 0) continue;
    const uint key = order_key(s[i]);
    above += key > kth ? 1 : 0;
    equal += key == kth ? 1 : 0;
  }
  const uint before_above = tg_exclusive_sum(above, t, lane, sg, sums);
  const uint before_equal = tg_exclusive_sum(equal, t, lane, sg, sums);
  uint taken_equal = before_equal;
  uint pos = before_above + min(before_equal, ties);
  for (int i = lo; i < hi; i++) {
    if (!isfinite(float(s[i])) || k == 0) continue;
    const uint key = order_key(s[i]);
    if (key > kth || (key == kth && taken_equal < ties)) {
      out[pos++] = i;
    }
    if (key == kth) taken_equal++;
  }
  for (int i = int(k) + int(t); i < a.k; i += 1024) out[i] = -1;
}

template [[host_name("ds_topk")]] [[kernel]] decltype(ds_topk<float>) ds_topk<float>;
template [[host_name("ds_topk_bf")]] [[kernel]] decltype(ds_topk<bfloat>) ds_topk<bfloat>;

struct ShortIndexArgs {
  uint rows, base, ratio, fill_keep, force_replay;
};

// All lanes bound the original FP32/BF16 score inputs while the first

// lanes emit the dense prefix and its -1 tail in the existing buffers.
template <bool AUDIT>
[[kernel]] void tf_short_index(const device uint4* Q [[buffer(0)]],
                                const device uint4* X [[buffer(1)]],
                                const device uint4* KEY [[buffer(2)]],
                                device int* PICKS [[buffer(3)]],
                                device int* KEEP [[buffer(4)]],
                                device atomic_uint* state [[buffer(5)]],
                                constant ShortIndexArgs& a [[buffer(6)]],
                                uint i [[thread_position_in_grid]]) {
  const uint q_count = a.rows * 1280 / 8;
  const uint x_count = a.rows * D / 8;
  const uint key_count = ((a.base + a.rows) / a.ratio) * ID / 8;
  uint4 data = 0;
  if (i < q_count) data = Q[i];
  else if (i < q_count + x_count) data = X[i - q_count];
  else if (i < q_count + x_count + key_count) data = KEY[i - q_count - x_count];
  constexpr uint4 low = 0x00007fffu;
  const bool outside = any((data & low) > 0x4380u) || any(((data >> 16) & low) > 0x4380u);
  if (outside || (a.force_replay && i == 0)) atomic_fetch_or_explicit(state, 1u, memory_order_relaxed);
  if (i < a.rows * IK) {
    const uint r = i / IK, key = i % IK;
    const uint visible = (a.base + r + 1) / a.ratio;
    const int pick = key < visible && !(a.force_replay && key == 0) ? int(key) : -1;
    if (AUDIT) {
      if (PICKS[i] != pick) atomic_fetch_or_explicit(state, 2u, memory_order_relaxed);
    } else PICKS[i] = pick;
  }
  constexpr uint blocks = 2048, block_size = 8;
  if (a.fill_keep && i < a.rows * blocks) {
    const uint r = i / blocks, part = i % blocks;
    const uint visible = (a.base + r + 1) / a.ratio;
    const int pick = part < (visible + block_size - 1) / block_size ? int(part) : -1;
    if (AUDIT) {
      if (KEEP[i] != pick) atomic_fetch_or_explicit(state, 2u, memory_order_relaxed);
    } else KEEP[i] = pick;
  }
}

template [[host_name("tf_short_index_fill")]] [[kernel]] decltype(tf_short_index<false>) tf_short_index<false>;
template [[host_name("tf_short_index_audit")]] [[kernel]] decltype(tf_short_index<true>) tf_short_index<true>;
