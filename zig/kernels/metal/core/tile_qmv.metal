// The tiles' twin for decode and windows (after tile_qmm.metal): fma chains in the MMA order, by row or by expert.

// The twin's columns a simdgroup and simdgroups a threadgroup.
template <int BITS> struct TfTqCols { static constant constexpr int value = 4; };
constant constexpr int TQ_SG = 2;
// Rows a scalar twin threadgroup takes over one load of its columns' codes.
constant constexpr int TQ_RT = 4;
// Picks a window call takes at most (tf_tile_qmv_group's slots).
constant constexpr int TQ_SLOTS = 128;

// Group g's 64-step fma chains (the MMA order) and input sum; each code decoded to its integer, as in the tiles.
template <int BITS, int NC>
inline float tf_tq_group(const device bfloat* x, const device uint8_t* w, size_t stride, int g, thread float* p,
                         float one) {
  static_assert(BITS == 4 || BITS == 6 || BITS == 8, "codes of 4, 6 or 8 bits");
  constexpr int GW = 2 * BITS;  // words a group of 64 codes
  float t = 0.0f;
  for (int c = 0; c < NC; ++c) p[c] = 0.0f;
  uint wq[BITS == 4 ? NC : 1][8];  // 4 bits: each column's group, two 16-byte loads
  if (BITS == 4)
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
      const device uint4* src = (const device uint4*)(w + c * stride + size_t(g) * (GW * 4));
      const uint4 a = src[0], b = src[1];
      wq[c][0] = a.x; wq[c][1] = a.y; wq[c][2] = a.z; wq[c][3] = a.w;
      wq[c][4] = b.x; wq[c][5] = b.y; wq[c][6] = b.z; wq[c][7] = b.w;
    }
  #pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    const uint4 xv = ((const device uint4*)(x + size_t(g) * 64))[kb];
    const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
    float xf[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = as_type<float>((j % 2) ? (xw[j / 2] & 0xFFFF0000u) : (xw[j / 2] << 16));
      t = kb == 0 && j == 0 ? v : fma(v, one, t);
      xf[j] = v;
    }
    if (BITS == 4) {
      #pragma unroll
      for (int c = 0; c < NC; ++c) {
        const uint word = wq[c][kb];
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[c] = fma(xf[j], float((word >> (4 * j)) & 0xFu), p[c]);
      }
    } else {
      // Codes 8 kb.. of this group sit in the two words from bit 8 kb BITS (48 or 64 bits, never a third word).
      const int bit = kb * 8 * BITS, at = bit / 32, sh = bit % 32;
      #pragma unroll
      for (int c = 0; c < NC; ++c) {
        const device uint* wc = (const device uint*)(w + c * stride + size_t(g) * (GW * 4)) + at;
        const ulong lo = (ulong(wc[1]) << 32) | wc[0];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint q = uint(lo >> (sh + BITS * j));
          p[c] = fma(xf[j], float(q & ((1u << BITS) - 1u)), p[c]);
        }
      }
    }
  }
  return t;
}

// A column's codes 8 kb.. of its group as integers in floats (4 bits from the group's words wq, else the stream).
template <int BITS>
inline void tf_tq_cf(const device uint8_t* wg, const thread uint* wq, int kb, thread float* cf) {
  static_assert(BITS == 4 || BITS == 6 || BITS == 8, "codes of 4, 6 or 8 bits");
  if (BITS == 4) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) cf[j] = float((wq[kb] >> (4 * j)) & 0xFu);
  } else {
    const int bit = kb * 8 * BITS, at = bit / 32, sh = bit % 32;
    const device uint* wc = (const device uint*)wg + at;
    const ulong lo = (ulong(wc[1]) << 32) | wc[0];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cf[j] = float(uint(lo >> (sh + BITS * j)) & ((1u << BITS) - 1u));
  }
}

// tf_tq_group for ROWS rows X[xrow[r]]: one code load and conversion a group, each row its own chains.
template <int BITS, int NC, int ROWS>
inline void tf_tq_group_rows(const device bfloat* X, int K, const threadgroup int* xrow, const device uint8_t* w,
                             size_t stride, int g, thread float (*p)[NC], thread float* t, float one) {
  constexpr int GW = 2 * BITS;
  uint wq[NC][BITS == 4 ? 8 : 1];
  if (BITS == 4)
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
      const device uint4* src = (const device uint4*)(w + c * stride + size_t(g) * 32);
      const uint4 a = src[0], b = src[1];
      wq[c][0] = a.x; wq[c][1] = a.y; wq[c][2] = a.z; wq[c][3] = a.w;
      wq[c][4] = b.x; wq[c][5] = b.y; wq[c][6] = b.z; wq[c][7] = b.w;
    }
  const device uint4* xr[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    xr[r] = (const device uint4*)(X + size_t(xrow[r]) * K + size_t(g) * 64);
    t[r] = 0.0f;
    for (int c = 0; c < NC; ++c) p[r][c] = 0.0f;
  }
  #pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    float cf[NC][8];
    #pragma unroll
    for (int c = 0; c < NC; ++c) tf_tq_cf<BITS>(w + c * stride + size_t(g) * (GW * 4), wq[c], kb, cf[c]);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint4 xv = xr[r][kb];
      const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
      float xf[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = as_type<float>((j % 2) ? (xw[j / 2] & 0xFFFF0000u) : (xw[j / 2] << 16));
        t[r] = kb == 0 && j == 0 ? v : fma(v, one, t[r]);
        xf[j] = v;
      }
      #pragma unroll
      for (int c = 0; c < NC; ++c)
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[r][c] = fma(xf[j], cf[c][j], p[r][c]);
    }
  }
}

// NC columns of ROWS rows, 32 groups a round: a group a lane, then lane r * NC + c chains row r's column c.
template <int BITS, int NC, int ROWS>
inline float tf_tq_columns_rows(const device bfloat* X, int K, const threadgroup int* xrow, float one,
                                const device uint8_t* w, const device bfloat* sc, const device bfloat* bi,
                                uint lane, threadgroup float* part) {
  const int G = K / 64;
  const size_t stride = size_t(K) * BITS / 8;
  const int r = int(lane) / NC, c = int(lane) % NC;
  float acc = 0.0f;
  for (int g0 = 0; g0 < G; g0 += 32) {  // a round's partials and sums staged in part [ROWS][NC + 1][32]
    const int g = g0 + int(lane);
    if (g < G) {
      float p[ROWS][NC], t[ROWS];
      tf_tq_group_rows<BITS, NC, ROWS>(X, K, xrow, w, stride, g, p, t, one);
      for (int i = 0; i < ROWS; ++i) {
        part[(i * (NC + 1) + NC) * 32 + lane] = t[i];
        for (int j = 0; j < NC; ++j) part[(i * (NC + 1) + j) * 32 + lane] = p[i][j];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (r < ROWS) {
      const device bfloat* s = sc + c * G + g0;
      const device bfloat* b = bi + c * G + g0;
      const threadgroup float* pc = part + (r * (NC + 1) + c) * 32;
      const threadgroup float* sums = part + (r * (NC + 1) + NC) * 32;
      const int n = min(32, G - g0);
      #pragma unroll(8)
      for (int j = 0; j < n; ++j) acc = fma(float(b[j]), sums[j], fma(float(s[j]), pc[j], acc));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  return acc;
}

// tf_tq_columns_rows at `rows` (1 to TQ_RT) rows.
template <int BITS, int NC>
inline float tf_tq_rows(const device bfloat* X, int K, const threadgroup int* xrow, int rows, float one,
                        const device uint8_t* w, const device bfloat* sc, const device bfloat* bi,
                        uint lane, threadgroup float* part) {
  switch (rows) {
    case 1: return tf_tq_columns_rows<BITS, NC, 1>(X, K, xrow, one, w, sc, bi, lane, part);
    case 2: return tf_tq_columns_rows<BITS, NC, 2>(X, K, xrow, one, w, sc, bi, lane, part);
    case 3: return tf_tq_columns_rows<BITS, NC, 3>(X, K, xrow, one, w, sc, bi, lane, part);
    default: return tf_tq_columns_rows<BITS, NC, 4>(X, K, xrow, one, w, sc, bi, lane, part);
  }
}

// NC columns of a row at the tiles' bits, 32 groups a round: a group a lane, then lane c chains column c in order.
template <int BITS, int NC>
inline float tf_tq_columns(const device bfloat* x, float one, const device uint8_t* w, const device bfloat* sc,
                           const device bfloat* bi, int K, uint lane, threadgroup float* part) {
  const int G = K / 64;
  const size_t stride = size_t(K) * BITS / 8;
  threadgroup float* sums = part + NC * 32;
  const int c = min(int(lane), NC - 1);
  float acc = 0.0f;
  for (int g0 = 0; g0 < G; g0 += 32) {
    const int g = g0 + int(lane);
    if (g < G) {
      float p[NC];
      sums[lane] = tf_tq_group<BITS, NC>(x, w, stride, g, p, one);
      for (int j = 0; j < NC; ++j) part[j * 32 + lane] = p[j];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const device bfloat* s = sc + c * G + g0;
    const device bfloat* b = bi + c * G + g0;
    const threadgroup float* pc = part + c * 32;
    const int n = min(32, G - g0);
    #pragma unroll(8)
    for (int j = 0; j < n; ++j) acc = fma(float(b[j]), sums[j], fma(float(s[j]), pc[j], acc));
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  return acc;
}

// Y = X W^T at the tiles' bits for decode: a threadgroup row a row (its own kernel keeps its registers few).
template <int BITS>
[[kernel]] void tf_tile_qmv(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                            const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                            constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                            threadgroup float* part [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  const int r = int(tg.y);
  const float y = tf_tq_columns<BITS, NC>(X + size_t(r) * K, a.one, W + size_t(n) * (K * BITS / 8), SC + size_t(n) * G,
                                         BI + size_t(n) * G, K, lane, part + sg * (NC + 1) * 32);
  if (int(lane) < NC) OUT[size_t(r) * a.N + n + lane] = bfloat(y);
}

// Y = X W^T for small windows: tg.y takes TQ_RT rows over one code load.
template <int BITS>
[[kernel]] void tf_tile_qmv_few(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                threadgroup float* part [[threadgroup(0)]], uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  threadgroup int rowmap[TQ_RT];
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  const int r0 = int(tg.y) * TQ_RT, rows = min(TQ_RT, a.R - r0);
  if (tid < TQ_RT) rowmap[tid] = r0 + int(tid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (n >= a.N) return;  // after the threadgroup's only barrier: every thread reaches it
  const float y = tf_tq_rows<BITS, NC>(X, K, rowmap, rows, a.one, W + size_t(n) * (K * BITS / 8), SC + size_t(n) * G,
                                       BI + size_t(n) * G, lane, part + sg * TQ_RT * (NC + 1) * 32);
  const int r = int(lane) / NC;
  if (r < rows) OUT[size_t(r0 + r) * a.N + n + lane % NC] = bfloat(y);
}

// The twin by slot (tg.y): expert ids[slot]'s matrix on x row slot / a.R, out row slot.
template <int BITS>
[[kernel]] void tf_tile_qmv_indexed(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                    const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                    constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                    const device uint* ids [[buffer(6)]], threadgroup float* part [[threadgroup(0)]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  const size_t e = ids[tg.y], row = tg.y / uint(a.R), m = e * size_t(a.N) + n;
  const float y = tf_tq_columns<BITS, NC>(X + row * K, a.one, W + m * (K * BITS / 8), SC + m * G, BI + m * G, K, lane,
                                         part + sg * (NC + 1) * 32);
  if (int(lane) < NC) OUT[size_t(tg.y) * a.N + n + lane] = bfloat(y);
}

// Gate and up by slot at the tiles' bits (W/SC/BI and W2/SC2/BI2, the same shape), out = swiglu of their bf16 rows.
template <int BITS>
[[kernel]] void tf_tile_qmv_glu(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                const device uint* ids [[buffer(6)]], const device uint8_t* W2 [[buffer(7)]],
                                const device bfloat* SC2 [[buffer(8)]], const device bfloat* BI2 [[buffer(9)]],
                                threadgroup float* part [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  const size_t e = ids[tg.y], row = tg.y / uint(a.R), m = e * size_t(a.N) + n;
  threadgroup float* own = part + sg * (NC + 1) * 32;
  const float g = tf_tq_columns<BITS, NC>(X + row * K, a.one, W + m * (K * BITS / 8), SC + m * G, BI + m * G, K,
                                         lane, own);
  const float u = tf_tq_columns<BITS, NC>(X + row * K, a.one, W2 + m * (K * BITS / 8), SC2 + m * G, BI2 + m * G, K,
                                         lane, own);
  if (int(lane) < NC) OUT[size_t(tg.y) * a.N + n + lane] = tf_tq_swiglu(bfloat(g), bfloat(u));
}

// Picks by expert, one threadgroup: GROUPS = count, (id, base, count) an expert in first-pick order, then their slots.
[[kernel]] void tf_tile_qmv_group(const device uint* ids [[buffer(0)]], constant int& slots [[buffer(1)]],
                                  device uint* groups [[buffer(2)]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup int first[TQ_SLOTS], cnt[TQ_SLOTS];
  const int s = int(t);
  int lead = s, count = 0, pos = 0;
  if (s < slots)
    for (int j = 0; j < slots; ++j)
      if (ids[j] == ids[s]) {
        ++count;
        if (j < s) {
          lead = min(lead, j);
          ++pos;
        }
      }
  first[s] = s < slots && lead == s;
  cnt[s] = count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (s >= slots) return;
  int item = 0, base = 0;
  for (int j = 0; j < lead; ++j)
    if (first[j]) {
      ++item;
      base += cnt[j];
    }
  groups[1 + 3 * slots + base + pos] = uint(s);
  if (lead == s) {
    groups[1 + 3 * item] = ids[s];
    groups[2 + 3 * item] = uint(base);
    groups[3 + 3 * item] = uint(count);
  }
  if (s == 0) {
    int n = 0;
    for (int j = 0; j < slots; ++j) n += first[j];
    groups[0] = uint(n);
  }
}

// The twin by picked expert (item tg.y of GROUPS): its matrix read once for all its slots, TQ_RT slots a pass.
template <int BITS, bool GLU>
[[kernel]] void tf_tile_qmv_grouped(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                    const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                    constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                    const device uint* groups [[buffer(6)]], const device uint8_t* W2 [[buffer(7)]],
                                    const device bfloat* SC2 [[buffer(8)]], const device bfloat* BI2 [[buffer(9)]],
                                    constant int& slots [[buffer(10)]], threadgroup float* part [[threadgroup(0)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  threadgroup int xrow[TQ_RT], orow[TQ_RT];
  if (tg.y >= groups[0]) return;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  const uint e = groups[1 + 3 * tg.y], base = groups[2 + 3 * tg.y], count = groups[3 + 3 * tg.y];
  const device uint* list = groups + 1 + 3 * slots + base;
  const size_t m = size_t(e) * size_t(a.N) + n;
  threadgroup float* own = part + sg * TQ_RT * (NC + 1) * 32;
  for (uint at = 0; at < count; at += TQ_RT) {
    const int rows = min(TQ_RT, int(count - at));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(rows)) {
      const int s = int(list[at + tid]);
      xrow[tid] = s / a.R;
      orow[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g, u = 0.0f;
    if (rows == 1) {
      const device bfloat* x = X + size_t(xrow[0]) * K;
      g = tf_tq_columns<BITS, NC>(x, a.one, W + m * (K * BITS / 8), SC + m * G, BI + m * G, K, lane, own);
      if (GLU) u = tf_tq_columns<BITS, NC>(x, a.one, W2 + m * (K * BITS / 8), SC2 + m * G, BI2 + m * G, K, lane, own);
    } else {
      g = tf_tq_rows<BITS, NC>(X, K, xrow, rows, a.one, W + m * (K * BITS / 8), SC + m * G, BI + m * G, lane, own);
      if (GLU)
        u = tf_tq_rows<BITS, NC>(X, K, xrow, rows, a.one, W2 + m * (K * BITS / 8), SC2 + m * G, BI2 + m * G, lane, own);
    }
    const int r = int(lane) / NC;
    if (r < rows) OUT[size_t(orow[r]) * a.N + n + lane % NC] = GLU ? tf_tq_swiglu(bfloat(g), bfloat(u)) : bfloat(g);
  }
}

template [[host_name("tf_tile_qmv_q4")]] [[kernel]] decltype(tf_tile_qmv<4>) tf_tile_qmv<4>;
template [[host_name("tf_tile_qmv_q6")]] [[kernel]] decltype(tf_tile_qmv<6>) tf_tile_qmv<6>;
template [[host_name("tf_tile_qmv_q8")]] [[kernel]] decltype(tf_tile_qmv<8>) tf_tile_qmv<8>;
template [[host_name("tf_tile_qmv_few_q4")]] [[kernel]] decltype(tf_tile_qmv_few<4>) tf_tile_qmv_few<4>;
template [[host_name("tf_tile_qmv_few_q6")]] [[kernel]] decltype(tf_tile_qmv_few<6>) tf_tile_qmv_few<6>;
template [[host_name("tf_tile_qmv_few_q8")]] [[kernel]] decltype(tf_tile_qmv_few<8>) tf_tile_qmv_few<8>;
template [[host_name("tf_tile_qmv_indexed_q4")]] [[kernel]] decltype(tf_tile_qmv_indexed<4>) tf_tile_qmv_indexed<4>;
template [[host_name("tf_tile_qmv_indexed_q6")]] [[kernel]] decltype(tf_tile_qmv_indexed<6>) tf_tile_qmv_indexed<6>;
template [[host_name("tf_tile_qmv_indexed_q8")]] [[kernel]] decltype(tf_tile_qmv_indexed<8>) tf_tile_qmv_indexed<8>;
template [[host_name("tf_tile_qmv_glu_q4")]] [[kernel]] decltype(tf_tile_qmv_glu<4>) tf_tile_qmv_glu<4>;
template [[host_name("tf_tile_qmv_glu_q6")]] [[kernel]] decltype(tf_tile_qmv_glu<6>) tf_tile_qmv_glu<6>;
template [[host_name("tf_tile_qmv_glu_q8")]] [[kernel]] decltype(tf_tile_qmv_glu<8>) tf_tile_qmv_glu<8>;
template [[host_name("tf_tile_qmv_grouped_q4")]] [[kernel]]
    decltype(tf_tile_qmv_grouped<4, false>) tf_tile_qmv_grouped<4, false>;
template [[host_name("tf_tile_qmv_grouped_q6")]] [[kernel]]
    decltype(tf_tile_qmv_grouped<6, false>) tf_tile_qmv_grouped<6, false>;
template [[host_name("tf_tile_qmv_grouped_q8")]] [[kernel]]
    decltype(tf_tile_qmv_grouped<8, false>) tf_tile_qmv_grouped<8, false>;
template [[host_name("tf_tile_qmv_grouped_glu_q4")]] [[kernel]]
    decltype(tf_tile_qmv_grouped<4, true>) tf_tile_qmv_grouped<4, true>;
template [[host_name("tf_tile_qmv_grouped_glu_q6")]] [[kernel]]
    decltype(tf_tile_qmv_grouped<6, true>) tf_tile_qmv_grouped<6, true>;
template [[host_name("tf_tile_qmv_grouped_glu_q8")]] [[kernel]]
    decltype(tf_tile_qmv_grouped<8, true>) tf_tile_qmv_grouped<8, true>;
