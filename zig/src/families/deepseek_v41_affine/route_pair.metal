// Pair only slots that already chose the same existing expert. Store back to the original slots.
[[kernel]] void tf_pair_expert_slots(const device uint* ids [[buffer(0)]],
                                    device int* mates [[buffer(1)]],
                                    constant uint& slots [[buffer(2)]],
                                    uint slot [[thread_position_in_grid]]) {
  const uint expert = ids[slot];
  uint ordinal = 0;
  for (uint i = 0; i < slot; ++i) ordinal += uint(ids[i] == expert);
  if (ordinal & 1) {
    mates[slot] = -1;
    return;
  }
  for (uint i = slot + 1; i < slots; ++i) {
    if (ids[i] == expert) {
      mates[slot] = int(i);
      return;
    }
  }
  mates[slot] = int(slot);
}

// For a small rectangular grid, derive each tile's partners from the actual

// router IDs. The masks follow slot order so scatters keep their original slots.
static int tf_take_route(thread uint& lo, thread uint& hi, int missing) {
  if (lo != 0) {
    const int slot = int(ctz(lo));
    lo &= lo - 1;
    return slot;
  }
  if (hi != 0) {
    const int slot = 32 + int(ctz(hi));
    hi &= hi - 1;
    return slot;
  }
  return missing;
}

template <int CAP>
static int2 tf_route_peers(const device uint* ids, int slots, uint lane, int slot) {
  const uint expert = ids[slot];
  uint lo = simd_or(int(lane) < slots && ids[lane] == expert ? 1u << lane : 0u);
  uint hi = slots > 32 ? simd_or(int(lane + 32) < slots && ids[lane + 32] == expert ? 1u << lane : 0u) : 0u;
  const uint before = (1u << uint(slot & 31)) - 1;
  const bool low_slot = slot < 32;
  const uint rank = low_slot ? popcount(lo & before) : popcount(lo) + popcount(hi & before);
  const uint group = CAP == 0 ? 1 : CAP;
  if (rank % group != 0) return int2(-1);
  const uint own = 1u << uint(slot & 31);
  lo = low_slot ? lo & ~before & ~own : 0;
  if (!low_slot) hi &= ~before & ~own;
  const int second = tf_take_route(lo, hi, slot);
  const int third = CAP == 3 ? tf_take_route(lo, hi, slot) : slot;
  return int2(second, third);
}

template <int BITS, int INLINE_CAP = 0, int FIXED_K = 0, int FIXED_N = 0>
[[kernel]] void tf_row_projection_pair(
    const device bfloat* X [[buffer(0)]],
    const device uint32_t* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]],
    const device bfloat* B [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    device bfloat* OUT [[buffer(5)]],
    const device uint* ids [[buffer(6)]],
    const device int* mates [[buffer(7)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 block [[threadgroup_position_in_grid]]) {
  const int slot = int(block.y);
  int2 others;
  if (INLINE_CAP == 0) others = int2(mates[slot], slot);
  else others = tf_route_peers<INLINE_CAP>(ids, dims[3], lane, slot);
  const int mate = others.x, third = others.y;
  if (mate < 0) return;
  constexpr bool fixed = FIXED_K != 0;
  const int K = fixed ? FIXED_K : dims[0], N = fixed ? FIXED_N : dims[1];
  const int GS = fixed ? 64 : dims[2], repeat = fixed ? (FIXED_K == 5120 ? 6 : 1) : dims[5];
  const int columns = 4, row0 = (int(block.x) * 2 + int(sg)) * columns;
  const uint expert = ids[slot];
  device bfloat* y0 = OUT + size_t(slot) * N + row0;
  device bfloat* y1 = OUT + size_t(mate) * N + row0;
  if (expert >= uint(dims[4])) {
    if (lane < uint(columns)) {
      y0[lane] = bfloat(0.0f);
      if (mate != slot) y1[lane] = bfloat(0.0f);
      if (INLINE_CAP == 3 && third != slot) OUT[size_t(third) * N + row0 + lane] = bfloat(0.0f);
    }
    return;
  }
  const size_t at = size_t(expert) * N + row0;
  const device uint8_t* wr = (const device uint8_t*)W + at * (K * BITS / 8);
  const device bfloat* sr = S + at * (K / GS);
  const device bfloat* br = B + at * (K / GS);
  const device bfloat* x0 = X + size_t(slot / repeat) * K;
  if (INLINE_CAP == 3 && third != slot) {
    device bfloat* y2 = OUT + size_t(third) * N + row0;
    TfRpRows<3> inputs = { x0, X + size_t(mate / repeat) * K, X + size_t(third / repeat) * K };
    float acc[3][columns];
    tf_rp_rows_at<columns, 3, BITS, false, fixed>(wr, sr, br, inputs, K, GS, lane, acc);
    if (lane == 0) for (int j = 0; j < columns; ++j) {
      y0[j] = bfloat(acc[0][j]);
      y1[j] = bfloat(acc[1][j]);
      y2[j] = bfloat(acc[2][j]);
    }
  } else if (mate == slot) {
    float acc[1][columns];
    tf_rp_rows<columns, 1, BITS, false, fixed>(wr, sr, br, x0, K, GS, lane, acc);
    if (lane == 0) for (int j = 0; j < columns; ++j) y0[j] = bfloat(acc[0][j]);
  } else {
    TfRpRows<2> inputs = { x0, X + size_t(mate / repeat) * K };
    float acc[2][columns];
    tf_rp_rows_at<columns, 2, BITS, false, fixed>(wr, sr, br, inputs, K, GS, lane, acc);
    if (lane == 0) for (int j = 0; j < columns; ++j) {
      y0[j] = bfloat(acc[0][j]);
      y1[j] = bfloat(acc[1][j]);
    }
  }
}

template [[host_name("tf_row_projection_pair_q2")]] [[kernel]] decltype(tf_row_projection_pair<2>) tf_row_projection_pair<2>;
template [[host_name("tf_row_projection_pair_q3")]] [[kernel]] decltype(tf_row_projection_pair<3>) tf_row_projection_pair<3>;

template [[host_name("tf_row_projection_inline2_q2")]] [[kernel]] decltype(tf_row_projection_pair<2, 2>) tf_row_projection_pair<2, 2>;

template [[host_name("tf_row_projection_inline2_q3")]] [[kernel]] decltype(tf_row_projection_pair<3, 2>) tf_row_projection_pair<3, 2>;

template [[host_name("tf_row_projection_inline3_q2")]] [[kernel]] decltype(tf_row_projection_pair<2, 3>) tf_row_projection_pair<2, 3>;

template [[host_name("tf_row_projection_inline3_q3")]] [[kernel]] decltype(tf_row_projection_pair<3, 3>) tf_row_projection_pair<3, 3>;

template [[host_name("tf_row_shape_gate_q2")]] [[kernel]] decltype(tf_row_projection_pair<2, 2, 5120, 2304>) tf_row_projection_pair<2, 2, 5120, 2304>;

template [[host_name("tf_row_shape_down_q2")]] [[kernel]] decltype(tf_row_projection_pair<2, 2, 2304, 5120>) tf_row_projection_pair<2, 2, 2304, 5120>;

template [[host_name("tf_row_shape_gate_q3")]] [[kernel]] decltype(tf_row_projection_pair<3, 2, 5120, 2304>) tf_row_projection_pair<3, 2, 5120, 2304>;

template [[host_name("tf_row_shape_down_q3")]] [[kernel]] decltype(tf_row_projection_pair<3, 2, 2304, 5120>) tf_row_projection_pair<3, 2, 2304, 5120>;
