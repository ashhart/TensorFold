// Four gathered rows share a pass over the weights, keeping glm_rows_sorted's per-row reduction and output rounding.
struct GlmGatherDims {
  int k, n, group, bits;
  int up_group, up_bits, pairs, repeat;
  int experts, ldo;
  float lim;
};

// Compact expert tiles, using the shared affine gather's SIMD prefix algorithm without matrix arithmetic.
inline bool glm_gather_tile(const device int* offsets, int experts, int total, int tile, uint lane,
                            thread int& expert, thread int& row, thread int& rows) {
  int before = 0;
  for (int base = 0; base < experts; base += 32) {
    const int e = base + int(lane);
    const int first = e < experts ? offsets[e] : total, last = e + 1 < experts ? offsets[e + 1] : total;
    const int count = (last - first + 3) / 4;
    const int upto = before + simd_prefix_inclusive_sum(count);
    const int owner = simd_sum(int(upto <= tile));
    if (owner < 32) {
      expert = base + owner;
      row = simd_shuffle(first, ushort(owner)) + (tile - simd_shuffle(upto - count, ushort(owner))) * 4;
      rows = min(4, simd_shuffle(last, ushort(owner)) - row);
      return true;
    }
    before = simd_shuffle(upto, ushort(31));
  }
  return false;
}

template <int BITS>
inline void glm_gather_dot(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                            thread const TfRpRows<4>& x, int k, int group, size_t at, uint lane,
                            thread float (*acc)[4]) {
  tf_rp_rows_at<4, 4, BITS, false>((const device uint8_t*)W + at * (k * BITS / 8),
                                  S + at * (k / group), B + at * (k / group), x, k, group, lane, acc);
}

// Each projection selects its width independently; group size is an operand of the shared row reduction.
inline void glm_gather_width(const device uint32_t* W, const device bfloat* S, const device bfloat* B,
                              thread const TfRpRows<4>& x, int bits, int k, int group, size_t at, uint lane,
                              thread float (*acc)[4]) {
  switch (bits) {
    case 2: glm_gather_dot<2>(W, S, B, x, k, group, at, lane, acc); break;
    case 3: glm_gather_dot<3>(W, S, B, x, k, group, at, lane, acc); break;
    case 4: glm_gather_dot<4>(W, S, B, x, k, group, at, lane, acc); break;
    case 5: glm_gather_dot<5>(W, S, B, x, k, group, at, lane, acc); break;
    case 6: glm_gather_dot<6>(W, S, B, x, k, group, at, lane, acc); break;
    case 8: glm_gather_dot<8>(W, S, B, x, k, group, at, lane, acc); break;
  }
}

template <bool GLU>
[[max_total_threads_per_threadgroup(64)]]
[[kernel]] void glm_rows_gather(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                                const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                                constant GlmGatherDims& d [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                const device int* offsets [[buffer(6)]], const device int* order [[buffer(7)]],
                                const device uint32_t* W2 [[buffer(8)]], const device bfloat* S2 [[buffer(9)]],
                                const device bfloat* B2 [[buffer(10)]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const int col = (int(tg.x) * 2 + int(sg)) * 4;
  if (col >= d.n) return;
  int expert, row, rows;
  if (!glm_gather_tile(offsets, d.experts, d.pairs, int(tg.y), lane, expert, row, rows)) return;
  TfRpRows<4> x;
  for (int b = 0; b < 4; b++) {
    const int at = row + (b < rows ? b : 0);
    const int src = GLU ? order[at] / d.repeat : at;
    x.x[b] = X + size_t(src) * d.k;
  }
  const size_t at = size_t(expert) * d.n + col;
  float acc[4][4];
  glm_gather_width(W, S, B, x, d.bits, d.k, d.group, at, lane, acc);
  if (GLU) {
    float up[4][4];
    glm_gather_width(W2, S2, B2, x, d.up_bits, d.k, d.up_group, at, lane, up);
    if (lane == 0) for (int b = 0; b < rows; b++) for (int j = 0; j < min(4, d.n - col); j++)
      OUT[size_t(row + b) * d.ldo + col + j] = glm_rows_glu(acc[b][j], up[b][j], d.lim);
  } else if (lane == 0) {
    for (int b = 0; b < rows; b++) for (int j = 0; j < min(4, d.n - col); j++)
      OUT[size_t(order[row + b]) * d.ldo + col + j] = bfloat(acc[b][j]);
  }
}

template [[host_name("glm_gather_glu")]] [[kernel]] decltype(glm_rows_gather<true>) glm_rows_gather<true>;
template [[host_name("glm_gather_down")]] [[kernel]] decltype(glm_rows_gather<false>) glm_rows_gather<false>;
