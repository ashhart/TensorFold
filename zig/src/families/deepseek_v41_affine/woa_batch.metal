constant constexpr int WoInput = 4096;
constant constexpr int WoOutput = 1024;
constant constexpr int WoGroups = 8;
constant constexpr int WoThreads = 2;

template <int COLS, int ROWS>
inline void tf_woa_tile(const device bfloat* X, const device uint8_t* W, const device bfloat* S,
                        const device bfloat* B, device bfloat* OUT, uint first, uint group,
                        uint column, uint lane) {
  const size_t wrow = size_t(group) * WoOutput + column;
  const device uint8_t* words = W + wrow * (WoInput / 2);
  const device bfloat* scales = S + wrow * (WoInput / 64);
  const device bfloat* biases = B + wrow * (WoInput / 64);
  TfRpRows<ROWS> input;
  for (int i = 0; i < ROWS; ++i)
    input.x[i] = X + (size_t(first + i) * WoGroups + group) * WoInput;
  float values[ROWS][COLS];
  tf_rp_rows_at<COLS, ROWS, 4, false>(words, scales, biases, input, WoInput, 64, lane, values);
  if (lane != 0) return;
  for (int i = 0; i < ROWS; ++i) {
    device bfloat* out = OUT + (size_t(first + i) * WoGroups + group) * WoOutput + column;
    for (int j = 0; j < COLS; ++j) out[j] = bfloat(values[i][j]);
  }
}

template <int COLS, int ROWS>
[[kernel]] void tf_woa(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                        const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                        device bfloat* OUT [[buffer(4)]], constant uint& rows [[buffer(5)]],
                        uint3 grid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint first = grid.z * ROWS;
  const uint column = (grid.x * WoThreads + sg) * COLS;
  if (ROWS == 3 && rows - first >= 3)
    tf_woa_tile<COLS, 3>(X, W, S, B, OUT, first, grid.y, column, lane);
  else if (ROWS >= 2 && rows - first >= 2)
    tf_woa_tile<COLS, 2>(X, W, S, B, OUT, first, grid.y, column, lane);
  else tf_woa_tile<COLS, 1>(X, W, S, B, OUT, first, grid.y, column, lane);
}

template [[host_name("tf_woa_c4_r1")]] [[kernel]] decltype(tf_woa<4, 1>) tf_woa<4, 1>;
template [[host_name("tf_woa_c2_r2")]] [[kernel]] decltype(tf_woa<2, 2>) tf_woa<2, 2>;
template [[host_name("tf_woa_c4_r2")]] [[kernel]] decltype(tf_woa<4, 2>) tf_woa<4, 2>;
template [[host_name("tf_woa_c8_r2")]] [[kernel]] decltype(tf_woa<8, 2>) tf_woa<8, 2>;
template [[host_name("tf_woa_c4_r3")]] [[kernel]] decltype(tf_woa<4, 3>) tf_woa<4, 3>;

[[kernel]] void tf_woa_audit(const device ushort* actual [[buffer(0)]],
                              const device ushort* expected [[buffer(1)]],
                              device atomic_uint* mismatch [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
  if (actual[i] != expected[i]) atomic_fetch_or_explicit(mismatch, 1u, memory_order_relaxed);
}
