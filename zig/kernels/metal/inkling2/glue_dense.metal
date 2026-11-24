// Multi-segment decode projections matching core/bf16_matvec.metal's exact 0-ULP summation order.
#pragma METAL fp contract(on)

struct InkDenseArgs { uint rows, k; uint end[6]; };

template <uint ROWS, bool EXACT>
inline void tf_ink2_dense_k(
    const device bfloat* X, const device bfloat* w,
    thread float (&acc)[ROWS], uint rows, uint k, uint lane) {
  for (uint i = 8 * lane; i < k; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i));
    const float4 w1 = float4(*(const device bfloat4*)(w + i + 4));
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      const uint xr = EXACT ? r : (r < rows ? r : 0);
      const device bfloat* x = X + (size_t)xr * k + i;
      acc[r] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));
    }
  }
}

template <uint ROWS>
inline void tf_ink2_dense_core(
    const device bfloat* X, const device bfloat* W, device bfloat* Y,
    uint rows, uint j, uint k, uint n_local, uint lane) {
  const device bfloat* w = W + (size_t)j * k;
  float acc[ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  if (ROWS == 1 || rows >= ROWS) {
    tf_ink2_dense_k<ROWS, true>(X, w, acc, rows, k, lane);
  } else {
    tf_ink2_dense_k<ROWS, false>(X, w, acc, rows, k, lane);
  }
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r < rows) Y[(size_t)r * n_local + j] = bfloat(s);
  }
}

template <uint ROWS>
[[kernel, max_total_threads_per_threadgroup(256)]]
void tf_ink2_dense4_t(
    const device bfloat* X [[buffer(0)]],
    const device bfloat* W0 [[buffer(1)]], const device bfloat* W1 [[buffer(2)]],
    const device bfloat* W2 [[buffer(3)]], const device bfloat* W3 [[buffer(4)]],
    device bfloat* Y0 [[buffer(5)]], device bfloat* Y1 [[buffer(6)]],
    device bfloat* Y2 [[buffer(7)]], device bfloat* Y3 [[buffer(8)]],
    constant InkDenseArgs& a [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg.x * 8 + sg;
  if (g >= a.end[3]) return;
  uint seg = 0;
  _Pragma("clang loop unroll(full)")
  for (uint s = 0; s < 3; ++s) if (g >= a.end[s]) seg = s + 1;
  const uint prev = seg == 0 ? 0 : a.end[seg - 1];
  const uint j = g - prev, n_local = a.end[seg] - prev;
  const device bfloat* W = seg == 0 ? W0 : seg == 1 ? W1 : seg == 2 ? W2 : W3;
  device bfloat* Y = seg == 0 ? Y0 : seg == 1 ? Y1 : seg == 2 ? Y2 : Y3;
  tf_ink2_dense_core<ROWS>(X, W, Y, a.rows, j, a.k, n_local, lane);
}

template <uint ROWS>
[[kernel, max_total_threads_per_threadgroup(256)]]
void tf_ink2_dense6_t(
    const device bfloat* Xq [[buffer(0)]], const device bfloat* Xk [[buffer(1)]],
    const device bfloat* Xv [[buffer(2)]],
    const device bfloat* W0 [[buffer(3)]], const device bfloat* W1 [[buffer(4)]],
    const device bfloat* W2 [[buffer(5)]], const device bfloat* W3 [[buffer(6)]],
    const device bfloat* W4 [[buffer(7)]], const device bfloat* W5 [[buffer(8)]],
    device bfloat* Y0 [[buffer(9)]], device bfloat* Y1 [[buffer(10)]],
    device bfloat* Y2 [[buffer(11)]], device bfloat* Y3 [[buffer(12)]],
    device bfloat* Y4 [[buffer(13)]], device bfloat* Y5 [[buffer(14)]],
    constant InkDenseArgs& a [[buffer(15)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tg.x * 8 + sg;
  if (g >= a.end[5]) return;
  uint seg = 0;
  _Pragma("clang loop unroll(full)")
  for (uint s = 0; s < 5; ++s) if (g >= a.end[s]) seg = s + 1;
  const uint prev = seg == 0 ? 0 : a.end[seg - 1];
  const uint j = g - prev, n_local = a.end[seg] - prev;
  const device bfloat* X = seg == 4 ? Xk : seg == 5 ? Xv : Xq;
  const device bfloat* W = W0;
  device bfloat* Y = Y0;
  switch (seg) {
    case 0: W = W0; Y = Y0; break;
    case 1: W = W1; Y = Y1; break;
    case 2: W = W2; Y = Y2; break;
    case 3: W = W3; Y = Y3; break;
    case 4: W = W4; Y = Y4; break;
    default: W = W5; Y = Y5; break;
  }
  tf_ink2_dense_core<ROWS>(X, W, Y, a.rows, j, a.k, n_local, lane);
}

#define TF_INK2_DENSE4_INST(ROWS, NAME)                                                                            \
  template [[host_name(NAME)]] kernel void tf_ink2_dense4_t<ROWS>(                                                 \
      const device bfloat*, const device bfloat*, const device bfloat*, const device bfloat*, const device bfloat*, \
      device bfloat*, device bfloat*, device bfloat*, device bfloat*, constant InkDenseArgs&, uint3, uint, uint);
#define TF_INK2_DENSE6_INST(ROWS, NAME)                                                                            \
  template [[host_name(NAME)]] kernel void tf_ink2_dense6_t<ROWS>(                                                 \
      const device bfloat*, const device bfloat*, const device bfloat*, const device bfloat*, const device bfloat*, \
      const device bfloat*, const device bfloat*, const device bfloat*, const device bfloat*, device bfloat*,       \
      device bfloat*, device bfloat*, device bfloat*, device bfloat*, device bfloat*, constant InkDenseArgs&,       \
      uint3, uint, uint);

TF_INK2_DENSE4_INST(1, "tf_ink2_dense4_1")
TF_INK2_DENSE4_INST(2, "tf_ink2_dense4_2")
TF_INK2_DENSE4_INST(4, "tf_ink2_dense4_4")
TF_INK2_DENSE4_INST(8, "tf_ink2_dense4_8")
TF_INK2_DENSE4_INST(16, "tf_ink2_dense4_16")
TF_INK2_DENSE4_INST(32, "tf_ink2_dense4_32")

TF_INK2_DENSE6_INST(1, "tf_ink2_dense6_1")
TF_INK2_DENSE6_INST(2, "tf_ink2_dense6_2")
TF_INK2_DENSE6_INST(4, "tf_ink2_dense6_4")
TF_INK2_DENSE6_INST(8, "tf_ink2_dense6_8")
TF_INK2_DENSE6_INST(16, "tf_ink2_dense6_16")
TF_INK2_DENSE6_INST(32, "tf_ink2_dense6_32")
