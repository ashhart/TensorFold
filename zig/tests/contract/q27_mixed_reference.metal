// Frozen from supplied base ec344c2bab: the head norm before mixed-row input addressing.
struct Q27OldHead { uint width, rows, stride, activation, gain; float eps; };

// Head RMSNorm retains its native-input rounding before gain multiplication, as the shared Metal RMS implementation.
kernel void q27_head_norm_before_mixed(device const uchar* X [[buffer(0)]], device const uchar* W [[buffer(1)]],
                          constant Q27OldHead& a [[buffer(2)]], device uchar* Y [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[32], inverse[1];
  float square = 0.0f;
  uint first = 4 * t;
  while (first < a.width) {
    for (uint j = 0; j != 4; ++j) {
      const uint column = first + j;
      const float value = column < a.width ? q27_read(X, row * a.stride + column, a.activation) : 0.0f;
      square += value * value;
    }
    first += 4 * threads;
  }
  square = simd_sum(square);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float total = simd_sum(sums[lane]);
    if (lane == 0) inverse[0] = precise::rsqrt(total / float(a.width) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint column = 4 * t; column < a.width; column += 4 * threads) {
    for (uint j = 0; j != 4 && column + j < a.width; ++j) {
      const uint at = column + j;
      const float value = q27_read(X, row * a.stride + at, a.activation);
      const float normal = q27_round(value * inverse[0], a.activation);
      q27_write(Y, row * a.width + at, q27_read(W, at, a.gain) * normal, a.activation);
    }
  }
}
