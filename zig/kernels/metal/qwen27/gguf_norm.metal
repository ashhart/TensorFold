// GGUF's residual RMSNorm with FP32 gains; compiled after glue.metal's q27_read, q27_round and q27_write.
struct Norm { uint width, rows, activation, gain, residual; float eps; };

// Each thread owns 16 contiguous values, then simd_sum and simdgroups in increasing order as row_glue.add_norm.
[[max_total_threads_per_threadgroup(1024)]]
kernel void q27_norm(device const uchar* X [[buffer(0)]], device const uchar* R [[buffer(1)]],
                     device const uchar* W [[buffer(2)]], constant Norm& a [[buffer(3)]],
                     device uchar* H [[buffer(4)]], device uchar* Y [[buffer(5)]],
                     uint2 group [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[32];
  const uint row = group.y, base = row * a.width + 16 * t;
  float values[16], square = 0.0f;
  for (uint j = 0; j < 16; ++j) {
    float h = q27_read(X, base + j, a.activation);
    if (a.residual) h = q27_round(h + q27_read(R, base + j, a.activation), a.activation);
    values[j] = h;
    q27_write(H, base + j, h, a.activation);
    square = fma(h, h, square);
  }
  square = simd_sum(square);
  if (lane == 0) sums[sg] = square;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < a.width / 512; ++s) total += sums[s];
  const float inv = rsqrt(total / float(a.width) + a.eps);
  for (uint j = 0; j < 16; ++j) q27_write(Y, base + j, q27_read(W, 16 * t + j, a.gain) * (values[j] * inv), a.activation);
}
