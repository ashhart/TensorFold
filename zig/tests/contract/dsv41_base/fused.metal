// The base's fused per-slot kernels, renamed tf_base_*: a test appends them to today's source to compare bytes.

// glue_tmp's rmsnorm, rope, actq and scatter of a window key row (dim 512) in a group of 256, into D[reps r + k].
[[kernel, max_total_threads_per_threadgroup(256)]] void tf_base_dsv41_f_kv(
    const device bfloat* X [[buffer(0)]], const device float* Wt [[buffer(1)]], device bfloat* KVN [[buffer(2)]],
    device bfloat* KVR [[buffer(3)]], device bfloat* KVQ [[buffer(4)]], device bfloat* WIN [[buffer(5)]],
    const device int* P [[buffer(6)]], const device float* CS [[buffer(7)]], const device int* D [[buffer(8)]],
    constant TfFKvArgs& a [[buffer(9)]], uint r [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8];
  const int pos = P[r];
  if (pos < 0) return;
  const device bfloat* x = X + ulong(r) * a.dim;
  const float y0 = tf_f_in(float(x[t])), y1 = tf_f_in(float(x[t + 256u]));
  float s = 0.0f;
  s = s + y0 * y0;
  s = s + y1 * y1;
  const float tot = tf_f_sum(s, part, lane, sg);
  const float inv = precise::rsqrt(tot / float(a.dim) + a.eps);
  for (uint j = 0; j < 2u; ++j) {
    const uint i = t + 256u * j;
    const float kn = float(bfloat(Wt[i] * ((j == 0 ? y0 : y1) * inv)));
    KVN[ulong(r) * a.dim + i] = bfloat(kn);
    const float other = simd_shuffle_xor(kn, ushort(1));
    float kr = kn;
    if (i >= a.dim - 64u) {
      const uint pr = (i - (a.dim - 64u)) / 2u;
      const float c = CS[ulong(pos) * 64u + pr], sn = CS[ulong(pos) * 64u + 32u + pr];
      kr = (i & 1u) == 0u ? float(bfloat(kn * c + (TF_CTL == 2 ? other * sn : -other * sn))) : float(bfloat(other * sn + kn * c)); // control 2: the rotation's sign lost
    }
    KVR[ulong(r) * a.dim + i] = bfloat(kr);
    const bfloat q = tf_f_actq(kr);
    KVQ[ulong(r) * a.dim + i] = q;
    for (uint k = 0; k < a.reps; ++k) {
      const int d = D[a.reps * r + k];
      if (d >= 0) WIN[ulong(d) * a.dim + i] = q;
    }
  }
  if (TF_CTL == 4 && r == 0 && t == 0) WIN[ulong(a.rows) * a.dim] = bfloat(1.0f); // the first bf16 past the ring
}
