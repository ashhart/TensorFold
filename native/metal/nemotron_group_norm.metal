
  // one threadgroup of GS / 4 threads per (row, group): thread t owns 4 consecutive elements
  const uint t = thread_position_in_threadgroup.x;
  const uint grp = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  constexpr int T = GS / 4;
  threadgroup float partial[T / 32];
  const int base = int(r) * XD + int(grp) * GS + int(t) * 4;
  float v[4];
  float ss = 0.0f;
  for (int i = 0; i < 4; i++) { v[i] = float(X[base + i]); ss = fma(v[i], v[i], ss); }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::precise::rsqrt(total / float(GS) + eps[0]);
  for (int i = 0; i < 4; i++) {
    const int c = int(grp) * GS + int(t) * 4 + i;
    OUT[base + i] = bfloat(float(W[c]) * float(bfloat(v[i] * scale)));
  }
