
  // Threadgroup (s, r), 256 threads: stream s of row r. keys = norm(key projection), queries = norm(streams), each
  // (1 + w) RMSNorm in fp32 to bf16; gate = sum of bf16(key * query) (fp32, bf16), / bf16(sqrt D), signed sqrt,
  // sigmoid (bf16 each); gated = bf16(sigmoid * value); normed = the conv norm of gated. Sums: each thread's dims in
  // order, simd_sum, the simdgroups in order.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
  const int s = int(threadgroup_position_in_grid.x), r = int(threadgroup_position_in_grid.y);
  constexpr int W = S * D, PER = D / 256;
  threadgroup float red[3][8];
  float k[PER], q[PER], v[PER];
  float sk = 0.0f, sq = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int d = int(t) + 256 * i;
    k[i] = float(KV[size_t(r) * (W + D) + s * D + d]);
    q[i] = float(H[size_t(r) * W + s * D + d]);
    v[i] = float(KV[size_t(r) * (W + D) + W + d]);
    sk = fma(k[i], k[i], sk);
    sq = fma(q[i], q[i], sq);
  }
  sk = simd_sum(sk); sq = simd_sum(sq);
  if (lane == 0) { red[0][sg] = sk; red[1][sg] = sq; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tk = 0.0f, tq = 0.0f;
  for (int j = 0; j < 8; j++) { tk += red[0][j]; tq += red[1][j]; }
  const float rk = metal::rsqrt(tk / float(D) + eps[0]), rq = metal::rsqrt(tq / float(D) + eps[0]);
  float dot = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    const float kn = float(bfloat((k[i] * rk) * KS[e])), qn = float(bfloat((q[i] * rq) * QS[e]));
    dot += float(bfloat(kn * qn));
  }
  dot = simd_sum(dot);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[2][sg] = dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gd = 0.0f;
  for (int j = 0; j < 8; j++) gd += red[2][j];
  const float g1 = float(bfloat(float(bfloat(gd)) / float(bfloat(metal::precise::sqrt(float(D))))));
  const float root = float(bfloat(metal::precise::sqrt(metal::max(metal::abs(g1), 1e-6f))));
  const float g2 = float(bfloat(metal::sign(g1) * root));
  const float sig = bsig(g2);
  float sc = 0.0f;
  for (int i = 0; i < PER; i++) {
    v[i] = float(bfloat(sig * v[i]));
    sc = fma(v[i], v[i], sc);
  }
  sc = simd_sum(sc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[0][sg] = sc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tc = 0.0f;
  for (int j = 0; j < 8; j++) tc += red[0][j];
  const float rc = metal::rsqrt(tc / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int e = s * D + int(t) + 256 * i;
    GATED[size_t(r) * W + e] = bfloat(v[i]);
    NORMED[size_t(r) * W + e] = bfloat((v[i] * rc) * CS[e]);
  }
