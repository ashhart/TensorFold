
  // Threadgroup (h, r, p): query head h of row r over part p of the row's key list (SPARSE[r]: the NK[r] ids
  // IDS[r]; else keys 0 .. NK[r] - 1), entries [p n / P, (p + 1) n / P); 8 simdgroups, simdgroup g taking every
  // 8th entry from the part's start, a lane D / 32 dims. fp32: scores q . k with q pre-scaled, an online softmax
  // per simdgroup, the simdgroups combined in order into the part's (max, sum, output).
  constexpr int PER = D / 32;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int part = int(threadgroup_position_in_grid.z);
  const int kvh = h / (H / KVH);
  const int n = NK[r];
  const int lo = int((long(part) * n) / P), hi = int((long(part + 1) * n) / P);
  const bool sparse = SPARSE[r] != 0;
  const size_t cap = size_t(Kc_shape[2]);
  const device bfloat* kb = Kc + size_t(kvh) * cap * D + lane * PER;
  const device bfloat* vb = Vc + size_t(kvh) * cap * D + lane * PER;
  const auto ids = IDS + size_t(r) * IDS_shape[1];      // device, or constant when MLX binds a small array so
  const device bfloat* qp = Q + (size_t(r) * H + h) * D + lane * PER;
  float q[PER], o[PER];
  for (int i = 0; i < PER; i++) { q[i] = SCALE[0] * float(qp[i]); o[i] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  for (int j = lo + int(g); j < hi; j += 8) {
    const size_t key = size_t(sparse ? ids[j] : j) * D;
    float sc = 0.0f;
    for (int i = 0; i < PER; i++) sc = fma(q[i], float(kb[key + i]), sc);
    sc = simd_sum(sc);
    const float mn = metal::max(m, sc);
    const float f = metal::precise::exp(m - mn), e = metal::precise::exp(sc - mn);
    l = fma(l, f, e);
    for (int i = 0; i < PER; i++) o[i] = fma(e, float(vb[key + i]), o[i] * f);
    m = mn;
  }
  threadgroup float ms[8], ls[8];
  threadgroup float tile[8][D];
  if (lane == 0) { ms[g] = m; ls[g] = l; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float top = -INFINITY;
  for (int k = 0; k < 8; k++) top = metal::max(top, ms[k]);
  const float mine = m == -INFINITY ? 0.0f : metal::precise::exp(m - top);
  for (int i = 0; i < PER; i++) tile[g][lane * PER + i] = o[i] * mine;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const size_t at = (size_t(r) * H + h) * P + part;
  for (int d = int(thread_position_in_threadgroup.x); d < D; d += 256) {
    float acc = 0.0f;
    for (int k = 0; k < 8; k++) acc += tile[k][d];
    PO[at * D + d] = acc;
  }
  if (thread_position_in_threadgroup.x == 0) {
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += ms[k] == -INFINITY ? 0.0f : ls[k] * metal::precise::exp(ms[k] - top);
    PM[at * 2] = top;
    PM[at * 2 + 1] = total;
  }
