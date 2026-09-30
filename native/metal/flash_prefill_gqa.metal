
  // Threadgroup (kvh, r, part): KV head kvh's G query heads, HS a simdgroup (each key and value read from threadgroup
  // memory serves HS heads), over one part of row r's key list. Each head's sums run as with one head a simdgroup.
  constexpr int G = H / KVH;
  constexpr int PER = D / 32;                 // 8 output dims a lane
  constexpr int HALF = D / 2;
  constexpr int KP = D + 8;                   // padded key rows, 16-byte aligned
  constexpr int VEC = D / 8;                  // 16-byte vectors a row
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int t = int(thread_position_in_threadgroup.x);
  const int nt = int(threads_per_threadgroup.x);
  const int kvh = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int part = int(threadgroup_position_in_grid.z);
  const int n = NK[r];
  const int lo = int((long(part) * n) / P), hi = int((long(part + 1) * n) / P);
  const bool sparse = SPARSE[r] != 0;
  const size_t cap = size_t(Kc_shape[2]);
  const device uint4* kbase = (const device uint4*)(Kc + size_t(kvh) * cap * D);
  const device uint4* vbase = (const device uint4*)(Vc + size_t(kvh) * cap * D);
  const auto ids = IDS + size_t(r) * IDS_shape[1];
  threadgroup float4 qs[G][D / 4];
  threadgroup uint4 ks[TK][KP / 8];
  threadgroup uint4 vs[TK][VEC];
  for (int j = 0; j < HS; j++) {
    const int g = int(sg) * HS + j;
    const device bfloat* qp = Q + (size_t(r) * H + kvh * G + g) * D;
    for (int i = int(lane); i < D / 4; i += 32) {
      const float s0 = SCALE[0];
      qs[g][i] = float4(s0 * float(qp[4 * i]), s0 * float(qp[4 * i + 1]), s0 * float(qp[4 * i + 2]), s0 * float(qp[4 * i + 3]));
    }
  }
  float o[HS][PER];
  float m[HS], l[HS];
  for (int j = 0; j < HS; j++) {
    for (int i = 0; i < PER; i++) o[j][i] = 0.0f;
    m[j] = -INFINITY; l[j] = 0.0f;
  }
  const int kt = int(lane) / 2, hf = int(lane) & 1;   // lanes 2k and 2k + 1 score the tile's key k, half each
  for (int base = lo; base < hi; base += TK) {
    const int cnt = metal::min(TK, hi - base);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = t; e < cnt * VEC; e += nt) {
      const int k = e / VEC, c = e - k * VEC;
      const int jj = base + k;
      const size_t row = size_t(sparse ? ids[jj] : jj) * VEC;
      ks[k][c] = kbase[row + c];
      vs[k][c] = vbase[row + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int kk = metal::min(kt, cnt - 1);
    const threadgroup bfloat4* kr = (const threadgroup bfloat4*)(&ks[kk][0]) + hf * (HALF / 4);
    float a[HS];
    for (int j = 0; j < HS; j++) a[j] = 0.0f;
    for (int i = 0; i < HALF / 4; i++) {
      const float4 kv = float4(kr[i]);
      for (int j = 0; j < HS; j++) {
        const float4 qv = qs[int(sg) * HS + j][hf * (HALF / 4) + i];
        a[j] = fma(qv.x, kv.x, a[j]); a[j] = fma(qv.y, kv.y, a[j]);
        a[j] = fma(qv.z, kv.z, a[j]); a[j] = fma(qv.w, kv.w, a[j]);
      }
    }
    const bool live = kt < cnt;
    float e[HS];
    for (int j = 0; j < HS; j++) {
      const float aj = a[j] + simd_shuffle_xor(a[j], ushort(1));
      const float sc = live ? aj : -INFINITY;
      const float mn = metal::max(m[j], simd_max(sc));
      const float f = metal::exp(m[j] - mn);
      e[j] = (live && hf == 0) ? metal::exp(sc - mn) : 0.0f;
      l[j] = fma(l[j], f, simd_sum(e[j]));
      for (int i = 0; i < PER; i++) o[j][i] *= f;
      m[j] = mn;
    }
    for (int k = 0; k < cnt; k++) {
      const threadgroup bfloat4* vr = (const threadgroup bfloat4*)(&vs[k][lane]);
      const float4 v0 = float4(vr[0]), v1 = float4(vr[1]);
      for (int j = 0; j < HS; j++) {
        const float ek = simd_shuffle(e[j], ushort(2 * k));
        o[j][0] = fma(ek, v0.x, o[j][0]); o[j][1] = fma(ek, v0.y, o[j][1]);
        o[j][2] = fma(ek, v0.z, o[j][2]); o[j][3] = fma(ek, v0.w, o[j][3]);
        o[j][4] = fma(ek, v1.x, o[j][4]); o[j][5] = fma(ek, v1.y, o[j][5]);
        o[j][6] = fma(ek, v1.z, o[j][6]); o[j][7] = fma(ek, v1.w, o[j][7]);
      }
    }
  }
  for (int j = 0; j < HS; j++) {
    const int h = kvh * G + int(sg) * HS + j;
    const size_t at = (size_t(r) * H + h) * P + part;
    for (int i = 0; i < PER; i++) PO[at * D + int(lane) * PER + i] = o[j][i];
    if (lane == 0) { PM[at * 2] = m[j]; PM[at * 2 + 1] = l[j]; }
  }
