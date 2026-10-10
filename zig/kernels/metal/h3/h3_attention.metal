// MiniMax H3 / FastH3 on Metal: tile-routed int8 attention with an online softmax across key tiles.

// softmax(q k^T) v over a query tile's key tiles: int8 scores and values, half weights. [query tiles, H] of 256.
[[kernel]] void h3_attention_tiles(
  const device int8_t* Q [[buffer(0)]],
  const device float* QS [[buffer(1)]],
  const device int8_t* K [[buffer(2)]],
  const device float* KS [[buffer(3)]],
  const device int8_t* V [[buffer(4)]],
  const device float* VS [[buffer(5)]],
  const device int32_t* IDX [[buffer(6)]],
  const device int32_t* SIZES [[buffer(7)]],
  const device int32_t* ROWOF [[buffer(8)]],
  const constant int32_t* P [[buffer(9)]],
  const constant float* SC [[buffer(10)]],
  device bfloat* Y [[buffer(11)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int D = 128, TQ = 64, KEYS = 64, TS = H3_SLOTS, H = H3P_HEADS;
  constexpr int TPR = 256 / TQ;
  constexpr int C1 = TQ * KEYS / 256, C2 = TQ * D / 256;
  constexpr float LIFT = 1024.0f;
  const int NQ = P[0], KSEL = P[1];
  const int head = int(tg.y), r0 = (P[2] + int(tg.x)) * TQ;
  const long hb = long(head) * TS;
  const device int32_t* list = IDX + (P[3] ? (long(head) * NQ + tg.x) * KSEL : 0);
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> q((device int8_t*)Q + hb * D, dextents<int32_t, 2>(D, TS));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> k((device int8_t*)K + hb * D, dextents<int32_t, 2>(D, TS));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> v((device int8_t*)V + hb * D, dextents<int32_t, 2>(D, TS));
  threadgroup half pt[TQ * KEYS];
  threadgroup float tops[256];
  threadgroup float fac[TQ];
  threadgroup float qsc[TQ];
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> p((threadgroup half*)pt, dextents<int32_t, 2>(KEYS, TQ));
  constexpr auto d1 = matmul2d_descriptor(TQ, KEYS, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto d2 = matmul2d_descriptor(TQ, D, KEYS, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d1, execution_simdgroups<8>> op1;
  matmul2d<d2, execution_simdgroups<8>> op2;
  auto a = q.slice<D, TQ>(0, r0);
  auto b0 = k.slice<D, KEYS>(0, 0);
  auto v0 = v.slice<D, KEYS>(0, 0);
  auto acc = op1.template get_destination_cooperative_tensor<decltype(a), decltype(b0), int32_t>();
  auto out = op2.template get_destination_cooperative_tensor<decltype(p), decltype(v0), float>();
  short srow[C1], scol[C1], orow[C2], ocol[C2];
  H3P_UNROLL
  for (ushort i = 0; i < C1; i++) {
    auto ids = acc.get_multidimensional_index(i);
    scol[i] = ids[0];
    srow[i] = ids[1];
  }
  H3P_UNROLL
  for (ushort i = 0; i < C2; i++) {
    auto ids = out.get_multidimensional_index(i);
    ocol[i] = ids[0];
    orow[i] = ids[1];
    out[i] = 0.0f;
  }
  for (int j = int(tid); j < TQ; j += 256) qsc[j] = QS[hb + r0 + j] * SC[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int row = int(tid) / TPR, first = (int(tid) % TPR) * C1;
  const int mine = row * KEYS + first;
  float top = -1e30f, total = 0.0f;
  float s[C1];
  for (int sel = 0; sel < KSEL; sel++) {
    const int tile = list[sel];
    const int k0 = tile * KEYS, size = SIZES[tile];
#ifdef H3_TILE_SCALES
    const float ks = KS[hb + k0], vs = VS[hb + k0];
#endif
    H3P_UNROLL
    for (ushort i = 0; i < C1; i++) acc[i] = 0;
    auto b = k.slice<D, KEYS>(0, k0);
#ifndef H3_KO_SCORES
    op1.run(a, b, acc);
#endif
    H3P_UNROLL
    for (ushort i = 0; i < C1; i++) {
#ifdef H3_TILE_SCALES
      const float score = scol[i] < size ? float(acc[i]) * qsc[srow[i]] * ks : -60000.0f;
#else
      const float score = scol[i] < size ? float(acc[i]) * qsc[srow[i]] * KS[hb + k0 + scol[i]] : -60000.0f;
#endif
      pt[srow[i] * KEYS + scol[i]] = half(score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local = -1e30f;
    H3P_UNROLL
    for (ushort j = 0; j < C1; j++) {
      s[j] = float(pt[mine + j]);
      local = max(local, s[j]);
    }
    tops[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float next = top;
    H3P_UNROLL
    for (ushort j = 0; j < TPR; j++) next = max(next, tops[row * TPR + j]);
    const float shrink = exp(top - next);
    if (tid % TPR == 0) fac[row] = shrink;
    total *= shrink;
    top = next;
    H3P_UNROLL
    for (ushort j = 0; j < C1; j++) {
#ifdef H3_KO_EXP
      const float weight = 1.0f + s[j] - next;
#else
      const float weight = exp(s[j] - next);
#endif
      total += weight;
#ifdef H3_TILE_SCALES
      pt[mine + j] = half(weight * vs * LIFT);
#else
      pt[mine + j] = half(weight * VS[hb + k0 + first + j] * LIFT);
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    H3P_UNROLL
    for (ushort i = 0; i < C2; i++) out[i] *= fac[orow[i]];
    auto vb = v.slice<D, KEYS>(0, k0);
#ifndef H3_KO_VALUES
    op2.run(p, vb, out);
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  tops[tid] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid % TPR == 0) {
    float sum = 0.0f;
    H3P_UNROLL
    for (ushort j = 0; j < TPR; j++) sum += tops[tid + j];
    fac[row] = 1.0f / (sum * LIFT);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  H3P_UNROLL
  for (ushort i = 0; i < C2; i++) {
    const int to = ROWOF[r0 + orow[i]];
    if (to >= 0) Y[(long(to) * H + head) * D + ocol[i]] = bfloat(out[i] * fac[orow[i]]);
  }
}

// Each key tile's int8 values summed over its 64 slots: what h3_attention_w8's offset weights add back.
[[kernel]] void h3_value_sums(
  const device int8_t* V8 [[buffer(0)]],
  device int32_t* VSUM [[buffer(1)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int c0 = 4 * int(lane);
  const device int8_t* v = V8 + (long(tg.y) * H3_SLOTS + long(tg.x) * 64) * 128 + c0;
  int4 sum = 0;
  for (int s = 0; s < 64; s++, v += 128) sum += int4(v[0], v[1], v[2], v[3]);
  device int32_t* out = VSUM + (long(tg.y) * H3_TILES + tg.x) * 128 + c0;
  for (int j = 0; j < 4; j++) out[j] = sum[j];
}

// h3_attention_tiles with 8-bit weights stored less 128, so the value product is int8 too. Needs H3_TILE_SCALES.
[[kernel]] void h3_attention_w8(
  const device int8_t* Q [[buffer(0)]],
  const device float* QS [[buffer(1)]],
  const device int8_t* K [[buffer(2)]],
  const device float* KS [[buffer(3)]],
  const device int8_t* V [[buffer(4)]],
  const device float* VS [[buffer(5)]],
  const device int32_t* IDX [[buffer(6)]],
  const device int32_t* SIZES [[buffer(7)]],
  const device int32_t* ROWOF [[buffer(8)]],
  const constant int32_t* P [[buffer(9)]],
  const constant float* SC [[buffer(10)]],
  device bfloat* Y [[buffer(11)]],
  const device int32_t* VSUM [[buffer(12)]],
  uint tid [[thread_index_in_threadgroup]],
  uint sg [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int D = 128, TQ = 64, KEYS = 64, TS = H3_SLOTS, H = H3P_HEADS;
  constexpr int C1 = TQ * KEYS / 256, C2 = TQ * D / 256;
#ifdef H3_W7
  constexpr float TOPW = 127.0f;
  constexpr int TOPI = 127, OFF = 0;
#else
  constexpr float TOPW = 255.0f;
  constexpr int TOPI = 255, OFF = 128;
#endif
  const int NQ = P[0], KSEL = P[1];
  const int head = int(tg.y), r0 = (P[2] + int(tg.x)) * TQ;
  const long hb = long(head) * TS;
  const device int32_t* list = IDX + (P[3] ? (long(head) * NQ + tg.x) * KSEL : 0);
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> q((device int8_t*)Q + hb * D, dextents<int32_t, 2>(D, TS));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> k((device int8_t*)K + hb * D, dextents<int32_t, 2>(D, TS));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> v((device int8_t*)V + hb * D, dextents<int32_t, 2>(D, TS));
  threadgroup int8_t pt[TQ * KEYS];
  threadgroup float tops[TQ * 4];
  threadgroup float staged[2 * D];                 // the key tile's summed values, read from the device once
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> p((threadgroup int8_t*)pt, dextents<int32_t, 2>(KEYS, TQ));
  constexpr auto d1 = matmul2d_descriptor(TQ, KEYS, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto d2 = matmul2d_descriptor(TQ, D, KEYS, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d1, execution_simdgroups<8>> op1;
  matmul2d<d2, execution_simdgroups<8>> op2;
  auto a = q.slice<D, TQ>(0, r0);
  auto b0 = k.slice<D, KEYS>(0, 0);
  auto v0 = v.slice<D, KEYS>(0, 0);
  auto acc = op1.template get_destination_cooperative_tensor<decltype(a), decltype(b0), int32_t>();
  auto prod = op2.template get_destination_cooperative_tensor<decltype(p), decltype(v0), int32_t>();
  // Score element i: row group i / 4, column c0 + i % 4. Value element i: row group 2 (i / 16) + (i / 4) % 2.
  const auto first = acc.get_multidimensional_index(ushort(0));
  const int c0 = first[0];
  int rws[4];
  H3P_UNROLL
  for (ushort g = 0; g < 4; g++) rws[g] = acc.get_multidimensional_index(ushort(4 * g))[1];
  const bool writer = (lane & 9) == 0;
  const int slot = int(sg) & 3;
  float out[C2];
  H3P_UNROLL
  for (ushort i = 0; i < C2; i++) out[i] = 0.0f;
  float4 qs, top = -1e30f, mass = 0.0f;
  H3P_UNROLL
  for (ushort g = 0; g < 4; g++) qs[g] = QS[hb + r0 + rws[g]] * SC[0] * 1.4426950408889634f;
#ifdef H3_PROBE_PAIR
  // the two products over 128 keys at a time with no softmax between: what pairing key tiles could reach
  {
    threadgroup int8_t pt2[TQ * 2 * KEYS];
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> p2((threadgroup int8_t*)pt2, dextents<int32_t, 2>(2 * KEYS, TQ));
    constexpr auto e1 = matmul2d_descriptor(TQ, 2 * KEYS, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto e2 = matmul2d_descriptor(TQ, D, 2 * KEYS, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<e1, execution_simdgroups<8>> o1;
    matmul2d<e2, execution_simdgroups<8>> o2;
    auto kb0 = k.slice<D, 2 * KEYS>(0, 0);
    auto wide = o1.template get_destination_cooperative_tensor<decltype(a), decltype(kb0), int32_t>();
    auto both = o2.template get_destination_cooperative_tensor<decltype(p2), decltype(kb0), int32_t>();
    for (int sel = 0; sel + 1 < KSEL; sel += 2) {
      const int k0 = min(list[sel], H3_TILES - 2) * KEYS;
      H3P_UNROLL
      for (ushort i = 0; i < C2; i++) wide[i] = 0;
      auto kb = k.slice<D, 2 * KEYS>(0, k0);
#ifndef H3_KO_SCORES
      o1.run(a, kb, wide);
#endif
      H3P_UNROLL
      for (ushort i = 0; i < C2; i++) both[i] = 0;
      auto vb = v.slice<D, 2 * KEYS>(0, k0);
#ifndef H3_KO_VALUES
      o2.run(p2, vb, both);
#endif
      H3P_UNROLL
      for (ushort i = 0; i < C2; i++) out[i] += float(both[i]) + float(wide[i]);
    }
    H3P_UNROLL
    for (ushort i = 0; i < C2; i++) {
      const ushort g = 2 * (i / 16) + (i / 4) % 2;
      const int to = ROWOF[r0 + rws[g]];
      if (to >= 0) Y[(long(to) * H + head) * D + c0 + (i & 3) + 64 * ((i / 8) % 2)] = bfloat(out[i]);
    }
    return;
  }
#endif
  // With H3_PAIR two neighbouring key tiles go through one 128-key product; off by default.
  threadgroup int8_t pt2[TQ * 2 * KEYS];
  tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> p2((threadgroup int8_t*)pt2, dextents<int32_t, 2>(2 * KEYS, TQ));
  constexpr auto e1 = matmul2d_descriptor(TQ, 2 * KEYS, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto e2 = matmul2d_descriptor(TQ, D, 2 * KEYS, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<e1, execution_simdgroups<8>> o1;
  matmul2d<e2, execution_simdgroups<8>> o2;
  auto kb0 = k.slice<D, 2 * KEYS>(0, 0);
  auto wide = o1.template get_destination_cooperative_tensor<decltype(a), decltype(kb0), int32_t>();
  auto both = o2.template get_destination_cooperative_tensor<decltype(p2), decltype(kb0), int32_t>();
#define H3_MERGE(T) \
  H3P_UNROLL \
  for (ushort i = 0; i < C2; i++) { \
    const ushort g = 2 * (i / 16) + (i / 4) % 2; \
    out[i] = out[i] * f[g] + (float(T[i]) + sv[(i & 3) + 4 * ((i / 8) % 2)]) * c[g]; \
  }
  for (int sel = 0, turn = 0; sel < KSEL; turn++) {
    const int tile = list[sel];
#ifdef H3_PAIR
    const bool pair = sel + 1 < KSEL && list[sel + 1] == tile + 1;
#else
    const bool pair = false;
#endif
    const int k0 = tile * KEYS, size = SIZES[tile];
    const float ks = KS[hb + k0], vs = VS[hb + k0];
    float4 m, ls, lm;
    float span = vs;
#ifndef H3_W7
    // alternate halves: a thread still merging the turn before reads the other one
    if (tid < D) {
      const device int32_t* sums = VSUM + (long(head) * H3_TILES + tile) * D + tid;
      staged[(turn & 1) * D + tid] = 128.0f * float(pair ? sums[0] + sums[D] : sums[0]);
    }
#endif
    if (pair) {
      const int size1 = SIZES[tile + 1];
      const float ks1 = KS[hb + k0 + KEYS], vs1 = VS[hb + k0 + KEYS];
      span = max(vs, vs1);
      H3P_UNROLL
      for (ushort i = 0; i < C2; i++) wide[i] = 0;
      auto kb = k.slice<D, 2 * KEYS>(0, k0);
      o1.run(a, kb, wide);
      // s[2 g + h]: row group g against the first (h = 0) or second tile's four columns
      const float4 open0 = float4(c0 < size ? 0.0f : -1e30f, c0 + 1 < size ? 0.0f : -1e30f, c0 + 2 < size ? 0.0f : -1e30f, c0 + 3 < size ? 0.0f : -1e30f);
      const float4 open1 = float4(c0 < size1 ? 0.0f : -1e30f, c0 + 1 < size1 ? 0.0f : -1e30f, c0 + 2 < size1 ? 0.0f : -1e30f, c0 + 3 < size1 ? 0.0f : -1e30f);
      float4 s[8];
      H3P_UNROLL
      for (ushort j = 0; j < 8; j++) {
        const ushort g = 2 * (j / 4) + j % 2, h = (j / 2) % 2;
        s[2 * g + h] = float4(float(wide[4 * j]), float(wide[4 * j + 1]), float(wide[4 * j + 2]), float(wide[4 * j + 3])) * (qs[g] * (h ? ks1 : ks)) + (h ? open1 : open0);
      }
      H3P_UNROLL
      for (ushort g = 0; g < 4; g++) {
        const float4 t = max(s[2 * g], s[2 * g + 1]);
        lm[g] = max(max(t.x, t.y), max(t.z, t.w));
      }
      lm = max(lm, simd_shuffle_xor(lm, 1));
      lm = max(lm, simd_shuffle_xor(lm, 8));
      if (writer) {
        H3P_UNROLL
        for (ushort g = 0; g < 4; g++) tops[rws[g] * 4 + slot] = lm[g];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // one value scale for the pair: the tile with the smaller one has its weights scaled down to match
      const float2 share = float2(vs, vs1) / span;
      H3P_UNROLL
      for (ushort g = 0; g < 4; g++) {
        const threadgroup float* t = tops + rws[g] * 4;
        m[g] = max(max(t[0], t[1]), max(t[2], t[3]));
        ls[g] = 0.0f;
        H3P_UNROLL
        for (ushort h = 0; h < 2; h++) {
          const int4 u = min(int4(TOPI), int4(fast::exp2(s[2 * g + h] - m[g]) * (TOPW * share[h]) + 0.5f));
          ls[g] += float(u.x + u.y + u.z + u.w) / share[h];
          threadgroup int8_t* w = pt2 + rws[g] * 2 * KEYS + h * KEYS + c0;
          w[0] = int8_t(u.x - OFF);
          w[1] = int8_t(u.y - OFF);
          w[2] = int8_t(u.z - OFF);
          w[3] = int8_t(u.w - OFF);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      H3P_UNROLL
      for (ushort i = 0; i < C2; i++) both[i] = 0;
      auto vb = v.slice<D, 2 * KEYS>(0, k0);
      o2.run(p2, vb, both);
    } else {
      H3P_UNROLL
      for (ushort i = 0; i < C1; i++) acc[i] = 0;
      auto b = k.slice<D, KEYS>(0, k0);
#ifndef H3_KO_SCORES
      op1.run(a, b, acc);
#endif
      const float4 open = float4(c0 < size ? 0.0f : -1e30f, c0 + 1 < size ? 0.0f : -1e30f, c0 + 2 < size ? 0.0f : -1e30f, c0 + 3 < size ? 0.0f : -1e30f);
      float4 s[4];
      H3P_UNROLL
      for (ushort g = 0; g < 4; g++) {
        s[g] = float4(float(acc[4 * g]), float(acc[4 * g + 1]), float(acc[4 * g + 2]), float(acc[4 * g + 3])) * (qs[g] * ks) + open;
        lm[g] = max(max(s[g].x, s[g].y), max(s[g].z, s[g].w));
      }
      lm = max(lm, simd_shuffle_xor(lm, 1));
      lm = max(lm, simd_shuffle_xor(lm, 8));
      if (writer) {
        H3P_UNROLL
        for (ushort g = 0; g < 4; g++) tops[rws[g] * 4 + slot] = lm[g];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      H3P_UNROLL
      for (ushort g = 0; g < 4; g++) {
        const threadgroup float* t = tops + rws[g] * 4;
        m[g] = max(max(t[0], t[1]), max(t[2], t[3]));
        const int4 u = min(int4(TOPI), int4(fast::exp2(s[g] - m[g]) * TOPW + 0.5f));
        ls[g] = float(u.x + u.y + u.z + u.w);
        threadgroup int8_t* w = pt + rws[g] * KEYS + c0;
        w[0] = int8_t(u.x - OFF);
        w[1] = int8_t(u.y - OFF);
        w[2] = int8_t(u.z - OFF);
        w[3] = int8_t(u.w - OFF);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      H3P_UNROLL
      for (ushort i = 0; i < C2; i++) prod[i] = 0;
      auto vb = v.slice<D, KEYS>(0, k0);
#ifndef H3_KO_VALUES
      op2.run(p, vb, prod);
#endif
    }
    const float4 next = max(top, m);
    const float4 f = fast::exp2(top - next), e = fast::exp2(m - next);
    const float4 c = e * span;
    mass = mass * f + e * ls;
    top = next;
    float sv[8];
#ifdef H3_W7
    H3P_UNROLL
    for (ushort j = 0; j < 8; j++) sv[j] = 0.0f;
#else
    const threadgroup float* sums = staged + (turn & 1) * D + c0;
    H3P_UNROLL
    for (ushort j = 0; j < 8; j++) sv[j] = sums[(j & 3) + 64 * (j >> 2)];
#endif
    if (pair) {
      H3_MERGE(both)
    } else {
      H3_MERGE(prod)
    }
    sel += pair ? 2 : 1;
  }
#undef H3_MERGE
  // a row's mass is the sum of its sixteen holders' parts
  mass += simd_shuffle_xor(mass, 1);
  mass += simd_shuffle_xor(mass, 8);
  if (writer) {
    H3P_UNROLL
    for (ushort g = 0; g < 4; g++) tops[rws[g] * 4 + slot] = mass[g];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 inv;
  H3P_UNROLL
  for (ushort g = 0; g < 4; g++) {
    const threadgroup float* t = tops + rws[g] * 4;
    inv[g] = 1.0f / max(t[0] + t[1] + t[2] + t[3], 1e-30f);
  }
  H3P_UNROLL
  for (ushort i = 0; i < C2; i++) {
    const ushort g = 2 * (i / 16) + (i / 4) % 2;
    const int to = ROWOF[r0 + rws[g]];
    if (to >= 0) Y[(long(to) * H + head) * D + c0 + (i & 3) + 64 * ((i / 8) % 2)] = bfloat(out[i] * inv[g]);
  }
}

