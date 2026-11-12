[[kernel, max_total_threads_per_threadgroup(128)]]
void im_attention_scores256(const device float* Q [[buffer(0)]], const device float* K [[buffer(1)]],
    device float* probabilities [[buffer(2)]], device float* stats [[buffer(3)]],
    constant int* P [[buffer(4)]], constant float* F [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using namespace tfp;
  const int row = int(tg.x) * 64 + int(sg) * 16, rows = P[2], blocks = P[6], columns = blocks * 32;
  const int kv = int(tg.y) / P[4], stride = blocks + 1;
  Q += long(tg.z) * P[12] + long(tg.y) * P[13] + long(row) * P[14];
  K += long(tg.z) * P[15] + long(kv) * P[16];
  probabilities += ((long(tg.z) * P[1] + tg.y) * rows + row) * columns;
  stats += ((long(tg.z) * P[1] + tg.y) * rows + row) * stride;
  const short2 home = frag_home(ushort(lane));
  const float scale2 = F[0] * 1.44269504089f;
  float top[2] = {kMasked, kMasked}, total[2] = {0.0f, 0.0f};
  for (int kb = 0; kb < blocks; kb++) {
    const bool last = kb == P[8];
    frag<float> s[2] = {frag<float>(0), frag<float>(0)};
#pragma clang loop unroll_count(4)
    for (short d = 0; d < 16; d++) {
      frag<float> q, k0, k1;
      frag_get_in(q, Q, P[14], 0, 16 * d, home, rows - row, kAnyCols);
      if (last) {
        frag_get_t_in(k0, K, P[17], 0, 16 * d, home, P[10], kAnyCols);
        frag_get_t_in(k1, K, P[17], 16, 16 * d, home, P[10], kAnyCols);
      } else {
        frag_get_t(k0, K, P[17], 0, 16 * d, home);
        frag_get_t(k1, K, P[17], 16, 16 * d, home);
      }
      mma_16x32<false, true>(s[0], s[1], q, k0, k1);
    }
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[f][e] *= scale2;
        if (last && 16 * f + home.x + TF_COL(e) >= P[10]) s[f][e] = kMasked;
      }
    }
    float next[2] = {top[0], top[1]};
    row_fold<true>(s[0], next);
    row_fold<true>(s[1], next);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = fast::exp2(s[f][e] - next[e >> 2]);
    }
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      const float previous = fast::exp2(top[h] - next[h]);
      top[h] = next[h];
      total[h] = total[h] * previous;
      const int r = home.y + 8 * h;
      if (home.x == 0 && row + r < rows) stats[r * stride + kb] = previous;
    }
    row_fold<false>(s[0], total);
    row_fold<false>(s[1], total);
    frag_put_in(s[0], probabilities, columns, 0, kb * 32, home, rows - row, columns);
    frag_put_in(s[1], probabilities, columns, 0, kb * 32 + 16, home, rows - row, columns);
    K += 32 * P[17];
    threadgroup_barrier(mem_flags::mem_none);
  }
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const int r = home.y + 8 * h;
    if (home.x == 0 && row + r < rows) stats[r * stride + blocks] = 1.f / total[h];
  }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void im_attention_values256(const device float* V [[buffer(0)]], const device float* probabilities [[buffer(1)]],
    const device float* stats [[buffer(2)]], device float* O [[buffer(3)]], constant int* P [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using namespace tfp;
  const int batch = int(tg.z) / 2, part = int(tg.z) % 2;
  const int row = int(tg.x) * 64 + int(sg) * 16, rows = P[2], blocks = P[6], columns = blocks * 32;
  const int kv = int(tg.y) / P[4], stride = blocks + 1;
  V += long(batch) * P[18] + long(kv) * P[19] + part * 128;
  O += long(batch) * P[21] + long(tg.y) * P[22] + long(row) * P[23] + part * 128;
  probabilities += ((long(batch) * P[1] + tg.y) * rows + row) * columns;
  stats += ((long(batch) * P[1] + tg.y) * rows + row) * stride;
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[8];
  TF_UNROLL
  for (short d = 0; d < 8; d++) acc[d] = frag<float>(0);
  for (int kb = 0; kb < blocks; kb++) {
    frag<float> s[2];
    frag_get_in(s[0], probabilities, columns, 0, kb * 32, home, rows - row, columns);
    frag_get_in(s[1], probabilities, columns, 0, kb * 32 + 16, home, rows - row, columns);
    float previous[2];
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = home.y + 8 * h;
      previous[h] = row + r < rows ? stats[r * stride + kb] : 0.0f;
    }
    TF_UNROLL
    for (short d = 0; d < 8; d++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[d][e] = acc[d][e] * previous[e >> 2];
    }
    simdgroup_barrier(mem_flags::mem_none);
    TF_UNROLL
    for (short d = 0; d < 8; d += 2) {
      if (d == 4) threadgroup_barrier(mem_flags::mem_none);
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        frag<float> v0, v1;
        if (kb == P[8]) {
          frag_get_in(v0, V, P[20], 16 * k, 16 * d, home, P[10], kAnyCols);
          frag_get_in(v1, V, P[20], 16 * k, 16 * d + 16, home, P[10], kAnyCols);
        } else {
          frag_get(v0, V, P[20], 16 * k, 16 * d, home);
          frag_get(v1, V, P[20], 16 * k, 16 * d + 16, home);
        }
        tensor_attention_pv(acc[d], acc[d + 1], s[k], v0, v1, ushort(lane));
      }
    }
    V += 32 * P[20];
  }
  threadgroup_barrier(mem_flags::mem_none);
  float inv[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const int r = home.y + 8 * h;
    inv[h] = row + r < rows ? stats[r * stride + blocks] : 0.0f;
  }
  TF_UNROLL
  for (short d = 0; d < 8; d++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) acc[d][e] = acc[d][e] * inv[e >> 2];
    frag_put_in(acc[d], O, P[23], 0, 16 * d, home, rows - row, kAnyCols);
  }
}
