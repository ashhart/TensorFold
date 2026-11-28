// Original prompt kernel with a runtime tail count. The contract appends it to the shared attention helpers.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(32 * TF_ATTN_RR * TF_ATTN_HG)]] void tf_base_dsv41_attn_pre(
    TF_ATTN_ARGS, uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint ra = sg / TF_ATTN_HG, r = tg.x * TF_ATTN_RR + ra;
  const uint h0 = (tg.y * TF_ATTN_HG + sg % TF_ATTN_HG) * TF_ATTN_PH;
  if (r >= a.rows || h0 >= a.heads) return;
  const uint nh = min(TF_ATTN_PH, a.heads - h0);
  const TfAttnMeta m = M[r];
  const float scale = (TF_CTL == 5 && ra == 1) ? a.scale * 1.0000001f : a.scale;
  float q[TF_ATTN_PH][16];
  for (uint hh = 0; hh < TF_ATTN_PH; ++hh) {
    const ulong at = (ulong(r) * a.heads + h0 + min(hh, nh - 1)) * (TF_ATTN_D / 8) + 2 * lane;
    tf_attn_x16(TfKey{Q[at], Q[at + 1]}, q[hh]);
  }
  TfAccN<TF_ATTN_PH> s;
  tf_attn_zero(s);
  const int lo = max(m.pos - int(TF_ATTN_WIN) + 1, 0);
  const device uint4* wk = W + ulong(m.wofs + lo - m.wbase) * (TF_ATTN_D / 8) + 2 * lane;
  for (int t = lo; t <= m.pos; ++t, wk += TF_ATTN_D / 8) tf_attn_pkey(s, q, TfKey{wk[0], wk[1]}, scale, lane);
  const device int4* ix = (const device int4*)(I + ulong(r) * TF_ATTN_SEL);
  const device uint4* ck = C + ulong(m.cofs) * (TF_ATTN_D / 8) + 2 * lane;
  for (uint run = 0; run < TF_ATTN_NCH - 1; ++run) {
    if (int(run * TF_ATTN_RUN) >= m.ncomp) break;
    const int end = min(m.ncomp, int((run + 1) * TF_ATTN_RUN));
    TfAccN<TF_ATTN_PH> c;
    tf_attn_zero(c);
    for (int j = int(run * TF_ATTN_RUN); j < end; j += 4) {
      const int4 iv = ix[j / 4];
      for (uint u = 0; u < 4; ++u) {
        const int jj = j + int(u);
        if (jj >= end || iv[u] < 0 || (TF_CTL == 2 && jj == m.ncomp - 1)) continue;
        const device uint4* kp = ck + uint(iv[u]) * (TF_ATTN_D / 8);
        tf_attn_pkey(c, q, TfKey{kp[0], kp[1]}, scale, lane);
      }
    }
    if (c.n == 0) continue;
    for (uint hh = 0; hh < TF_ATTN_PH; ++hh) tf_attn_join(s, hh, c.m[hh], c.l[hh], c.acc[hh]);
  }
  for (uint hh = 0; hh < nh; ++hh)
    tf_attn_store(s, hh, SK[h0 + hh], CS + ulong(m.pos) * 64, O + (ulong(r) * a.heads + h0 + hh) * TF_ATTN_D, lane);
}
template [[host_name("tf_base_dsv41_attn_pre_bf16")]] [[kernel]] decltype(tf_base_dsv41_attn_pre<bfloat>) tf_base_dsv41_attn_pre<bfloat>;
template [[host_name("tf_base_dsv41_attn_pre_f32")]] [[kernel]] decltype(tf_base_dsv41_attn_pre<float>) tf_base_dsv41_attn_pre<float>;
