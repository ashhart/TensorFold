// The base's csa2_attn per-slot kernels, renamed tf_base_*: a test appends them to today's source to compare bytes.

// Decode: a threadgroup is one row, HS heads and a simdgroup for each of the five chunks; simdgroup 0 joins them.

// A selected run's simdgroup first copies its 128 indices to threadgroup memory, so no key load waits on an index load.
template <typename T, bool PF>
[[kernel, max_total_threads_per_threadgroup(32 * TF_ATTN_NCH)]] void tf_base_dsv41_attn_dec(
    TF_ATTN_ARGS, uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tacc[TF_ATTN_NCH][TF_ATTN_HS][TF_ATTN_D];
  threadgroup float tml[TF_ATTN_NCH][TF_ATTN_HS][2];
  threadgroup uint tn[TF_ATTN_NCH];
  threadgroup int tidx[TF_ATTN_SEL];
  const uint r = tg.x, h0 = tg.y * TF_ATTN_HS;
  if (r >= a.rows || h0 >= a.heads) return;
  if (sg != 0)
    for (uint u = 0; u < TF_ATTN_RUN / 32; ++u) {
      const uint j = (sg - 1) * TF_ATTN_RUN + 32 * u + lane;
      tidx[j] = I[ulong(r) * TF_ATTN_SEL + j];
    }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const TfAttnMeta m = M[r];
  TfQ q;
  tf_attn_loadq(q, Q, r, h0, a.heads, lane);
  TfAcc s;
  tf_attn_zero(s);
  if (sg == 0) tf_attn_window<PF>(s, q, W, m, a.scale, lane);
  else tf_attn_selected<PF>(s, q, C, (const threadgroup int*)tidx, m, sg - 1, a.scale, lane);
  for (uint hh = 0; hh < TF_ATTN_HS; ++hh)
    for (uint i = 0; i < 16; ++i) tacc[sg][hh][16 * lane + i] = s.acc[hh][i];
  if (lane < 2 * TF_ATTN_HS) tml[sg][lane / 2][lane % 2] = (lane % 2) ? s.l[lane / 2] : s.m[lane / 2];
  if (lane == 0) tn[sg] = s.n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (uint c = 1; c < TF_ATTN_NCH; ++c) {
    if (tn[c] == 0) continue;
    for (uint hh = 0; hh < TF_ATTN_HS; ++hh) {
      float ac[16];
      for (uint i = 0; i < 16; ++i) ac[i] = tacc[c][hh][16 * lane + i];
      tf_attn_join(s, hh, tml[c][hh][0], tml[c][hh][1], ac);
    }
  }
  for (uint hh = 0; hh < TF_ATTN_HS; ++hh)
    tf_attn_store(s, hh, SK[h0 + hh], CS + ulong(m.pos + m.rope) * 64, O + (ulong(r) * a.heads + h0 + hh) * TF_ATTN_D,
                  lane);
  if (TF_CTL == 4 && r == 0 && h0 == 0 && lane == 0) O[ulong(a.rows) * a.heads * TF_ATTN_D] = T(1.0f);
}

template [[host_name("tf_base_dsv41_attn_dec_bf16")]] [[kernel]] decltype(tf_base_dsv41_attn_dec<bfloat, false>) tf_base_dsv41_attn_dec<bfloat, false>;
template [[host_name("tf_base_dsv41_attn_dec_f32")]] [[kernel]] decltype(tf_base_dsv41_attn_dec<float, false>) tf_base_dsv41_attn_dec<float, false>;
template [[host_name("tf_base_dsv41_attn_dec_pf_bf16")]] [[kernel]] decltype(tf_base_dsv41_attn_dec<bfloat, true>) tf_base_dsv41_attn_dec<bfloat, true>;
template [[host_name("tf_base_dsv41_attn_dec_pf_f32")]] [[kernel]] decltype(tf_base_dsv41_attn_dec<float, true>) tf_base_dsv41_attn_dec<float, true>;
