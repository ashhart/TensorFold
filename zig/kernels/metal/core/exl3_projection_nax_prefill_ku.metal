// KU k blocks a barrier pair: the next group decodes while this one multiplies; accumulation order is unchanged.
constexpr constant uint TF_KU = 2;
template <uint RG, uint CG, uint KU, uint Rpg, bool SkipPad>
inline void tf_exl3_prefill_body_ku(threadgroup half* wx, const device half* x, const device uchar* trellis,
                                    device float* sums, constant TfExl3Args& a, const device uint* routes,
                                    const device TfExl3Expert* descriptors, uint lane, uint sg, uint tid,
                                    uint tptg, uint3 tg) {
  const uint trows = RG * 16u;
  const uint rb = tg.y * trows, nb = tg.x * (64u * CG);
  if (a.indexed == 1 || rb >= a.rows) return;
  const uint tile_expert = a.indexed ? routes[rb] : a.expert_id;
  if (a.indexed && tile_expert >= a.experts) return;
  const TfExl3DigJob job = tf_exl3_dig_job<CG>(trellis, a, routes, descriptors, rb, nb, trows);
  constexpr uint slab = 16u * 64u * CG;
  const short2 home = frag_home(ushort(lane));
  constexpr uint wpg = RG / (4u * Rpg);
  constexpr uint states = 8u / wpg;
  const uint gwarp = tid / 32u, group = gwarp / wpg, wg = gwarp % wpg;
  const uint h = group % 4u, hg = (group / 4u) * 64u;
  const uint p0 = (wg * 32u + tid % 32u) * states;
  const bool decodes = group < 4u * CG and nb + hg + h * 16u < a.n;
  const TfExl3DigWindows walk = tf_exl3_dig_windows<states, CG, 64u * CG>(p0, 6u, hg, h, false);
  const uint words = walk.words;
  const uint wi0 = walk.first_word;
  const uint sgr = Rpg * (sg % (RG / Rpg)), sgc = sg / (RG / Rpg);
  // The routed tile's padded rows need no multiply.
  uint real_mask = (1u << Rpg) - 1u;
  if (SkipPad and a.indexed != 0) {
    real_mask = 0u;
    if (lane < 16u * Rpg) {
      const uint grow = rb + (sgr + lane / 16u) * 16u + (lane % 16u);
      if (grow < a.rows and routes[grow] != 0xffffffffu) real_mask = 1u << (lane / 16u);
    }
  }
  bool group_live[Rpg];
  bool any_live = false;
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    group_live[r] = simd_any((real_mask & (1u << r)) != 0u);
    any_live = any_live or group_live[r];
  }
  frag<float> lo0[Rpg], hi0[Rpg], lo1[Rpg], hi1[Rpg];
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    lo0[r] = frag<float>(0);
    hi0[r] = frag<float>(0);
    lo1[r] = frag<float>(0);
    hi1[r] = frag<float>(0);
  }
  // Prologue: the first KU blocks into buffer 0.
  if (decodes) {
    TF_UNROLL
    for (uint s = 0; s < KU; ++s) {
      const uint w0 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, 16u * s, nb, hg, h, wi0, words);
      const uint w1 = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, 16u * s, nb, hg, h, wi0 + 1, words);
      tf_exl3_decode_funnel<states>(wx + s * slab, nullptr, nullptr, job.codebook, walk.at, walk.first_bit, 3, w0, w1);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint cur_buf = 0;
  for (uint kb = 0; kb < a.k; kb += 16u * KU) {
    const uint nkb = kb + 16u * KU;
    // The next group's stream words are issued before the multiply, so their latency hides behind the tensor work.
    uint nw0[KU], nw1[KU];
    if (nkb < a.k and decodes) {
      TF_UNROLL
      for (uint s = 0; s < KU; ++s) {
        nw0[s] = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, nkb + 16u * s, nb, hg, h, wi0, words);
        nw1[s] = tf_exl3_block_word(job.trellis, job.tile_k_stride, job.tile_n_stride, nkb + 16u * s, nb, hg, h, wi0 + 1, words);
      }
    }
    if (any_live) {
      frag<half> av[KU][Rpg];
      TF_UNROLL
      for (uint s = 0; s < KU; ++s) {
        TF_UNROLL
        for (uint r = 0; r < Rpg; ++r) {
          if (group_live[r]) tf_frag_get_dev4(av[s][r], (const device half*)x, int(a.k), int(rb + (sgr + r) * 16), int(kb + 16u * s), home, int(rb + job.rows_here));
        }
      }
      const int wc = int(sgc * 64u);
      TF_UNROLL
      for (uint s = 0; s < KU; ++s) {
        const threadgroup half* cur = wx + (cur_buf * KU + s) * slab;
        frag<half> b0, b1;
        tf_frag_get_tg4(b0, cur, int(64u * CG), 0, wc, home);
        tf_frag_get_tg4(b1, cur, int(64u * CG), 0, wc + 16, home);
        TF_UNROLL
        for (uint r = 0; r < Rpg; ++r) {
          if (group_live[r]) mma_16x32<false, false>(lo0[r], hi0[r], av[s][r], b0, b1);
        }
        tf_frag_get_tg4(b0, cur, int(64u * CG), 0, wc + 32, home);
        tf_frag_get_tg4(b1, cur, int(64u * CG), 0, wc + 48, home);
        TF_UNROLL
        for (uint r = 0; r < Rpg; ++r) {
          if (group_live[r]) mma_16x32<false, false>(lo1[r], hi1[r], av[s][r], b0, b1);
        }
      }
    }
    if (nkb < a.k and decodes) {
      threadgroup half* nxt = wx + (cur_buf ^ 1u) * KU * slab;
      TF_UNROLL
      for (uint s = 0; s < KU; ++s)
        tf_exl3_decode_funnel<states>(nxt + s * slab, nullptr, nullptr, job.codebook, walk.at, walk.first_bit, 3, nw0[s], nw1[s]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cur_buf ^= 1u;
  }
  const int r0 = int(rb + sgr * 16);
  const int c0 = int(nb + sgc * 64u);
  TF_UNROLL
  for (uint r = 0; r < Rpg; ++r) {
    if (group_live[r]) {
      frag_put_in(lo0[r], sums, int(a.n), r0 + int(r * 16), c0, home, int(rb + job.rows_here), int(nb + job.cols_here));
      frag_put_in(hi0[r], sums, int(a.n), r0 + int(r * 16), c0 + 16, home, int(rb + job.rows_here), int(nb + job.cols_here));
      frag_put_in(lo1[r], sums, int(a.n), r0 + int(r * 16), c0 + 32, home, int(rb + job.rows_here), int(nb + job.cols_here));
      frag_put_in(hi1[r], sums, int(a.n), r0 + int(r * 16), c0 + 48, home, int(rb + job.rows_here), int(nb + job.cols_here));
    }
  }
}
