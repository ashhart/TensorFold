
  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);
  threadgroup bfloat16_t Xs[BM * BK_padded];
  threadgroup bfloat16_t Ws[BN * BK_padded];
  qmm_t_impl<bfloat16_t, GS, BITS, ALIGNED != 0, BM, BK, BN>(W, S, B, X, Y, Xs, Ws, KK[0], NN[0], MM[0], KK[0],
      threadgroup_position_in_grid, thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
