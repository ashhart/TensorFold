
  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);
  threadgroup bfloat16_t Xs[BM * BK_padded];
  threadgroup bfloat16_t Ws[BN * BK_padded];
  tf_gather_qmm_tiles<bfloat16_t, GS, 4, BM, BN, BK, WM, WN>(Xs, Ws, X, W, S, B, OFF, Y, MM[0], NN[0], KK[0], EE[0],
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
