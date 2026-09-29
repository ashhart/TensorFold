
  constexpr int blockM = BM * SM * TM;
  constexpr int blockN = BN * SN * TN;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int row = int(threadgroup_position_in_grid.z);
  const int in_vec_size = int(X_shape[1]);
  const int out_vec_size = int(M_shape[1]);
  const device T* in_vec = X + size_t(row) * in_vec_size;
  device T* out_vec = OUT + size_t(row) * out_vec_size;
  threadgroup float tgp_memory[BM > 1 ? BM * (blockN + TN) : 1];
  float result[TN] = {0};
  T inter[TN];
  float v_coeff[TM];
  const int thrM = SN != 32 ? simd_lid / SN : 0;
  const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
  const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
  const int sgN = BN != 1 ? (simd_gid % BN) : 0;
  const int cm = SM * sgM + thrM;
  const int cn = SN * sgN + thrN;
  int bm = cm * TM;
  const int bn = cn * TN;
  int out_col = int(threadgroup_position_in_grid.x) * blockN + bn;
  const int n_iter = in_vec_size / blockM;
  const int leftover = in_vec_size - blockM * n_iter;
  if (out_col < out_vec_size) {
    out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;
    for (int i = 0; i < n_iter; ++i) {
      threadgroup_barrier(mem_flags::mem_none);
      for (int tm = 0; tm < TM; tm++) v_coeff[tm] = static_cast<float>(in_vec[bm + tm]);
      for (int tm = 0; tm < TM; tm++) {
        const float vc = v_coeff[tm];
        for (int tn = 0; tn < TN; tn++) inter[tn] = M[size_t(bm + tm) * out_vec_size + out_col + tn];
        for (int tn = 0; tn < TN; tn++) result[tn] += vc * inter[tn];
      }
      bm += blockM;
    }
    if (leftover > 0) {
      for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
        v_coeff[tm] = static_cast<float>(in_vec[bm + tm]);
        for (int tn = 0; tn < TN; tn++) inter[tn] = M[size_t(bm + tm) * out_vec_size + out_col + tn];
        for (int tn = 0; tn < TN; tn++) result[tn] += v_coeff[tm] * inter[tn];
      }
    }
  }
  for (int tn = 0; tn < TN; tn++)
    for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) result[tn] += simd_shuffle_down(result[tn], SN * sm);
  if (BM > 1) {
    threadgroup float* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
    if (thrM == 0) {
      for (int tn = 0; tn < TN; tn++) tgp_results[tn] = result[tn];
      threadgroup_barrier(mem_flags::mem_none);
      if (sgM == 0)
        for (int sgm = 1; sgm < BM; sgm++)
          for (int tn = 0; tn < TN; tn++) result[tn] += tgp_results[sgm * (blockN + TN) + tn];
    }
  }
  if (cm == 0 && out_col < out_vec_size)
    for (int j = 0; j < TN; j++) out_vec[out_col + j] = static_cast<T>(result[j]);
