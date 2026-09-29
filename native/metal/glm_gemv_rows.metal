
  constexpr int blockM = BM * SM * TM;
  constexpr int blockN = BN * SN * TN;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int row = int(threadgroup_position_in_grid.z);
  const int in_vec_size = int(X_shape[1]);
  const int out_vec_size = int(M_shape[0]);
  const device T* in_vec = X + size_t(row) * in_vec_size;
  device T* out_vec = OUT + size_t(row) * out_vec_size;
  threadgroup float tgp_memory[BN > 1 ? BN * (blockM + TM) : 1];
  float result[TM] = {0};
  T inter[TN];
  float v_coeff[TN];
  const int thrM = SN != 32 ? simd_lid / SN : 0;
  const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
  const int sgN = BN != 1 ? (simd_gid % BN) : 0;
  const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
  const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;
  const int bm = (simdM + thrM) * TM;
  int bn = (simdN + thrN) * TN;
  int out_row = int(threadgroup_position_in_grid.x) * blockM + bm;
  if (out_row >= out_vec_size) return;
  out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;
  const device T* mat = M + size_t(out_row) * in_vec_size;
  const int n_iter = in_vec_size / blockN;
  const int leftover = in_vec_size - blockN * n_iter;
  for (int i = 0; i < n_iter; ++i) {
    for (int tn = 0; tn < TN; tn++) v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
    int mat_offset = 0;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) inter[tn] = mat[mat_offset + bn + tn];
      for (int tn = 0; tn < TN; tn++) result[tm] += inter[tn] * v_coeff[tn];
      mat_offset += in_vec_size;
    }
    bn += blockN;
  }
  if (leftover > 0) {
    for (int tn = 0; tn < TN; tn++) v_coeff[tn] = bn + tn < in_vec_size ? static_cast<float>(in_vec[bn + tn]) : 0.0f;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) inter[tn] = bn + tn < in_vec_size ? mat[tm * in_vec_size + bn + tn] : T(0);
      for (int tn = 0; tn < TN; tn++) result[tm] += inter[tn] * v_coeff[tn];
    }
  }
  for (int tm = 0; tm < TM; tm++)
    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) result[tm] += simd_shuffle_down(result[tm], sn);
  if (BN > 1) {
    threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
    if (thrN == 0) {
      for (int tm = 0; tm < TM; tm++) tgp_results[tm] = result[tm];
      threadgroup_barrier(mem_flags::mem_none);
      if (sgN == 0)
        for (int sgn = 1; sgn < BN; sgn++)
          for (int tm = 0; tm < TM; tm++) result[tm] += tgp_results[sgn * (blockM + TM) + tm];
    }
  }
  if (simdN == 0 && thrN == 0)
    for (int tm = 0; tm < TM; tm++) out_vec[out_row + tm] = static_cast<T>(result[tm]);
