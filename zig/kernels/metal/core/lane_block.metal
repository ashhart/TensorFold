
// TF_TM 16-row blocks share each group's weight read; each row keeps its products, FMAs and split-K sums in order.
constant constexpr int TM = TF_TM;

// Register 16 kc + 4 nj + e holds code K0 + 16 kc + e of column c0 + 8 nj (regBlock's order), as an exact half.
inline void tf_block_codes(const device uint* W, int tile, int g, int lane, thread half* b) {
#if TF_BITS == 4
  const device uint4* wl = (const device uint4*)W + ((size_t)tile * KG + g) * 64 + lane;
  const uint4 w0 = wl[0], w1 = wl[32];
  thread half2* b2 = (thread half2*)b;
  const half2 bias = half2(1024.0h);
  for (int w = 0; w < 8; w++) {
    const int e = 16 * (w & 1) + 2 * (w >> 1);
    const uint word = w < 4 ? w0[w] : w1[w - 4], lo = word & 0x0F0F0F0Fu, hi = (word >> 4) & 0x0F0F0F0Fu;
    b2[e] = as_type<half2>((lo & 0x00FF00FFu) | 0x64006400u) - bias;
    b2[e + 1] = as_type<half2>((hi & 0x00FF00FFu) | 0x64006400u) - bias;
    b2[e + 8] = as_type<half2>(((lo >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
    b2[e + 9] = as_type<half2>(((hi >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
  }
#else
  const int k0 = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), c0 = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  const device uint* group = W + ((size_t)tile * KG + g) * NT * WP;
  for (int nj = 0; nj < 4; nj++) {
    const device uint* col = group + (c0 + 8 * nj) * WP;
    for (int kc = 0; kc < 4; kc++) {
#if TF_BITS == 8
      const uint v = col[(k0 + 16 * kc) / 4];
#else
      const int bit = 6 * (k0 + 16 * kc), i = bit >> 5, sh = bit & 31;
      uint v = col[i] >> sh;
      if (sh > 8) v |= col[i + 1] << (32 - sh);
#endif
      for (int e = 0; e < 4; e++) b[16 * kc + 4 * nj + e] = half((v >> (TF_BITS * e)) & ((1u << TF_BITS) - 1u));
    }
  }
#endif
}

[[kernel]] void tf_lane_block(const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]], const device bfloat* SB [[buffer(2)]],
    const device int* mdims [[buffer(3)]], device TF_OUT* Y [[buffer(4)]], const device float* XS [[buffer(5)]],
    uint sgi [[simdgroup_index_in_threadgroup]], uint lanei [[thread_index_in_simdgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  const int r0 = int(tg.y) * 16 * TM, rows = mdims[0] - r0;
  if (rows <= 0) return;
  const int blocks = min((rows + 15) / 16, TM), lane = int(lanei), ks = int(sgi), stride = mdims[1];
  const int tile = tf_tile(int(tg.x)), n0 = tile * NT;
  const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  const int first = TF_G0 + (ks * TF_GN) / TF_SK, last = TF_G0 + ((ks + 1) * TF_GN) / TF_SK;
  constexpr auto desc = matmul2d_descriptor(16, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> shape((device bfloat*)X, dextents<int32_t, 2>(TF_K, 16));
  auto a0 = shape.slice(0, 0);
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto P = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a0)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  const device uint4* sbv = (const device uint4*)SB;
  float C[16 * TM];
  for (int i = 0; i < 16 * TM; i++) C[i] = 0.0f;
  for (int g = first; g < last; g++) {
    tf_block_codes(W, tile, g, lane, (thread half*)&cb[0]);
    const uint4 q0 = sbv[((size_t)g * TF_N + n0 + nb) / 4], q1 = sbv[((size_t)g * TF_N + n0 + nb + 16) / 4];
    for (int b = 0; b < blocks; b++) {
      const int base = r0 + 16 * b, M = min(rows - 16 * b, 16);
      tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)(X + (size_t)base * TF_K), dextents<int32_t, 2>(TF_K, M));
      auto x = a.slice(g * GS, 0);
      op.run(x, cb, P);
      const float x0 = mb < M ? XS[g * stride + base + mb] : 0.0f, x1 = mb + 8 < M ? XS[g * stride + base + mb + 8] : 0.0f;
      thread float* R = C + 16 * b;
      for (int j = 0; j < 4; j++) {
        const float s0 = as_type<float>(q0[j] << 16), b0 = as_type<float>(q0[j] & 0xFFFF0000u);
        const float s1 = as_type<float>(q1[j] << 16), b1 = as_type<float>(q1[j] & 0xFFFF0000u);
        R[j] = fma(s0, P[j], fma(b0, x0, R[j]));
        R[4 + j] = fma(s0, P[4 + j], fma(b0, x1, R[4 + j]));
        R[8 + j] = fma(s1, P[8 + j], fma(b1, x0, R[8 + j]));
        R[12 + j] = fma(s1, P[12 + j], fma(b1, x1, R[12 + j]));
      }
    }
  }
  threadgroup float part[(TF_SK > 1 ? TF_SK - 1 : 1) * 16 * TM * 32];
  if (TF_SK > 1) {
    if (ks > 0) for (int i = 0; i < 16 * TM; i++) part[((ks - 1) * 16 * TM + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) for (int s2 = 1; s2 < TF_SK; s2++) for (int i = 0; i < 16 * TM; i++) C[i] += part[((s2 - 1) * 16 * TM + i) * 32 + lane];
  }
  if (ks == 0)
    for (int b = 0; b < blocks; b++)
      for (int i = 0; i < 16; i++) {
        const int m = mb + 8 * ((i >> 2) & 1), n = n0 + nb + 16 * (i >> 3) + (i & 3);
        if (m < min(rows - 16 * b, 16) && n < TF_N) Y[(size_t)(r0 + 16 * b + m) * TF_YS + n] = static_cast<TF_OUT>(C[16 * b + i]);
      }
}
