// Inkling 2 prompt projections on simdgroup MMA, bit-equal to the one-row bf16 matvec (mv.metal): 32 chains, xor tree.

#ifndef TF_CTL_DENSE
#define TF_CTL_DENSE 0 // planted faults (contract controls): 1 flush non-finite x, 2 drop the last chunk, 3 fp16 sums
#endif

#define I2D_UNR _Pragma("clang loop unroll(full)")

struct I2DenseArgs {
  uint k, n, rows;
};

// X [rows][k], W [n][k] bf16, k % 256 == 0, 16-byte aligned: 16 x 16 outputs a threadgroup, simdgroup g chains 8g..8g+7
template <typename T>
inline void i2_dense_mma(const device uint4* X, const device uint4* W, device T* Y, constant I2DenseArgs& a,
                         uint2 tg, uint lane, uint g, threadgroup uint (&tg_x)[4][8][2][8][4],
                         threadgroup uint (&tg_w)[4][8][2][8][4], threadgroup float (&part)[4][2][2][8][8]) {
  const uint n0 = tg.x * 16u, r0 = tg.y * 16u;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  simdgroup_matrix<float, 8, 8> acc[2][2][8];
  I2D_UNR for (uint i = 0; i < 2; ++i) I2D_UNR for (uint j = 0; j < 2; ++j) I2D_UNR for (uint c = 0; c < 8; ++c)
    acc[i][j][c] = simdgroup_matrix<float, 8, 8>(0.0f);
  const uint kv = a.k >> 3u;
  const uint pp = lane & 3u;
  const uint k_lim = TF_CTL_DENSE == 2 ? a.k - 256u : a.k;
  for (uint k0 = 0; k0 < k_lim; k0 += 256u) {
    I2D_UNR for (uint t = 0; t < 2; ++t) {
      const uint idx = lane + 32u * t;
      {
        const uint m = idx >> 2u;
        const uint row = r0 + m;
        uint4 v0 = uint4(0), v1 = uint4(0);
        if (row < a.rows) {
          const device uint4* p = X + ulong(row) * kv + ((k0 + 64u * pp + g * 8u) >> 3u);
          v0 = p[0];
          v1 = p[4];
        }
        I2D_UNR for (uint w = 0; w < 4; ++w) {
          tg_x[g][2 * w][m >> 3u][m & 7u][pp] = (v0[w] & 0xFFFFu) | (v1[w] << 16u);
          tg_x[g][2 * w + 1][m >> 3u][m & 7u][pp] = (v0[w] >> 16u) | (v1[w] & 0xFFFF0000u);
        }
      }
      {
        const uint p8 = (idx >> 2u) & 7u, nr = idx >> 5u;
        const uint col = n0 + nr * 8u + 2u * pp;
        uint4 v0 = uint4(0), v1 = uint4(0);
        const uint off = (k0 + 32u * p8 + g * 8u) >> 3u;
        if (col < a.n) v0 = W[ulong(col) * kv + off];
        if (col + 1u < a.n) v1 = W[ulong(col + 1u) * kv + off];
        I2D_UNR for (uint w = 0; w < 4; ++w) {
          tg_w[g][2 * w][nr][p8][pp] = (v0[w] & 0xFFFFu) | (v1[w] << 16u);
          tg_w[g][2 * w + 1][nr][p8][pp] = (v0[w] >> 16u) | (v1[w] & 0xFFFF0000u);
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    I2D_UNR for (uint c = 0; c < 8; ++c) {
      simdgroup_matrix<bfloat, 8, 8> xa[2], wb[2];
      I2D_UNR for (uint i = 0; i < 2; ++i) {
        uint xp = tg_x[g][c][i][fm][fn >> 1u];
        const uint wp = tg_w[g][c][i][fm][fn >> 1u];
        if (TF_CTL_DENSE == 1) {
          if ((xp & 0x7F80u) == 0x7F80u) xp &= 0xFFFF0000u;
          if ((xp & 0x7F800000u) == 0x7F800000u) xp &= 0x0000FFFFu;
        }
        xa[i].thread_elements()[0] = as_type<bfloat>(ushort(xp));
        xa[i].thread_elements()[1] = as_type<bfloat>(ushort(xp >> 16u));
        wb[i].thread_elements()[0] = as_type<bfloat>(ushort(wp));
        wb[i].thread_elements()[1] = as_type<bfloat>(ushort(wp >> 16u));
      }
      I2D_UNR for (uint i = 0; i < 2; ++i) I2D_UNR for (uint j = 0; j < 2; ++j) {
        simdgroup_multiply_accumulate(acc[i][j][c], xa[i], wb[j], acc[i][j][c]);
        if (TF_CTL_DENSE == 3) {
          acc[i][j][c].thread_elements()[0] = float(half(acc[i][j][c].thread_elements()[0]));
          acc[i][j][c].thread_elements()[1] = float(half(acc[i][j][c].thread_elements()[1]));
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  // tree levels 1-3 inside a group, levels 4-5 across groups through threadgroup memory
  I2D_UNR for (uint i = 0; i < 2; ++i) I2D_UNR for (uint j = 0; j < 2; ++j) I2D_UNR for (uint e = 0; e < 2; ++e) {
    float v[8];
    I2D_UNR for (uint c = 0; c < 8; ++c) v[c] = acc[i][j][c].thread_elements()[e];
    I2D_UNR for (uint s = 1u; s < 8u; s *= 2u) I2D_UNR for (uint c = 0; c < 8u; c += 2u * s) v[c] = v[c] + v[c + s];
    part[g][i][j][fm][fn + e] = v[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = g * 32u + lane; o < 256u; o += 128u) {
    const uint i = o >> 7u, j = (o >> 6u) & 1u, m = (o >> 3u) & 7u, n = o & 7u;
    const float t = (part[0][i][j][m][n] + part[1][i][j][m][n]) + (part[2][i][j][m][n] + part[3][i][j][m][n]);
    const uint row = r0 + i * 8u + m, col = n0 + j * 8u + n;
    if (row < a.rows && col < a.n) Y[ulong(row) * a.n + col] = T(t);
  }
}

#define I2_DENSE_KERNEL(NAME, T)                                                                                   \
  [[kernel, max_total_threads_per_threadgroup(128)]] void NAME(                                                  \
      const device uint4* X [[buffer(0)]], const device uint4* W [[buffer(1)]], device T* Y [[buffer(2)]],        \
      constant I2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],                          \
      uint lane [[thread_index_in_simdgroup]], uint g [[simdgroup_index_in_threadgroup]]) {                      \
    threadgroup uint tg_x[4][8][2][8][4];                                                                        \
    threadgroup uint tg_w[4][8][2][8][4];                                                                        \
    threadgroup float part[4][2][2][8][8];                                                                       \
    i2_dense_mma<T>(X, W, Y, a, tg, lane, g, tg_x, tg_w, part);                                                  \
  }

I2_DENSE_KERNEL(inkling2_dense_mma_bf16, bfloat)
I2_DENSE_KERNEL(inkling2_dense_mma_f32, float)
