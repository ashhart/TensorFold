// Inkling 2 prompt projections on simdgroup MMA, bit-equal to the one-row bf16 matvec (mv.metal): 32 chains, xor tree.

#ifndef TF_CTL_DENSE
#define TF_CTL_DENSE 0 // planted faults (controls): 1 flush non-finite x, 2 drop the last chunk, 3 fp16 sums, 4 swap kd
#endif

#define I2D_UNR _Pragma("clang loop unroll(full)")

// Chain = k mod 32; an MMA sums k = chain + 32 kd. A words hold kd 2pp, 2pp+1 of a row, B words two columns at one kd.
struct I2DenseArgs {
  uint k, n, rows;
};

// Two rows' four consecutive bf16 as four fragment words: word v holds element v of `a` low and of `b` high.
inline void i2_dense_zip(uint2 a, uint2 b, thread uint (&w)[4]) {
  w[0] = (a.x & 0xFFFFu) | (b.x << 16u);
  w[1] = (a.x >> 16u) | (b.x & 0xFFFF0000u);
  w[2] = (a.y & 0xFFFFu) | (b.y << 16u);
  w[3] = (a.y >> 16u) | (b.y & 0xFFFF0000u);
}

// Puts X in fragment order: a 256-wide chunk of a row becomes the words [group 8][pp 4][chain 4] of two kd.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_dense_perm(
    const device uint2* X [[buffer(0)]], device uint4* Y [[buffer(1)]], constant I2DenseArgs& a [[buffer(2)]],
    uint2 id [[thread_position_in_grid]]) {
  const uint chunk = id.x >> 5u, g = (id.x >> 2u) & 7u, pp = id.x & 3u;
  const device uint2* p = X + ulong(id.y) * (a.k >> 2u) + chunk * 64u + 16u * pp + g;
  uint w[4];
  if (TF_CTL_DENSE == 4) i2_dense_zip(p[8], p[0], w);
  else i2_dense_zip(p[0], p[8], w);
  Y[ulong(id.y) * (a.k >> 3u) + id.x] = uint4(w[0], w[1], w[2], w[3]);
}

// 16 x 16 outputs a threadgroup; simdgroup g owns chains 4g..4g+3 and reads its fragments straight from device memory.
template <typename T>
inline void i2_dense_mma(const device uint4* X, const device uint2* W, device T* Y, constant I2DenseArgs& a, uint2 tg,
                         uint lane, uint g, threadgroup float (&part)[8][2][2][8][8]) {
  const uint n0 = tg.x * 16u, r0 = tg.y * 16u;
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  simdgroup_matrix<float, 8, 8> acc[2][2][4];
  I2D_UNR for (uint i = 0; i < 2; ++i) I2D_UNR for (uint j = 0; j < 2; ++j) I2D_UNR for (uint c = 0; c < 4; ++c)
    acc[i][j][c] = simdgroup_matrix<float, 8, 8>(0.0f);
  // A fragment: row fm, kd fn and fn + 1. B fragment: kd fm, columns fn and fn + 1. Edge tiles read a valid neighbour.
  const device uint4* xr[2];
  const device uint2* wc0[2];
  const device uint2* wc1[2];
  I2D_UNR for (uint i = 0; i < 2; ++i)
    xr[i] = X + ulong(min(r0 + 8u * i + fm, a.rows - 1u)) * (a.k >> 3u) + g * 4u + (fn >> 1u);
  I2D_UNR for (uint j = 0; j < 2; ++j) {
    wc0[j] = W + ulong(min(n0 + 8u * j + fn, a.n - 1u)) * (a.k >> 2u) + 8u * fm + g;
    wc1[j] = W + ulong(min(n0 + 8u * j + fn + 1u, a.n - 1u)) * (a.k >> 2u) + 8u * fm + g;
  }
  const uint chunks = TF_CTL_DENSE == 2 ? (a.k >> 8u) - 1u : a.k >> 8u;
  for (uint ch = 0; ch < chunks; ++ch) {
    uint4 xv[2];
    uint wv[2][4];
    I2D_UNR for (uint i = 0; i < 2; ++i) xv[i] = xr[i][ch * 32u];
    I2D_UNR for (uint j = 0; j < 2; ++j) i2_dense_zip(wc0[j][ch * 64u], wc1[j][ch * 64u], wv[j]);
    I2D_UNR for (uint c = 0; c < 4; ++c) {
      simdgroup_matrix<bfloat, 8, 8> xa[2], wb[2];
      I2D_UNR for (uint i = 0; i < 2; ++i) {
        uint xp = xv[i][c];
        if (TF_CTL_DENSE == 1) {
          if ((xp & 0x7F80u) == 0x7F80u) xp &= 0xFFFF0000u;
          if ((xp & 0x7F800000u) == 0x7F800000u) xp &= 0x0000FFFFu;
        }
        xa[i].thread_elements()[0] = as_type<bfloat>(ushort(xp));
        xa[i].thread_elements()[1] = as_type<bfloat>(ushort(xp >> 16u));
        wb[i].thread_elements()[0] = as_type<bfloat>(ushort(wv[i][c]));
        wb[i].thread_elements()[1] = as_type<bfloat>(ushort(wv[i][c] >> 16u));
      }
      I2D_UNR for (uint i = 0; i < 2; ++i) I2D_UNR for (uint j = 0; j < 2; ++j) {
        simdgroup_multiply_accumulate(acc[i][j][c], xa[i], wb[j], acc[i][j][c]);
        if (TF_CTL_DENSE == 3) {
          acc[i][j][c].thread_elements()[0] = float(half(acc[i][j][c].thread_elements()[0]));
          acc[i][j][c].thread_elements()[1] = float(half(acc[i][j][c].thread_elements()[1]));
        }
      }
    }
  }
  // tree levels 1-2 inside a group, levels 3-5 across the eight groups through threadgroup memory
  I2D_UNR for (uint i = 0; i < 2; ++i) I2D_UNR for (uint j = 0; j < 2; ++j) I2D_UNR for (uint e = 0; e < 2; ++e) {
    float v[4];
    I2D_UNR for (uint c = 0; c < 4; ++c) v[c] = acc[i][j][c].thread_elements()[e];
    part[g][i][j][fm][fn + e] = (v[0] + v[1]) + (v[2] + v[3]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint o = g * 32u + lane, i = o >> 7u, j = (o >> 6u) & 1u, m = (o >> 3u) & 7u, n = o & 7u;
  float s[8];
  I2D_UNR for (uint q = 0; q < 8; ++q) s[q] = part[q][i][j][m][n];
  const float t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
  const uint row = r0 + i * 8u + m, col = n0 + j * 8u + n;
  if (row < a.rows && col < a.n) Y[ulong(row) * a.n + col] = T(t);
}

#define I2_DENSE_KERNEL(NAME, T)                                                                                   \
  [[kernel, max_total_threads_per_threadgroup(256)]] void NAME(                                                  \
      const device uint4* X [[buffer(0)]], const device uint2* W [[buffer(1)]], device T* Y [[buffer(2)]],        \
      constant I2DenseArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],                          \
      uint lane [[thread_index_in_simdgroup]], uint g [[simdgroup_index_in_threadgroup]]) {                      \
    threadgroup float part[8][2][2][8][8];                                                                       \
    i2_dense_mma<T>(X, W, Y, a, tg, lane, g, part);                                                              \
  }

I2_DENSE_KERNEL(inkling2_dense_mma_bf16, bfloat)
I2_DENSE_KERNEL(inkling2_dense_mma_f32, float)
