// music3 GEMMs Y[b] = X[b] W^T (+ bias) (+ Y[b]), fp32 sums, one rounding: bf16 / fp16 on the tensor units.
#include "../prefill/gemm_nax.metal"

inline float rh(float x) { return float(half(x)); }

struct LinearArgs {
  int M, N, K, ldx, ldy, has_bias, accumulate, pad;
  long sx, sy;  // batch strides (elements) of X and Y
};

// Epilogues: plain (+ bias, + Y); kGlu (W rows in 32s: 16 values, their 16 gates) the GLU at col / 2; kRope q, k.
enum Mode { kPlain = 0, kGlu = 1, kRope = 2 };

// A 64x128 block a threadgroup, a 32x32 block a simdgroup (2 x 4), as MLX's steel_gemm_fused_nax.
template <typename T, int MODE>
inline void linear(const device T* X, const device T* W, const device T* bias, const device float* cs, device T* Y,
                   constant LinearArgs& a, uint3 tg, uint sg, uint lane) {
  const int row = int(tg.y) * 64 + 32 * int(sg / 4), col = int(tg.x) * 128 + 32 * int(sg % 4);
  X += long(tg.z) * a.sx + long(row) * a.ldx;
  W += long(col) * a.K;
  Y += long(tg.z) * a.sy + long(row) * a.ldy + (MODE == kGlu ? col / 2 : col);
  const int rows = min(32, a.M - row), cols = min(32, a.N - col);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc[2][2];
  if (rows == 32 && cols == 32) {
    tfp::gemm_block<T, false, true, true, 256>(acc, X, W, a.ldx, a.K, a.K, a.K / 256, rows, cols, home);
  } else {
    tfp::gemm_block<T, false, true, false, 256>(acc, X, W, a.ldx, a.K, a.K, a.K / 256, rows, cols, home);
  }
  if (rows <= 0 || cols <= 0) return;
  if (MODE != kPlain) {
    const bool rot = MODE == kRope && col < 4096 && col % 64 == 0;  // dims (i, i + 16) of a q or k head, at row % Lseq
    for (short i = 0; i < 2; i++) {
      for (short e = 0; e < 8; e++) {
        const int r = 16 * i + home.y + (e >> 2) * 8, c = home.x + TF_COL(e);
        if (r >= rows) continue;
        const float x0 = float(T(acc[i][0][e])), x1 = float(T(acc[i][1][e]));  // the GEMM's rounded outputs
        device T* y = Y + long(r) * a.ldy + c;
        if (MODE == kGlu) {
          const int p = col / 2 + c;
          const float v = rh(x0 + float(bias[p])), g = rh(x1 + float(bias[a.N / 2 + p]));
          *y = T(v * rh(g / (1.0f + fast::exp(-g))));  // fast: precise exp here spills the GEMM (2.5x slower)
        } else if (rot) {
          const int j = (row + r) % a.pad;
          const float cn = cs[(j * 16 + c) * 2], sn = cs[(j * 16 + c) * 2 + 1];
          y[0] = T(x0 * cn - x1 * sn);
          y[16] = T(x1 * cn + x0 * sn);
        } else {
          y[0] = T(x0);
          y[16] = T(x1);
        }
      }
    }
    return;
  }
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < 2; j++) {
      for (short e = 0; e < 8; e++) {
        const int r = 16 * i + home.y + (e >> 2) * 8, c = 16 * j + home.x + TF_COL(e);
        if (r >= rows || c >= cols) continue;
        float v = acc[i][j][e];
        if (a.has_bias) v += float(bias[col + c]);
        device T* y = Y + long(r) * a.ldy + c;
        if (a.accumulate) v += float(*y);
        *y = T(v);
      }
    }
  }
}

#define LINEAR(NAME, T, MODE)                                                                                      \
  [[max_total_threads_per_threadgroup(256)]] kernel void NAME(                                                    \
      const device T* X [[buffer(0)]], const device T* W [[buffer(1)]], const device T* bias [[buffer(2)]],       \
      device T* Y [[buffer(3)]], const device float* cs [[buffer(4)]], constant LinearArgs& a [[buffer(5)]],      \
      uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]) {                                                                  \
    linear<T, MODE>(X, W, bias, cs, Y, a, tg, sg, lane);                                                          \
  }
LINEAR(m3_linear_b, bfloat, kPlain)
LINEAR(m3_linear_h, half, kPlain)
LINEAR(m3_linear_glu_h, half, kGlu)
LINEAR(m3_linear_rope_h, half, kRope)

// fp32 Y (+)= X W^T from bf16 halves x = xh + xl, w = wh + wl: xh wh + (xh wl + xl wh), about 16 bits an operand.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_linear_s(
    const device bfloat* Xh [[buffer(0)]], const device bfloat* Xl [[buffer(1)]], const device bfloat* Wh [[buffer(2)]],
    const device bfloat* Wl [[buffer(3)]], device float* Y [[buffer(4)]], constant LinearArgs& a [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tg.y) * 64 + 32 * int(sg / 4), col = int(tg.x) * 128 + 32 * int(sg % 4);
  const long xo = long(tg.z) * a.sx + long(row) * a.ldx, wo = long(col) * a.K;
  Y += long(tg.z) * a.sy + long(row) * a.ldy + col;
  const int rows = min(32, a.M - row), cols = min(32, a.N - col);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> hh[2][2], cross[2][2];
  for (short i = 0; i < 2; i++)
    for (short j = 0; j < 2; j++) hh[i][j] = cross[i][j] = tfp::frag<float>(0);
  if (rows <= 0 || cols <= 0) return;
  const bool full = rows == 32 && cols == 32 && a.K % 16 == 0;
  Xh += xo;
  Xl += xo;
  Wh += wo;
  Wl += wo;
  for (int k0 = 0; k0 < a.K; k0 += 16) {  // each operand's halves read once a step for all three products
    tfp::frag<bfloat> ah[2], al[2], bh[2], bl[2];
    for (short i = 0; i < 2; i++) {
      if (full) {
        tfp::frag_get(ah[i], Xh, a.ldx, 16 * i, k0, home);
        tfp::frag_get(al[i], Xl, a.ldx, 16 * i, k0, home);
        tfp::frag_get_t(bh[i], Wh, a.K, 16 * i, k0, home);
        tfp::frag_get_t(bl[i], Wl, a.K, 16 * i, k0, home);
      } else {
        tfp::frag_get_in(ah[i], Xh, a.ldx, 16 * i, k0, home, rows, a.K);
        tfp::frag_get_in(al[i], Xl, a.ldx, 16 * i, k0, home, rows, a.K);
        tfp::frag_get_t_in(bh[i], Wh, a.K, 16 * i, k0, home, cols, a.K);
        tfp::frag_get_t_in(bl[i], Wl, a.K, 16 * i, k0, home, cols, a.K);
      }
    }
    for (short i = 0; i < 2; i++) {
      tfp::mma_16x32<false, true>(hh[i][0], hh[i][1], ah[i], bh[0], bh[1]);
      tfp::mma_16x32<false, true>(cross[i][0], cross[i][1], ah[i], bl[0], bl[1]);
      tfp::mma_16x32<false, true>(cross[i][0], cross[i][1], al[i], bh[0], bh[1]);
    }
  }
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < 2; j++) {
      for (short e = 0; e < 8; e++) {
        const int r = 16 * i + home.y + (e >> 2) * 8, c = 16 * j + home.x + TF_COL(e);
        if (r >= rows || c >= cols) continue;
        float v = hh[i][j][e] + cross[i][j][e];
        device float* y = Y + long(r) * a.ldy + c;
        if (a.accumulate) v += *y;
        *y = v;
      }
    }
  }
}

// x = hi + lo in bf16 halves (hi = bf16(x), lo = bf16(x - hi)), n values.
[[max_total_threads_per_threadgroup(256)]] kernel void m3_split(const device float* x [[buffer(0)]], device bfloat* hi [[buffer(1)]],
                     device bfloat* lo [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const bfloat h = bfloat(x[i]);
  hi[i] = h;
  lo[i] = bfloat(x[i] - float(h));
}
