// tf-glm-rows-speed's candidates: glm_rows's one-row sums at RPS columns a simdgroup, SG a group, K compiled in.
template <int BITS, int RPS, int SG, int K>
[[kernel]] void glm_rows_x(const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
                           const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
                           constant GlmRowDims& d [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint3 tg [[threadgroup_position_in_grid]]) {
  const int row0 = (int(tg.x) * SG + int(sg)) * RPS;
  if (row0 >= d.n) return;
  const int k = K > 0 ? K : d.k;
  const int h = int(tg.y);
  const size_t at = size_t(h) * d.hsw + row0;
  const device uint8_t* wr = (const device uint8_t*)W + at * (k * BITS / 8);
  const device bfloat* sr = S + at * (k / 64);
  const device bfloat* br = B + at * (k / 64);
  const device bfloat* x0 = X + size_t(h) * d.hsx;
  device bfloat* o0 = OUT + size_t(h) * d.hso + row0;
  const int cols = metal::min(RPS, d.n - row0);
  #pragma clang loop unroll(disable)
  for (int m = 0; m < d.rows; m++) {
    float acc[1][RPS];
    tf_rp_rows<RPS, 1, BITS, false>(wr, sr, br, x0 + size_t(m) * d.ldx, k, 64, lane, acc);
    if (lane == 0)
      for (int j = 0; j < cols; j++) o0[size_t(m) * d.ldo + j] = bfloat(acc[0][j]);
  }
}

#define GX(B, R, S, K) \
  template [[host_name("x_q" #B "_r" #R "_s" #S "_k" #K)]] [[kernel]] decltype(glm_rows_x<B, R, S, K>) glm_rows_x<B, R, S, K>;
#define GXK(B, K) GX(B, 4, 2, K) GX(B, 8, 2, K) GX(B, 4, 4, K) GX(B, 8, 4, K) GX(B, 2, 4, K) GX(B, 2, 8, K) GX(B, 4, 8, K)
#define GXB(B) GXK(B, 0) GXK(B, 4096) GXK(B, 8192) GXK(B, 16384) GXK(B, 12288) GXK(B, 1536) GXK(B, 2048) GXK(B, 128) GXK(B, 256) GXK(B, 512)
GXB(8)

