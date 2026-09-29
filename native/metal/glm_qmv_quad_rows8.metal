
  constexpr int QUADS = 8;
  constexpr int PER = K / 4;                       // inputs (and weight bytes) a lane
  constexpr int KB = K;                            // bytes a weight row
  constexpr int KG = K / 64;
  const uint lane = thread_index_in_simdgroup;
  const int quad_lid = int(lane % 4), quad_gid = int(lane / 4);
  const int r = int(threadgroup_position_in_grid.x);
  const int out_row = int(threadgroup_position_in_grid.y) * QUADS * 8 + quad_gid;
  const device uint8_t* w = (const device uint8_t*)W + size_t(out_row) * KB + quad_lid * PER;
  const device bfloat* sc = S + size_t(out_row) * KG + quad_lid / (64 / PER);
  const device bfloat* bi = B + size_t(out_row) * KG + quad_lid / (64 / PER);
  const device bfloat* x = X + size_t(r) * K + quad_lid * PER;
  float xt[PER];
  const float sum = loadv<8, PER>(x, xt);
  float result[8];
  for (int row = 0; row < 8; row++) {
    result[row] = 0.0f;
    if (row * QUADS + out_row < N) {
      const float s = float(sc[row * QUADS * KG]), bb = float(bi[row * QUADS * KG]);
      result[row] += qdotv<8, PER>(w + size_t(row) * QUADS * KB, xt, s, bb, sum);
    }
  }
  for (int row = 0; row < 8; row++) {
    const float v = quad_sum(result[row]);
    if (quad_lid == 0 && row * QUADS + out_row < N) OUT[size_t(r) * N + out_row + row * QUADS] = bfloat(v);
  }
