
  constexpr int QUADS = 8;
  constexpr int PER = K / 4;                       // inputs a lane
  constexpr int KB = K / 2;                        // bytes a weight row
  constexpr int KG = K / 64;
  const uint lane = thread_index_in_simdgroup;
  const int quad_lid = int(lane % 4), quad_gid = int(lane / 4);
  const int r = int(threadgroup_position_in_grid.x);
  const int out_row = int(threadgroup_position_in_grid.y) * QUADS * 8 + quad_gid;
  const device uint8_t* w = (const device uint8_t*)W + size_t(out_row) * KB + quad_lid * (PER / 2);
  const device bfloat* sc = S + size_t(out_row) * KG + quad_lid / (64 / PER);
  const device bfloat* bi = B + size_t(out_row) * KG + quad_lid / (64 / PER);
  const device bfloat* x = X + size_t(r) * K + quad_lid * PER;
  float xt[PER];
  float sum = 0.0f;
  for (int i = 0; i < PER; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  float result[8];
  for (int row = 0; row < 8; row++) {
    result[row] = 0.0f;
    if (row * QUADS + out_row < N) {
      const device uint16_t* ws = (const device uint16_t*)(w + size_t(row) * QUADS * KB);
      const float s = float(sc[row * QUADS * KG]), bb = float(bi[row * QUADS * KG]);
      float accum = 0.0f;
      for (int i = 0; i < PER / 4; i++)
        accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
                 xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
      result[row] += s * accum + sum * bb;
    }
  }
  for (int row = 0; row < 8; row++) {
    const float v = quad_sum(result[row]);
    if (quad_lid == 0 && row * QUADS + out_row < N) OUT[size_t(r) * N + out_row + row * QUADS] = bfloat(v);
  }
