
  // Thread (c, r): channel c of row r. The depthwise conv over [tail ; normed] rows r + DIL j (fp32 in j order,
  // bf16), SiLU (bf16 sigmoid, bf16 product), then h + (gated + silu) in bf16.
  const int c = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float y = 0.0f;
  for (int j = 0; j < TAPS; j++) y = fma(CW[c * TAPS + j], float(CIN[size_t(r + DIL * j) * W + c]), y);
  const float yb = float(bfloat(y));
  const float silu = float(bfloat(yb * bsig(yb)));
  const size_t at = size_t(r) * W + c;
  HOUT[at] = bfloat(float(H[at]) + float(bfloat(float(GATED[at]) + silu)));
