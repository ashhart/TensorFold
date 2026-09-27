
  // Thread (d, r): dim d of token row r (quantized embedding, mx.dequantize's bf16(bf16(scale * q) + bias)),
  // written to each of the TILE copies of the row (the residual streams start as copies of the embedding).
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  const size_t row = size_t(IDS[r]);
  const uint word = W[row * (DIMS / 8) + d / 8];
  const bfloat q = bfloat(float((word >> (4 * (d % 8))) & 0xFu));
  const bfloat v = SC[row * (DIMS / 32) + d / 32] * q + BI[row * (DIMS / 32) + d / 32];
  for (int t = 0; t < TILE; t++) OUT[(r * TILE + t) * DIMS + d] = v;
