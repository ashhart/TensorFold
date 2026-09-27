
  // Thread (d, h, r): dim d of head h of row r. Row id IDS[r][h] lies in one of 8 table groups (row starts GSTART);
  // its 4-bit value q, scale and bias give bf16(bf16(scale * q) + bias) (mx.dequantize on bf16 scales).
  const int d = int(thread_position_in_grid.x);
  const int h = int(thread_position_in_grid.y);
  const int r = int(thread_position_in_grid.z);
  const uint id = IDS[r * H + h];
  int g = 0;
  for (int j = 1; j < 8; j++) g += id >= GSTART[j] ? 1 : 0;
  const size_t row = size_t(id - GSTART[g]);
  const device uint32_t* W; const device bfloat* SC; const device bfloat* BI;
  switch (g) {
    case 0: W = W0; SC = S0; BI = B0; break;
    case 1: W = W1; SC = S1; BI = B1; break;
    case 2: W = W2; SC = S2; BI = B2; break;
    case 3: W = W3; SC = S3; BI = B3; break;
    case 4: W = W4; SC = S4; BI = B4; break;
    case 5: W = W5; SC = S5; BI = B5; break;
    case 6: W = W6; SC = S6; BI = B6; break;
    default: W = W7; SC = S7; BI = B7; break;
  }
  const uint word = W[row * (DIMS / 8) + d / 8];
  const bfloat q = bfloat(float((word >> (4 * (d % 8))) & 0xFu));
  const bfloat sc = SC[row * (DIMS / 32) + d / 32], bi = BI[row * (DIMS / 32) + d / 32];
  OUT[(r * H + h) * DIMS + d] = sc * q + bi;
