
  // Thread (c, r): output c of the down + inject rows, / S, then SiLU (c < LOW) or the gate 2 sigmoid (hc_project's).
  const int c = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  const float v4 = float(bfloat(float(DN[size_t(r) * ND + c]) / float(S)));
  if (c < LOW) ACT[size_t(r) * LOW + c] = bfloat(bsilu(v4));
  else INJ[size_t(r) * S + (c - LOW)] = bfloat(2.0f * bsig(v4));
