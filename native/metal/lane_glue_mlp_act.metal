
  // 64 threads per (row, 64-group): h = SiLU(gate) * up, and the down projection's group sums
  const uint t = thread_position_in_threadgroup.x;
  const uint g = threadgroup_position_in_grid.x;
  const uint m = threadgroup_position_in_grid.y;
  const int M = dims[0], MP = dims[1];
  threadgroup bfloat hb[64];
  if (int(m) >= M) {
    if (t == 0) XS[g * MP + m] = 0.0f;
    return;
  }
  const int e = int(m) * N + int(g) * 64 + int(t);
  const float gf = float(GATE[e]);
  const bfloat h = bfloat(gf / (1.0f + metal::precise::exp(-gf)) * float(UP[e]));
  HOUT[e] = h;
  hb[t] = h;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(hb[i]);
    XS[g * MP + m] = acc;
  }
