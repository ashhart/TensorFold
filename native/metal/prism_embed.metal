
  const uint t = thread_position_in_threadgroup.x;
  const uint blk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  threadgroup float buf[1024];

  const size_t row = size_t(IDS[r]);
  for (uint e = t; e < 1024; e += 512) {
    const uint k = blk * 1024 + e;
    const uint q = (W[row * (K / 16) + k / 16] >> (2 * (k % 16))) & 3u;
    const size_t g = row * (K / G) + k / G;
    buf[e] = float(SC[g]) * float(q) + float(BI[g]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint h = 1; h < 1024; h <<= 1) {
    const uint i = (t / h) * (2 * h) + (t % h);
    const float a = buf[i], b = buf[i + h];
    buf[i] = a + b;
    buf[i + h] = a - b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const size_t base = size_t(r) * K + blk * 1024;
  for (uint e = t; e < 1024; e += 512) OUT[base + e] = bfloat((buf[e] * 0.03125f) * SG[blk * 1024 + e]);
