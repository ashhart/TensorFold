
  const uint t = thread_position_in_threadgroup.x;
  const uint blk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;
  threadgroup float buf[1024];

  const size_t base = size_t(r) * K + blk * 1024;
  for (uint e = t; e < 1024; e += 512) buf[e] = float(X[base + e]) * SG[blk * 1024 + e];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint h = 1; h < 1024; h <<= 1) {
    const uint i = (t / h) * (2 * h) + (t % h);
    const float a = buf[i], b = buf[i + h];
    buf[i] = a + b;
    buf[i + h] = a - b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint e = t; e < 1024; e += 512) OUT[base + e] = bfloat(buf[e] * 0.03125f);
