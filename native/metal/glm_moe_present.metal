
  const uint t = thread_position_in_threadgroup.x;
  device uint32_t* box = (device uint32_t*)BOX;
  for (uint e = t; e < uint(NE); e += 1024) box[e] = 0u;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = t; i < uint(N); i += 1024) box[IDS[i]] = 1u;
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0) OUT[0] = 1u;
