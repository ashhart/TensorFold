
  const uint i = thread_position_in_grid.x;
  device uint32_t* box = (device uint32_t*)BOX;
  if (i < uint(N)) box[i] = uint32_t(PICK[i]);
  if (i == 0) OUT[0] = 1u;
