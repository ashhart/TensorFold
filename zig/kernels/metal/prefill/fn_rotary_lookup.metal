
  if (LOG2BASE[TF_ROPE_ENABLE] != 0.0f && position >= 0) {
    uint hash = uint(position);
    hash = (hash ^ (hash >> 16)) * 0x7feb352du;
    hash = (hash ^ (hash >> 15)) * 0x846ca68bu;
    hash ^= hash >> 16;
    const constant int* tags = reinterpret_cast<const constant int*>(LOG2BASE + TF_ROPE_KEYS);
    for (uint probe = 0; probe < TF_ROPE_SLOTS; ++probe) {
      const uint slot = (hash + probe) & (TF_ROPE_SLOTS - 1);
      const int tag = tags[slot];
      if (tag < 0) break;
      if (tag == position) {
        const constant float4* values = reinterpret_cast<const constant float4*>(LOG2BASE + TF_ROPE_DATA);
        const float4 value = values[slot * 32 + uint(i)];
        const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
        return {value.x, value.y, value.z, value.w, value.z * ys, value.w * ys};
      }
    }
  }
