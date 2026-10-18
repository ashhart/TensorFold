[[kernel]] void tf_rotary_sweep(const constant float* table [[buffer(0)]],
    const constant uint4& range [[buffer(1)]], device TF_RotaryScalar* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= range.y) return;
  const uint linear = range.x + tid;
  const int i = int(linear % 32), position = int((linear / 32) * range.z);
  out[tid] = tf_rotary_scalar(i, position, table);
}
