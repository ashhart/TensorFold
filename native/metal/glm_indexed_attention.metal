
  const uint gid = threadgroup_position_in_grid.y;          // row * HEADS + head
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  constexpr int SIMD_GROUPS = 32;
  constexpr int SIMD_WIDTH = 32;
  constexpr int qk_per_thread = QK_DIM / SIMD_WIDTH;
  constexpr int v_per_thread = QK_DIM / SIMD_WIDTH;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread];
  threadgroup U outputs[SIMD_GROUPS * SIMD_WIDTH];
  threadgroup U max_scores[SIMD_GROUPS];
  threadgroup U sum_exp_scores[SIMD_GROUPS];
  const int row = int(gid) / HEADS;
  const int key_length = int(meta[0]);
  const size_t stride = size_t(QK_DIM);
  const device bfloat* qptr = queries + size_t(gid) * QK_DIM + int(simd_lid) * qk_per_thread;
  device bfloat* optr = out + size_t(gid) * QK_DIM + int(simd_gid) * v_per_thread;
  const U s = U(scale[0]);
  for (int i = 0; i < qk_per_thread; i++) q[i] = s * static_cast<U>(qptr[i]);
  for (int i = 0; i < v_per_thread; i++) o[i] = 0;
  const int indices_offset = row * TOPK;
  U max_score = -3.4028234663852886e38f;
  U sum_exp_score = 0;
  for (int selected_idx = int(simd_gid); selected_idx < TOPK; selected_idx += SIMD_GROUPS) {
    const int key_pos = int(indices[indices_offset + selected_idx]);
    const bool valid = key_pos >= 0 && key_pos < key_length;
    U score = -3.4028234663852886e38f;
    if (valid) {
      const device bfloat* kptr = keys + size_t(key_pos) * stride + int(simd_lid) * qk_per_thread;
      score = 0;
      for (int j = 0; j < qk_per_thread; j++) score += q[j] * static_cast<U>(kptr[j]);
      score = simd_sum(score);
    }
    const U new_max = max(max_score, score);
    const U factor = fast::exp(max_score - new_max);
    const U exp_score = valid ? fast::exp(score - new_max) : U(0);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    if (valid) {
      const device bfloat* vptr = keys + size_t(key_pos) * stride + int(simd_lid) * v_per_thread;
      for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
    } else {
      for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor;
    }
  }
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  const U new_max = simd_max(max_score);
  const U factor = fast::exp(max_score - new_max);
  const U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * SIMD_WIDTH + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * SIMD_WIDTH + simd_lid] * factor);
    o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) optr[i] = static_cast<bfloat>(o[i]);
  }
