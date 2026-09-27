
  // One threadgroup of NE threads per row, a thread an expert: softmax in fp32 (sums in simdgroup order), then
  // each expert's rank = experts with a larger probability, or an equal one and a lower id; ranks below TOPK are the
  // picks, in rank order; weights / their sum (in rank order) in fp32, bf16 out.
  const uint e = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int r = int(threadgroup_position_in_grid.x);
  threadgroup float red[NE / 32];
  threadgroup float probs[NE];
  threadgroup float picked[TOPK];
  const float logit = float(L[r * NL + int(e)]);
  float m = simd_max(logit);
  if (lane == 0) red[g] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0];
  for (int k = 1; k < NE / 32; k++) m = metal::max(m, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float x = metal::exp(logit - m);
  const float zs = simd_sum(x);
  if (lane == 0) red[g] = zs;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float z = 0.0f;
  for (int k = 0; k < NE / 32; k++) z += red[k];
  const float p = x / z;
  probs[e] = p;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int rank = 0;
  for (int j = 0; j < NE; j++) {
    const float q = probs[j];
    rank += (q > p || (q == p && j < int(e))) ? 1 : 0;
  }
  if (rank < TOPK) {
    EXPERTS[r * TOPK + rank] = uint32_t(e);
    picked[rank] = p;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (e == 0) {
    float total = 0.0f;
    for (int k = 0; k < TOPK; k++) total += picked[k];
    for (int k = 0; k < TOPK; k++) WEIGHTS[r * TOPK + k] = bfloat(picked[k] / total);
  }
