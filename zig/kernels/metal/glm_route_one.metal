// tf_route_logits for a one-row window (appended to core/moe_route.metal): row 0's sums alone, in the same order.
[[kernel]] void glm_route_logits_one(const device bfloat* X [[buffer(0)]], const device uint4* RP [[buffer(1)]],
                                     constant int& rows [[buffer(2)]], device TF_OUT_T* OUT [[buffer(3)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                     uint s [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[8][4];
  const int q = int(tg.x);
  const device uint4* w = RP + ((size_t(q) * 8 + s) * ITERS + PERL * lane) * 2;
  float acc[4]; // [expert]
  for (int tn = 0; tn < 4; tn++) acc[tn] = 0.0f;
  for (int ii = 0; ii < PERL; ii++) {
    float inter[4][4];
    for (int h = 0; h < 2; h++) {
      const uint4 v = w[ii * 2 + h];
      const uint words[4] = {v.x, v.y, v.z, v.w};
      for (int j = 0; j < 4; j++) {
        const int e = h * 8 + j * 2;
        inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
        inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
      }
    }
    const device bfloat* x = X + 32 * (PERL * int(lane) + ii) + 4 * int(s);
    for (int tm = 0; tm < 4; tm++) {
      const float xv = float(x[tm]);
      for (int tn = 0; tn < 4; tn++) acc[tn] += xv * inter[tm][tn];
    }
  }
  for (int tn = 0; tn < 4; tn++) {
    const float v = simd_sum(acc[tn]);
    if (lane == 0) part[s][tn] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int o = int(tpos.x);
  if (o < 4) {
    float v = part[0][o];
    for (int k = 1; k < 8; k++) v += part[k][o];
#ifdef TF_ROUND_LOGITS
    v = float(bfloat(v));
#endif
    OUT[4 * q + o] = static_cast<TF_OUT_T>(v);
  }
}
