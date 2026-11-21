// A window's dense-attention softmaxes in one launch (appended to ops/softmax.metal): row z (grid y) n0 + z keys.

// Each row tf_softmax_bf16's bits: threads past its keys hold -inf, adding zero as a smaller group's empty slots do.
struct GlmSoftmaxRows {
  int n0, stride;
};

[[kernel]] void glm_softmax_rows(const device bfloat* X [[buffer(0)]], constant GlmSoftmaxRows& a [[buffer(1)]],
                                 device bfloat* OUT [[buffer(2)]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint3 t [[thread_position_in_threadgroup]],
                                 uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float tops[32];
  threadgroup float sums[32];
  const size_t at = size_t(tg.y) * a.stride;
  tf_softmax_row(X + at, OUT + at, a.n0 + int(tg.y), tg.x, t.x, sg, lane, tops, sums);
}
