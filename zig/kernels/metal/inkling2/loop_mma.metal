// The loop blend on simdgroup matrices, for GPUs without tensor units: the scalar blend's bytes.
[[kernel, max_total_threads_per_threadgroup(32)]] void tf_ink2_loop_blend_mma(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
    const device bfloat* zeta [[buffer(2)]], const device bfloat* forget [[buffer(3)]],
    const device bfloat* gain [[buffer(4)]], device float* out [[buffer(5)]],
    constant InkLoopArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  // Threadgroup (x, y): outputs 16x .. 16x + 15 of rows 16y .. 16y + 15; forget and gain chain over the ranks.
  const uint r0 = tg.y * 16u, d0 = tg.x * 16u;
  simdgroup_matrix<float, 8, 8> dotf[2][2], dotg[2][2];
  _Pragma("clang loop unroll(full)") for (uint i = 0u; i < 2u; ++i)
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < 2u; ++j) {
      dotf[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
      dotg[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  // The ranks go in order, eight to an MMA, and an MMA adds its eight into C in order.
  _Pragma("clang loop unroll(full)") for (uint o = 0u; o < INKLING_LOOP_RANK / 8u; ++o) {
    simdgroup_matrix<bfloat, 8, 8> z[2], wf[2], wg[2];
    _Pragma("clang loop unroll(full)") for (uint i = 0u; i < 2u; ++i)
      simdgroup_load(z[i], zeta + size_t(r0 + 8u * i) * INKLING_LOOP_RANK + 8u * o, INKLING_LOOP_RANK);
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < 2u; ++j) {
      simdgroup_load(wf[j], forget + size_t(d0 + 8u * j) * INKLING_LOOP_RANK + 8u * o, INKLING_LOOP_RANK, ulong2(0, 0), true);
      simdgroup_load(wg[j], gain + size_t(d0 + 8u * j) * INKLING_LOOP_RANK + 8u * o, INKLING_LOOP_RANK, ulong2(0, 0), true);
    }
    _Pragma("clang loop unroll(full)") for (uint i = 0u; i < 2u; ++i)
      _Pragma("clang loop unroll(full)") for (uint j = 0u; j < 2u; ++j) {
        simdgroup_multiply_accumulate(dotf[i][j], z[i], wf[j], dotf[i][j]);
        simdgroup_multiply_accumulate(dotg[i][j], z[i], wg[j], dotg[i][j]);
      }
  }
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  _Pragma("clang loop unroll(full)") for (uint i = 0u; i < 2u; ++i)
    _Pragma("clang loop unroll(full)") for (uint j = 0u; j < 2u; ++j)
      _Pragma("clang loop unroll(full)") for (uint e = 0u; e < 2u; ++e) {
        const uint row = r0 + 8u * i + fm, d = d0 + 8u * j + fn + e;
        if (row >= a.rows) continue;
        const float f = precise::tanh(dotf[i][j].thread_elements()[e]);
        const float g = precise::tanh(dotg[i][j].thread_elements()[e]);
        // Four separately rounded binary32 steps, as the scalar kernel does.
        const float keep = 1.0f - f;
        const float left = keep * h[size_t(row) * INKLING_LOOP_H + d];
        const float right = g * m[size_t(row) * a.stride + d];
        out[size_t(row) * INKLING_LOOP_H + d] = left + right;
      }
}
