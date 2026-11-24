// The loop gates of eight rows a threadgroup on simdgroup matrices, for GPUs without tensor units: the scalar bytes.
struct InkLoopMmaArgs { uint rows, m_stride; };

// A threadgroup's tile is eight rows; the staging holds 256 columns of their normalized inputs.
#define INKLING_LOOP_MMA_ROWS 8u
#define INKLING_LOOP_MMA_STAGE 256u

[[kernel, max_total_threads_per_threadgroup(256)]] void tf_inkling2_loop_gates_mma(
    const device float* h [[buffer(0)]], const device float* m [[buffer(1)]],
    const device bfloat* down [[buffer(2)]], const device bfloat* forget [[buffer(3)]],
    const device bfloat* gain [[buffer(4)]], device float* out [[buffer(5)]],
    constant InkLoopMmaArgs& a [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row0 = tg * INKLING_LOOP_MMA_ROWS;
  threadgroup float sums[2 * INKLING_LOOP_MMA_ROWS][8], inverse[2 * INKLING_LOOP_MMA_ROWS];
  threadgroup bfloat staged[INKLING_LOOP_MMA_ROWS * INKLING_LOOP_MMA_STAGE];
  threadgroup bfloat zeta[INKLING_LOOP_MMA_ROWS * INKLING_LOOP_RANK];
  // Each row's two sums of squares reduce as the scalar gates do: 256 strided chains, simd_sum, then the eight.
  for (uint i = 0; i < 2 * INKLING_LOOP_MMA_ROWS; ++i) {
    const uint row = row0 + i / 2;
    float ss = 0.0f;
    if (row < a.rows) {
      const device float* input = (i & 1u) ? m + size_t(row) * a.m_stride : h + size_t(row) * INKLING_LOOP_H;
      for (uint d = t; d < INKLING_LOOP_H; d += 256) {
        const float x = input[d];
        ss += x * x;
      }
    }
    const float reduced = simd_sum(ss);
    if (lane == 0) sums[i][sg] = reduced;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 2 * INKLING_LOOP_MMA_ROWS) {
    float ss = 0.0f;
    for (uint j = 0; j < 8; ++j) ss += sums[t][j];
    inverse[t] = precise::rsqrt(precise::divide(ss, 2304.0f) + INKLING_LOOP_EPS);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint qid = lane >> 2u;
  const uint fm = (qid & 4u) + ((lane >> 1u) & 3u);
  const uint fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
  // Output rank 8 sg .. 8 sg + 7 over the 2 H inputs in order: an MMA adds its eight products into C in order.
  simdgroup_matrix<float, 8, 8> zacc(0.0f);
  const device bfloat* zrow = down + size_t(8u * sg) * (2 * INKLING_LOOP_H);
  for (uint chunk = 0; chunk < 2 * INKLING_LOOP_H / INKLING_LOOP_MMA_STAGE; ++chunk) {
    const uint source = chunk / (INKLING_LOOP_H / INKLING_LOOP_MMA_STAGE);
    const uint d0 = (chunk % (INKLING_LOOP_H / INKLING_LOOP_MMA_STAGE)) * INKLING_LOOP_MMA_STAGE;
    for (uint i = 0; i < INKLING_LOOP_MMA_ROWS; ++i) {
      const uint row = row0 + i;
      float x = 0.0f;
      if (row < a.rows)
        x = (source == 0 ? h[size_t(row) * INKLING_LOOP_H + d0 + t] : m[size_t(row) * a.m_stride + d0 + t]) *
            inverse[2 * i + source];
      staged[i * INKLING_LOOP_MMA_STAGE + t] = bfloat(x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)") for (uint o = 0; o < INKLING_LOOP_MMA_STAGE / 8u; ++o) {
      simdgroup_matrix<bfloat, 8, 8> fa, fb;
      simdgroup_load(fa, staged + 8u * o, INKLING_LOOP_MMA_STAGE);
      simdgroup_load(fb, zrow + chunk * INKLING_LOOP_MMA_STAGE + 8u * o, 2 * INKLING_LOOP_H, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(zacc, fa, fb, zacc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  _Pragma("clang loop unroll(full)") for (uint e = 0; e < 2u; ++e)
    zeta[fm * INKLING_LOOP_RANK + 8u * sg + fn + e] = bfloat(zacc.thread_elements()[e]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The forget and gain dots chain over the ranks in order, eight to an MMA; the blend is the scalar one's four steps.
  for (uint tile = sg; tile < INKLING_LOOP_H / 8u; tile += 8u) {
    const uint d0 = tile * 8u;
    simdgroup_matrix<float, 8, 8> dotf(0.0f), dotg(0.0f);
    _Pragma("clang loop unroll(full)") for (uint o = 0; o < INKLING_LOOP_RANK / 8u; ++o) {
      simdgroup_matrix<bfloat, 8, 8> z, wf, wg;
      simdgroup_load(z, zeta + 8u * o, INKLING_LOOP_RANK);
      simdgroup_load(wf, forget + size_t(d0) * INKLING_LOOP_RANK + 8u * o, INKLING_LOOP_RANK, ulong2(0, 0), true);
      simdgroup_load(wg, gain + size_t(d0) * INKLING_LOOP_RANK + 8u * o, INKLING_LOOP_RANK, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(dotf, z, wf, dotf);
      simdgroup_multiply_accumulate(dotg, z, wg, dotg);
    }
    const uint row = row0 + fm;
    if (row >= a.rows) continue;
    _Pragma("clang loop unroll(full)") for (uint e = 0; e < 2u; ++e) {
      const uint d = d0 + fn + e;
      const float f = precise::tanh(dotf.thread_elements()[e]);
      const float g = precise::tanh(dotg.thread_elements()[e]);
      // Four separately rounded binary32 steps, as the scalar kernel does.
      const float keep = 1.0f - f;
      const float left = keep * h[size_t(row) * INKLING_LOOP_H + d];
      const float right = g * m[size_t(row) * a.m_stride + d];
      out[size_t(row) * INKLING_LOOP_H + d] = left + right;
    }
  }
}
