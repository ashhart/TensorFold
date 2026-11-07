kernel void tf_exl3_matmul_prefill_nax31(const device half* const x [[buffer(0)]],
                                        const device uchar* const trellis [[buffer(1)]],
                                        device float* const sums [[buffer(2)]],
                                        constant TfExl3Args& a [[buffer(3)]],
                                        const device uint* const routes [[buffer(4)]],
                                        const device TfExl3Expert* const descriptors [[buffer(5)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half wx[2 * 16 * 64];
  tf_exl3_prefill_body_db<8, 1, true, 4, false>(wx, x, trellis, sums, a, routes, descriptors, nullptr, lane, sg, tid, tptg.x, tg);
}

// The register-only decode on our map: one threadgroup a sixteen-column tile, one k walk, no split-K or reduce.
#ifndef TF_FMA_WARPS
#define TF_FMA_WARPS 1
#endif

// THE TWO-WORD FUNNEL AT halves = 6 IN THE STATIC BODY: every lane run fits two words, so the third is dead.
#ifndef TF_FMA_TWO
#define TF_FMA_TWO 0
#endif
// THE VALUE PATH'S FORM: the arithmetic decoder and the LUT hold the same values, so the choice is pure cost.
#ifndef TF_FMA_VALUE_ARITH
#define TF_FMA_VALUE_ARITH 0
#endif
