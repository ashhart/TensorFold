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

// The register-only decode structure on our map -- one threadgroup a sixteen-column tile, one k walk, no split-K, no
// partials and no reduce dispatch, and the decoded weights stay in REGISTERS: there is no threadgroup weight tile, no
// barrier in the k loop and no threadgroup memory round trip. One warp owns a sixteen-column tile of one row, each
// lane holds two output columns and the four k rows it reads, the eight decoded weights stay in registers, and the
// four lanes that share a column reduce once, at the end of the walk. TF_FMA_WARPS: the warps a threadgroup. Each
// warp walks its own slice of the k blocks and the warps' partial sums are added once, at the end of the walk,
// through a threadgroup reduction -- so a split-K dispatch's parallelism without its partials round trip and its
// reduce dispatch. Our map is unchanged: the lane's eight states are the same eight the tensor-unit body reads (the
// same stream words, the same bit offsets, the same codebook values), so the products are the same products. The
// reduction order is the FP32 walk's, not the tensor unit's: the sums are FP32 either way, but the tensor unit
// reduces sixteen k at a time and this body adds them one at a time, so the two are not bit-identical (the numerics
// band shows the difference).
#ifndef TF_FMA_WARPS
#define TF_FMA_WARPS 1
#endif

// THE TWO-WORD FUNNEL AT halves = 6 IN THE STATIC BODY. The static body reads three stream words a lane a block (a
// run of 7*S + 16 bits; at S = 3 that is 37 bits), but at halves = 6 every lane run starts at off0 in {3,11,19,27},
// so the run is contained in two 32-bit words and only the top five bits of the funnel second register are read
// (states J = 6, 7 at bit 18/21 use c_next >> 30 / >> 27). The third word feeds only the low off0 <= 27 bits of
// c_next, which the extraction never reads. Dropping it is bit-identical BY CONSTRUCTION (the same argument as routed
// funnel); The two forms are hash-identical.
#ifndef TF_FMA_TWO
#define TF_FMA_TWO 0
#endif
// THE VALUE PATH'S FORM. The alternative form is the ARITHMETIC one the format's own decoder uses (tf_exl3_value: the
// same byte sum, the same half, the same fma) -- the LUT's entries ARE those values, entry for entry, so the two
// forms are BIT-IDENTICAL and the choice is a pure cost question. TF_FMA_VALUE_ARITH = 1 takes the arithmetic form.
#ifndef TF_FMA_VALUE_ARITH
#define TF_FMA_VALUE_ARITH 0
#endif
