// D = A B on the M5 tensor units, bit-identical to MLX 0.32.3's steel_gemm_fused_nax_nn (bf16, fp32 accumulate).
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
// tf:header
#include "../nax.h"

// Simdgroup s: columns 32 s .. 32 s + 31, every 16-row band; 2 ceil(K / 32) chunks of 16, zeros past K, as MLX's.
inline void tf_nax_gemm(const device bfloat* a, const device bfloat* b, device bfloat* d, const constant int32_t* p,
                        const constant int64_t* batch, uint3 tg, uint sg, uint lane) {
  const int M = p[0], N = p[1], K = p[2];
  const int lda = p[3], ldb = p[4], ldd = p[5];
  const long zo = long(tg.z) / batch[0], zi = long(tg.z) % batch[0];
  a += zo * batch[1] + zi * batch[2];
  b += zo * batch[3] + zi * batch[4];
  d += long(tg.z) * batch[5];
  const int col0 = int(tg.x) * 128 + 32 * int(sg);
  if (col0 >= N) return;
  const short2 o = tfp::frag_home(ushort(lane));
  const int chunks = (K / 256) * 16 + ((K % 256) + 31) / 32 * 2;
  for (int row0 = int(tg.y) * 64; row0 < min(M, int(tg.y) * 64 + 64); row0 += 16) {
    tfp::frag<float> c0 = 0.0f, c1 = 0.0f;
    for (int ch = 0; ch < chunks; ++ch) {
      const int k0 = 16 * ch;
      tfp::frag<bfloat> fa, f0, f1;
      tfp::frag_get_in(fa, a, lda, row0, k0, o, M, K);
      tfp::frag_get_in(f0, b, ldb, k0, col0, o, K, N);
      tfp::frag_get_in(f1, b, ldb, k0, col0 + 16, o, K, N);
      tfp::mma_16x32<false, false>(c0, c1, fa, f0, f1);
    }
    tfp::frag_put_in(c0, d, ldd, row0, col0, o, M, N);
    tfp::frag_put_in(c1, d, ldd, row0, col0 + 16, o, M, N);
  }
}

// tf:kernel tf_nax_gemm_nn_bf16 inputs=A,B,P,BATCH outputs=D
[[kernel]] void tf_nax_gemm_nn_bf16(
  const device bfloat16_t* A [[buffer(0)]],
  const device bfloat16_t* B [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant int64_t* BATCH [[buffer(3)]],
  device bfloat16_t* D [[buffer(4)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  tf_nax_gemm(A, B, D, P, BATCH, threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
              thread_index_in_simdgroup);
}
