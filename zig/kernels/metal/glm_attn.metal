// GLM-5.3-Flash latent attention for rows reading all keys: the 64 query heads as one matrix against the one latent head.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

// fp32 parts of one output block added as one run would leave them: (0 + first) + second.
inline void glm_join(thread frag<float>& c, thread const frag<float>& d) {
  c = (frag<float>(0.0f) + c) + d;
}

// S [64, n] = Q K^T in 16-wide steps (simdgroup g: heads 16 g.., a threadgroup 32 keys); arg.y: two 256-dim halves joined.
[[kernel]] void glm_latent_scores(const device bfloat* q [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                  device bfloat* s [[buffer(2)]], constant int2& arg [[buffer(3)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const int n = arg.x, col0 = int(tg.x) * 32, row0 = int(g) * 16;
  const short2 home = frag_home(ushort(lane));
  frag<float> c0 = 0.0f, c1 = 0.0f, d0 = 0.0f, d1 = 0.0f;
  for (int k0 = 0; k0 < 512; k0 += 16) {
    frag<bfloat> a, b0, b1;
    frag_get_in(a, q, 512, row0, k0, home, 64, 512);
    frag_get_t_in(b0, keys, 512, col0, k0, home, n, 512);
    frag_get_t_in(b1, keys, 512, col0 + 16, k0, home, n, 512);
    if (arg.y != 0 && k0 >= 256) {
      mma_16x32<false, true>(d0, d1, a, b0, b1);
    } else {
      mma_16x32<false, true>(c0, c1, a, b0, b1);
    }
  }
  if (arg.y != 0) {
    glm_join(c0, d0);
    glm_join(c1, d1);
  }
  frag_put_in(c0, s, n, row0, col0, home, 64, n);
  frag_put_in(c1, s, n, row0, col0 + 16, home, 64, n);
}

// Keys [lo, hi) of O += P K in 16-wide steps: whole 256-key blocks, then the rest rounded up to 32 (zeros past hi).
inline void glm_values_run(thread frag<float>& c0, thread frag<float>& c1, const device bfloat* p,
                           const device bfloat* keys, int n, int lo, int hi, int row0, int col0, short2 home) {
  const int len = hi - lo;
  const int steps = (len / 256) * 16 + ((len % 256) + 31) / 32 * 2;
  for (int i = 0; i < steps; ++i) {
    const int k0 = lo + 16 * i;
    frag<bfloat> a, b0, b1;
    frag_get_in(a, p, n, row0, k0, home, 64, hi);
    frag_get_in(b0, keys, 512, k0, col0, home, hi, 512);
    frag_get_in(b1, keys, 512, k0, col0 + 16, home, hi, 512);
    mma_16x32<false, false>(c0, c1, a, b0, b1);
  }
}

// O [64, 512] = P K in 16-wide steps (simdgroup g: heads 16 g.., a threadgroup 32 dims); arg.y > 0: keys [0, arg.y) apart.
[[kernel]] void glm_latent_values(const device bfloat* p [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                  device bfloat* o [[buffer(2)]], constant int2& arg [[buffer(3)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const int n = arg.x, part = arg.y, col0 = int(tg.x) * 32, row0 = int(g) * 16;
  const short2 home = frag_home(ushort(lane));
  frag<float> c0 = 0.0f, c1 = 0.0f;
  glm_values_run(c0, c1, p, keys, n, 0, part > 0 ? part : n, row0, col0, home);
  if (part > 0) {
    frag<float> d0 = 0.0f, d1 = 0.0f;
    glm_values_run(d0, d1, p, keys, n, part, n, row0, col0, home);
    glm_join(c0, d0);
    glm_join(c1, d1);
  }
  frag_put_in(c0, o, 512, row0, col0, home, 64, 512);
  frag_put_in(c1, o, 512, row0, col0 + 16, home, 64, 512);
}
