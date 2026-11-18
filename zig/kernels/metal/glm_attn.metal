// GLM-5.3-Flash latent attention, rows reading all keys: the 64 query heads as one matrix against the one latent head.
#include <metal_stdlib>
using namespace metal;
#include "../nax.h"
using namespace tfp;

// fp32 parts of one output block added as one run would leave them: (0 + first) + second.
inline void glm_join(thread frag<float>& c, thread const frag<float>& d) {
  c = (frag<float>(0.0f) + c) + d;
}

// S[64,n] = QK^T in 16-wide steps (heads 16 hg.., keys 32 kb..); halves: two 256-dim halves joined.
inline void glm_scores_at(const device bfloat* q, const device bfloat* keys, device bfloat* s, int n, bool halves,
                          int kb, int hg, uint lane) {
  const int col0 = kb * 32, row0 = hg * 16;
  const short2 home = frag_home(ushort(lane));
  frag<float> c0 = 0.0f, c1 = 0.0f, d0 = 0.0f, d1 = 0.0f;
  for (int k0 = 0; k0 < 512; k0 += 16) {
    frag<bfloat> a, b0, b1;
    frag_get_in(a, q, 512, row0, k0, home, 64, 512);
    frag_get_t_in(b0, keys, 512, col0, k0, home, n, 512);
    frag_get_t_in(b1, keys, 512, col0 + 16, k0, home, n, 512);
    if (halves && k0 >= 256) {
      mma_16x32<false, true>(d0, d1, a, b0, b1);
    } else {
      mma_16x32<false, true>(c0, c1, a, b0, b1);
    }
  }
  if (halves) {
    glm_join(c0, d0);
    glm_join(c1, d1);
  }
  frag_put_in(c0, s, n, row0, col0, home, 64, n);
  frag_put_in(c1, s, n, row0, col0 + 16, home, 64, n);
}

// One row: simdgroup g heads 16 g.., a threadgroup 32 keys; arg.y: two 256-dim halves joined (fewer than 256 keys).
[[kernel]] void glm_latent_scores(const device bfloat* q [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                  device bfloat* s [[buffer(2)]], constant int2& arg [[buffer(3)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  glm_scores_at(q, keys, s, arg.x, arg.y != 0, int(tg.x), int(g), lane);
}

// A window's rows at once: row z (grid z) has n0 + z keys; its query, scores and output `stride` apart.
struct GlmAttnRows {
  int n0, q_stride, s_stride, o_stride;
};

// Each row as glm_latent_scores leaves it: heads 16 (tg.y * simdgroups + g).., keys 32 tg.x.. (past n: nothing).
[[kernel]] void glm_latent_scores_rows(const device bfloat* q [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                       device bfloat* s [[buffer(2)]], constant GlmAttnRows& a [[buffer(3)]],
                                       uint3 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
                                       uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int z = int(tg.z), n = a.n0 + z;
  if (int(tg.x) * 32 >= n) return;
  glm_scores_at(q + size_t(z) * a.q_stride, keys, s + size_t(z) * a.s_stride, n, n < 256, int(tg.x),
                int(tg.y * sgs + g), lane);
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

// O[64,512] = PK in 16-wide steps (heads 16 hg.., dims 32 db..); part > 0: keys [0, part) apart.
inline void glm_values_at(const device bfloat* p, const device bfloat* keys, device bfloat* o, int n, int part, int db,
                          int hg, uint lane) {
  const int col0 = db * 32, row0 = hg * 16;
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

// One row: simdgroup g heads 16 g.., a threadgroup 32 dims; arg.y > 0: keys [0, arg.y) apart.
[[kernel]] void glm_latent_values(const device bfloat* p [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                  device bfloat* o [[buffer(2)]], constant int2& arg [[buffer(3)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  glm_values_at(p, keys, o, arg.x, arg.y, int(tg.x), int(g), lane);
}

// Each row as glm_latent_values leaves it (past 1,024 keys the first 1,024 apart, as the one-row launch sets).
[[kernel]] void glm_latent_values_rows(const device bfloat* p [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                       device bfloat* o [[buffer(2)]], constant GlmAttnRows& a [[buffer(3)]],
                                       uint3 tg [[threadgroup_position_in_grid]], uint g [[simdgroup_index_in_threadgroup]],
                                       uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int z = int(tg.z), n = a.n0 + z;
  glm_values_at(p + size_t(z) * a.s_stride, keys, o + size_t(z) * a.o_stride, n, n > 1024 ? 1024 : 0, int(tg.x),
                int(tg.y * sgs + g), lane);
}
