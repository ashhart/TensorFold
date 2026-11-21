// Intact 8-bit/64 kv_b decode and verify absorb/unabsorb kernels in row-projection arithmetic order.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_CTL_GLM_KVB8
#define TF_CTL_GLM_KVB8 0
#endif

struct GlmKv8Args {
  uint stride, heads, rows, k, n, per_head, row_off;
};

[[max_total_threads_per_threadgroup(512)]]
[[kernel]] void glm_absorb_8b(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]],
                              const device bfloat* biases [[buffer(2)]], const device bfloat* x [[buffer(3)]],
                              device bfloat* y [[buffer(4)]], constant GlmKv8Args& a [[buffer(5)]],
                              uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const int r = int(sg);
  if (r >= int(a.rows)) return;
  const int h = int(tg.z), out_col = 32 * int(tg.y);
  if (out_col >= int(a.n)) return;
  const int cols = (TF_CTL_GLM_KVB8 == 2 && out_col + 32 >= int(a.n)) ? 0 : min(32, int(a.n) - out_col);
  if (TF_CTL_GLM_KVB8 == 1 && r == 0 && lane == 0u && tg.y == 0u && tg.z == 0u) ((device uint16_t*)x)[1] ^= 1u;
  const size_t nw = a.n / 4u, ng = a.n / 64u, w_head = size_t(h) * a.per_head + a.row_off;
  const device uint32_t* ws = w + w_head * nw + out_col / 4 + lane * nw;
  const device bfloat* sc = scales + w_head * ng + out_col / 64 + lane * ng;
  const device bfloat* bi = biases + w_head * ng + out_col / 64 + lane * ng;
  const device bfloat* xp = x + size_t(r) * a.stride + size_t(h) * a.k + lane;
  device bfloat* yp = y + (size_t(r) * a.heads + h) * a.n + out_col;
  float result[32] = {0};
  for (uint i = 0; i < a.k; i += 32u) {
    if (i + lane < a.k) {
      const float xl = *xp;
      const float scale = *sc;
      const float bias = *bi;
      const uint4 wl0 = ((const device uint4*)ws)[0];
      const uint4 wl1 = ((const device uint4*)ws)[1];
      const thread uint8_t* wb0 = (const thread uint8_t*)&wl0;
      const thread uint8_t* wb1 = (const thread uint8_t*)&wl1;
      for (int j = 0; j < 16; j++) {
        result[j] += xl * (scale * float(wb0[j]) + bias);
        result[16 + j] += xl * (scale * float(wb1[j]) + bias);
      }
    }
    xp += 32;
    sc += 32 * ng;
    bi += 32 * ng;
    ws += 32 * nw;
  }
  for (int k = 0; k < cols; k++) {
    const float v = simd_sum(result[k]);
    if (lane == 0u) yp[k] = static_cast<bfloat>(v);
  }
}

[[max_total_threads_per_threadgroup(512)]]
[[kernel]] void glm_unabsorb_8b(const device uint32_t* W [[buffer(0)]], const device bfloat* S [[buffer(1)]],
                                const device bfloat* B [[buffer(2)]], const device bfloat* X [[buffer(3)]],
                                device bfloat* OUT [[buffer(4)]], constant GlmKv8Args& a [[buffer(5)]],
                                uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  constexpr int RPS = 4;
  const int r = int(sg);
  if (r >= int(a.rows)) return;
  const int h = int(tg.z), out_row = RPS * int(tg.y);
  if (out_row >= int(a.n)) return;
  const int cols = (TF_CTL_GLM_KVB8 == 2 && out_row + RPS >= int(a.n)) ? 0 : min(RPS, int(a.n) - out_row);
  if (TF_CTL_GLM_KVB8 == 1 && r == 0 && lane == 0u && tg.y == 0u && tg.z == 0u) ((device uint16_t*)X)[1] ^= 1u;
  const size_t kg = size_t(a.k / 64u);
  const size_t wrow0 = size_t(h) * a.per_head + a.row_off + out_row;
  const device uint8_t* w = (const device uint8_t*)W + wrow0 * a.k + lane * 16u;
  const device bfloat* sc = S + wrow0 * kg + lane / 4u;
  const device bfloat* bi = B + wrow0 * kg + lane / 4u;
  const device bfloat* x = X + size_t(r) * a.stride + size_t(h) * a.k + lane * 16u;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (uint k0 = 0; k0 < a.k; k0 += 512u) {
    if (k0 + lane * 16u < a.k) {
      float xt[16];
      float sum = 0.0f;
      for (int i = 0; i < 16; i += 4) {
        const bfloat c0 = x[i], c1 = x[i + 1], c2 = x[i + 2], c3 = x[i + 3];
        sum += float(c0);
        sum += float(c1);
        sum += float(c2);
        sum += float(c3);
        xt[i] = float(c0);
        xt[i + 1] = float(c1);
        xt[i + 2] = float(c2);
        xt[i + 3] = float(c3);
      }
      for (int j = 0; j < cols; j++) {
        const uint4 raw = *(const device uint4*)(w + size_t(j) * a.k);
        float accum = 0.0f;
        for (int i = 0; i < 4; i++) {
          const uchar4 q = as_type<uchar4>(raw[i]);
          accum += (xt[4 * i] * float(q.x) + xt[4 * i + 1] * float(q.y) +
                    xt[4 * i + 2] * float(q.z) + xt[4 * i + 3] * float(q.w));
        }
        acc[j] += float(sc[size_t(j) * kg]) * accum + sum * float(bi[size_t(j) * kg]);
      }
    }
    w += 512;
    sc += 8;
    bi += 8;
    x += 512;
  }
  for (int j = 0; j < cols; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0u) OUT[(size_t(r) * a.heads + h) * a.n + out_row + j] = bfloat(v);
  }
}
