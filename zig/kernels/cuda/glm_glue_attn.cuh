// latent.py's MLA attention and sparse.py's index scores by hand: the chunk kernels' online softmax, the chunk merge.
#pragma once

#include "glm_glue_router.cuh"
#include <cuda_fp8.h>
#include "common/mla.cuh"

// The 16-head form (decode windows) and 32-head form (prompt chunks), each in glm_attn::wide_smem<HB>() bytes of smem.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok, const int* cnt,
                     float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w, float scale, int sparse,
                     const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<16>(qa, lc, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, pt, pstep);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok, const int* cnt,
                     float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w, float scale, int sparse,
                     const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<32>(qa, lc, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse, pt, pstep);
}

// E4M3 cache variants reconstruct the existing BF16 shared tile before the unchanged attention math.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                         const int* tok, const int* cnt, float* po, float* pm, float* pl,
                         int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn::chunks_wide<16, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
                                 po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0, scales);
}
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                         const int* tok, const int* cnt, float* po, float* pm, float* pl,
                         int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn::chunks_wide<32, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
                                 po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0, scales);
}

// E4M3 page addresses enter the unchanged chunk arithmetic.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_16_fp8_paged(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                               const int* tok, const int* cnt, float* po, float* pm, float* pl,
                               int64_t rows, int64_t heads, int64_t w, float scale, int sparse,
                               const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<16, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
                                  po, pm, pl, rows, heads, w, scale, sparse, pt, pstep, scales);
}

// E4M3 page addresses enter the unchanged chunk arithmetic.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_chunks_32_fp8_paged(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
                               const int* tok, const int* cnt, float* po, float* pm, float* pl,
                               int64_t rows, int64_t heads, int64_t w, float scale, int sparse,
                               const int* pt, int64_t pstep) {
  glm_attn::chunks_wide<32, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
                                  po, pm, pl, rows, heads, w, scale, sparse, pt, pstep, scales);
}

// latent._merge, block (row, head), 128 threads: chunks merged in order, o / l by div.full; skips sparse count-0 rows.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_merge(const float* po, const float* pm, const float* pl, __nv_bfloat16* out, const int* cnt, int64_t rows,
                 int64_t heads, int nch, int sparse) {
  constexpr int LW = glm_attn::LW;
  const int64_t r = blockIdx.x, h = blockIdx.y;
  if (sparse && cnt[r] == 0) return;
  const int k0 = threadIdx.x * 4;
  float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int c = 0; c < nch; ++c) {
    const int64_t base = (c * rows + r) * heads + h;
    const float cm = pm[base], cl = pl[base];
    const bool active = cl > 0.0f;
    const float nm = active ? fmaxf(m, cm) : m;
    const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
    if (c == 0) {
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(b, po[base * LW + k0 + j], 0.0f);
      l = __fmaf_rn(b, cl, 0.0f);
    } else {
      const float a = active ? (m == -INFINITY ? 0.0f : glm_triton::exp_approx(m - nm)) : 1.0f;
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(o[j], a, __fmul_rn(b, po[base * LW + k0 + j]));
      l = __fmaf_rn(l, a, __fmul_rn(b, cl));
    }
    m = nm;
  }
  for (int j = 0; j < 4; ++j) out[(r * heads + h) * LW + k0 + j] = __float2bfloat16_rn(glm_triton::div_full(o[j], l));
}

// sparse._scores, block (row, L pool blocks) of 4 warps, 16 pools a warp: sum_h w relu(dots scale), -inf past visible.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_scores(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, const __nv_bfloat16* pk,
                  float* out, const int* pos, int64_t np, int64_t nb, float scale, float wscale, int64_t heads,
                  int loop, const int* pt, int64_t ps) {
  constexpr int D = 128, HP = 32, BP = 64;
  __shared__ __align__(16) __nv_bfloat16 sm[4 * 2 * 16 * D];  // q's 32 rows, then each warp's two 16-pool buffers
  auto at = [](int row, int u) { return row * D + ((u ^ (row & 7)) << 3); };
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int64_t r = blockIdx.x, npool = ((int64_t)pos[0] + r + 1) / 4;
  for (int idx = threadIdx.x; idx < HP * (D / 8); idx += 128) {
    const int hh = idx / (D / 8), u = idx % (D / 8);
    const bool ok = hh < heads;
    glm_attn::cp16(sm + at(hh, u), qi + (r * heads + (ok ? hh : 0)) * D + u * 8, ok);
  }
  glm_attn::commit();
  float wv[4];
  for (int i = 0; i < 4; ++i) {
    const int64_t hh = g + 8 * i;
    wv[i] = hh < heads ? glm_triton::fmul(__bfloat162float(wts[r * w_stride + hh]), wscale) : 0.0f;
  }
  // A block's pools for this warp: rows p0 .. p0 + 15 of pk (16 | p0: one page), 16 chunks of 16 bytes, 8 a lane.
  auto stage = [&](int buf, int64_t p0) {
    const __nv_bfloat16* src = pk + glm_pages::at<128>(pt, ps / 2, p0 < npool ? p0 : 0, D);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int idx = lane + t * 32, row = idx / (D / 8), u = idx % (D / 8);
      const int64_t p = p0 + row;
      const bool ok = p < npool;
      glm_attn::cp16(sm + (warp * 2 + buf) * 16 * D + at(row, u), src + (ok ? row : 0) * D + u * 8, ok);
    }
  };
  const int64_t first = (int64_t)blockIdx.y * loop;
  // Blocks that score: those under nb with visible pools (pb * BP < npool); the rest only write -inf.
  auto live = [&](int j) { const int64_t pb = first + j; return j < loop && pb < nb && pb * BP < npool; };
  glm_attn::wait<0>();
  __syncthreads();
  uint32_t aq[D / 16][2][4];
#pragma unroll
  for (int kq = 0; kq < D / 16; ++kq)
#pragma unroll
    for (int i = 0; i < 2; ++i) glm_attn::ldm4<false>(aq[kq][i], sm + at(i * 16 + (lane & 15), kq * 2 + (lane >> 4)));
  __syncthreads();  // q's fragments are in registers: the key buffers take its place
  if (live(0)) stage(0, first * BP + warp * 16);
  glm_attn::commit();
  for (int j = 0; j < loop; ++j) {
    const int64_t pb = first + j;
    if (pb >= nb) break;
    const int64_t p0 = pb * BP + warp * 16;
    if (!live(j)) {
      for (int e = lane; e < 16; e += 32)
        if (p0 + e < np) out[r * np + p0 + e] = -INFINITY;
      continue;
    }
    if (live(j + 1)) stage((j + 1) & 1, p0 + BP);
    glm_attn::commit();
    glm_attn::wait<1>();
    __syncwarp();
    float acc[2][2][4] = {};
    const __nv_bfloat16* keys = sm + (warp * 2 + (j & 1)) * 16 * D;
#pragma unroll
    for (int kq = 0; kq < D / 16; ++kq) {
      uint32_t b[4];
      glm_attn::ldm4<false>(b, keys + at((lane >> 4) * 8 + (lane & 7), kq * 2 + ((lane >> 3) & 1)));
      for (int nt = 0; nt < 2; ++nt)
        for (int i = 0; i < 2; ++i) glm_router::mma16816(acc[i][nt], aq[kq][i], b + 2 * nt);
    }
    __syncwarp();
    for (int nt = 0; nt < 2; ++nt)
      for (int e = 0; e < 2; ++e) {
        const float v0 = fmaxf(__fmul_rn(acc[0][nt][e], scale), 0.0f),
                    v1 = fmaxf(__fmul_rn(acc[0][nt][e + 2], scale), 0.0f);
        const float v2 = fmaxf(__fmul_rn(acc[1][nt][e], scale), 0.0f),
                    v3 = fmaxf(__fmul_rn(acc[1][nt][e + 2], scale), 0.0f);
        float s = glm_triton::fmul(wv[1], v1);
        s = __fmaf_rn(wv[0], v0, s);
        s = __fmaf_rn(wv[2], v2, s);
        s = __fmaf_rn(wv[3], v3, s);
        for (int o = 16; o >= 4; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
        const int64_t p = p0 + nt * 8 + q * 2 + e;
        if (g == 0 && p < np) out[r * np + p] = p < npool ? s : -INFINITY;
      }
  }
}
