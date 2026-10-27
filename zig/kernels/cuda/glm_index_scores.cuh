// GLM-5.3-Flash DSA indexer, a prompt chunk's pool scores: tf_glm_scores' arithmetic with each key tile staged once.
#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace glm_index {

constexpr int D = 128, BP = 64, RB = 8, KQ = D / 16;  // key width, pools a tile, rows a CTA (a warp each), k16 steps

#ifndef GLM_INDEX_HOST  // tools/glm_index_emulate supplies these on the host
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldm4(uint32_t* r, const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif

// Element offset of pool p's key: in order (pt null), or row p % 128 of page pt[p / 128] (glm_pages::at<128>).
__device__ __forceinline__ int64_t key_at(const int* pt, int64_t page_elems, int64_t p) {
  return pt ? (int64_t)pt[p / 128] * page_elems + (p % 128) * D : p * D;
}

// Logical 16-byte chunk u of key row `row` (0..63) in the tile: the 16-row groups' layout of tf_glm_scores.
__device__ __forceinline__ int tile_at(int row, int u) { return row * D + ((u ^ (row & 7)) << 3); }

}  // namespace glm_index

// tf_glm_scores bit for bit, 8 rows (a warp each) sharing each 64-pool key tile; grid (cdiv(rows, 8), cdiv(nb, loop)).
extern "C" __global__ void __launch_bounds__(glm_index::RB * 32)
    tf_glm_index_scores(const __nv_bfloat16* qi, const __nv_bfloat16* wts, int64_t w_stride, const __nv_bfloat16* pk,
                        float* out, const int* pos, int64_t rows, int64_t np, int64_t nb, float scale, float wscale,
                        int64_t heads, int loop, const int* pt, int64_t ps) {
  using namespace glm_index;
  __shared__ __align__(16) __nv_bfloat16 tile[2][BP * D];  // two key tiles
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int64_t r = (int64_t)blockIdx.x * RB + warp, P = pos[0];
  const bool live = r < rows;
  const int64_t npool = live ? (P + r + 1) / 4 : 0;
  const int64_t last = min(rows, (int64_t)(blockIdx.x + 1) * RB) - 1, nmax = (P + last + 1) / 4;  // the CTA's largest
  // this row's query fragments for the 8 k16 steps and 2 head tiles (ldmatrix's A layout), and a lane's 4 head weights
  uint32_t aq[KQ][2][4];
#pragma unroll
  for (int kq = 0; kq < KQ; ++kq)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int h0 = i * 16 + g, h1 = h0 + 8;
      const __nv_bfloat16* b0 = qi + ((live ? r : 0) * heads + (h0 < heads ? h0 : 0)) * D + kq * 16 + 2 * q;
      const __nv_bfloat16* b1 = qi + ((live ? r : 0) * heads + (h1 < heads ? h1 : 0)) * D + kq * 16 + 2 * q;
      const bool o0 = live && h0 < heads, o1 = live && h1 < heads;
      aq[kq][i][0] = o0 ? *reinterpret_cast<const uint32_t*>(b0) : 0u;
      aq[kq][i][1] = o1 ? *reinterpret_cast<const uint32_t*>(b1) : 0u;
      aq[kq][i][2] = o0 ? *reinterpret_cast<const uint32_t*>(b0 + 8) : 0u;
      aq[kq][i][3] = o1 ? *reinterpret_cast<const uint32_t*>(b1 + 8) : 0u;
    }
  float wv[4];
  for (int i = 0; i < 4; ++i) {
    const int64_t hh = g + 8 * i;
    wv[i] = live && hh < heads ? __fmul_rn(__bfloat162float(wts[r * w_stride + hh]), wscale) : 0.0f;
  }
  // The CTA's 256 threads copy tile pb's keys (pools past the CTA's visible ones read as 0).
  auto stage = [&](int buf, int64_t pb) {
    for (int idx = threadIdx.x; idx < BP * (D / 8); idx += RB * 32) {
      const int row = idx / (D / 8), u = idx % (D / 8);
      const int64_t p = pb * BP + row;
      const bool ok = p < nmax;
      cp16(&tile[buf][tile_at(row, u)], pk + key_at(pt, ps / 2, ok ? p : 0) + u * 8, ok);
    }
  };
  const int64_t first = (int64_t)blockIdx.y * loop;
  auto any = [&](int j) { return j < loop && first + j < nb && (first + j) * BP < nmax; };
  if (any(0)) stage(0, first);
  cp_commit();
  for (int j = 0; j < loop; ++j) {
    const int64_t pb = first + j, p0 = pb * BP;
    if (pb >= nb) break;
    cp_wait();
    __syncthreads();  // tile j is in, and every warp is done with tile j - 1
    if (any(j + 1)) stage((j + 1) & 1, pb + 1);
    cp_commit();
    if (!live) continue;
    if (p0 >= npool) {  // the row sees none of these pools: -inf
      for (int e = lane; e < BP; e += 32)
        if (p0 + e < np) out[r * np + p0 + e] = -INFINITY;
      continue;
    }
    for (int gi = 0; gi < BP / 16; ++gi) {
      float acc[2][2][4] = {};
      const __nv_bfloat16* keys = tile[j & 1] + gi * 16 * D;
#pragma unroll
      for (int kq = 0; kq < KQ; ++kq) {
        uint32_t b[4];
        ldm4(b, keys + tile_at((lane >> 4) * 8 + (lane & 7), kq * 2 + ((lane >> 3) & 1)));
        for (int nt = 0; nt < 2; ++nt)
          for (int i = 0; i < 2; ++i) mma16816(acc[i][nt], aq[kq][i], b + 2 * nt);
      }
      for (int nt = 0; nt < 2; ++nt)
        for (int e = 0; e < 2; ++e) {
          const float v0 = fmaxf(__fmul_rn(acc[0][nt][e], scale), 0.0f),
                      v1 = fmaxf(__fmul_rn(acc[0][nt][e + 2], scale), 0.0f);
          const float v2 = fmaxf(__fmul_rn(acc[1][nt][e], scale), 0.0f),
                      v3 = fmaxf(__fmul_rn(acc[1][nt][e + 2], scale), 0.0f);
          float s = __fmul_rn(wv[1], v1);
          s = __fmaf_rn(wv[0], v0, s);
          s = __fmaf_rn(wv[2], v2, s);
          s = __fmaf_rn(wv[3], v3, s);
          for (int o = 16; o >= 4; o >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, o);
          const int64_t p = p0 + gi * 16 + nt * 8 + q * 2 + e;
          if (g == 0 && p < np) out[r * np + p] = p < npool ? s : -INFINITY;
        }
    }
  }
}
