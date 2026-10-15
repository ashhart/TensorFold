// Bolt's ggml blocks (IQ2_XXS, Q8_0) on mma.sync: grouped experts and dense projections, decode rows and prompt rows.
//
// The weight is mma's A operand (16 output columns a warp tile), the rows are B (8 rows an n8 tile), so a lane's
// output columns are the ones it decoded and each 32-input group's scale stays in the lane. The "k" order is ours,
// the same in both operands: in a 32-input group, lane t takes inputs 8t .. 8t + 3 in step 0 and 8t + 4 .. 8t + 7 in
// step 1, so an IQ2_XXS lane decodes one whole grid entry a column. Codes become exact bf16 integers; each group
// sums into fp32 by mma, then takes its column's fp32 scale.
//
// Weights are repacked at load (tf_ggml_pack), a pure byte permutation of the checkpoint's rows, same size:
//   IQ2_XXS [n/16][k/256][1056]: 8 groups of [gq 8][col gq, gq + 8][8 bytes qs], then [gq 8][2] fp16 d
//   Q8_0    [n/16][k/32][544]:   [lane 32][(step 0, col gq), (0, gq + 8), (1, gq), (1, gq + 8)][4 int8], then d as above
//                                (lane (gq, t) step s holds inputs 8t + 4s .. + 3)
// A matrix's 16-column tiles follow each other; a grouped layer's experts follow each other.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "experts.cuh"
#include "ggml_iq2xxs_tables.cuh"

namespace tf_ggml {

__device__ __forceinline__ uint4 ld_nc(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_nc32(const uint8_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ float half_f(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short)bits)); }

// The sign bits of a bf16 pair from bits 0 and 1 of sb (< 2^16): bit 0 lands on 15, bit 1 on 31, no other bit there.
__device__ __forceinline__ uint32_t sign_mask(uint32_t sb) { return (sb * 0x40008000u) & 0x80008000u; }

// Two grid bytes of g (selector sel) as exact bf16 integers, signs from bits 0 and 1 of sb.
__device__ __forceinline__ uint32_t grid_pair(uint32_t g, uint32_t sel, uint32_t sb) {
  uint32_t v = __byte_perm(g, 0x43u, sel);
  const uint32_t k = K128;
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), *reinterpret_cast<const __nv_bfloat162*>(&k));
  return *reinterpret_cast<uint32_t*>(&r) ^ sign_mask(sb);
}

// IQ2_XXS: a group is a 32-weight sub-block (4 grid indices, 4 x 7 sign bits, a 4-bit scale); 8 groups a block.
struct Iq2 {
  static constexpr int G = 8, BYTES = 1056, Q = 4;  // Q: groups a decode load brings, a lane's 16 bytes each
  static constexpr bool TABLE = true;
  // lane (gq, t) loads group 4q + t's bytes of columns gq, gq + 8: a warp's load is 512 distinct bytes
  static __device__ __forceinline__ const uint8_t* uptr(const uint8_t* tile, int q, int lane) {
    return tile + (q >> 1) * BYTES + (q & 1) * 512 + (lane & 3) * 128 + (lane >> 2) * 16;
  }
  static __device__ __forceinline__ const uint8_t* wptr(const uint8_t* tile, int g, int lane) {
    return tile + (g >> 3) * BYTES + (g & 7) * 128 + (lane >> 2) * 16;
  }
  static __device__ __forceinline__ const uint8_t* dptr(const uint8_t* tile, int g, int lane) {
    return tile + (g >> 3) * BYTES + 1024 + (lane >> 2) * 4;
  }
  static __device__ __forceinline__ void decode(const uint4& w, uint32_t dp, int t, const uint2* tab,
                                                uint32_t (&a)[2][4], float (&sc)[2]) {
    sc[0] = half_f(dp & 0xFFFFu) * (0.5f + (float)(w.y >> 28)) * 0.25f;
    sc[1] = half_f(dp >> 16) * (0.5f + (float)(w.w >> 28)) * 0.25f;
#pragma unroll
    for (int c = 0; c < 2; ++c) {  // lane t decodes grid entry t of the group: 8 weights, both steps
      const uint32_t aux0 = c ? w.z : w.x, aux1 = c ? w.w : w.y;
      const uint2 g = tab[(aux0 >> (8 * t)) & 255u];
      const uint32_t si = (aux1 >> (7 * t)) & 127u;
      const uint32_t sb = si | ((__popc(si) & 1u) << 7);  // ksigns_iq2xs: even parity in bit 7
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const uint32_t gs = s ? g.y : g.x, bs = sb >> (4 * s);
        a[s][c] = grid_pair(gs, 0x4140u, bs);
        a[s][2 + c] = grid_pair(gs, 0x4342u, bs >> 2);
      }
    }
  }
};

// Two int8 of u (bytes 2h, 2h + 1) as exact bf16.
__device__ __forceinline__ uint32_t i8_pair(uint32_t u, int h) {
  const uint32_t v = u ^ 0x80808080u;
  const float f0 = __uint_as_float(__byte_perm(v, 0x4B000000u, 0x7440u | (2 * h))) - 8388736.f;
  const float f1 = __uint_as_float(__byte_perm(v, 0x4B000000u, 0x7440u | (2 * h + 1))) - 8388736.f;
  __nv_bfloat162 r = __floats2bfloat162_rn(f0, f1);
  return *reinterpret_cast<uint32_t*>(&r);
}

// Q8_0: a group is one block (fp16 d, 32 int8).
struct Q8 {
  static constexpr int G = 1, BYTES = 544, Q = 1;
  static constexpr bool TABLE = false;
  static __device__ __forceinline__ const uint8_t* uptr(const uint8_t* tile, int q, int lane) {
    return tile + q * BYTES + lane * 16;
  }
  static __device__ __forceinline__ const uint8_t* wptr(const uint8_t* tile, int g, int lane) {
    return tile + g * BYTES + lane * 16;
  }
  static __device__ __forceinline__ const uint8_t* dptr(const uint8_t* tile, int g, int lane) {
    return tile + g * BYTES + 512 + (lane >> 2) * 4;
  }
  static __device__ __forceinline__ void decode(const uint4& w, uint32_t dp, int, const uint2*,
                                                uint32_t (&a)[2][4], float (&sc)[2]) {
    sc[0] = half_f(dp & 0xFFFFu);
    sc[1] = half_f(dp >> 16);
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint32_t u = comp(w, 2 * s + c);
        a[s][c] = i8_pair(u, 0);
        a[s][2 + c] = i8_pair(u, 1);
      }
  }
};

// IQ2_XXS's grid in shared memory, 256 entries of 8 bytes: indices diverge across a warp. Signs and the bf16
// conversion are arithmetic; tables of them cost more in bank conflicts than they save.
constexpr int TABLE_WORDS = 256;

template <class F>
__device__ __forceinline__ void load_tables(uint2* tab) {
  if constexpr (F::TABLE) {
    for (int i = threadIdx.x; i < TABLE_WORDS; i += blockDim.x) {
      const uint64_t g = tf_iq2xxs_grid[i];
      tab[i] = make_uint2((uint32_t)g, (uint32_t)(g >> 32));
    }
    __syncthreads();
  }
}

// One item of rows: grouped (expert, first member, count) from the plan, or a dense run of rows (members null).
struct Item {
  int e, first, cnt;
};

__device__ __forceinline__ int pair_of(const int* __restrict__ members, const Item& it, int r) {
  return members ? __ldg(members + it.first + r) : it.first + r;
}

template <int EPI>
__device__ __forceinline__ void store(void* out, size_t at, float v) {
  if constexpr (EPI == 0) {
    reinterpret_cast<float*>(out)[at] = v;
  } else {
    reinterpret_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(EPI == 1 ? relu2(v) : v);
  }
}

// One 32-input group into acc: NT n8 row tiles, B fragments by `ldb(nt, s)` (uint2: b0, b1).
template <class F, int NT, class LDB>
__device__ __forceinline__ void group(float (&acc)[NT][4], const uint4& w, uint32_t dp, int t, const uint2* tab,
                                      LDB ldb) {
  uint32_t a[2][4];
  float sc[2];
  F::decode(w, dp, t, tab, a, sc);
  float p[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) p[nt][0] = p[nt][1] = p[nt][2] = p[nt][3] = 0.f;
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const uint2 b = ldb(nt, s);
      mma(p[nt], a[s][0], a[s][1], a[s][2], a[s][3], b.x, b.y);
    }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    acc[nt][0] = fmaf(p[nt][0], sc[0], acc[nt][0]);
    acc[nt][1] = fmaf(p[nt][1], sc[0], acc[nt][1]);
    acc[nt][2] = fmaf(p[nt][2], sc[1], acc[nt][2]);
    acc[nt][3] = fmaf(p[nt][3], sc[1], acc[nt][3]);
  }
}

// acc[nt] holds columns col0 + gq (0, 1) and col0 + gq + 8 (2, 3) of rows nt * 8 + 2t (0, 2) and + 1 (1, 3).
template <int EPI, int NT>
__device__ __forceinline__ void epilogue_rows(const float (&acc)[NT][4], void* out, int N, int col0, int lane,
                                              const int* __restrict__ members, const Item& it) {
  const int gq = lane >> 2, t = lane & 3;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int r = nt * 8 + 2 * t + j;
      if (r >= it.cnt) continue;
      const size_t row = (size_t)pair_of(members, it, r) * N;
      store<EPI>(out, row + col0 + gq, acc[nt][j]);
      store<EPI>(out, row + col0 + gq + 8, acc[nt][2 + j]);
    }
}

constexpr int U = 2;  // loads a warp has in flight, twice over

// Decode rows: a CTA per TPC (item, 16-column tile) units, KW warps each over K's loads (Q groups a load), summed in
// a fixed order.
template <class F, int EPI, int KW, int TPC>
__device__ __forceinline__ void gemv_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                          const uint8_t* __restrict__ W, int KG, int NT16,
                                          const int* __restrict__ items, const int* __restrict__ counts,
                                          const int* __restrict__ members, void* __restrict__ out, int N, int rows,
                                          int skip) {
  __shared__ uint2 tab[F::TABLE ? TABLE_WORDS : 1];
  __shared__ float red[TPC][KW > 1 ? KW - 1 : 1][8][32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gq = lane >> 2, t = lane & 3;
  const int sub = warp / KW, kw = warp - sub * KW;
  const int units = (items ? __ldg(counts) : 1) * NT16;
  if (blockIdx.x * TPC >= units) return;
  const int unit = blockIdx.x * TPC + sub, iu = unit / NT16, tile = unit - iu * NT16;
  Item it = {0, 0, 0};
  if (unit < units) it = items ? Item{__ldg(items + 3 * iu), __ldg(items + 3 * iu + 1), __ldg(items + 3 * iu + 2)}
                               : Item{0, 0, rows};
  const bool live = unit < units && it.e != skip;
  const uint8_t* wt = W + ((size_t)it.e * NT16 + tile) * (size_t)(KG / F::G) * F::BYTES;
  const int KQ = KG / F::Q;
  uint4 w[U];
  uint32_t dp[U];
  auto fetch = [&](int q0) {
#pragma unroll
    for (int u = 0; u < U; ++u)
      if (q0 + u * KW < KQ) {
        w[u] = ld_nc(F::uptr(wt, q0 + u * KW, lane));
        dp[u] = ld_nc32(F::dptr(wt, (q0 + u * KW) * F::Q, lane));
      }
  };
  if (live) fetch(kw);  // the weights are on their way while the tables load
  load_tables<F>(tab);
  const bool v0 = gq < it.cnt, v1 = gq + 8 < it.cnt, two = it.cnt > 8;
  const int p0 = v0 ? pair_of(members, it, gq) : 0, p1 = v1 ? pair_of(members, it, gq + 8) : 0;
  const __nv_bfloat16* x0 = X + (size_t)(slots ? p0 / slots : p0) * x_stride + 8 * t;
  const __nv_bfloat16* x1 = X + (size_t)(slots ? p1 / slots : p1) * x_stride + 8 * t;
  float acc[2][4];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
  // a load's rows come in at once, Q groups' b fragments (steps 0 and 1) for row tiles 0 and 1
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  auto xrows = [&](const __nv_bfloat16* x, bool v, int g) {
    return v ? __ldg(reinterpret_cast<const uint4*>(x + g * 32)) : zero;
  };
  for (int q = live ? kw : KQ; q < KQ; q += U * KW) {
    uint4 cw[U];
    uint32_t cd[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      cw[u] = w[u];
      cd[u] = dp[u];
    }
    fetch(q + U * KW);
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int qu = q + u * KW;
      if (qu >= KQ) break;
      uint4 xr[F::Q][2];
#pragma unroll
      for (int j = 0; j < F::Q; ++j) {
        xr[j][0] = xrows(x0, v0, qu * F::Q + j);
        xr[j][1] = two ? xrows(x1, v1, qu * F::Q + j) : zero;
      }
#pragma unroll
      for (int j = 0; j < F::Q; ++j) {
        uint4 wv = cw[u];
        if constexpr (F::Q > 1) {  // group 4qu + j of columns gq, gq + 8 sits in lane (gq, j)
          const int src = (lane & ~3) | j;
          wv = make_uint4(__shfl_sync(~0u, wv.x, src), __shfl_sync(~0u, wv.y, src), __shfl_sync(~0u, wv.z, src),
                          __shfl_sync(~0u, wv.w, src));
        }
        auto ldb = [&](int nt, int s) {
          const uint4& v = xr[j][nt];
          return s ? make_uint2(v.z, v.w) : make_uint2(v.x, v.y);
        };
        if (two) {
          group<F, 2>(acc, wv, cd[u], t, tab, ldb);
        } else {
          float(&a1)[1][4] = *reinterpret_cast<float(*)[1][4]>(&acc[0]);
          group<F, 1>(a1, wv, cd[u], t, tab, ldb);
        }
      }
    }
  }
  if constexpr (KW > 1) {
    if (kw > 0)
#pragma unroll
      for (int q = 0; q < 8; ++q) red[sub][kw - 1][q][lane] = acc[q >> 2][q & 3];
    __syncthreads();
    if (kw > 0) return;
    for (int wi = 0; wi < KW - 1; ++wi)
#pragma unroll
      for (int q = 0; q < 8; ++q) acc[q >> 2][q & 3] += red[sub][wi][q][lane];
  }
  if (!live) return;
  epilogue_rows<EPI, 2>(acc, out, N, tile * 16, lane, members, it);
}

// Prompt rows: a CTA per (item of up to 64 rows, WN 16-column tiles), rows staged 128 inputs a stage.
constexpr int BM = 64, SK = 128, XC = SK / 8, STAGES = 3;  // XC: 16-byte chunks a staged row; 48 KiB of stages

__device__ __forceinline__ int chunk(int r, int c) { return r * XC + (c ^ (r & 7)); }

template <class F, int EPI, int WN, int NT>
__device__ __forceinline__ void prompt_tiles(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                             const uint8_t* __restrict__ W, int KG, int NT16,
                                             const int* __restrict__ members, void* __restrict__ out, int N,
                                             const Item& it, int ct, const uint2* tab) {
  constexpr int THREADS = WN * 32, XPT = BM * XC / THREADS;
  extern __shared__ uint4 sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, t = lane & 3;
  const int tile = ct * WN + warp;
  const bool active = tile < NT16;
  const __nv_bfloat16* xsrc[XPT];
  int xdst[XPT];
#pragma unroll
  for (int i = 0; i < XPT; ++i) {
    const int q = tid + i * THREADS, r = q / XC, c = q - r * XC;
    xdst[i] = chunk(r, c);
    xsrc[i] = nullptr;
    if (r < it.cnt) {
      const int p = pair_of(members, it, r);
      xsrc[i] = X + (size_t)(slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  auto stage = [&](int s, int k) {
    uint4* xs = sm + s * BM * XC;
#pragma unroll
    for (int i = 0; i < XPT; ++i)
      if (xsrc[i]) cp16(xs + xdst[i], xsrc[i] + (size_t)k * SK);
  };
  const uint8_t* wt = W + ((size_t)it.e * NT16 + (active ? tile : 0)) * (size_t)(KG / F::G) * F::BYTES;
  constexpr int GS = SK / 32;  // groups a stage
  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
  const int steps = KG / GS;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) stage(s, s);
    cp_commit();
  }
  uint4 w[GS];
  uint32_t dp[GS];
  if (active)
#pragma unroll
    for (int gi = 0; gi < GS; ++gi) {
      w[gi] = ld_nc(F::wptr(wt, gi, lane));
      dp[gi] = ld_nc32(F::dptr(wt, gi, lane));
    }
  for (int k = 0; k < steps; ++k) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (k + STAGES - 1 < steps) stage((k + STAGES - 1) % STAGES, k + STAGES - 1);
    cp_commit();
    if (!active) continue;
    uint4 wn[GS];
    uint32_t dn[GS];
    if (k + 1 < steps)
#pragma unroll
      for (int gi = 0; gi < GS; ++gi) {
        wn[gi] = ld_nc(F::wptr(wt, (k + 1) * GS + gi, lane));
        dn[gi] = ld_nc32(F::dptr(wt, (k + 1) * GS + gi, lane));
      }
    const uint4* xs = sm + (k % STAGES) * BM * XC;
#pragma unroll
    for (int gi = 0; gi < GS; ++gi)
      group<F, NT>(acc, w[gi], dp[gi], t, tab, [&](int nt, int s) {
        const int r = nt * 8 + gq;
        return reinterpret_cast<const uint2*>(xs + chunk(r, gi * 4 + t))[s];
      });
#pragma unroll
    for (int gi = 0; gi < GS; ++gi) {
      w[gi] = wn[gi];
      dp[gi] = dn[gi];
    }
  }
  cp_wait<0>();
  if (active) epilogue_rows<EPI, NT>(acc, out, N, tile * 16, lane, members, it);
}

template <class F, int EPI, int WN>
__device__ __forceinline__ void prompt_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                            const uint8_t* __restrict__ W, int KG, int NT16,
                                            const int* __restrict__ items, const int* __restrict__ counts,
                                            const int* __restrict__ members, void* __restrict__ out, int N, int rows,
                                            int skip) {
  __shared__ uint2 tab[F::TABLE ? TABLE_WORDS : 1];
  const int nct = (NT16 + WN - 1) / WN, iu = blockIdx.x / nct, ct = blockIdx.x - iu * nct;
  Item it;
  if (items) {
    if (iu >= __ldg(counts)) return;
    it = {__ldg(items + 3 * iu), __ldg(items + 3 * iu + 1), __ldg(items + 3 * iu + 2)};
    if (it.e == skip) return;
  } else {
    it = {0, iu * BM, min(BM, rows - iu * BM)};
  }
  load_tables<F>(tab);
  const int nt = min(8, (it.cnt + 7) >> 3);
#define TF_TILES(NT) \
  prompt_tiles<F, EPI, WN, NT>(X, x_stride, slots, W, KG, NT16, members, out, N, it, ct, tab)
  switch (nt) {
    case 8: TF_TILES(8); break;
    case 7: TF_TILES(7); break;
    case 6: TF_TILES(6); break;
    case 5: TF_TILES(5); break;
    case 4: TF_TILES(4); break;
    case 3: TF_TILES(3); break;
    case 2: TF_TILES(2); break;
    default: TF_TILES(1);
  }
#undef TF_TILES
}

}  // namespace tf_ggml

// The entries: format x (decode with 4 warps over K a tile, decode with a warp a tile and 4 tiles a CTA, prompt) x
// epilogue (0 fp32, 1 relu^2 bf16, 3 bf16); items null for dense rows.
#define TF_GGML_GEMV(NAME, F, EPI, KW, TPC)                                                                            \
  extern "C" __global__ void __launch_bounds__(KW * TPC * 32) NAME(                                                   \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint8_t* __restrict__ W, int KG, int NT16,   \
      const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,                \
      void* __restrict__ out, int N, int rows, int skip) {                                                           \
    tf_ggml::gemv_body<tf_ggml::F, EPI, KW, TPC>(X, x_stride, slots, W, KG, NT16, items, counts, members, out, N,    \
                                                 rows, skip);                                                        \
  }
#define TF_GGML_ENTRIES(FMT, F, EPI, SUFFIX)                                                                          \
  TF_GGML_GEMV(tf_ggml_##FMT##_gemv_##SUFFIX, F, EPI, 4, 1)                                                          \
  TF_GGML_GEMV(tf_ggml_##FMT##_gemv4_##SUFFIX, F, EPI, 1, 4)                                                         \
  extern "C" __global__ void __launch_bounds__(256) tf_ggml_##FMT##_prompt_##SUFFIX(                                \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint8_t* __restrict__ W, int KG, int NT16,   \
      const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,                \
      void* __restrict__ out, int N, int rows, int skip) {                                                           \
    tf_ggml::prompt_body<tf_ggml::F, EPI, 8>(X, x_stride, slots, W, KG, NT16, items, counts, members, out, N, rows,  \
                                             skip);                                                                  \
  }
TF_GGML_ENTRIES(iq2, Iq2, 1, relu2)
TF_GGML_ENTRIES(iq2, Iq2, 0, f32)
TF_GGML_ENTRIES(iq2, Iq2, 3, bf16)
TF_GGML_ENTRIES(q8, Q8, 1, relu2)
TF_GGML_ENTRIES(q8, Q8, 0, f32)
TF_GGML_ENTRIES(q8, Q8, 3, bf16)
#undef TF_GGML_ENTRIES
#undef TF_GGML_GEMV

// The load repack: `count` matrices [n][k / KB][B] of checkpoint blocks into the 16-column tiles above, a byte a thread.
extern "C" __global__ void tf_ggml_pack(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int iq2, int n,
                                        int kb, long long total) {
  const int B = iq2 ? 66 : 34, TB = iq2 ? 1056 : 544;
  const long long per = (long long)n * kb * B;
  for (long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x; o < total; o += (long long)gridDim.x * blockDim.x) {
    const long long m = o / per, rem = o - m * per;
    const long long tb = rem / TB;
    const int r = (int)(rem - tb * TB), tile = (int)(tb / kb), b = (int)(tb - (long long)tile * kb);
    int col, at;
    if (r >= TB - 32) {
      const int q = r - (TB - 32);
      col = tile * 16 + (q >> 2) + 8 * ((q >> 1) & 1);
      at = q & 1;
    } else if (iq2) {
      col = tile * 16 + ((r >> 4) & 7) + 8 * ((r >> 3) & 1);
      at = 2 + (r >> 7) * 8 + (r & 7);
    } else {
      const int ln = r >> 4, q = (r >> 2) & 3;
      col = tile * 16 + (ln >> 2) + 8 * (q & 1);
      at = 2 + 8 * (ln & 3) + 4 * (q >> 1) + (r & 3);
    }
    dst[o] = src[m * per + ((long long)col * kb + b) * B + at];
  }
}
