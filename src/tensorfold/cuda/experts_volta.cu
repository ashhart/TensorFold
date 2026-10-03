// Grouped 4-bit experts on Volta (sm_70): fp16 tensor cores (mma.m8n8k4) with fp32 sums, reading the same packed
// blocks as the sm_80 kernels (experts.py's ``pack``), decode and prompt forms alike.
//
// Rows. Each x row is scaled by a power of two so its largest value lands in [2^14, 2^15): every bf16 value then
// converts to fp16 exactly, and the scale comes back on the fp32 sums (as qmm_volta.cu does). A pair is an mma row,
// and a pair's sums run over K in group order whichever pairs share its item, so its bits never depend on them.
//
// Weights. A column's group expands to fp16(q * s + b) (the nibble in an fp16 mantissa, 1024 subtracted, one fma).
// A warp owns 32 columns: quadpair q the block's n8 tile q, lane idx its column in the tile. A thread reads its
// column's 4 * GS / 32 words of the group from the packed block (nibble slots j and j + 4 are inputs 2j and 2j + 1,
// which is one fp16 pair of the mma's B fragment), and two row tiles of 8 pairs a turn share each dequantized word.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

namespace {

constexpr int NTW = 4;                       // n8 tiles a warp: 32 output columns
constexpr int COLS = 8 * NTW;
constexpr int MT = 2;                        // row tiles of 8 pairs a pass
constexpr int WARPS = 4;

__device__ __forceinline__ float bf2f(unsigned short v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// SwiGLU as the families define it: bf16(bf16(silu(g)) * u) on bf16 g and u, clipped first when limit > 0
__device__ __forceinline__ float swiglu(float g, float u, float limit) {
  float gv = bf(g), uv = bf(u);
  if (limit > 0.f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  return bf(gv / (1.f + expf(-gv))) * uv;
}

__device__ __forceinline__ float relu2(float a) {
  const float u = fmaxf(bf(a), 0.f);
  return u * u;
}

// ---- rows -------------------------------------------------------------------------------------------------------

// One row a block: its largest value, the exponent that puts it in [2^14, 2^15) (an empty or non-finite row keeps 1),
// then the row as fp16 and its scale 2^e.
__global__ void prep_rows(const unsigned short* __restrict__ x, long ldx, int K, __half* __restrict__ out,
                          float* __restrict__ rs) {
  const int row = blockIdx.x;
  const unsigned short* xr = x + row * ldx;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(bf2f(xr[k])));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = red[0];
  for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w) amax = fmaxf(amax, red[w]);
  const int e = amax > 0.f && isfinite(amax) ? ilogbf(amax) - 14 : 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) out[(long)row * K + k] = __float2half_rn(ldexpf(bf2f(xr[k]), -e));
  if (threadIdx.x == 0) rs[row] = ldexpf(1.f, e);
}

// ---- the matmul ---------------------------------------------------------------------------------------------------

// Fragment layout, measured on a V100 (see qmm_volta.cu): quadpair q = (lane >> 2) & 3, idx = lane % 4 + 4 * (lane >= 16).
// A .row: the thread holds A[idx][0..3]; B .col: B[0..3][idx]; D f32 d[i] at row (lane & 1) + 2 * ((i >> 1) & 1) +
// 4 * (lane >= 16), column (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2).
__device__ __forceinline__ void mma884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
               "{%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// A word's eight nibbles as four fp16 pairs q * s + b in input order: (inputs 0, 1), (2, 3), (4, 5), (6, 7).
__device__ __forceinline__ uint4 dequant(unsigned w, __half2 s2, __half2 b2) {
  const __half2 k1024 = __halves2half2(__ushort_as_half(0x6400), __ushort_as_half(0x6400));
  unsigned o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    unsigned u = ((w >> (4 * j)) & 0x000F000Fu) | 0x64006400u;      // nibbles j and j + 4 are inputs 2j and 2j + 1
    const __half2 v = __hfma2(__hsub2(*reinterpret_cast<__half2*>(&u), k1024), s2, b2);
    o[j] = *reinterpret_cast<const unsigned*>(&v);
  }
  return make_uint4(o[0], o[1], o[2], o[3]);
}

// EPI 0: fp32 out (down); 1: bf16 relu(bf16(v))^2; 2: bf16 SwiGLU(matrix 0, matrix 1); 3: bf16 out (prefill down)
template <int EPI>
__device__ __forceinline__ void store(void* out, size_t at, float a, float b, float limit) {
  if constexpr (EPI == 0) {
    reinterpret_cast<float*>(out)[at] = a;
  } else {
    float o;
    if constexpr (EPI == 1) o = relu2(a);
    else if constexpr (EPI == 2) o = swiglu(a, b, limit);
    else o = a;
    reinterpret_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(o);
  }
}

// Pair p reads x row p / slots when slots > 0 (x holds tokens), else x row p; out row p.
template <int GS, int M, int EPI>
__global__ void __launch_bounds__(WARPS * 32)
    expert_kernel(const __half* __restrict__ X, const float* __restrict__ RS, int slots, const unsigned* __restrict__ W,
                  int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,
                  const int* __restrict__ members, void* __restrict__ out, int N, float limit) {
  constexpr int H = GS / 32, WORDS = 128 * H, BLOCK = WORDS + 32;       // a block's words: weights, then scales and biases
  const int lane = threadIdx.x & 31, q = (lane >> 2) & 3, idx = (lane & 3) + 4 * (lane >= 16);
  const int K = KG * GS;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    const unsigned* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK);
    for (int r0 = 0; r0 < cnt; r0 += 8 * MT) {
      const __half* xr[MT];
      bool rok[MT];
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const int m = r0 + 8 * t + idx;
        rok[t] = m < cnt;
        const int p = rok[t] ? __ldg(members + first + m) : 0;
        xr[t] = X + (size_t)(slots ? p / slots : p) * K;
      }
      float acc[MT][M][8];
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[t][m][i] = 0.f;
      for (int g = 0; g < KG; ++g) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const unsigned* b = blk + ((size_t)g * M + m) * BLOCK;
          const unsigned sw = __ldg(b + WORDS + (idx >> 1) * 8 + q), bw = __ldg(b + WORDS + (idx >> 1) * 8 + 4 + q);
          const __half sh = __float2half_rn(bf2f((idx & 1 ? sw >> 16 : sw) & 0xFFFF));
          const __half bh = __float2half_rn(bf2f((idx & 1 ? bw >> 16 : bw) & 0xFFFF));
          const __half2 s2 = __halves2half2(sh, sh), b2 = __halves2half2(bh, bh);
#pragma unroll
          for (int kk = 0; kk < 4 * H; ++kk) {                          // word kk: inputs 8 kk .. 8 kk + 7 of the group
            const int qq = kk / H, j = kk % H, tj = q * H + j;
            const unsigned w = __ldg(b + (tj >> 2) * 128 + (idx * 4 + qq) * 4 + (tj & 3));
            const uint4 o = dequant(w, s2, b2);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
              const uint4 a = rok[t] ? __ldg(reinterpret_cast<const uint4*>(xr[t] + g * GS + kk * 8)) : make_uint4(0, 0, 0, 0);
              mma884(acc[t][m], a.x, a.y, o.x, o.y);
              mma884(acc[t][m], a.z, a.w, o.z, o.w);
            }
          }
        }
      }
#pragma unroll
      for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int m = r0 + 8 * t + (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
          if (m >= cnt) continue;
          const int p = __ldg(members + first + m);
          const float rs = __ldg(RS + (slots ? p / slots : p));
          const int col = cb * COLS + q * 8 + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
          store<EPI>(out, (size_t)p * N + col, acc[t][0][i] * rs, acc[t][M - 1][i] * rs, limit);
        }
    }
  }
}

template <int GS, int M, int EPI>
void launch(const at::Tensor& x16, const at::Tensor& rs, int slots, const at::Tensor& w, int kg, int nb,
            const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int n,
            float limit, int64_t max_units) {
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < 65535 ? need : 65535);
  if (grid < 1) return;
  expert_kernel<GS, M, EPI><<<grid, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(x16.data_ptr()), rs.data_ptr<float>(), slots,
      reinterpret_cast<const unsigned*>(w.data_ptr()), kg, nb, items.data_ptr<int>(), counts.data_ptr<int>(),
      members.data_ptr<int>(), out.data_ptr(), n, limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_any(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
             int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
             at::Tensor& out, int64_t n, double limit, int64_t max_units) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t rows = x.size(0), k = kg * gs;
  const auto stream = at::cuda::getCurrentCUDAStream();
  at::Tensor x16 = at::empty({rows, k}, x.options().dtype(at::kHalf));
  at::Tensor rs = at::empty({rows}, x.options().dtype(at::kFloat));
  prep_rows<<<static_cast<unsigned>(rows), 256, 0, stream>>>(reinterpret_cast<const unsigned short*>(x.data_ptr()),
                                                             x_stride, static_cast<int>(k),
                                                             reinterpret_cast<__half*>(x16.data_ptr()),
                                                             rs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int sl = static_cast<int>(slots), kk = static_cast<int>(kg), b = static_cast<int>(nb);
  const int nn = static_cast<int>(n);
  const float lim = static_cast<float>(limit);
#define TF_RUN(GS_, M_, EPI_)                                                                         \
  if (gs == GS_ && epi == EPI_) {                                                                     \
    launch<GS_, M_, EPI_>(x16, rs, sl, w, kk, b, items, counts, members, out, nn, lim, max_units);   \
    return;                                                                                           \
  }
  TF_RUN(32, 1, 0) TF_RUN(32, 1, 3) TF_RUN(32, 2, 2) TF_RUN(64, 1, 0) TF_RUN(64, 1, 3) TF_RUN(64, 1, 1)
  TF_RUN(64, 2, 2)
#undef TF_RUN
  TORCH_CHECK(false, "experts: no kernel for group ", gs, " and epilogue ", epi);
}

}  // namespace

void experts_run_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                      const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items, const at::Tensor& counts,
                      const at::Tensor& members, at::Tensor& out, int64_t n, double limit, int64_t max_units) {
  run_any(gs, epi, x, x_stride, slots, w, kg, nb, items, counts, members, out, n, limit, max_units);
}

// A prompt plan's items hold up to 64 pairs; the same kernel takes them eight a pass.
void experts_prefill_cuda(int64_t gs, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                          const at::Tensor& w, int64_t kg, int64_t nb, const at::Tensor& items,
                          const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                          double limit, int64_t max_items) {
  run_any(gs, epi, x, x_stride, slots, w, kg, nb, items, counts, members, out, n, limit, max_items * nb);
}
