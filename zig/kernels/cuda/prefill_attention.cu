// Device code of src/tensorfold/cuda/kernels/prefill_attention.cu (lines 1-250, comments and ATen includes dropped), checked by zig/tests/cuda/copies.py.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace tf_prefill_attention {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int BN = 64, HALF = 32;             // keys a tile (the fold's unit), keys a staging slot
constexpr float LOG2E = 1.4426950408889634f;
constexpr int SMEM_MAX = 96 * 1024;

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(qmm_frag::smem(p)));
}

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ float tdiv(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

__device__ __forceinline__ float rowsum(const float (&c)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = __fadd_rn(c[j], __shfl_xor_sync(0xffffffffu, c[j], 2));
        w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                     __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
pattn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
             const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
             int G, float scale) {
    constexpr bool causal = true;
    constexpr int skip = 0;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// One segment's rows from row `first` on, keys from p0 on `skip` rows further on: the body for tree_kernel.
template <int D, int WARPS, int HPC, int NS>
__device__ __forceinline__ void
pattn_rows(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
           const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
           int G, float scale, int first, int skip) {
    constexpr bool causal = true;
    const int HO = H;
    auto first_row = [=] { return first; };
#include "prefill_attention_rows.cuh"
}

// Packed rows: a causal trunk of P rows, then branches at P + cu[b], each over the trunk and its own earlier rows.
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
tree_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
            const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, const int* __restrict__ cu, int P,
            int n, int H, int HK, int G, float scale) {
    constexpr int ROWS = 16 * (WARPS / HPC);
    int tile = blockIdx.x, b = -1, W = P;                // the trunk's row tiles, then each branch's
    while (tile >= (W + ROWS - 1) / ROWS) {
        tile -= (W + ROWS - 1) / ROWS;
        if (++b == n) return;
        W = cu[b + 1] - cu[b];
    }
    const int64_t at = (int64_t)(b < 0 ? 0 : P + cu[b]) * H * D;
    pattn_rows<D, WARPS, HPC, NS>(q + at, kc, vc, out + at, b < 0 ? 0 : P, W, H, HK, G, scale, tile * ROWS,
                                  b < 0 ? 0 : cu[b]);
}

// One sequence, every row over every key (no mask), a block a head: q, k, v heads of qkv [W, 3 HO D] -> out [W, HO D].
template <int D, int WARPS, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
full_kernel(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out, int W, int HO, float scale) {
    constexpr int HPC = 1, G = 1, p0 = 0, skip = 0;
    constexpr bool causal = false;
    const int H = 3 * HO, HK = H;
    const __nv_bfloat16* __restrict__ q = qkv;
    const __nv_bfloat16* __restrict__ kc = qkv + HO * D;
    const __nv_bfloat16* __restrict__ vc = qkv + 2 * HO * D;
    auto first_row = [] { return blockIdx.x * 16 * WARPS; };
#include "prefill_attention_rows.cuh"
}

} // namespace tf_prefill_attention

// Head dim 128, eight warps, eight query heads a block, eight staging slots (Nemotron's 16 heads a KV head).
template __global__ void tf_prefill_attention::pattn_kernel<128, 8, 8, 8>(const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, float);
// Head dim 64, eight warps, four query heads a block, eight staging slots (LFM2's 4 heads a KV head, causal or trees).
template __global__ void tf_prefill_attention::tree_kernel<64, 8, 4, 8>(const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int*, int, int, int, int, int, float);
// Head dim 72, eight warps of one head, eight staging slots (SigLIP2's tower, 72 dims padded to 80 inside).
template __global__ void tf_prefill_attention::full_kernel<72, 8, 8>(const __nv_bfloat16*, __nv_bfloat16*, int, int,
    float);
// Head dim 80, eight warps of one head, eight staging slots (RADIO's tower: 1280 wide, 16 heads).
template __global__ void tf_prefill_attention::full_kernel<80, 8, 8>(const __nv_bfloat16*, __nv_bfloat16*, int, int,
    float);
