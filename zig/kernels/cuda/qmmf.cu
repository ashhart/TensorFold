// Device code of nvfp4/qmmf.cu (lines 14-298, comments and ATen dropped), its FP4, FP8 and FP8G modes plus BF16.

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/qmm_frag.cuh"

namespace tf_qmmf {

using namespace qmm_frag;

enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, FP8G = 3, BF16 = 4 };

constexpr int GS = 64;

__device__ __forceinline__ uint32_t fp8pair(uint32_t w) {
    const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
    const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
    return r;
}

// UE8M0 has no zero encoding: byte 0 is 2^-127, byte 255 is NaN.
__device__ __forceinline__ float e8m0f(uint8_t b) {
    return __uint_as_float(b == 0 ? 0x00400000u : b == 255 ? 0x7FC00000u : uint32_t(b) << 23);
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

__device__ __forceinline__ float e4m3f(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}

template <int MODE, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == BF16 ? BN * GS * 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int BARRIERS = MODE == BF16 && BN > 64 ? 128 : 0;
    static constexpr int SMEM = (STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS) + BARRIERS;
};

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool FUSE = false>
__device__ __forceinline__ void qmmf_body(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group, int ldy) {
#define TF_QMMF_STRIDED
#include "qmmf_body.cuh"
#undef TF_QMMF_STRIDED
}


template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool FUSE = false>
__global__ void __launch_bounds__(WM * WN * 32) qmmf_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group) {
#include "qmmf_body.cuh"
}

// The shared down has one K slice; only the output address changes from the existing BF16-weight product.
#define TF_SHARED_DOWN(NAME, BM, BN) \
extern "C" __global__ void __launch_bounds__(128) NAME( \
        const __nv_bfloat16* x, const unsigned char* w, float scale, float* out, \
        int M, int N, int K, int npad, int ldx, int group, int ldy) { \
    qmmf_body<BF16, BM, BN, 1, 4, 4, true, false, false>( \
        x, w, nullptr, scale, out, nullptr, M, N, K, 1, npad, ldx, group, ldy); \
}
TF_SHARED_DOWN(tf_qmmf_shared16_half, 16, 32)
TF_SHARED_DOWN(tf_qmmf_shared16, 16, 64)
TF_SHARED_DOWN(tf_qmmf_shared32, 32, 64)
TF_SHARED_DOWN(tf_qmmf_shared64, 64, 64)
#undef TF_SHARED_DOWN

template <bool F32>
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long long total, int SK,
                              float scale) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int s = 1; s < SK; ++s) acc = acc + part[s * total + i];
    acc = acc * scale;
    if (F32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

template __global__ void qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);

// The instances the Zig launchers load by name (the Python host instantiates these through its launch switch).
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, true, false, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 16, 64, 1, 4, 4, true, true, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, true, false, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 32, 64, 1, 4, 4, true, true, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, true, false, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, true, true, false>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP4, 64, 64, 1, 4, 4, true, false, true>(
    const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float,
    void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, false, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 32, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 32, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 64, 64, 1, 4, 4, true, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 16, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 32, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8G, 64, 64, 1, 4, 4, true, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);







template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 16, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 32, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<FP8, 64, 64, 1, 4, 4, true, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);

// Half tiles (32 columns a block) of the 16-row tile: the 64-column tile's sums, twice the blocks for a small grid.
template __global__ void qmmf_kernel<MXFP8, 16, 32, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 32, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 32, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<MXFP8, 16, 32, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 32, 1, 4, 4, false, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 32, 1, 4, 4, false, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 32, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 16, 32, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);

template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 32, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, true, false, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, true, true, false>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
template __global__ void qmmf_kernel<BF16, 64, 64, 1, 4, 4, true, false, true>(const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
// Prompt tiles retain the same m16n8k16 chains and fused slice boundaries as the 64x64 instances.
#define BF16_PROMPT(F32, FUSE) \
    template __global__ void qmmf_kernel<BF16, 128, 128, 2, 4, 3, F32, false, FUSE>( \
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
BF16_PROMPT(false, false)
BF16_PROMPT(false, true)
BF16_PROMPT(true, false)
BF16_PROMPT(true, true)
#undef BF16_PROMPT

template __global__ void reduce_kernel<true>(const float* __restrict__, void* __restrict__, long long, int, float);
template __global__ void reduce_kernel<false>(const float* __restrict__, void* __restrict__, long long, int, float);

}
