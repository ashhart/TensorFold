// Native FP4 products over TensorFold's decode packs, with independent TMA producer and compute warps.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "common/qmm_frag.cuh"
#include "common/native_fp4.cuh"

namespace tf_nvfp4_tma {

template<int Branches> struct alignas(64) Maps {
    CUtensorMap activation;
    CUtensorMap activation_scales;
    CUtensorMap weights[Branches];
    CUtensorMap weight_scales[Branches];
};
static_assert(sizeof(CUtensorMap) == 128);
static_assert(sizeof(Maps<1>) == 512 && sizeof(Maps<2>) == 768);

__device__ __forceinline__ void initialize(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                 "r"(qmm_frag::smem(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void expect(uint64_t* bar, unsigned bytes) {
    asm volatile("{ .reg .b64 state; mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
                 "state, [%0], %1; }" :: "r"(qmm_frag::smem(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void arrive(uint64_t* bar) {
    asm volatile("{ .reg .b64 state; mbarrier.arrive.release.cta.shared::cta.b64 "
                 "state, [%0], 1; }" :: "r"(qmm_frag::smem(bar)) : "memory");
}

__device__ __forceinline__ void wait(uint64_t* bar, unsigned phase) {
    asm volatile("{ .reg .pred ready; WAIT_%=: "
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1; "
                 "@!ready bra WAIT_%=; }" ::
                 "r"(qmm_frag::smem(bar)), "r"(phase) : "memory");
}

__device__ __forceinline__ void copy2(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3}], [%4];" :: "r"(qmm_frag::smem(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y),
                 "r"(qmm_frag::smem(bar)) : "memory");
}

__device__ __forceinline__ void copy3(void* dst, const CUtensorMap* map,
                                     int x, int y, int z, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3,%4}], [%5];" :: "r"(qmm_frag::smem(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z),
                 "r"(qmm_frag::smem(bar)) : "memory");
}

__device__ __forceinline__ void copy4(void* dst, const CUtensorMap* map,
                                     int x, int y, int z, int w, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3,%4,%5}], [%6];" :: "r"(qmm_frag::smem(dst)),
                 "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(w),
                 "r"(qmm_frag::smem(bar)) : "memory");
}

template<int BM, int BN, int Branches> struct Layout {
    static constexpr int Stages = 3;
    static constexpr int WarpM = 4, WarpN = 2, Consumers = 256, Threads = 288;
    static constexpr int MT = BM / WarpM / 16, NT = BN / WarpN / 8;
    static constexpr int A = BM * 64, AS = BM * 16, B = BN * 64, BS = BN * 8;
    static constexpr int Stage = A + AS + Branches * (B + BS);
    static constexpr int Shared = Stages * Stage + Stages * 2 * sizeof(uint64_t);
    static_assert(Stage % 512 == 0);
};

// The documented 64-byte swizzle XORs address bits 7:8 into bits 4:5; each stage starts on a 512-byte boundary.
__device__ __forceinline__ int activation_offset(int row, int byte) {
    const int logical = row * 64 + byte;
    return logical ^ ((logical >> 3) & 0x30);
}

template<int BM, int BN, int Branches, bool F32, bool SwiGlu, bool Packed = false>
__device__ __forceinline__ void product(const Maps<Branches>& maps, void* output,
                                      int m, int n, int k, float alpha0, float alpha1) {
    using L = Layout<BM, BN, Branches>;
    static_assert(!SwiGlu || Branches == 2);
    extern __shared__ __align__(512) unsigned char memory[];
    uint64_t* ready = reinterpret_cast<uint64_t*>(memory + L::Stages * L::Stage);
    uint64_t* empty = ready + L::Stages;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int tiles = k / 128;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < L::Stages; ++s) {
            initialize(ready + s, 1);
            initialize(empty + s, L::Consumers);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (tid < 32) {
        if (tid == 0) for (int tile = 0; tile < tiles; ++tile) {
            const int s = tile % L::Stages;
            if (tile >= L::Stages) wait(empty + s, ((tile / L::Stages) - 1) & 1);
            unsigned char* stage = memory + s * L::Stage;
            expect(ready + s, L::Stage);
            copy2(stage, &maps.activation, tile * 64, m0, ready + s);
            copy2(stage + L::A, &maps.activation_scales, (tile / 2) * 4, m0, ready + s);
#pragma unroll
            for (int branch = 0; branch < Branches; ++branch) {
                unsigned char* weights = stage + L::A + L::AS + branch * (L::B + L::BS);
                copy4(weights, &maps.weights[branch], 0, 0, tile * 2, n0 / 64, ready + s);
                copy3(weights + L::B, &maps.weight_scales[branch], 0, tile * 2, n0 / 64, ready + s);
            }
        }
        return;
    }
    const int thread = tid - 32, lane = thread & 31, warp = thread >> 5;
    const int wm = warp / L::WarpN, wn = warp % L::WarpN;
    float accum[Branches][L::MT][L::NT][4] = {};
    for (int tile = 0; tile < tiles; ++tile) {
        const int s = tile % L::Stages;
        wait(ready + s, (tile / L::Stages) & 1);
        const unsigned char* stage = memory + s * L::Stage;
        const uint32_t* a_scales = reinterpret_cast<const uint32_t*>(stage + L::A);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint32_t a[L::MT][4], sa[L::MT];
#pragma unroll
            for (int i = 0; i < L::MT; ++i) {
                const int row = wm * (BM / L::WarpM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int byte = half * 32 + (lane >> 4) * 16;
                qmm_frag::ldmatrix4(a[i], stage + activation_offset(row, byte));
                const int scale_row = wm * (BM / L::WarpM) + i * 16 + (lane >> 2) + (lane & 1) * 8;
                sa[i] = a_scales[scale_row * 4 + (tile & 1) * 2 + half];
            }
#pragma unroll
            for (int branch = 0; branch < Branches; ++branch) {
                const unsigned char* b = stage + L::A + L::AS + branch * (L::B + L::BS);
                const uint32_t* scales = reinterpret_cast<const uint32_t*>(b + L::B);
#pragma unroll
                for (int j = 0; j < L::NT; ++j) {
                    const int col = wn * (BN / L::WarpN) + j * 8;
                    const int nt = col / 64, within = (col % 64) / 8;
                    const int at = ((nt * 2 + half) * 8 + within) * 32 + lane;
                    const uint2 packed = reinterpret_cast<const uint2*>(b)[at];
                    const uint32_t b0 = Packed ? packed.x : tf_nvfp4::contiguous8(packed.x, lane);
                    const uint32_t b1 = Packed ? packed.y : tf_nvfp4::contiguous8(packed.y, lane);
                    const uint32_t sb = scales[(nt * 2 + half) * 64 + col % 64 + (lane >> 2)];
#pragma unroll
                    for (int i = 0; i < L::MT; ++i)
                        tf_nvfp4::mma(accum[branch][i][j], a[i], b0, b1, sa[i], sb);
                }
            }
        }
        arrive(empty + s);
    }
#pragma unroll
    for (int i = 0; i < L::MT; ++i)
#pragma unroll
        for (int j = 0; j < L::NT; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int row = m0 + wm * (BM / L::WarpM) + i * 16 + (lane >> 2) + h * 8;
                    const int col = n0 + wn * (BN / L::WarpN) + j * 8 + (lane & 3) * 2 + pair;
                    if (row >= m || col >= n) continue;
                    float value = accum[0][i][j][h * 2 + pair] * alpha0;
                    if constexpr (SwiGlu) {
                        const float gate = __bfloat162float(__float2bfloat16_rn(value));
                        const float up = __bfloat162float(__float2bfloat16_rn(accum[1][i][j][h * 2 + pair] * alpha1));
                        value = gate / (1.0f + expf(-gate)) * up;
                    }
                    const size_t at = static_cast<size_t>(row) * n + col;
                    if constexpr (F32) reinterpret_cast<float*>(output)[at] = value;
                    else reinterpret_cast<__nv_bfloat16*>(output)[at] = __float2bfloat16_rn(value);
                }
}
}

#define TF_TMA_SINGLE(NAME, BM, F32) \
extern "C" __global__ __launch_bounds__(288) void NAME( \
    const __grid_constant__ tf_nvfp4_tma::Maps<1> maps, void* output, int m, int n, int k, float alpha) { \
    tf_nvfp4_tma::product<BM, 128, 1, F32, false>(maps, output, m, n, k, alpha, 0); \
}
TF_TMA_SINGLE(tf_nvfp4_tma_128, 128, false)
TF_TMA_SINGLE(tf_nvfp4_tma_128_f32, 128, true)
TF_TMA_SINGLE(tf_nvfp4_tma_256, 256, false)
TF_TMA_SINGLE(tf_nvfp4_tma_256_f32, 256, true)
#undef TF_TMA_SINGLE

#define TF_TMA_PAIR(NAME, BM) \
extern "C" __global__ __launch_bounds__(288) void NAME( \
    const __grid_constant__ tf_nvfp4_tma::Maps<2> maps, __nv_bfloat16* output, \
    int m, int n, int k, float gate_alpha, float up_alpha) { \
    tf_nvfp4_tma::product<BM, 64, 2, false, true>(maps, output, m, n, k, gate_alpha, up_alpha); \
}
TF_TMA_PAIR(tf_nvfp4_tma_swiglu_128, 128)
TF_TMA_PAIR(tf_nvfp4_tma_swiglu_256, 256)
#undef TF_TMA_PAIR

extern "C" __global__ __launch_bounds__(256) void tf_nvfp4_pack_mma(
    const uint2* source, uint2* output, int pairs) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pairs) return;
    const int lane = threadIdx.x & 31;
    const uint2 value = source[index];
    output[index] = make_uint2(tf_nvfp4::contiguous8(value.x, lane), tf_nvfp4::contiguous8(value.y, lane));
}

#define TF_PACKED_SINGLE(NAME, BM, F32) \
extern "C" __global__ __launch_bounds__(288) void NAME( \
    const __grid_constant__ tf_nvfp4_tma::Maps<1> maps, void* output, int m, int n, int k, float alpha) { \
    tf_nvfp4_tma::product<BM, 128, 1, F32, false, true>(maps, output, m, n, k, alpha, 0); \
}
TF_PACKED_SINGLE(tf_nvfp4_tma_packed_128, 128, false)
TF_PACKED_SINGLE(tf_nvfp4_tma_packed_128_f32, 128, true)
TF_PACKED_SINGLE(tf_nvfp4_tma_packed_256, 256, false)
TF_PACKED_SINGLE(tf_nvfp4_tma_packed_256_f32, 256, true)
#undef TF_PACKED_SINGLE

#define TF_PACKED_PAIR(NAME, BM) \
extern "C" __global__ __launch_bounds__(288) void NAME( \
    const __grid_constant__ tf_nvfp4_tma::Maps<2> maps, __nv_bfloat16* output, \
    int m, int n, int k, float gate_alpha, float up_alpha) { \
    tf_nvfp4_tma::product<BM, 64, 2, false, true, true>(maps, output, m, n, k, gate_alpha, up_alpha); \
}
TF_PACKED_PAIR(tf_nvfp4_tma_packed_swiglu_128, 128)
TF_PACKED_PAIR(tf_nvfp4_tma_packed_swiglu_256, 256)
#undef TF_PACKED_PAIR
