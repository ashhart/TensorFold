// One-row bf16 y = W x in cuBLASLt 13.1.1's gemvx order (16 strided fma chains, halving tree); qualified at init.
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kRows = 8;
constexpr int kChunk = 256;  // k a stage
constexpr int kStages = 3;
constexpr int kRowBytes = kChunk * 2;
constexpr int kStageBytes = kRows * kRowBytes;
static_assert(kRowBytes == 32 * 16, "a stage row is one 16-byte copy a lane");

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

// bf16 to fp32 is exact: the low and the high element of a little-endian pair.
__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

}  // namespace

// Lane (g, c) runs chains 4c..4c + 3 of row g; rows are swizzled by p ^ 2(r % 4) so reads hit distinct banks.
extern "C" __global__ void __launch_bounds__(32) drama3_staged_gemv(const __nv_bfloat16* __restrict__ x,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  __nv_bfloat16* __restrict__ y, int k) {
    __shared__ __align__(128) unsigned char ring[kStages * kStageBytes];
    const int lane = threadIdx.x, g = lane >> 2, c = lane & 3;
    const unsigned char* rows = reinterpret_cast<const unsigned char*>(w + static_cast<size_t>(blockIdx.x) * kRows * k);
    const int chunks = k / kChunk;
    auto issue = [&](int s) {
        unsigned char* stage = ring + (s % kStages) * kStageBytes;
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            cp_async16(stage + r * kRowBytes + 16 * (lane ^ (2 * (r & 3))),
                       rows + static_cast<size_t>(r) * k * 2 + static_cast<size_t>(s) * kRowBytes + 16 * lane);
    };
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < chunks) issue(s);
        asm volatile("cp.async.commit_group;\n");
    }
    float p[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < chunks; ++s) {
        if (s + kStages - 1 < chunks) issue(s + kStages - 1);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 1));
        __syncwarp();
        const unsigned char* row = ring + (s % kStages) * kStageBytes + g * kRowBytes;
        const uint2* xs = reinterpret_cast<const uint2*>(x + s * kChunk) + c;
#pragma unroll
        for (int j = 0; j < kChunk / 16; ++j) {
            // k = 256 s + 16 j + 4 c ..: bytes 32 j + 8 c of the stage row, piece 2 j + c / 2
            const uint2 a = *reinterpret_cast<const uint2*>(row + 16 * ((2 * j + (c >> 1)) ^ (2 * (g & 3))) + 8 * (c & 1));
            const uint2 b = __ldg(xs + 4 * j);
            p[0] = fmaf(lo(a.x), lo(b.x), p[0]);
            p[1] = fmaf(hi(a.x), hi(b.x), p[1]);
            p[2] = fmaf(lo(a.y), lo(b.y), p[2]);
            p[3] = fmaf(hi(a.y), hi(b.y), p[3]);
        }
        __syncwarp();
    }
    // chain t = 4c + i: t + 8 is lane c + 2's p[i], t + 4 is lane c + 1's
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] += __shfl_down_sync(0xffffffffu, p[i], 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] += __shfl_down_sync(0xffffffffu, p[i], 1);
    if (c != 0) return;
    const float sum = (p[0] + p[2]) + (p[1] + p[3]);
    // the epilogue's fma(alpha = 1, sum, 0) and its +0 bias are exactly sum + 0 (-0 becomes +0)
    y[blockIdx.x * kRows + g] = __float2bfloat16_rn(sum + 0.0f);
}
