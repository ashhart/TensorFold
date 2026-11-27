// Inkling 2's vision tower glue: patch rows to the sequence and axial rotary.
#include <cuda_bf16.h>
#include <stdint.h>

namespace ink2v {

constexpr int kWidth = 1024; // the tower's hidden size
constexpr int kHeads = 16;
constexpr int kDim = 64;
constexpr int kPairs = kDim / 2;
constexpr int kThreads = 256;

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ __nv_bfloat16 tobf(float v) { return __float2bfloat16_rn(v); }

}  // namespace ink2v

using namespace ink2v;

// Block a sequence row: row 0 the class token bf16(cls + cls_pos), row r the patch r - 1 plus its resampled position.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2v_embed(const __nv_bfloat16* __restrict__ pe,
                                                                      const __nv_bfloat16* __restrict__ table,
                                                                      const uint32_t* __restrict__ taps,
                                                                      const float* __restrict__ wts,
                                                                      const __nv_bfloat16* __restrict__ cls,
                                                                      const __nv_bfloat16* __restrict__ cls_pos,
                                                                      __nv_bfloat16* __restrict__ x) {
    const int r = blockIdx.x;
    for (int c = threadIdx.x; c < kWidth; c += kThreads) {
        float v;
        if (r == 0) {
            v = __fadd_rn(bf(cls[c]), bf(cls_pos[c]));
        } else {
            const int p = r - 1;
            float s = 0.0f;
            for (int t = 0; t < 4; ++t) s = __fadd_rn(s, __fmul_rn(bf(table[(int64_t)taps[p * 4 + t] * kWidth + c]), wts[p * 4 + t]));
            v = __fadd_rn(bf(pe[(int64_t)p * kWidth + c]), bf(tobf(s)));
        }
        x[(int64_t)r * kWidth + c] = tobf(v);
    }
}

// Block a row: q, k pairs (2j, 2j + 1) turned in fp32; MiniMax M3 and dots pair (i, i + half), so it is not shared.
extern "C" __global__ void __launch_bounds__(kThreads) tf_ink2v_rope(__nv_bfloat16* __restrict__ qkv, const float* __restrict__ cs,
                                                                     const float* __restrict__ sn) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < 2 * kHeads * kPairs; i += kThreads) {
        const int j = i % kPairs;
        __nv_bfloat16* p = qkv + (int64_t)r * 3 * kWidth + (i / kPairs) * kDim + 2 * j;
        const float c = cs[r * kPairs + j], s = sn[r * kPairs + j], a = bf(p[0]), b = bf(p[1]);
        p[0] = tobf(__fadd_rn(__fmul_rn(a, c), __fmul_rn(-b, s)));
        p[1] = tobf(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
    }
}
