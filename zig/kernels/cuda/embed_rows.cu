// Token rows for every CUDA family (float64 reference: core/act_ref.zig's affine4): 4-bit affine or bf16 tables.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_embed {

constexpr int NT = 256;

}  // namespace tf_embed

using namespace tf_embed;

// out[r, k, i] = bf16(q * s + b) of token ids[r], k < copies: q 4 bits (low first), s and b a group's (32 or 64).
extern "C" __global__ void __launch_bounds__(NT) tf_embed_rows_affine4(const int* __restrict__ ids,
                                                                       const uint32_t* __restrict__ w,
                                                                       const __nv_bfloat16* __restrict__ s,
                                                                       const __nv_bfloat16* __restrict__ b,
                                                                       __nv_bfloat16* __restrict__ out, int d, int group,
                                                                       int copies) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * NT + threadIdx.x; // 8 values, one word
    if (8 * c >= d) return;
    const int64_t tok = ids[r];
    const uint32_t word = w[tok * (d / 8) + c];
    const int64_t g = tok * (d / group) + 8 * c / group;
    const float sc = __bfloat162float(s[g]), bi = __bfloat162float(b[g]);
    uint4 o;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        h[i] = __floats2bfloat162_rn(__fmaf_rn(static_cast<float>((word >> (8 * i)) & 0xFu), sc, bi),
                                     __fmaf_rn(static_cast<float>((word >> (8 * i + 4)) & 0xFu), sc, bi));
    for (int k = 0; k < copies; ++k) *reinterpret_cast<uint4*>(out + (r * copies + k) * d + 8 * c) = o;
}

// out[r] = w[ids[r]] for a bf16 table, 16 bytes a thread (d % 8 == 0).
extern "C" __global__ void __launch_bounds__(NT) tf_embed_rows_bf16(const int* __restrict__ ids,
                                                                    const __nv_bfloat16* __restrict__ w,
                                                                    __nv_bfloat16* __restrict__ out, int d) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * NT + threadIdx.x;
    if (8 * c >= d) return;
    *reinterpret_cast<uint4*>(out + r * d + 8 * c) = *reinterpret_cast<const uint4*>(w + static_cast<int64_t>(ids[r]) * d + 8 * c);
}
