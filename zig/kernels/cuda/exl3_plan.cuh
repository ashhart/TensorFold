// The Grouped plan's stage arguments and plan_finish: exl3_experts.cu serves it, contract_controls.cu plants faults.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_decode.cuh"

namespace tf_exl3 {

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// The SwiGLU clamps as torch.clamp has them: a NaN stays a NaN, where fminf and fmaxf hand back the limit.
__device__ __forceinline__ float upTo(float x, float l) { return x > l ? l : x; }
__device__ __forceinline__ float within(float x, float l) { return x > l ? l : (x < -l ? -l : x); }

}  // namespace tf_exl3

// The shared Grouped plan supplies triples (expert, first, count) and direct pair indices.
struct PlanStage {
    const int *items, *counts, *members;
    const __nv_bfloat16* x;
    const half *su0, *su1, *sv0, *sv1;
    half *x0, *x1;
    const float* z;
    void* out;
    int k, n, p, sk, slots, stride, skip, epi, act_mode;  // act_mode 2: SwiGLU-OAI in bf16 steps (needs limit > 0)
    float limit, alpha;
};

namespace tf_exl3 {

// plan_finish as served, or one of the contract's planted SwiGLU-OAI faults: no + 1 on the up, alpha 1.
enum FinishFault { FINISH_SERVED, FINISH_NO_PLUS_ONE, FINISH_ALPHA_ONE };

template <int FAULT>
__device__ __forceinline__ void plan_finish(const PlanStage a) {
    const int it = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    if (it >= a.counts[0]) return;
    const int e = a.items[3 * it], first = a.items[3 * it + 1], count = a.items[3 * it + 2];
    if (e == a.skip) return;
    const int n = blk * 128 + 4 * lane;
    for (int j = 0; j < count; ++j) {
        const int p = a.members[first + j];
        float g[4] = {}, u[4] = {};
        for (int s = 0; s < a.sk; ++s)
            for (int q = 0; q < 4; ++q) {
                g[q] += a.z[((size_t)s * a.p + p) * a.n + n + q];
                if (a.epi == 2) u[q] += a.z[((size_t)(a.sk + s) * a.p + p) * a.n + n + q];
            }
        fwht128(g, lane);
        if (a.epi == 2) fwht128(u, lane);
        for (int q = 0; q < 4; ++q) {
            float v = g[q] * HAD_SCALE * __half2float(a.sv0[(size_t)e * a.n + n + q]);
            if (a.epi == 2) {
                float up = u[q] * HAD_SCALE * __half2float(a.sv1[(size_t)e * a.n + n + q]);
                if (a.act_mode == 2) {  // (clamp(up) + 1) * g * sigmoid(alpha * g), g = min(gate, limit), MiniMax's
                    const float alpha = FAULT == FINISH_ALPHA_ONE ? 1.f : a.alpha;
                    const float one = FAULT == FINISH_NO_PLUS_ONE ? 0.f : 1.f;
                    const float g2 = upTo(bf16r(v), a.limit), u2 = within(bf16r(up), a.limit);
                    const float sg = bf16r(1.f / (1.f + expf(-bf16r(g2 * alpha))));
                    v = bf16r(u2 + one) * bf16r(g2 * sg);
                    const size_t at = (size_t)p * a.n + n + q;
                    reinterpret_cast<__nv_bfloat16*>(a.out)[at] = __float2bfloat16_rn(v);
                    continue;
                }
                if (a.act_mode == 0) { v = bf16r(v); up = bf16r(up); }
                if (a.limit > 0) { v = upTo(v, a.limit); up = within(up, a.limit); }
                v = v / (1.f + expf(-v));
                if (a.act_mode == 0) v = bf16r(v);
                v *= up;
            }
            const size_t at = (size_t)p * a.n + n + q;
            if (a.epi == 0) reinterpret_cast<float*>(a.out)[at] = v;
            else reinterpret_cast<__nv_bfloat16*>(a.out)[at] = __float2bfloat16_rn(v);
        }
    }
}

}  // namespace tf_exl3
