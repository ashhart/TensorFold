// Frozen pre-SiLU norm backward, used only by the contract replacement and wrong-gate controls.
#include "common/reduce.cuh"
#include "common/act.cuh"
namespace tf_gdn_legacy {
using bf16 = __nv_bfloat16;
constexpr int DV = 128;
struct GdnShape { int rows, nk, nv, conv, proj; };
__device__ __forceinline__ float f32(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float bf(float v) { return f32(__float2bfloat16_rn(v)); }
using tf_common::warp_sum;
using tf_common::block_sum_tree;
using tf_common::sigmoidf_;
extern "C" __global__ void __launch_bounds__(128) tf_ctl_gdn_norm_back_sigmoid(const bf16* __restrict__ y, const bf16* __restrict__ P,
                                                                   const bf16* __restrict__ norm_w, const float* __restrict__ dout,
                                                                   float* __restrict__ dy, float* __restrict__ dproj, GdnShape s,
                                                                   float eps) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV], scratch[32];
    __shared__ float rinv;
    const size_t row = ((size_t)r * s.nv + h) * DV;
    ys[t] = f32(y[row + t]);
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float v = ys[lane * 4 + i]; ss = ss + v * v; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float w = f32(norm_w[t]), yv = ys[t], ri = rinv;
    const float yn = bf(bf(yv * ri) * w);
    const size_t zc = (size_t)r * s.proj + s.conv + h * DV + t;
    const float sg = sigmoidf_(f32(P[zc])), d = dout[row + t];
    dproj[zc] = d * yn * sg * (1.0f - sg);
    const float dn = d * sg * w;
    const float dot = block_sum_tree(yv * dn, scratch, 4);
    dy[row + t] = ri * dn - ri * ri * ri * yv * dot / (float)DV;
}

}
