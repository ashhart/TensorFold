// Hy4-flash's LatentMoE routing (families/hy4/cuda_route.zig); the formula and its bits are in hy4_route.cuh.

#include "route.cuh"

// LOG fp32 [R, 512], BIAS fp32 [512] -> IDS int32 [R, 16], W fp32 [R, 16]. <<<R, 128>>>
extern "C" __global__ void __launch_bounds__(128, 1)
    tf_hy4_route_m4(const float* __restrict__ LOG, const float* __restrict__ BIAS, int R, float scale,
                    int* __restrict__ IDS, float* __restrict__ W) {
    tf_hy4_route::route_m4<tf_hy4_route::kNone>(LOG, BIAS, R, scale, IDS, W);
}

// Single-row fused top-16 + grouped MoE plan (P=16, E=512, T=16). <<<1, 128>>>
extern "C" __global__ void __launch_bounds__(128, 1)
    tf_hy4_route_m4_plan1(const float* __restrict__ LOG, const float* __restrict__ BIAS, float scale,
                          int* __restrict__ IDS, float* __restrict__ W, int* __restrict__ MEMBERS,
                          int* __restrict__ ITEMS, int* __restrict__ COUNTS) {
    tf_hy4_route::route_m4_plan1(LOG, BIAS, scale, IDS, W, MEMBERS, ITEMS, COUNTS);
}
