// Gate 3's control for tf_moe_router_tiles: the same tiles that lose the last live row (a dropped tile edge).
#pragma once

#include "common/router_tiles.cuh"

extern "C" __global__ void __launch_bounds__(256) tf_ctl_moe_router_tiles_row(const __nv_bfloat16* __restrict__ X, int x_stride,
                                                                                const __nv_bfloat16* __restrict__ W,
                                                                                float* __restrict__ OUT, int R, int D, int E,
                                                                                const int* __restrict__ P) {
    tf_router_tiles::tiles(X, W, OUT, R - 1, D, E, 1);
}
