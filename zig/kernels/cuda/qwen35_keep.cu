// Keeping a verified window's path (cuda_cache.zig): the recurrences take its logged updates and its keys move down.

#include "qwen35.cuh"

using namespace tf_q35;

// Each linear layer's recurrence through the path's (decay, key, write), as the forward's chain would have left it.
extern "C" __global__ void __launch_bounds__(128) tf_q35k_apply(Geo geo, Seg sg, Path path) {
    const int NK = geo.nk, NV = geo.nv;
    const int layer = blockIdx.z / NV, hv = blockIdx.z % NV;
    const int hk = hv / (NV / NK);
    const int dk = threadIdx.x;
    const int dv = blockIdx.y * 4 + threadIdx.y;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv) * DK + PT * dk;
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
    float state[PT];
#pragma unroll
    for (int i = 0; i < PT; i++) state[i] = st[i];
    for (int j = 0; j < path.n; j++) {
        const LogRow l = log_row(geo, sg, layer, path.rows[j]);
        const float g = l.g[hv], d = l.delta[hv * DK + dv];
#pragma unroll
        for (int i = 0; i < PT; i++) {
            state[i] = state[i] * g;
            state[i] = fmaf(bf(l.k[hk * DK + PT * dk + i]), d, state[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < PT; i++) st[i] = state[i];
}

// Each linear layer's convolution tail as logged after window row `last`. Grid (nl, tail values / 256), 256 threads.
extern "C" __global__ void tf_q35k_conv(Geo geo, Seg sg, int last) {
    const int layer = blockIdx.x;
    const size_t n = geo.conv_rows();
    const size_t i = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16* CS = reinterpret_cast<__nv_bfloat16*>(sg.conv) + (size_t)layer * n;
    CS[i] = log_row(geo, sg, layer, last).conv[i];
}

// The path's keys and values move down to the slots after the cache; row j lands at or below j, never on one unread.
extern "C" __global__ void __launch_bounds__(256) tf_q35k_compact(Geo geo, Seg sg, Path path) {
    const int KVH = geo.kvh;
    const int layer = blockIdx.x, h = blockIdx.y;
    __nv_bfloat16* kv = reinterpret_cast<__nv_bfloat16*>(sg.kv);
    const int d = threadIdx.x;
    for (int j = 0; j < path.n; j++) {
        const int from = sg.len + path.rows[j], to = sg.len + j;
        if (from != to) {
            kv[kv_row(sg, KVH, layer, 0, h, to) + d] = kv[kv_row(sg, KVH, layer, 0, h, from) + d];
            kv[kv_row(sg, KVH, layer, 1, h, to) + d] = kv[kv_row(sg, KVH, layer, 1, h, from) + d];
        }
        __syncthreads();
    }
}
