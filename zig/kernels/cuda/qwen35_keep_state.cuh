// apply, apply_pend and conv. qwen35_keep.cu includes this twice: TF_Q35_TABLE 0 (main's), 1 (_table, TF_Q35_K)

// Each linear layer's recurrence through the path's (decay, key, write), as the forward's chain would have left it.
extern "C" __global__ void __launch_bounds__(128) TF_Q35_K(tf_q35k_apply)(Geo geo, Seg sg, Path path) {
    const int NK = geo.nk, NV = geo.nv;
    const int layer = blockIdx.z / NV, hv = blockIdx.z % NV;
    const int hk = hv / (NV / NK);
    const int dk = threadIdx.x;
    const int dv = blockIdx.y * 4 + threadIdx.y;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv) * DK + PT * dk;
#if TF_Q35_TABLE
    float* st = rec_of(states_of(sg), layer) + at;
#else
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
#endif
    float state[PT];
#pragma unroll
    for (int i = 0; i < PT; i++) state[i] = st[i];
    for (int j = 0; j < path.n; j++) {
#if TF_Q35_TABLE
        const LogRow l = log_row(geo, states_of(sg), layer, path.rows[j]);
#else
        const LogRow l = log_row(geo, sg, layer, path.rows[j]);
#endif
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

// Pending kept rows 0..n-1 into each linear layer's recurrence in order, with tf_q35k_apply's fp32 steps.
extern "C" __global__ void __launch_bounds__(128) TF_Q35_K(tf_q35k_apply_pend)(Geo geo, Seg sg, int n) {
    const int NK = geo.nk, NV = geo.nv;
    const int layer = blockIdx.z / NV, hv = blockIdx.z % NV;
    const int hk = hv / (NV / NK);
    const int dk = threadIdx.x;
    const int dv = blockIdx.y * 4 + threadIdx.y;
    constexpr int PT = DK / 32;
    const size_t at = ((size_t)hv * DK + dv) * DK + PT * dk;
#if TF_Q35_TABLE
    float* st = rec_of(states_of(sg), layer) + at;
#else
    float* st = reinterpret_cast<float*>(sg.rec) + (size_t)layer * geo.rec() + at;
#endif
    float state[PT];
#pragma unroll
    for (int i = 0; i < PT; i++) state[i] = st[i];
    for (int j = 0; j < n; j++) {
        const LogRow l = pend_row(geo, sg.plog, layer, j);
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
extern "C" __global__ void TF_Q35_K(tf_q35k_conv)(Geo geo, Seg sg, int last) {
    const int layer = blockIdx.x;
    const size_t n = geo.conv_rows();
    const size_t i = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) return;
#if TF_Q35_TABLE
    __nv_bfloat16* CS = conv_of(geo, states_of(sg), layer);
    CS[i] = log_row(geo, states_of(sg), layer, last).conv[i];
#else
    __nv_bfloat16* CS = reinterpret_cast<__nv_bfloat16*>(sg.conv) + (size_t)layer * n;
    CS[i] = log_row(geo, sg, layer, last).conv[i];
#endif
}
