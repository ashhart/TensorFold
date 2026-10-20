// qwen35.cu's row-wise kernels: the tap ring's residuals and the MLP activation, eight entries a thread.
#pragma once

// A tapped layer's residual bf16(h + r) into the segment's tap ring at the row's position; grid (rows, D / 2048).
extern "C" __global__ void __launch_bounds__(256) tf_q35_tap(const __nv_bfloat16* __restrict__ H, const __nv_bfloat16* __restrict__ R,
                                                              const Seg* __restrict__ segs, const int* __restrict__ rowseg, int tap, int D,
                                                              const int* __restrict__ P) {
    if (past_rows(P, blockIdx.x)) return;
    const size_t m = blockIdx.x;
    const Seg sg = segs[rowseg[m]];
    if (sg.tap == 0) return;
    const long long pos = sg.len + (long long)m - sg.row0;
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(sg.tap) + ((size_t)(pos % sg.tap_rows) * sg.taps + tap) * D;
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 8;  // eight entries a thread, grid (rows, D / 2048)
    if (i >= D) return;
    const uint4 hw = *reinterpret_cast<const uint4*>(H + m * D + i), rw = *reinterpret_cast<const uint4*>(R + m * D + i);
    const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hw);
    const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rw);
    uint4 ow;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ow);
#pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = __float2bfloat16_rn(bf(hb[j]) + bf(rb[j]));
    *reinterpret_cast<uint4*>(out + i) = ow;
}

// out = bf16(silu(gate) * up) over `total` values, rows D wide, eight a thread; XS (may be null) their group sums.
extern "C" __global__ void __launch_bounds__(256) tf_q35_mlp_act(const __nv_bfloat16* __restrict__ GATE, const __nv_bfloat16* __restrict__ UP,
                                            __nv_bfloat16* __restrict__ OUT, long long total, const int* __restrict__ P, int D, float* __restrict__ XS) {
    __shared__ __align__(16) __nv_bfloat16 done[2048];
    if (P != nullptr) total = (long long)P[2] * D;
    const long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i < total) {
        const uint4 gw = *reinterpret_cast<const uint4*>(GATE + i), uw = *reinterpret_cast<const uint4*>(UP + i);
        const __nv_bfloat16* gb = reinterpret_cast<const __nv_bfloat16*>(&gw);
        const __nv_bfloat16* ub = reinterpret_cast<const __nv_bfloat16*>(&uw);
        uint4 ow;
        __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ow);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float gf = bf(gb[j]);
            ob[j] = __float2bfloat16_rn(gf / (1.0f + expf(-gf)) * bf(ub[j]));
        }
        *reinterpret_cast<uint4*>(OUT + i) = ow;
        *reinterpret_cast<uint4*>(done + threadIdx.x * 8) = ow;
    }
    if (XS == nullptr) return;
    __syncthreads();
    const long long first = (long long)blockIdx.x * 2048 + threadIdx.x * 64;  // the block's 32 groups, each in order
    if (threadIdx.x < 32 && first < total) XS[first / 64] = group_sum(done + threadIdx.x * 64);
}
