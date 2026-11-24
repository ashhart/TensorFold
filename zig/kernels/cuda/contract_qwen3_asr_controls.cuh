// Gate 3: vit.cu's conv taps and flatten for Qwen3-ASR's tower, each with one planted fault the shapes gate must catch.
#pragma once

// tf_vit_conv_cols reading the clamped edge sample where the conv's padding is zero.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_vit_conv_cols_edge(const __nv_bfloat16* __restrict__ X,
                                                                           __nv_bfloat16* __restrict__ Y, int fi, int ti,
                                                                           int c, int ld) {
    const int fo = (fi - 1) / 2 + 1, to = (ti - 1) / 2 + 1;
    const int64_t r = blockIdx.x;
    const int64_t n = r / (static_cast<int64_t>(fo) * to);
    const int f = static_cast<int>(r / to % fo), t = static_cast<int>(r % to);
    const int j = blockIdx.y * 256 + threadIdx.x;
    if (j >= ld) return;
    __nv_bfloat16 v = __float2bfloat16_rn(0.0f);
    if (j < 9 * c) {
        const int ci = j / 9, kh = j % 9 / 3, kw = j % 3;
        const int sf = min(max(2 * f - 1 + kh, 0), fi - 1), st = min(max(2 * t - 1 + kw, 0), ti - 1);
        v = X[((n * fi + sf) * ti + st) * c + ci];
    }
    Y[r * ld + j] = v;
}

// tf_vit_conv_flatten writing a row f-major (f c + ci) where torch flattens [c][f] (ci f + fb).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_vit_conv_flatten_fc(const __nv_bfloat16* __restrict__ X,
                                                                            __nv_bfloat16* __restrict__ Y, int f, int t,
                                                                            int c) {
    const int64_t row = blockIdx.x, n = row / t;
    const int tt = static_cast<int>(row % t);
    for (int e = threadIdx.x; e < c * f; e += 256) {
        const int fb = e / c, ci = e % c;
        Y[row * c * f + e] = X[((n * f + fb) * t + tt) * c + ci];
    }
}
