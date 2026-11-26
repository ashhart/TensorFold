// Production Hy4-flash hyper-connection kernels (`hy4_hc_pre`, `hy4_hc_head`, `hy4_hc_merge`).
#include "hc.cuh"

extern "C" __global__ void __launch_bounds__(hy4_hc::kThreads)
    hy4_hc_pre(const unsigned short* x, const float* fn, const float* base, const float* scale, float eps, float hc_eps,
               float magnitude, unsigned short* out, float* post, int hidden, int repeat) {
    hy4_hc::hc_mix<8, 0>(x, fn, base, scale, eps, hc_eps, magnitude, out, post, hidden, repeat);
}

extern "C" __global__ void __launch_bounds__(hy4_hc::kThreads)
    hy4_hc_head(const unsigned short* x, const float* fn, const float* base, const float* scale, float eps,
                float hc_eps, float magnitude, unsigned short* out, float* post, int hidden, int repeat) {
    hy4_hc::hc_mix<4, 0>(x, fn, base, scale, eps, hc_eps, magnitude, out, post, hidden, repeat);
}

extern "C" __global__ void __launch_bounds__(hy4_hc::kThreads)
    hy4_hc_merge(const unsigned short* residual, const unsigned short* f, const float* post, unsigned short* out,
                 int rows, int hidden, int repeat) {
    hy4_hc::hc_merge<0>(residual, f, post, out, rows, hidden, repeat);
}
