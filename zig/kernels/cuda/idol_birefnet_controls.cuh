// Gate 1-4 controls for Idol's background remover: idol_birefnet.cuh's bodies with one planted fault each.
#pragma once
#include "idol_birefnet.cuh"

// tf_br_conv adding its 256-term blocks in fp16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_br_conv_half_blocks(const float* x, const float* w, const float* b,
                                                                             const float* post, const float* r, const int* map,
                                                                             float* y, int rows, int cols, int cin, int ldx,
                                                                             int cout, int ldw, int k, int ldy, int yoff,
                                                                             int flags, int row0) {
    tf_br::conv<tf_br::CTL_CONV_HALF_BLOCKS>(x, w, b, post, r, map, y, rows, cols, cin, ldx, cout, ldw, k, ldy, yoff, flags, row0);
}

// tf_br_conv's ReLU by fmaxf, which turns a NaN into 0 where torch keeps it.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_br_conv_relu_max(const float* x, const float* w, const float* b,
                                                                          const float* post, const float* r, const int* map,
                                                                          float* y, int rows, int cols, int cin, int ldx,
                                                                          int cout, int ldw, int k, int ldy, int yoff,
                                                                          int flags, int row0) {
    tf_br::conv<tf_br::CTL_CONV_RELU_MAX>(x, w, b, post, r, map, y, rows, cols, cin, ldx, cout, ldw, k, ldy, yoff, flags, row0);
}

// tf_br_norm summing the row in fp16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_br_norm_half_sum(const float* x, const int* map, const float* g,
                                                                          const float* beta, float* y, int width, int mode,
                                                                          int src_w, int ldy, int yoff, float eps) {
    tf_br::norm<tf_br::CTL_NORM_HALF_SUM>(x, map, g, beta, y, width, mode, src_w, ldy, yoff, eps);
}

// tf_br_window rounding P to fp16 before P V.
extern "C" __global__ void __launch_bounds__(192) tf_ctl_br_window_half_p(const float* qkv, const float* table, float* o,
                                                                          int width, int heads, int rows, int cols, int shift,
                                                                          float scale) {
    tf_br::window<tf_br::CTL_WINDOW_HALF_P>(qkv, table, o, width, heads, rows, cols, shift, scale);
}

// tf_br_deform rounding its columns to fp16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_br_deform_half_col(const float* x, const float* offs, float* col,
                                                                            int rows, int cols, int k, int first, int count,
                                                                            int ldo) {
    tf_br::deform<tf_br::CTL_DEFORM_HALF_COL>(x, offs, col, rows, cols, k, first, count, ldo,
                                              blockIdx.x * blockDim.y + threadIdx.y, blockIdx.y * blockDim.x + threadIdx.x);
}

// tf_br_pool summing in fp16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_br_pool_half_sum(const float* x, float* mean, int pixels, int channels) {
    tf_br::pool<tf_br::CTL_POOL_HALF_SUM>(x, mean, pixels, channels);
}

// tf_br_resize staging its four samples in fp16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_br_resize_half(const float* x, float* y, int in_h, int in_w, int out_h,
                                                                        int out_w, int channels, int ldx, int xoff, int ldy,
                                                                        int yoff) {
    tf_br::resize<tf_br::CTL_RESIZE_HALF>(x, y, in_h, in_w, out_h, out_w, channels, ldx, xoff, ldy, yoff,
                                          blockIdx.x * blockDim.y + threadIdx.y, blockIdx.y * blockDim.x + threadIdx.x);
}
