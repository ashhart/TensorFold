#pragma once

__device__ inline uint16_t tf_columns_ctl_round(float v) {
    const uint32_t bits = __float_as_uint(v);
    return isnan(v) ? 0x7fc0 : static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

template <int Mode>
__device__ void tf_columns_ctl(const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t count, int width) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count || (Mode == 2 && i + 1 == count)) return;
    const int col = (i + (Mode < 2 ? 1 : 0)) % width;
    float v = tf_ctl_x(w, col);
    if constexpr (Mode == 0 || Mode == 3) v *= tf_ctl_x(x, i);
    if constexpr (Mode == 3) v = fmaxf(v, 0.f);
    out[i] = tf_columns_ctl_round(v);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_columns_shifted_multiply(
    const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t count, int width) {
    tf_columns_ctl<0>(x, w, out, count, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_columns_shifted_fill(
    const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t count, int width) {
    tf_columns_ctl<1>(x, w, out, count, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_columns_lost_tail(
    const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t count, int width) {
    tf_columns_ctl<2>(x, w, out, count, width);
}

extern "C" __global__ void __launch_bounds__(256) tf_ctl_columns_fmax_multiply(
    const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t count, int width) {
    tf_columns_ctl<3>(x, w, out, count, width);
}
