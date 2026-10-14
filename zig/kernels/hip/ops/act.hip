
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>

namespace tf {
namespace rocm {

__device__ float warp_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

__device__ inline float widen(float v) { return v; }
__device__ inline float widen(__half v) { return __half2float(v); }
__device__ inline float widen(hip_bfloat16 v) { return static_cast<float>(v); }
__device__ inline void narrow(float v, float* out) { *out = v; }
__device__ inline void narrow(float v, __half* out) { *out = __float2half_rn(v); }
__device__ inline void narrow(float v, hip_bfloat16* out) { *out = hip_bfloat16(v); }

// fp32 sum of squares by one wave (lane l adds l, l + 32, ... then the xor tree), so bits do not depend on blockDim.
constexpr int kRmsThreads = 256;
constexpr int kRmsWidth = 8192;

template <typename T>
__global__ void __launch_bounds__(kRmsThreads) rms_kernel(const T* __restrict__ x, const float* __restrict__ weight,
                                                         T* __restrict__ y, int width, float eps) {
    __shared__ float row_v[kRmsWidth];
    __shared__ float inv_s;
    const int row = blockIdx.x;
    const T* in = x + static_cast<long long>(row) * width;
    for (int i = threadIdx.x; i < width; i += kRmsThreads) row_v[i] = widen(in[i]);
    __syncthreads();
    if (threadIdx.x < 32) {
        float sum = 0.f;
        for (int i = threadIdx.x; i < width; i += 32) sum += row_v[i] * row_v[i];
        const float inv = rsqrtf(warp_sum(sum) / static_cast<float>(width) + eps);
        if (threadIdx.x == 0) inv_s = inv;
    }
    __syncthreads();
    const float inv = inv_s;
    T* out = y + static_cast<long long>(row) * width;
    for (int i = threadIdx.x; i < width; i += kRmsThreads) {
        float value = row_v[i] * inv;
        if (weight != nullptr) value *= weight[i];
        narrow(value, out + i);
    }
}

// Length 1. Taps run in order, then the state shifts toward the new sample.
extern "C" __global__ void tf_conv_decode(const float* __restrict__ x, const float* __restrict__ weight,
                                   float* __restrict__ state, float* __restrict__ y, int channels, int kernel) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    if (channel >= channels) return;
    const float* row = weight + static_cast<long long>(channel) * kernel;
    float history[7];
    int kept = kernel - 1;
    float acc = 0.f;
    for (int tap = 0; tap < kept; ++tap) {
        history[tap] = state[(static_cast<long long>(batch) * kept + tap) * channels + channel];
        acc += history[tap] * row[tap];
    }
    float sample = x[static_cast<long long>(batch) * channels + channel];
    acc += sample * row[kept];
    float gate = 1.f / (1.f + expf(-acc));
    y[(static_cast<long long>(batch) * channels) + channel] = acc * gate;
    for (int tap = 0; tap < kept - 1; ++tap) {
        state[(static_cast<long long>(batch) * kept + tap) * channels + channel] = history[tap + 1];
    }
    if (kept > 0) state[(static_cast<long long>(batch) * kept + kept - 1) * channels + channel] = sample;
}

// A window's rows in one launch: conv_decode's step for each row in turn, the state after each kept in ``states``.
extern "C" __global__ void tf_conv_rows(const float* __restrict__ x, const float* __restrict__ weight,
                                 float* __restrict__ state, float* __restrict__ y, float* __restrict__ states,
                                 int channels, int kernel, int rows) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const float* row = weight + static_cast<long long>(channel) * kernel;
    float history[7];
    int kept = kernel - 1;
    for (int tap = 0; tap < kept; ++tap) history[tap] = state[static_cast<long long>(tap) * channels + channel];
    for (int r = 0; r < rows; ++r) {
        float acc = 0.f;
        for (int tap = 0; tap < kept; ++tap) acc += history[tap] * row[tap];
        float sample = x[static_cast<long long>(r) * channels + channel];
        acc += sample * row[kept];
        float gate = 1.f / (1.f + expf(-acc));
        y[static_cast<long long>(r) * channels + channel] = acc * gate;
        for (int tap = 0; tap < kept - 1; ++tap) history[tap] = history[tap + 1];
        if (kept > 0) history[kept - 1] = sample;
        if (states != nullptr) {
            for (int tap = 0; tap < kept; ++tap) {
                states[(static_cast<long long>(r) * kept + tap) * channels + channel] = history[tap];
            }
        }
    }
    for (int tap = 0; tap < kept; ++tap) state[static_cast<long long>(tap) * channels + channel] = history[tap];
}

// ``per`` > 0: rows of a window, ``per`` rows to a position, row r at ``pos_dev[r / per]``.
extern "C" __global__ void tf_rope_decode(const float* __restrict__ x, float* __restrict__ y, int width, int rotary,
                                   int pos, float theta, const int* __restrict__ pos_dev, int per) {
    int row = blockIdx.x;
    if (pos_dev != nullptr) pos = per > 0 ? pos_dev[row / per] : *pos_dev;
    int lane = threadIdx.x;
    const float* in = x + static_cast<long long>(row) * width;
    float* out = y + static_cast<long long>(row) * width;
    int half = rotary >> 1;
    for (int i = lane; i < half; i += 32) {
        float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(half));
        float ang = static_cast<float>(pos) * freq;
        float c = cosf(ang);
        float s = sinf(ang);
        float x1 = in[i];
        float x2 = in[half + i];
        out[i] = x1 * c - x2 * s;
        out[half + i] = x1 * s + x2 * c;
    }
    for (int i = rotary + lane; i < width; i += 32) out[i] = in[i];
}

template <typename T>
__global__ void __launch_bounds__(256) moe_router_kernel(const T* __restrict__ x, const float* __restrict__ rows,
                                                         float* __restrict__ logits, int d, int e) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int row = blockIdx.y;
    if (expert >= e) return;
    const T* xr = x + static_cast<long long>(row) * d;
    const float* w = rows + static_cast<long long>(expert) * d;
    float acc = 0.f;
    for (int i = lane; i < d; i += 32) acc = fmaf(widen(xr[i]), w[i], acc);
    acc = warp_sum(acc);
    if (lane == 0) logits[static_cast<long long>(row) * e + expert] = acc;
}

constexpr int kSelectMax = 1024;

__device__ inline float round_bf16(float v) { return static_cast<float>(hip_bfloat16(v)); }

extern "C" __global__ void __launch_bounds__(32) tf_moe_select(const float* __restrict__ logits, int* __restrict__ pick,
                                                        float* __restrict__ wts, int* __restrict__ items,
                                                        int* __restrict__ members, int capacity, int experts,
                                                        int top_k) {
    __shared__ float v[kSelectMax];
    __shared__ float ex[32];
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int slots = top_k + 1;
    const float* l = logits + static_cast<long long>(row) * (experts + 1);
    for (int i = lane; i < experts; i += 32) v[i] = l[i];
    __syncthreads();
    float top = 0.f, total = 0.f;
    for (int k = 0; k < top_k; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
        for (int i = lane; i < experts; i += 32) {
            const float value = v[i];
            if (value > best || (value == best && i < idx)) {
                best = value;
                idx = i;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other = __shfl_xor(best, mask, 32);
            const int at = __shfl_xor(idx, mask, 32);
            if (other > best || (other == best && at < idx)) {
                best = other;
                idx = at;
            }
        }
        if (k == 0) top = best;
        const float e = expf(best - top);
        total += e;
        if (lane == 0) {
            pick[row * slots + k] = idx;
            ex[k] = e;
            v[idx] = -INFINITY;
        }
        __syncthreads();
    }
    if (lane == 0) {
        for (int k = 0; k < top_k; ++k) wts[row * slots + k] = round_bf16(ex[k] / total);
        const float gate = round_bf16(l[experts]);
        pick[row * slots + top_k] = experts;
        wts[row * slots + top_k] = round_bf16(1.f / (1.f + expf(-gate)));
    }
    if (items != nullptr) {
        __syncthreads();
        for (int k = lane; k < capacity; k += 32) {
            const bool real = k < slots;
            items[3 * k] = real ? pick[k] : 0;
            items[3 * k + 1] = real ? k : 0;
            items[3 * k + 2] = real ? 1 : 0;
            if (real) members[k] = k;
        }
    }
}

template <typename T>
__global__ void moe_act_kernel(const float* __restrict__ both, T* __restrict__ out, int width, float limit,
                               long long total) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long p = i / width;
    const int c = static_cast<int>(i - p * width);
    float g = both[p * 2 * width + c];
    float u = both[p * 2 * width + width + c];
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    narrow(g / (1.f + expf(-g)) * u, out + i);
}

template <typename T>
__global__ void moe_combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, T* __restrict__ out,
                                   int slots, int d) {
    const int row = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;
    float acc = 0.f;
    for (int s = 0; s < slots; ++s) {
        acc += y[(static_cast<long long>(row) * slots + s) * d + c] * wts[row * slots + s];
    }
    narrow(acc, out + static_cast<long long>(row) * d + c);
}

template <typename T>
__global__ void gdn_gate_kernel(const T* __restrict__ a, const T* __restrict__ b, const float* __restrict__ a_log,
                                const float* __restrict__ dt_bias, float* __restrict__ gate, float* __restrict__ beta,
                                int count, int heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int h = i % heads;
    beta[i] = 1.f / (1.f + expf(-widen(b[i])));
    const float x = widen(a[i]) + dt_bias[h];
    const float soft = x > 20.f ? x : log1pf(expf(x));
    gate[i] = expf(-expf(a_log[h]) * soft);
}

// The template kernels launches.zig resolves.
template __global__ void rms_kernel<float>(const float* __restrict__ x, const float* __restrict__ weight,
    float* __restrict__ y, int width, float eps);
template __global__ void rms_kernel<__half>(const __half* __restrict__ x, const float* __restrict__ weight,
    __half* __restrict__ y, int width, float eps);
template __global__ void rms_kernel<hip_bfloat16>(const hip_bfloat16* __restrict__ x, const float* __restrict__ weight,
    hip_bfloat16* __restrict__ y, int width, float eps);
template __global__ void moe_router_kernel<__half>(const __half* __restrict__ x, const float* __restrict__ rows,
    float* __restrict__ logits, int d, int e);
template __global__ void moe_router_kernel<hip_bfloat16>(const hip_bfloat16* __restrict__ x,
    const float* __restrict__ rows, float* __restrict__ logits, int d, int e);
template __global__ void moe_act_kernel<__half>(const float* __restrict__ both, __half* __restrict__ out, int width,
    float limit, long long total);
template __global__ void moe_act_kernel<hip_bfloat16>(const float* __restrict__ both, hip_bfloat16* __restrict__ out,
    int width, float limit, long long total);
template __global__ void moe_combine_kernel<float>(const float* __restrict__ y, const float* __restrict__ wts,
    float* __restrict__ out, int slots, int d);
template __global__ void moe_combine_kernel<__half>(const float* __restrict__ y, const float* __restrict__ wts,
    __half* __restrict__ out, int slots, int d);
template __global__ void moe_combine_kernel<hip_bfloat16>(const float* __restrict__ y, const float* __restrict__ wts,
    hip_bfloat16* __restrict__ out, int slots, int d);
template __global__ void gdn_gate_kernel<float>(const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ gate,
    float* __restrict__ beta, int count, int heads);
template __global__ void gdn_gate_kernel<__half>(const __half* __restrict__ a, const __half* __restrict__ b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ gate,
    float* __restrict__ beta, int count, int heads);
template __global__ void gdn_gate_kernel<hip_bfloat16>(const hip_bfloat16* __restrict__ a,
    const hip_bfloat16* __restrict__ b, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    float* __restrict__ gate, float* __restrict__ beta, int count, int heads);

}  // namespace rocm
}  // namespace tf
