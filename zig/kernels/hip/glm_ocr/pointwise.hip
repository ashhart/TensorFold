// Copyright (c) 2016- Facebook, Inc. and the other holders in LICENSES/PyTorch-BSD-3.txt.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>

// Reduction order and LayerNorm adapt PyTorch 8514f05; BSD-3-Clause, see LICENSES/PyTorch-BSD-3.txt.
using B = hip_bfloat16;
__device__ inline unsigned long long index() {
    return static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
}
__device__ inline float rounded(float v) { return float(B(v)); }
struct Moments { float mean, variance, count; };
__device__ Moments update(Moments a, float value) {
#pragma clang fp contract(fast)
    const float delta = value - a.mean;
    const float count = a.count + 1.f;
    const float mean = a.mean + delta * __builtin_amdgcn_rcpf(count);
    return {mean, a.variance + delta * (value - mean), count};
}
__device__ Moments combine(Moments b, Moments a) {
#pragma clang fp contract(fast)
    const float delta = b.mean - a.mean;
    const float count = a.count + b.count;
    if (!(count > 0.f)) return {0.f, 0.f, 0.f};
    const float reciprocal = __builtin_amdgcn_rcpf(count);
    const float na = a.count * reciprocal, nb = b.count * reciprocal;
    return {na * a.mean + nb * b.mean,
        a.variance + b.variance + delta * delta * a.count * nb, count};
}

// The Conv3d output is rounded to BF16 before its BF16 bias is added and rounded again.
extern "C" __global__ void __launch_bounds__(256) tf_glmocr_bias(
    hip_bfloat16* x, const hip_bfloat16* bias, unsigned long long count, unsigned int width) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) x[i] = hip_bfloat16(float(x[i]) + float(bias[i % width]));
}

extern "C" __global__ void __launch_bounds__(256) tf_glmocr_bias_fill(
    B* x, const B* bias, unsigned long long count, unsigned int width) {
    const auto i = index();
    if (i < count) x[i] = bias[i % width];
}

extern "C" __global__ void __launch_bounds__(256) tf_glmocr_cast(const float* x, B* out, unsigned long long count) {
    const auto i = index();
    if (i < count) out[i] = B(x[i]);
}

extern "C" __global__ void __launch_bounds__(256) tf_glmocr_add(const B* x, const B* y, B* out, unsigned long long count) {
    const auto i = index();
    if (i < count) out[i] = B(float(x[i]) + float(y[i]));
}

extern "C" __global__ void __launch_bounds__(256) tf_glmocr_silu_mul(
    const B* gate, const B* up, B* out, unsigned long long count) {
    const auto i = index();
    if (i < count) {
        const float g = float(gate[i]);
        const float activated = rounded(g / (1.f + expf(-g)));
        out[i] = B(activated * float(up[i]));
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_glmocr_gelu(B* x, unsigned long long count) {
    const auto i = index();
    if (i < count) {
        const float v = float(x[i]);
        x[i] = B((0.5f * v) * (1.f + erff(v * 0.7071067811865475f)));
    }
}

// Four source patches per merge cell, with channels moved before the patch's two spatial axes.
extern "C" __global__ void __launch_bounds__(256) tf_glmocr_merge_layout(
    const B* x, B* out, unsigned long long count, unsigned int width, int reverse) {
    const auto i = index();
    if (i < count) {
        const auto cell = i / (4 * width);
        const auto channel = (i / 4) % width;
        const auto patch = i % 4;
        const auto other = (cell * 4 + patch) * width + channel;
        if (reverse) out[other] = x[i];
        else out[i] = x[other];
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_glmocr_qkv_split(
    const B* x, B* q, B* k, B* v, unsigned long long count, unsigned int width) {
    const auto i = index();
    if (i < count) {
        const auto row = i / width;
        const auto col = i % width;
        q[i] = x[row * 3 * width + col];
        k[i] = x[row * 3 * width + width + col];
        v[i] = x[row * 3 * width + 2 * width + col];
    }
}

// RMS norm has a BF16 node before weight multiplication; LayerNorm keeps its affine step in FP32.
extern "C" __global__ void __launch_bounds__(256) tf_glmocr_norm(
    const B* x, const B* weight, const B* bias, B* out, unsigned int width, float eps, int layer) {
    __shared__ float part[256];
    const auto base = static_cast<unsigned long long>(blockIdx.x) * width;
    const int tid = threadIdx.x;
    if (!layer) {
        unsigned height = 1;
        while (height * 2 <= gridDim.x && height < 16) height *= 2;
        unsigned lanes = 1;
        const unsigned items = width >= 128 ? width / 4 : width;
        while (lanes * 2 <= items && lanes * 2 <= 512 / height && lanes < 256) lanes *= 2;
        float accum[4] = {0.f, 0.f, 0.f, 0.f};
        if (tid < lanes) {
            if (width >= 128) {
                for (unsigned i = tid * 4; i < width; i += lanes * 4) {
                    for (unsigned j = 0; j < 4 && i + j < width; ++j) {
                        const float z = float(x[base + i + j]);
                        accum[j] += z * z;
                    }
                }
            } else {
                for (unsigned i = tid; i < width; i += lanes * 4) {
                    for (unsigned j = 0; j < 4 && i + j * lanes < width; ++j) {
                        const float z = float(x[base + i + j * lanes]);
                        accum[j] += z * z;
                    }
                }
            }
        }
        float total = ((accum[0] + accum[1]) + accum[2]) + accum[3];
        part[tid] = total;
        for (unsigned step = lanes / 2; step >= 32; step /= 2) {
            __syncthreads();
            if (tid < step) part[tid] += part[tid + step];
        }
        __syncthreads();
        total = part[tid];
        for (unsigned step = 1; step < lanes && step < 32; step *= 2)
            total += __shfl_down(total, step, 32);
        if (tid == 0) part[0] = float(1.0 / sqrt(double(total * (1.f / width) + eps)));
        __syncthreads();
        const float inv = part[0];
        for (unsigned i = tid; i < width; i += 256)
            out[base + i] = B(rounded(float(x[base + i]) * inv) * float(weight[i]));
        return;
    }
    Moments state = {0.f, 0.f, 0.f};
    for (unsigned i = tid * 4; i < width; i += 256 * 4)
        for (unsigned j = 0; j < 4 && i + j < width; ++j) state = update(state, float(x[base + i + j]));
    for (int shift = 16; shift > 0; shift /= 2) {
        const Moments other = {__shfl_down(state.mean, shift, 32), __shfl_down(state.variance, shift, 32),
            __shfl_down(state.count, shift, 32)};
        state = combine(state, other);
    }
    const unsigned wave = tid / 32;
    if (tid % 32 == 0) { part[wave] = state.mean; part[8 + wave] = state.variance; part[16 + wave] = state.count; }
    __syncthreads();
    for (unsigned step = 4; step; step /= 2) {
        if (tid % 32 == 0 && wave < step) {
            const Moments other = {part[wave + step], part[8 + wave + step], part[16 + wave + step]};
            state = combine(state, other);
            part[wave] = state.mean; part[8 + wave] = state.variance; part[16 + wave] = state.count;
        }
        __syncthreads();
    }
    const float mean = part[0];
    const float variance = part[8] / width;
    const float inv = rsqrtf(variance + eps);
    for (unsigned i = tid; i < width; i += 256) {
        const float v = (float(x[base + i]) - mean) * inv;
        out[base + i] = B(fmaf(v, float(weight[i]), float(bias[i])));
    }
}

// A thread owns both rotary partners, so the source and destination may be the same buffer.
extern "C" __global__ void __launch_bounds__(256) tf_glmocr_vision_rope(
    B* x, const float* inverse, unsigned long long pairs, unsigned int grid_w) {
    const auto i = index();
    if (i >= pairs) return;
    const auto head = i / 32;
    const unsigned j = i % 32;
    const auto row = head / 16;
    const auto cell = row / 4;
    const unsigned h = (cell / (grid_w / 2)) * 2 + (row % 4) / 2;
    const unsigned w = (cell % (grid_w / 2)) * 2 + row % 2;
    const float phase = float(j < 16 ? h : w) * inverse[j % 16];
    const float co = cosf(phase), si = sinf(phase);
    const auto at = head * 64 + j;
    const float a = float(x[at]), b = float(x[at + 32]);
    x[at] = B(a * co - b * si);
    x[at + 32] = B(b * co + a * si);
}

extern "C" __global__ void __launch_bounds__(256) tf_glmocr_text_rope(
    B* x, const float* inverse, const int* positions, unsigned long long pairs, unsigned int heads) {
    const auto i = index();
    if (i >= pairs) return;
    const auto head = i / 64;
    const unsigned j = i % 64;
    const auto row = head / heads;
    const unsigned axis = j < 16 ? 0 : (j < 40 ? 1 : 2);
    const float phase = float(positions[row * 3 + axis]) * inverse[j];
    const float co = rounded(cosf(phase)), si = rounded(sinf(phase));
    const auto at = head * 128 + 2 * j;
    const float a = float(x[at]), b = float(x[at + 1]);
    x[at] = B(rounded(a * co) + rounded(-b * si));
    x[at + 1] = B(rounded(b * co) + rounded(a * si));
}
