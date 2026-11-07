#pragma once

#include "common.hpp"

// BF16 vision primitives. LayerNorm keeps both reductions and its affine epilogue in FP32.
extern "C" __global__ void __launch_bounds__(256)
tf_layer_norm_bf16(const uint16_t* x, const uint16_t* weight, const uint16_t* bias,
                   uint16_t* out, int width, float eps) {
    __shared__ float sums[256];
    const int t = threadIdx.x;
    const long long row = static_cast<long long>(blockIdx.x) * width;
    float sum = 0.0f;
    for (int c = t; c < width; c += 256) sum += bf16_to_float(x[row + c]);
    sums[t] = sum;
    __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (t < d) sums[t] += sums[t + d];
        __syncthreads();
    }
    const float mean = sums[0] / width;
    float var = 0.0f;
    for (int c = t; c < width; c += 256) {
        const float delta = bf16_to_float(x[row + c]) - mean;
        var += delta * delta;
    }
    __syncthreads();
    sums[t] = var;
    __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (t < d) sums[t] += sums[t + d];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / width + eps);
    for (int c = t; c < width; c += 256) {
        const float z = (bf16_to_float(x[row + c]) - mean) * inv;
        out[row + c] = float_to_bf16(z * bf16_to_float(weight[c]) + bf16_to_float(bias[c]));
    }
}

extern "C" __global__ void __launch_bounds__(256)
tf_gelu_bf16(const uint16_t* x, uint16_t* out, long long n) {
    const long long i = gid();
    if (i < n) {
        const float z = bf16_to_float(x[i]);
        out[i] = float_to_bf16((0.5f * z) * (1.0f + erff(z * 0.70710678118654752440f)));
    }
}

// Packed [row, 3, head, dim] becomes Q FP32 and K/V BF16, all head-major.
// The FP32 products of vision RoPE round once to BF16, including the widened Q output.
extern "C" __global__ void __launch_bounds__(256)
tf_vision_rope_bf16(const uint16_t* qkv, const unsigned* positions, float* q, uint16_t* k,
                    uint16_t* v, int rows, int heads, int dim, float theta) {
    const long long i = gid(), n = static_cast<long long>(rows) * heads * dim;
    if (i >= n) return;
    const int c = i % dim, h = (i / dim) % heads, r = i / (dim * heads);
    const int half = dim / 2, quarter = dim / 4, j = c % half;
    const float exponent = static_cast<float>(j % quarter) / quarter;
    const float inv = 1.0f / powf(theta, exponent);
    const float angle = static_cast<float>(positions[2 * r + j / quarter]) * inv;
    const float co = cosf(angle), si = sinf(angle);
    const long long src = static_cast<long long>(r) * 3 * heads * dim + h * dim;
    const int partner = c < half ? c + half : c - half;
    const float sign = c < half ? -1.0f : 1.0f;
    const long long dst = (static_cast<long long>(h) * rows + r) * dim + c;
    const float qx = bf16_to_float(qkv[src + c]), qp = bf16_to_float(qkv[src + partner]);
    const float kx = bf16_to_float(qkv[src + heads * dim + c]);
    const float kp = bf16_to_float(qkv[src + heads * dim + partner]);
    q[dst] = bf16_to_float(float_to_bf16(qx * co + (sign * qp) * si));
    k[dst] = float_to_bf16(kx * co + (sign * kp) * si);
    v[dst] = qkv[src + 2 * heads * dim + c];
}

extern "C" __global__ void __launch_bounds__(256)
tf_attention_rows_bf16(const float* x, uint16_t* out, int rows, int heads, int dim) {
    const long long i = gid(), n = static_cast<long long>(rows) * heads * dim;
    if (i >= n) return;
    const int c = i % dim, h = (i / dim) % heads, r = i / (dim * heads);
    out[i] = float_to_bf16(x[(static_cast<long long>(h) * rows + r) * dim + c]);
}

// Dense BF16 image/text operators; compensated FP32 sums keep their order at every prompt or verify width.

namespace {

extern "C" __global__ __launch_bounds__(256)
void tf_vl_embed(const void* table, const int* ids, void* out, int rows, int width, int kind, float scale) {
    const long long i = gid();
    if (i >= (long long)rows * width) return;
    const int row = i / width, column = i % width;
    store(out, kind, i, load(table, kind, (long long)ids[row] * width + column) * scale);
}

extern "C" __global__ __launch_bounds__(256)
void tf_vl_linear(const void* x, const void* w, const void* bias, void* out,
                  int rows, int n, int k, int kind, int out_kind) {
    __shared__ float xs[16][16];
    __shared__ float ws[64][16];
    const int tid = threadIdx.x, row = tid / 16, col = tid % 16;
    const int r = blockIdx.y * 16 + row, base = blockIdx.x * 64;
    float acc[4] = {0, 0, 0, 0};
    float correction[4] = {0, 0, 0, 0};
    for (int first = 0; first < k; first += 16) {
        xs[row][col] = r < rows && first + col < k ? load(x, kind, (long long)r * k + first + col) : 0;
        for (int j = tid; j < 64 * 16; j += 256) {
            int nc = base + j / 16, kc = first + j % 16;
            ws[j / 16][j % 16] = nc < n && kc < k ? load(w, kind, (long long)nc * k + kc) : 0;
        }
        __syncthreads();
        for (int j = 0; j < 16 && first + j < k; ++j) {
            float xv = xs[row][j];
            for (int tile = 0; tile < 4; ++tile) {
                float product = xv * ws[col + tile * 16][j];
                if (isfinite(acc[tile]) && isfinite(product)) {
                    float adjusted = product - correction[tile];
                    float next = acc[tile] + adjusted;
                    correction[tile] = isfinite(next) ? (next - acc[tile]) - adjusted : 0;
                    acc[tile] = next;
                } else {
                    acc[tile] += product;
                    correction[tile] = 0;
                }
            }
        }
        __syncthreads();
    }
    if (r >= rows) return;
    for (int tile = 0; tile < 4; ++tile) {
        int nc = base + col + tile * 16;
        if (nc < n) {
            float value = acc[tile];
            if (bias) value += load(bias, kind, nc);
            store(out, out_kind, (long long)r * n + nc, value);
        }
    }
}

// Not tf_layer_norm_bf16: that one takes rsqrtf, this one 1 / sqrtf. The sums are the same, yet about one
// BF16 output in 100,000 differs, so dots.mocr and Nemotron Parse each keep the norm their receipts were made on.
extern "C" __global__ __launch_bounds__(256)
void tf_vl_layernorm(const void* x, const void* weight, const void* bias, void* out,
                     int rows, int width, int kind, float eps) {
    __shared__ float sums[256];
    const int tid = threadIdx.x, row = blockIdx.x;
    if (row >= rows) return;
    const long long base = (long long)row * width;
    float sum = 0;
    for (int j = tid; j < width; j += 256) sum += load(x, kind, base + j);
    sums[tid] = sum;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (tid < step) sums[tid] += sums[tid + step];
        __syncthreads();
    }
    const float mean = sums[0] / width;
    __syncthreads();
    sum = 0;
    for (int j = tid; j < width; j += 256) {
        float centered = load(x, kind, base + j) - mean;
        sum += centered * centered;
    }
    sums[tid] = sum;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (tid < step) sums[tid] += sums[tid + step];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(sums[0] / width + eps);
    for (int j = tid; j < width; j += 256) {
        float value = (load(x, kind, base + j) - mean) * inv;
        value = value * load(weight, kind, j) + load(bias, kind, j);
        store(out, kind, base + j, value);
    }
}

extern "C" __global__ __launch_bounds__(256)
void tf_vl_pointwise(const void* x, const void* y, void* out, long long count, int kind, int mode, float scale) {
    const long long i = gid();
    if (i >= count) return;
    const float value = load(x, kind, i);
    float result;
    if (mode == 0) result = (0.5f * value) * (1.0f + erff(value * 0.7071067811865475f));
    else if (mode == 1) result = value + load(y, kind, i);
    else result = value * scale;
    store(out, kind, i, result);
}

extern "C" __global__ __launch_bounds__(256)
void tf_vl_layout(const void* x, void* out, int rows, int heads, int dim, int in_kind, int out_kind, int mode) {
    const long long i = gid(), count = (long long)rows * heads * dim;
    if (i >= count) return;
    long long source;
    if (mode == 0) {
        // Attention returns [head,row,dim]; projections consume [row,head,dim].
        const int d = i % dim, h = (i / dim) % heads, r = i / (heads * dim);
        source = ((long long)h * rows + r) * dim + d;
    } else {
        // Conv2's input groups four horizontal patches, channel before group member.
        const int member = i % 4, channel = (i / 4) % dim;
        source = ((i / (4 * dim)) * 4 + member) * dim + channel;
    }
    store(out, out_kind, i, load(x, in_kind, source));
}

}  // namespace

#include "vl_position.hpp"
