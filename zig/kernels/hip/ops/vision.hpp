#pragma once

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
