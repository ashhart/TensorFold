// MiniMax-H3-Image's VAE image decoder on CUDA: bf16 tensors, fp32 sums, the RMS norm's f64 sum.
#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include "common/reduce.cuh"

typedef __nv_bfloat16 bf16;

static __device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
static __device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

// Correctly rounded reciprocal square root, as the host's float64 1/sqrt rounded to f32.
static __device__ __forceinline__ float inv_sqrt(float x) { return __frsqrt_rn(x); }

// One parameter block for every kernel: the pass's tiles and their geometry, a product's size, the norms' epsilon.
struct H3dShape {
    unsigned tiles;
    unsigned h;
    unsigned w;
    unsigned tokens;
    unsigned rows;
    unsigned cols;
    unsigned m;
    unsigned n;
    unsigned k;
    unsigned mode;
    float eps;
    float scale;
};

// Keys a lane holds in the attention: 9 a lane, 288 a warp, past a 16 x 16 tile's 261 tokens.
#define H3D_SLOTS 9

extern "C" {

// Each tile's cells rounded to bf16, then post_quant_conv (24 to 24 with bias) to bf16: tok [tiles * h * w, 24].
__global__ void __launch_bounds__(256) h3d_post_quant(
        const float* lat, const uint2* org, const bf16* pw, const bf16* pb,
        bf16* tok, H3dShape p) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned cells = p.h * p.w;
    if (x >= 24 || y >= p.tiles * cells) return;
    const unsigned tile = y / cells;
    const unsigned i = y % cells;
    const size_t r = org[tile].x + i / p.w;
    const size_t c = org[tile].y + i % p.w;
    const size_t plane = (size_t)p.rows * p.cols;
    float acc = 0.0f;
    for (unsigned k = 0; k < 24; ++k) acc += f(pw[x * 24 + k]) * f(__float2bfloat16(lat[k * plane + r * p.cols + c]));
    tok[(size_t)y * 24 + x] = __float2bfloat16(acc + f(pb[x]));
}

// A tile's sequence: x_embedder (24 to 1,024 with bias) on its cells to bf16, then the 4 registers and a zero token.
__global__ void __launch_bounds__(256) h3d_embed(
        const bf16* tok, const bf16* ew, const bf16* eb, const bf16* reg,
        bf16* x, H3dShape p) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y;
    if (j >= 1024 || y >= p.tiles * p.tokens) return;
    const unsigned cells = p.h * p.w;
    const unsigned tile = y / p.tokens;
    const unsigned t = y % p.tokens;
    bf16 v = __float2bfloat16(0.0f);
    if (t < cells) {
        const bf16* in = tok + ((size_t)tile * cells + t) * 24;
        float acc = 0.0f;
        for (unsigned k = 0; k < 24; ++k) acc += f(ew[j * 24 + k]) * f(in[k]);
        v = __float2bfloat16(acc + f(eb[j]));
    } else if (t < cells + 4) {
        v = reg[(t - cells) * 1024 + j];
    }
    x[(size_t)y * 1024 + j] = v;
}

// The row's 1,024 squares in f64, so the mean rounds as the host reference's f64 sum does.
static __device__ __forceinline__ double h3d_sum_sq(double v, double* part) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    double total = 0.0;
    for (int w = 0; w < 8; ++w) total += part[w];
    __syncthreads();
    return total;
}

// RMSNorm, one row of 1,024 a block of 256: y = bf16((x * rsqrt(mean(x^2) + eps)) * g), the products in fp32.
__global__ void __launch_bounds__(256) h3d_rms_norm(
        const bf16* x, const bf16* g, bf16* y, H3dShape p) {
    __shared__ double part[8];
    const unsigned row = blockIdx.x;
    const unsigned t = threadIdx.x;
    const bf16* xr = x + (size_t)row * 1024;
    float v[4];
    double sq = 0.0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        v[i] = f(xr[t + 256 * i]);
        sq += (double)v[i] * (double)v[i];
    }
    const float inv = inv_sqrt((float)(h3d_sum_sq(sq, part) / 1024.0) + p.eps);
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) y[(size_t)row * 1024 + t + 256 * i] = __float2bfloat16(v[i] * inv * f(g[t + 256 * i]));
}

// norm_out: LayerNorm in fp32 (weight and bias), then bf16 as proj_out's input. One row a block of 256.
__global__ void __launch_bounds__(256) h3d_layer_norm(
        const bf16* x, const bf16* g, const bf16* b, bf16* y, H3dShape p) {
    __shared__ float sums[8];
    __shared__ float squares[8];
    const unsigned row = blockIdx.x;
    const unsigned t = threadIdx.x;
    const bf16* xr = x + (size_t)row * 1024;
    float v[4];
    float s = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        v[i] = f(xr[t + 256 * i]);
        s += v[i];
    }
    const float mean = tf_common::block_sum_serial(s, sums) / 1024.0f;
    float sq = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) sq += (v[i] - mean) * (v[i] - mean);
    const float rstd = inv_sqrt(tf_common::block_sum_serial(sq, squares) / 1024.0f + p.eps);
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned c = t + 256 * i;
        y[(size_t)row * 1024 + c] = __float2bfloat16((v[i] - mean) * rstd * f(g[c]) + f(b[c]));
    }
}

#define H3D_TILE 128
#define H3D_STEP 16

// y [m, n] = x [m, k] . w [n, k]^T fp32: each output summed in increasing K by unfused, individually rounded fp32 ops.
__global__ void __launch_bounds__(256) h3d_linear(
        const bf16* x, const bf16* w, float* y, H3dShape p) {
    // One float of padding a row spreads the K-major stores across banks.
    __shared__ float xs[H3D_STEP][H3D_TILE + 1];
    __shared__ float ws[H3D_STEP][H3D_TILE + 1];
    const unsigned tx = threadIdx.x % 16;
    const unsigned ty = threadIdx.x / 16;
    const unsigned row0 = blockIdx.y * H3D_TILE;
    const unsigned col0 = blockIdx.x * H3D_TILE;
    float acc[8][8] = {};
    for (unsigned k0 = 0; k0 < p.k; k0 += H3D_STEP) {
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            const unsigned e = threadIdx.x + 256 * j;
            const unsigned r = e / H3D_STEP;
            const unsigned kk = e % H3D_STEP;
            const bool in_k = k0 + kk < p.k;
            xs[kk][r] = in_k && row0 + r < p.m ? f(x[(size_t)(row0 + r) * p.k + k0 + kk]) : 0.0f;
            ws[kk][r] = in_k && col0 + r < p.n ? f(w[(size_t)(col0 + r) * p.k + k0 + kk]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (unsigned kk = 0; kk < H3D_STEP; ++kk) {
            float a[8], b[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) a[i] = xs[kk][ty + 16 * i];
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) b[j] = ws[kk][tx + 16 * j];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
#pragma unroll
                for (unsigned j = 0; j < 8; ++j) acc[i][j] = __fadd_rn(acc[i][j], __fmul_rn(a[i], b[j]));
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned row = row0 + ty + 16 * i;
        if (row >= p.m) continue;
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            const unsigned col = col0 + tx + 16 * j;
            if (col < p.n) y[(size_t)row * p.n + col] = acc[i][j];
        }
    }
}

// y = bf16(acc + b); mode 1 LayerScale y = bf16(y + bf16(bf16(acc + b) * s)) at the Metal epilogue's rounding.
__global__ void __launch_bounds__(256) h3d_finish(
        const float* acc, const bf16* b, const bf16* s, bf16* y, H3dShape p) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.n || row >= p.m) return;
    const size_t at = (size_t)row * p.n + x;
    const float out = bf(acc[at] + f(b[x]));
    if (p.mode == 0) {
        y[at] = __float2bfloat16(out);
    } else {
        y[at] = __float2bfloat16(f(y[at]) + bf(out * f(s[x])));
    }
}

// Per row and head: q and k RMS-normed (no weight) to bf16, their first 32 dims turned half-style in bf16; v copied.
__global__ void __launch_bounds__(32) h3d_heads(
        const bf16* qkv, const bf16* cs, const bf16* sn,
        bf16* q, bf16* k, bf16* v, H3dShape p) {
    const unsigned row = blockIdx.x;
    const unsigned head = blockIdx.y;
    const unsigned lane = threadIdx.x;
    const unsigned tile = row / p.tokens;
    const unsigned t = row % p.tokens;
    const bf16* src = qkv + (size_t)row * 3072 + head * 192;
    const size_t to = (((size_t)tile * 16 + head) * p.tokens + t) * 64;
    const float c = f(cs[t * 32 + lane]);
    const float s = f(sn[t * 32 + lane]);
#pragma unroll
    for (unsigned part = 0; part < 2; ++part) {
        const float a = f(src[part * 64 + lane]);
        const float z = f(src[part * 64 + 32 + lane]);
        const float inv = inv_sqrt(tf_common::warp_sum(a * a + z * z) / 64.0f + p.eps);
        const float na = bf(a * inv);
        const float other = __shfl_xor_sync(0xffffffffu, na, 16);
        const float turned = lane < 16 ? -other : other;
        bf16* out = (part == 0 ? q : k) + to;
        out[lane] = __float2bfloat16(bf(na * c) + bf(turned * s));
        out[32 + lane] = __float2bfloat16(z * inv);
    }
    v[to + lane] = src[128 + lane];
    v[to + 32 + lane] = src[160 + lane];
}

// softmax(q k^T / 8) v, a warp a query over its tile's keys; fp32 scores, softmax and sums. Grid [T / 8, 16, tiles].
__global__ void __launch_bounds__(256) h3d_attention(
        const bf16* q, const bf16* k, const bf16* v, bf16* o, H3dShape p) {
    __shared__ float qs[8 * 64];
    __shared__ float ps[8 * 32 * H3D_SLOTS];
    const unsigned lane = threadIdx.x & 31;
    const unsigned sg = threadIdx.x >> 5;
    const unsigned T = p.tokens;
    const unsigned query = blockIdx.x * 8 + sg;
    if (query >= T) return;
    const unsigned head = blockIdx.y;
    const unsigned tile = blockIdx.z;
    const size_t base = ((size_t)tile * 16 + head) * T * 64;
    float* qr = qs + sg * 64;
    float* pr = ps + sg * 32 * H3D_SLOTS;
    qr[lane] = f(q[base + (size_t)query * 64 + lane]);
    qr[lane + 32] = f(q[base + (size_t)query * 64 + 32 + lane]);
    __syncwarp();
    float sc[H3D_SLOTS];
    float top = -INFINITY;
#pragma unroll
    for (unsigned i = 0; i < H3D_SLOTS; ++i) {
        const unsigned key = lane + 32 * i;
        sc[i] = -INFINITY;
        if (key < T) {
            const bf16* kr = k + base + (size_t)key * 64;
            float dot = 0.0f;
            for (unsigned d = 0; d < 64; ++d) dot += qr[d] * f(kr[d]);
            sc[i] = dot * p.scale;
            top = fmaxf(top, sc[i]);
        }
    }
    top = tf_common::warp_max(top);
    float total = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < H3D_SLOTS; ++i) {
        const unsigned key = lane + 32 * i;
        if (key < T) {
            const float e = expf(sc[i] - top);
            pr[key] = e;
            total += e;
        }
    }
    total = tf_common::warp_sum(total);
    __syncwarp();
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (unsigned key = 0; key < T; ++key) {
        const float pk = pr[key];
        const bf16* vr = v + base + (size_t)key * 64 + 2 * lane;
        a0 += pk * f(vr[0]);
        a1 += pk * f(vr[1]);
    }
    const size_t at = ((size_t)tile * T + query) * 1024 + head * 64 + 2 * lane;
    o[at] = __float2bfloat16(a0 / total);
    o[at + 1] = __float2bfloat16(a1 / total);
}

// act = bf16(bf16(SiLU(gate)) * value), w1's rows read as [gate 4,096 | value 4,096]. Threads [4,096, m].
__global__ void __launch_bounds__(256) h3d_swiglu(const bf16* wide, bf16* act, H3dShape p) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 4096 || y >= p.m) return;
    const bf16* r = wide + (size_t)y * 8192;
    const float g = f(r[x]);
    const float silu = bf(g / (1.0f + expf(-g)));
    act[(size_t)y * 4096 + x] = __float2bfloat16(silu * f(r[4096 + x]));
}

} // extern "C"
