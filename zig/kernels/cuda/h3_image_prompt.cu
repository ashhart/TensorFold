// MiniMax-H3-Image's prompt stage on CUDA: bf16 in, fp32 products, the norms' squares in f64, bf16 out.
#include <cuda_bf16.h>
#include <math.h>
#include <stddef.h>

typedef __nv_bfloat16 bf16;

static __device__ float f(bf16 v) { return __bfloat162float(v); }
static __device__ bf16 h(float v) { return __float2bfloat16_rn(v); }

// exp(x) one rounding from exact as the reference computes it: expf is up to 2 ULPs off.
static __device__ float exact_exp(float x) { return (float)exp((double)x); }

// One RMSNorm value: mode 0 bf16(w * bf16(x * r)) as Qwen's text, mode 1 bf16(x * r * w) as H3; r f64, the weight fp32.
static __device__ float h3p_normed(float x, double r, float w, int mode) {
    const float p = (float)((double)x * r);
    return mode == 0 ? __fmul_rn(w, f(h(p))) : __fmul_rn(p, w);
}

// y = RMSNorm(x) by w: squares summed in f64 in the host's order, so the inverse root is the reference's bits.
extern "C" __global__ void __launch_bounds__(256) h3p_norm(
        const bf16* x, const float* w, bf16* y, int width, int mode, float eps) {
    __shared__ double r;
    const int row = blockIdx.x;
    const bf16* xr = x + (size_t)row * width;
    bf16* yr = y + (size_t)row * width;
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int i = 0; i < width; ++i) {
            const double v = f(xr[i]);
            s += v * v;
        }
        r = 1.0 / sqrt(s / (double)width + (double)eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += 256) yr[i] = h(h3p_normed(f(xr[i]), r, w[i], mode));
}

// q then k heads of 128 normed in place (mode as h3p_norm) and turned by cos, sin [rows, 64]; a warp a head, in order.
extern "C" __global__ void __launch_bounds__(256) h3p_heads(
        bf16* q, bf16* k, const float* wq, const float* wk, const float* cs, const float* sn,
        int heads, int kv_heads, int q_row, int k_row, int head, int mode, int rotary, float eps) {
    const int row = blockIdx.x;
    const int hh = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (hh >= heads + kv_heads) return;
    const int lane = threadIdx.x & 31;
    const bool is_q = hh < heads;
    bf16* x = is_q ? q + (size_t)row * q_row + (size_t)hh * head
                   : k + (size_t)row * k_row + (size_t)(hh - heads) * head;
    const float* w = is_q ? wq : wk;
    double ss = 0.0;
    for (int i = 0; i < 128; ++i) {
        const double v = f(x[i]);
        ss += v * v;
    }
    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = f(x[lane * 4 + i]);
    __syncwarp();
    const double r = 1.0 / sqrt(ss / 128.0 + (double)eps);
    for (int i = 0; i < 4; ++i) v[i] = h3p_normed(v[i], r, w[lane * 4 + i], mode);
    if (rotary) {
        // round as the host stores the norm; channel c pairs with c +- 64 (16 lanes away); products round first.
        for (int i = 0; i < 4; ++i) v[i] = f(h(v[i]));
        const float sign = lane < 16 ? -1.0f : 1.0f;
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], 16);
            const size_t at = (size_t)row * 64 + (lane & 15) * 4 + i;
            v[i] = __fadd_rn(f(h(__fmul_rn(v[i], cs[at]))), f(h(__fmul_rn(sign * o, sn[at]))));
        }
    }
    for (int i = 0; i < 4; ++i) x[lane * 4 + i] = h(v[i]);
}

// One key's score: q . k over 128 in order, times scale, as the host reference's.
static __device__ float h3p_score(const bf16* q, const bf16* k, float scale) {
    float s = 0.0f;
    for (int i = 0; i < 128; ++i) s = __fadd_rn(s, __fmul_rn(f(q[i]), f(k[i])));
    return __fmul_rn(s, scale);
}

// o = softmax(q k^T / sqrt(128)) v in bf16, a warp a row's head: the top score, then the host's in-order sums.
extern "C" __global__ void __launch_bounds__(256) h3p_attention(
        const bf16* q, const bf16* k, const bf16* v, bf16* o,
        int rows, int heads, int kv_heads, int q_row, int k_row, int v_row, int head,
        int causal, float scale) {
    const int row = blockIdx.x;
    const int hh = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (hh >= heads) return;
    const int lane = threadIdx.x & 31;
    const int kv = hh / (heads / kv_heads);
    const bf16* qp = q + (size_t)row * q_row + (size_t)hh * head;
    const bf16* kp = k + (size_t)kv * head;
    const bf16* vp = v + (size_t)kv * head + lane * 4;
    const int end = causal ? row + 1 : rows;
    float top = -INFINITY;
    for (int j = lane; j < end; j += 32) top = fmaxf(top, h3p_score(qp, kp + (size_t)j * k_row, scale));
    for (int k_ = 16; k_; k_ >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, k_));
    float l = 0.0f;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int j0 = 0; j0 < end; j0 += 32) {
        // the warp's next 32 keys' weights, one a lane, then read back in key order
        const int j = j0 + lane;
        const float mine = j < end ? exact_exp(__fadd_rn(h3p_score(qp, kp + (size_t)j * k_row, scale), -top)) : 0.0f;
        for (int u = 0; u < 32 && j0 + u < end; ++u) {
            const float e = __shfl_sync(0xffffffffu, mine, u);
            l = __fadd_rn(l, e);
            const bf16* vr = vp + (size_t)(j0 + u) * v_row;
            for (int i = 0; i < 4; ++i) acc[i] = __fadd_rn(acc[i], __fmul_rn(e, f(vr[i])));
        }
    }
    bf16* out = o + (size_t)row * heads * 128 + (size_t)hh * 128 + lane * 4;
    for (int i = 0; i < 4; ++i) out[i] = h(__fdiv_rn(acc[i], l));
}

// o = silu(g) * v bf16: mode 0 bf16(bf16(g / e) v), mode 1 bf16(bf16(g * bf16(1 / e)) v) as H3; strides may differ.
extern "C" __global__ void __launch_bounds__(256) h3p_glu(
        const bf16* g, const bf16* v, bf16* o, int width, int g_row, int v_row, int mode) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int row = blockIdx.y;
    if (c >= width) return;
    const float x = f(g[(size_t)row * g_row + c]);
    const float e = 1.0f + exact_exp(-x);
    const float s = mode == 0 ? f(h(__fdiv_rn(x, e))) : f(h(__fmul_rn(x, f(h(__fdiv_rn(1.0f, e))))));
    o[(size_t)row * width + c] = h(__fmul_rn(s, f(v[(size_t)row * v_row + c])));
}


// x = bf16(x + bf16(y)): round the fp32 product before the residual add.
extern "C" __global__ void __launch_bounds__(256) h3p_add(bf16* x, const float* y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] = h(__fadd_rn(f(x[i]), f(h(y[i]))));
}

// x rows idx[i] = bf16(x + y[i]): DeepStack's features added at the image pads, y bf16.
extern "C" __global__ void __launch_bounds__(256) h3p_scatter_add(
        bf16* x, const bf16* y, const unsigned* idx, int width) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int i = blockIdx.y;
    if (c >= width) return;
    const size_t to = (size_t)idx[i] * width + c;
    x[to] = h(__fadd_rn(f(x[to]), f(y[(size_t)i * width + c])));
}

// The embedding gather: x rows = table's rows `ids`, bf16 copied as stored.
extern "C" __global__ void __launch_bounds__(256) h3p_embed(
        const bf16* table, const unsigned* ids, bf16* x, int width) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int row = blockIdx.y;
    if (c >= width) return;
    x[(size_t)row * width + c] = table[(size_t)ids[row] * width + c];
}

// Vision rows spliced into the stream: x rows pads[i] = rows[i], whole bf16 rows.
extern "C" __global__ void __launch_bounds__(256) h3p_splice(
        bf16* x, const bf16* rows_, const unsigned* pads, int width) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int i = blockIdx.y;
    if (c >= width) return;
    x[(size_t)pads[i] * width + c] = rows_[(size_t)i * width + c];
}

// y [rows, n] fp32 = x [rows, k] . w [n, k]^T bf16: K summed sequentially, its bf16 rounding the reference's own.
extern "C" __global__ void __launch_bounds__(256) h3p_linear(
        const bf16* x, const bf16* w, float* y, int rows, int n, int k) {
    const int at = blockIdx.x * 256 + threadIdx.x;
    if (at >= rows * n) return;
    const int row = at / n;
    const int col = at % n;
    const bf16* xr = x + (size_t)row * k;
    const bf16* wr = w + (size_t)col * k;
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum = __fadd_rn(sum, __fmul_rn(f(xr[i]), f(wr[i])));
    y[at] = sum;
}

// y [n] = bf16(x [n]): a product's fp32 rows at their one rounding to bf16, the reference's every output.
extern "C" __global__ void __launch_bounds__(256) h3p_round(const float* x, bf16* y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = h(x[i]);
}
