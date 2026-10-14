// GLM-5.3-Flash on CUDA: the row steps between the linears, ours (embedding, RMSNorm, the hyper-connections, the
// residuals, SwiGLU, the router and its top-8, the experts' combine, the KDA conv window). Each row is one block's,
// with fp32 sums in a fixed order (threads strided over the row, then a fixed tree), so a row's bits depend on its own
// inputs alone, never on how many rows the window holds.

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// the block's sum of v (blockDim.x a multiple of 32, at most 1024): xor tree in each warp, then the warps in order
__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const int warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[warp] = v;
    __syncthreads();
    float s = red[0];
    for (int i = 1; i < nw; ++i) s += red[i];
    return s;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

}  // namespace

// out[r, c * D + d] = table[ids[r], d] for each of ``copies`` streams (bf16 rows, D a multiple of 8)
extern "C" __global__ void glm_embed(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
                                     __nv_bfloat16* __restrict__ out, int D, int copies) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int r = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(table + (long long)ids[r] * D);
    uint4* dst = reinterpret_cast<uint4*>(out + (long long)r * copies * D);
    for (int i = threadIdx.x; i < D / 8; i += blockDim.x) {
        const uint4 v = src[i];
        for (int c = 0; c < copies; ++c) dst[c * (D / 8) + i] = v;
    }
}

// RMSNorm (Llama: bf16(w * bf16(x * rinv))) of row r of x (stride xs_) -> out (stride os_), and the 64-groups' fp32
// sums of the outputs when xs is given. D <= 4096.
extern "C" __global__ void glm_rmsnorm(const __nv_bfloat16* __restrict__ x, long long x_stride, const __nv_bfloat16* __restrict__ w,
                                       __nv_bfloat16* __restrict__ out, long long o_stride, float* __restrict__ xs, float eps, int D) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    __shared__ float red[32];
    __shared__ float ys[4096];
    const int r = blockIdx.x;
    const __nv_bfloat16* xr = x + r * x_stride;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float v = __bfloat162float(xr[d]);
        ss = fmaf(v, v, ss);
    }
    const float rinv = 1.0f / sqrtf(block_sum(ss, red) / (float)D + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float y = bf(__bfloat162float(w[d]) * bf(__bfloat162float(xr[d]) * rinv));
        out[r * o_stride + d] = __float2bfloat16_rn(y);
        ys[d] = y;
    }
    if (xs == nullptr) return;
    __syncthreads();
    for (int g = threadIdx.x; g < D / 64; g += blockDim.x) {
        float s = 0.0f;
        for (int i = 0; i < 64; ++i) s += ys[g * 64 + i];
        xs[(long long)r * (D / 64) + g] = s;
    }
}

// hc_pre (hyper-connections, 4 streams of D = 4096), in two launches. The partials: row r's 24 mixing dots of the
// streams [16384] with fn [24, 16384] and their sum of squares over 16 fixed slices of 1024 inputs, one block a
// (row, slice) -> part [R, 16, 25]. 128 threads.
extern "C" __global__ void __launch_bounds__(128) glm_hc_part(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ fn,
        float* __restrict__ part) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int D = 4096, WIDE = 4 * D, M = 24, SL = 1024;
    __shared__ float red[32];
    const int r = blockIdx.x, sl = blockIdx.y, tid = threadIdx.x;
    const int k = sl * SL + tid * 8;   // each thread 8 consecutive inputs of the slice
    const uint4 xv = *reinterpret_cast<const uint4*>(x + (long long)r * WIDE + k);
    const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xv);
    float xf[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(xh[j]);
        xf[2 * j] = f.x;
        xf[2 * j + 1] = f.y;
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) ss = fmaf(xf[j], xf[j], ss);
    float* out = part + ((long long)r * 16 + sl) * (M + 1);
#pragma unroll 4
    for (int m = 0; m < M; ++m) {
        const uint4 wv = *reinterpret_cast<const uint4*>(fn + (long long)m * WIDE + k);
        const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&wv);
        float p = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(wh[j]);
            p = fmaf(f.x, xf[2 * j], p);
            p = fmaf(f.y, xf[2 * j + 1], p);
        }
        const float sm = block_sum(p, red);
        if (tid == 0) out[m] = sm;
    }
    const float sm = block_sum(ss, red);
    if (tid == 0) out[M] = sm;
}

// The finish: the slices' partials in slice order, the pre / post / comb weights (comb: ``iters`` Sinkhorn steps), the
// collapsed row RMSNormed by nw -> out [D] and its 64-groups' sums xs; post [4] and comb [4, 4] kept for hc_post.
// 512 threads.
extern "C" __global__ void __launch_bounds__(512) glm_hc_finish(const __nv_bfloat16* __restrict__ x, const float* __restrict__ part,
        const float* __restrict__ base, const float* __restrict__ scale, const __nv_bfloat16* __restrict__ nw,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* __restrict__ post, float* __restrict__ comb,
        float eps, float hc_eps, int iters) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int D = 4096, WIDE = 4 * D, M = 24;
    __shared__ float red[32];
    __shared__ float pre[4];
    __shared__ float ys[D];
    const int r = blockIdx.x, tid = threadIdx.x;
    const __nv_bfloat16* xr = x + (long long)r * WIDE;
    __shared__ float mixs[M + 1];
    if (tid <= M) { // each sum: the slices' partials in slice order
        float a = 0.0f;
        for (int sl = 0; sl < 16; ++sl) a += part[((long long)r * 16 + sl) * (M + 1) + tid];
        mixs[tid] = a;
    }
    __syncthreads();
    if (tid < 32) { // warp 0: lane i * 4 + j holds comb[i][j]; lanes 16 .. 23 the pre and post weights
        const int lane = tid;
        const float rinv = 1.0f / sqrtf(mixs[M] / (float)WIDE + eps);
        const float s_pre = scale[0], s_post = scale[1], s_comb = scale[2];
        if (lane >= 16 && lane < 20) pre[lane - 16] = sigmoid(mixs[lane - 16] * rinv * s_pre + base[lane - 16]) + hc_eps;
        if (lane >= 20 && lane < 24) post[r * 4 + lane - 20] = 2.0f * sigmoid(mixs[4 + lane - 20] * rinv * s_post + base[4 + lane - 20]);
        const int q = lane & 15;
        const float l = mixs[8 + q] * rinv * s_comb + base[8 + q];
        // row softmax over j (lanes of one i: xor 1, 2), then Sinkhorn: columns (xor 4, 8) and rows in turn
        float mx = fmaxf(l, __shfl_xor_sync(0xffffffffu, l, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float ex = expf(l - mx);
        float tot = ex + __shfl_xor_sync(0xffffffffu, ex, 1);
        tot = tot + __shfl_xor_sync(0xffffffffu, tot, 2);
        float c = ex / tot + hc_eps;
        auto col = [&](float v) {
            float t = v + __shfl_xor_sync(0xffffffffu, v, 4);
            t = t + __shfl_xor_sync(0xffffffffu, t, 8);
            return v / (t + hc_eps);
        };
        auto row = [&](float v) {
            float t = v + __shfl_xor_sync(0xffffffffu, v, 1);
            t = t + __shfl_xor_sync(0xffffffffu, t, 2);
            return v / (t + hc_eps);
        };
        c = col(c);
        for (int it = 0; it < iters - 1; ++it) c = col(row(c));
        if (lane < 16) comb[r * 16 + lane] = c;
    }
    __syncthreads();
    const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
    float cs = 0.0f;
    for (int d = tid; d < D; d += 512) {
        const float c = bf(fmaf(p3, __bfloat162float(xr[3 * D + d]), fmaf(p2, __bfloat162float(xr[2 * D + d]),
                           fmaf(p1, __bfloat162float(xr[D + d]), p0 * __bfloat162float(xr[d])))));
        ys[d] = c;
        cs = fmaf(c, c, cs);
    }
    const float rinv2 = 1.0f / sqrtf(block_sum(cs, red) / (float)D + eps);
    for (int d = tid; d < D; d += 512) {
        const float y = bf(__bfloat162float(nw[d]) * bf(ys[d] * rinv2));
        out[(long long)r * D + d] = __float2bfloat16_rn(y);
        ys[d] = y;
    }
    __syncthreads();
    for (int g = tid; g < D / 64; g += 512) {
        float s = 0.0f;
        for (int i = 0; i < 64; ++i) s += ys[g * 64 + i];
        xs[(long long)r * (D / 64) + g] = s;
    }
}

// hc_post: the branch (both ranks' fp32 partials g [2, R, D], rank 0 first, rounded once) into each of the 4 streams:
// x_s = bf16(post_s * branch + sum_i comb[i][s] x_i), x in place. grid (R, D / 1024), 256 threads.
extern "C" __global__ void glm_hc_post(__nv_bfloat16* __restrict__ x, const float* __restrict__ g, const float* __restrict__ post,
                                       const float* __restrict__ comb, int R) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int D = 4096;
    const int r = blockIdx.x;
    const float* cb = comb + r * 16;
    for (int d = blockIdx.y * 1024 + threadIdx.x; d < (blockIdx.y + 1) * 1024; d += blockDim.x) {
        const float branch = bf(g[(long long)r * D + d] + g[(long long)R * D + (long long)r * D + d]);
        __nv_bfloat16* xr = x + (long long)r * 4 * D;
        const float x0 = __bfloat162float(xr[d]), x1 = __bfloat162float(xr[D + d]), x2 = __bfloat162float(xr[2 * D + d]),
                    x3 = __bfloat162float(xr[3 * D + d]);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float mixed = fmaf(cb[12 + s], x3, fmaf(cb[8 + s], x2, fmaf(cb[4 + s], x1, cb[s] * x0)));
            xr[s * D + d] = __float2bfloat16_rn(fmaf(post[r * 4 + s], branch, mixed));
        }
    }
}

// the plain residual (the MTP block): x = bf16(x + bf16(g rank 0 + g rank 1)), x [R, D] in place
extern "C" __global__ void glm_residual_add(__nv_bfloat16* __restrict__ x, const float* __restrict__ g, int R) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int D = 4096;
    const int r = blockIdx.x;
    for (int d = blockIdx.y * 1024 + threadIdx.x; d < (blockIdx.y + 1) * 1024; d += blockDim.x) {
        const long long i = (long long)r * D + d;
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + bf(g[i] + g[(long long)R * D + i]));
    }
}

// the final collapse: bf16((x0 + x1 + x2 + x3) / 4) of the 4 streams
extern "C" __global__ void glm_stream_mean(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int D = 4096;
    const int r = blockIdx.x;
    for (int d = blockIdx.y * 1024 + threadIdx.x; d < (blockIdx.y + 1) * 1024; d += blockDim.x) {
        const __nv_bfloat16* xr = x + (long long)r * 4 * D;
        const float a = ((__bfloat162float(xr[d]) + __bfloat162float(xr[D + d])) + __bfloat162float(xr[2 * D + d])) +
                        __bfloat162float(xr[3 * D + d]);
        out[(long long)r * D + d] = __float2bfloat16_rn(a / 4.0f);
    }
}

// SwiGLU over [gate | up] (bf16, width W each): bf16(bf16(silu(min(g, L))) * clip(u, -L, L)) and its 64-groups' sums.
// grid (R, W / 512), 256 threads.
extern "C" __global__ void glm_swiglu(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out, float* __restrict__ xs,
                                      float limit, int W) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    __shared__ float ys[512];
    const int r = blockIdx.x, c0 = blockIdx.y * 512;
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        const int d = c0 + i;
        const float g = fminf(__bfloat162float(gu[(long long)r * 2 * W + d]), limit);
        const float u = fminf(fmaxf(__bfloat162float(gu[(long long)r * 2 * W + W + d]), -limit), limit);
        const float a = bf(bf(g / (1.0f + expf(-g))) * u);
        out[(long long)r * W + d] = __float2bfloat16_rn(a);
        ys[i] = a;
    }
    __syncthreads();
    if (threadIdx.x < 8) {
        float s = 0.0f;
        for (int i = 0; i < 64; ++i) s += ys[threadIdx.x * 64 + i];
        xs[(long long)r * (W / 64) + blockIdx.y * 8 + threadIdx.x] = s;
    }
}

// router logits: out[m, e] = fp32 x[m] . w[e] (bf16 [NE, D]). A block takes 2 experts, a warp a quarter of D (D a
// multiple of 1024): its lanes over 8-input pieces in order, a xor tree, then the quarters summed in order: each row's
// sum is its own. grid (NE / 2, ceil(M / 16)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) glm_router(const __nv_bfloat16* __restrict__ x, long long x_stride,
        const __nv_bfloat16* __restrict__ w, float* __restrict__ out, int M, int NE, int D) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    __shared__ float part[8][16];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e = blockIdx.x * 2 + (warp >> 2), q = warp & 3;
    const int m0 = blockIdx.y * 16;
    const int len = D >> 2;
    float acc[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) acc[m] = 0.0f;
    if (e < NE) {
        const __nv_bfloat16* wr = w + (long long)e * D + q * len;
        const __nv_bfloat16* xq = x + q * len;
        for (int k = lane * 8; k < len; k += 256) {
            const uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
            const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&wv);
            float wf[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __bfloat1622float2(wh[j]);
                wf[2 * j] = f.x;
                wf[2 * j + 1] = f.y;
            }
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                if (m0 + m < M) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(xq + (long long)(m0 + m) * x_stride + k);
                    const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xv);
                    float p = acc[m];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __bfloat1622float2(xh[j]);
                        p = fmaf(wf[2 * j], f.x, p);
                        p = fmaf(wf[2 * j + 1], f.y, p);
                    }
                    acc[m] = p;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        float a = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0) part[warp][m] = a;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int m = threadIdx.x & 15, h = threadIdx.x >> 4, ee = blockIdx.x * 2 + h;
        if (ee < NE && m0 + m < M)
            out[(long long)(m0 + m) * NE + ee] = ((part[4 * h][m] + part[4 * h + 1][m]) + part[4 * h + 2][m]) + part[4 * h + 3][m];
    }
}

// each row's 8 experts by sigmoid score plus bias (the lower id on ties), weights the picks' scores normalized in pick
// order times ``scale``, then the shared expert (id NE, weight 1) in slot 8. A warp a row (NE <= 320).
extern "C" __global__ void glm_topk(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ pick,
                                    float* __restrict__ wts, float scale, int NE) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int PER = 10, TOPK = 8, SLOTS = 9;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    float score[PER], choice[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = i * 32 + lane;
        if (e < NE) {
            score[i] = sigmoid(logits[(long long)r * NE + e]);
            choice[i] = score[i] + bias[e];
        } else {
            score[i] = 0.0f;
            choice[i] = -INFINITY;
        }
    }
    float total = 0.0f, wk[TOPK];
    int ids[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float best = -INFINITY;
        int at = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (choice[i] > best) {
                best = choice[i];
                at = i * 32 + lane;
            }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, off);
            const int oa = __shfl_xor_sync(0xffffffffu, at, off);
            if (ob > best || (ob == best && oa < at)) {
                best = ob;
                at = oa;
            }
        }
        float sk = 0.0f;
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (i * 32 + lane == at) {
                sk = score[i];
                choice[i] = -INFINITY;
            }
        sk = __shfl_sync(0xffffffffu, sk, at & 31);
        ids[k] = at;
        wk[k] = sk;
        total += sk;
    }
    if (lane == 0) {
        for (int k = 0; k < TOPK; ++k) {
            pick[r * SLOTS + k] = ids[k];
            wts[r * SLOTS + k] = wk[k] / (total + 1e-20f) * scale;
        }
        pick[r * SLOTS + TOPK] = NE;
        wts[r * SLOTS + TOPK] = 1.0f;
    }
}

// a rank's fp32 MoE share: sum over the 9 slots in order of w_k y_k (y bf16 in a prompt, fp32 in a window)
extern "C" __global__ void glm_combine(const void* __restrict__ y, int y_f32, const float* __restrict__ wts, float* __restrict__ out) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    constexpr int D = 4096, SLOTS = 9;
    const int r = blockIdx.x;
    for (int d = blockIdx.y * 1024 + threadIdx.x; d < (blockIdx.y + 1) * 1024; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < SLOTS; ++k) {
            const long long i = ((long long)r * SLOTS + k) * D + d;
            const float v = y_f32 ? reinterpret_cast<const float*>(y)[i] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(y)[i]);
            acc = fmaf(v, wts[r * SLOTS + k], acc);
        }
        out[(long long)r * D + d] = acc;
    }
}

// the KDA conv window: conv [layers, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows]
// (proj [layers, R, >= C], strides in elements). grid (layers, ceil(C / 1024)), 256 threads.
extern "C" __global__ void glm_conv_shift(__nv_bfloat16* __restrict__ conv, const __nv_bfloat16* __restrict__ proj, int keep,
                                          long long conv_layer, long long proj_layer, long long proj_row, int C) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const long long l = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < min(C, (int)(blockIdx.y + 1) * 1024); c += blockDim.x) {
        __nv_bfloat16 v[3];
#pragma unroll
        for (int t = 0; t < 3; ++t) {
            const int src = keep + t;
            v[t] = src < 3 ? conv[l * conv_layer + src * C + c] : proj[l * proj_layer + (src - 3) * proj_row + c];
        }
#pragma unroll
        for (int t = 0; t < 3; ++t) conv[l * conv_layer + t * C + c] = v[t];
    }
}

// rows whose position is 0 zeroed (the MTP head's first row reads no token before it): x [R, D] bf16
extern "C" __global__ void glm_zero_first(__nv_bfloat16* __restrict__ x, const int* __restrict__ pos, int D) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int r = blockIdx.x;
    if (pos[r] != 0) return;
    for (int d = threadIdx.x; d < D; d += blockDim.x) x[(long long)r * D + d] = __float2bfloat16_rn(0.0f);
}

// dst row i = src row idx[i] (D bf16 each): each stream's last row of a batched step, for its head
extern "C" __global__ void glm_gather_rows(const __nv_bfloat16* __restrict__ src, const int* __restrict__ idx,
                                           __nv_bfloat16* __restrict__ dst, int D) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const int i = blockIdx.x;
    const uint4* s = reinterpret_cast<const uint4*>(src + (long long)idx[i] * D);
    uint4* o = reinterpret_cast<uint4*>(dst + (long long)i * D);
    for (int d = threadIdx.x; d < D / 8; d += blockDim.x) o[d] = s[d];
}
