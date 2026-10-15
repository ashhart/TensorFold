// GLM KDA from python-0.6 glm5_next/cuda/kda.cu plus a segmented chain; HF definition, update shared with replay.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// One delta-rule step on this thread's 4 x 4 block of the state: per-channel decay, read (k), correct toward v.
__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float (&gg)[4],
                                       const float* vrow, int warp, float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * gg[i];
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

__device__ __forceinline__ void chain_body(int h,
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ state_in, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ state_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    const int C = 3 * H * DK;                         // conv channels: q | k | v
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV], gs[DK];
    __shared__ float beta_s, rinv;
    int c = -1;
    if (t < 3 * DK) c = (t / DK) * H * DK + h * DK + (t % DK);
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    const float decay_rate = expf(a_log[h]);
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
                acc = acc + __bfloat162float(cw[(size_t)c * TAPS + tap]) * x;
            }
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        } else if (t >= 512 && t < 512 + DK) {
            const int i = t - 512;
            const float a = __bfloat162float(A[(size_t)r * a_stride + h * DK + i]) + dt_bias[h * DK + i];
            gs[i] = expf(lower * sigmoidf_(decay_rate * a));
        } else if (t == 1023) {
            beta_s = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = v4[i] * inv;
                if (warp == 0) y = y * (1.0f / sqrtf((float)DK));
                x[lane * 4 + i] = y;
            }
        }
        __syncthreads();
        const float beta = beta_s;
        float kk[4], qq[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; gg[i] = gs[lane * 4 + i]; }
        update(s, kk, gg, vs, warp, beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        if (k_save != nullptr) {
            const size_t base = ((size_t)r * H + h) * DK;
            if (t < DK) { k_save[base + t] = ks[t]; g_save[base + t] = gs[t]; }
            if (t >= DK && t < DK + DV) v_save[base + t - DK] = __float2bfloat16_rn(vs[t - DK]);
            if (t == 0) b_save[r * H + h] = beta;
        }
        __syncthreads();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float y = ys[lane * 4 + i]; ss = ss + y * y; }
            ss = warp_sum(ss);
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        __syncthreads();
        if (t < DV) {
            const float yn = ys[t] * rinv;
            const float yw = __bfloat162float(norm_w[t]) * yn;
            const float gate = __bfloat162float(G[(size_t)r * g_stride + h * DV + t]);
            out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
        }
        __syncthreads();
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

__global__ void __launch_bounds__(1024) chain_kernel(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ state_in, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ state_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    chain_body(blockIdx.x, H, P, p_stride, b_off, A, a_stride, G, g_stride, cs, cw, state_in, a_log, dt_bias, norm_w, eps,
               lower, rows, out, state_out, k_save, v_save, g_save, b_save);
}

}  // namespace

// One stream's window in a batched step: rows from ``first``, states, conv, replay inputs and projections (layer 0).
struct KdaSeg {
    int first, rows;
    const float* rec_in;
    float* rec_out;
    const __nv_bfloat16* conv;
    float* k_save;
    __nv_bfloat16* v_save;
    float* g_save;
    float* b_save;
    __nv_bfloat16* proj;
};

// chain_kernel for every stream of a batched step at ``li``: grid (H, streams), projection rows kept for the shift.
extern "C" __global__ void __launch_bounds__(1024) glm_kda_chain_seg(const KdaSeg* __restrict__ segs, int li,
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, __nv_bfloat16* __restrict__ out,
        int max_rows, int proj_w) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // a dependent matmul may start on its weights
    const KdaSeg sg = segs[blockIdx.y];
    const int C = 3 * H * DK;
    const size_t rec = (size_t)li * H * DV * DK, conv = (size_t)li * 3 * C;
    const size_t kv = (size_t)li * max_rows * H * DK, bb = (size_t)li * max_rows * H;
    const __nv_bfloat16* p = P + (size_t)sg.first * p_stride;
    if (blockIdx.x == 0) {   // the window's conv channels, for the commit (one block a stream copies them)
        __nv_bfloat16* dst = sg.proj + (size_t)li * max_rows * proj_w;
        for (int i = threadIdx.x; i < sg.rows * C; i += blockDim.x) {
            const int r = i / C, c = i % C;
            dst[(size_t)r * proj_w + c] = p[(size_t)r * p_stride + c];
        }
    }
    chain_body(blockIdx.x, H, p, p_stride, b_off, A + (size_t)sg.first * a_stride, a_stride, G + (size_t)sg.first * g_stride,
               g_stride, sg.conv + conv, cw, sg.rec_in + rec, a_log, dt_bias, norm_w, eps, lower, sg.rows,
               out + (size_t)sg.first * H * DV, sg.rec_out + rec, sg.k_save + kv, sg.v_save + kv, sg.g_save + kv, sg.b_save + bb);
}

namespace {

__global__ void __launch_bounds__(1024) replay_kernel(
        int H, const float* __restrict__ state_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save,
        const float* __restrict__ b_save, int rows, float* __restrict__ state_out) {
    const int h = blockIdx.x;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(v_save[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = k_save[base + lane * 4 + i]; gg[i] = g_save[base + lane * 4 + i]; }
        update(s, kk, gg, vs, warp, b_save[r * H + h]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

// All layers at once: block (layer, head); per-layer strides of the state buffers and saved rows.
__global__ void __launch_bounds__(1024) replay_layers_kernel(
        int H, const float* __restrict__ state_in, size_t state_stride, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        size_t kv_stride, size_t b_stride, int rows, float* __restrict__ state_out) {
    const int layer = blockIdx.x / H, h = blockIdx.x % H;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const float* sin = state_in + layer * state_stride;
    float* sout = state_out + layer * state_stride;
    const float* ks = k_save + layer * kv_stride;
    const __nv_bfloat16* vsv = v_save + layer * kv_stride;
    const float* gsv = g_save + layer * kv_stride;
    const float* bsv = b_save + layer * b_stride;
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = sin[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(vsv[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[base + lane * 4 + i]; gg[i] = gsv[base + lane * 4 + i]; }
        update(s, kk, gg, vs, warp, bsv[r * H + h]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) sout[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

// Long windows: prep_kernel and out_kernel for all rows, the delta rule row by row (step_kernel); chain_kernel's bits.

// Block (row, head), 512 threads: conv + SiLU of the 3 x 128 q|k|v channels, the decay, beta, then the q/k norms.
__global__ void __launch_bounds__(512) prep_kernel(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ cs,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        float lower, float* __restrict__ q_out, float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save,
        float* __restrict__ g_save, float* __restrict__ b_save) {
    const int C = 3 * H * DK;
    const int r = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK];
    const size_t base = ((size_t)r * H + h) * DK;
    if (t < 3 * DK) {
        const int c = (t / DK) * H * DK + h * DK + (t % DK);
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < TAPS; ++tap) {
            const int at = r + tap;
            const float x = at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + c])
                                          : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
            acc = acc + __bfloat162float(cw[(size_t)c * TAPS + tap]) * x;
        }
        const float act = bf(acc / (1.0f + expf(-acc)));
        if (t < DK) qs[t] = act;
        else if (t < 2 * DK) ks[t - DK] = act;
        else v_save[base + t - 2 * DK] = __float2bfloat16_rn(act);
    } else {
        const int i = t - 3 * DK;
        const float decay_rate = expf(a_log[h]);
        const float a = __bfloat162float(A[(size_t)r * a_stride + h * DK + i]) + dt_bias[h * DK + i];
        g_save[base + i] = expf(lower * sigmoidf_(decay_rate * a));
        if (i == 0) b_save[r * H + h] = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
    }
    __syncthreads();
    if (warp < 2) {
        const float* x = warp == 0 ? qs : ks;
        float v4[4], ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
        ss = warp_sum(ss);
        float inv = 1.0f / sqrtf(ss + 1e-6f);
        float* dst = warp == 0 ? q_out : k_save;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float y = v4[i] * inv;
            if (warp == 0) y = y * (1.0f / sqrtf((float)DK));
            dst[base + lane * 4 + i] = y;
        }
    }
}

// Block (head, part): a warp owns 4 value rows; TR rows of saved k, q, decay, v and beta staged in shared at a time.
template <int WARPS, int TR>
__global__ void __launch_bounds__(WARPS * 32) step_kernel(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    constexpr int NT = WARPS * 32, VR = WARPS * 4;
    const int h = blockIdx.x, v0 = blockIdx.y * VR;
    const int lw = threadIdx.x >> 5, lane = threadIdx.x & 31, warp = blockIdx.y * WARPS + lw;
    __shared__ __align__(16) float kt[TR][DK];
    __shared__ __align__(16) float qt[TR][DK];
    __shared__ __align__(16) float gt[TR][DK];
    __shared__ __align__(16) float vt[TR][VR];
    __shared__ float bt[TR];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r0 = 0; r0 < rows; r0 += TR) {
        const int n = rows - r0 < TR ? rows - r0 : TR;
        __syncthreads();
        for (int x = threadIdx.x; x < n * DK; x += NT) {
            const int rr = x / DK, c = x % DK;
            const size_t base = ((size_t)(r0 + rr) * H + h) * DK + c;
            kt[rr][c] = k_save[base];
            qt[rr][c] = q_in[base];
            gt[rr][c] = g_save[base];
        }
        for (int x = threadIdx.x; x < n * VR; x += NT) {
            const int rr = x / VR, c = x % VR;
            vt[rr][c] = __bfloat162float(v_save[((size_t)(r0 + rr) * H + h) * DV + v0 + c]);
        }
        if (threadIdx.x < n) bt[threadIdx.x] = b_save[(r0 + threadIdx.x) * H + h];
        __syncthreads();
        for (int rr = 0; rr < n; ++rr) {
            const float4 k4 = reinterpret_cast<const float4*>(kt[rr])[lane];      // one 16-byte load a lane:
            const float4 q4 = reinterpret_cast<const float4*>(qt[rr])[lane];      // no bank conflicts
            const float4 g4 = reinterpret_cast<const float4*>(gt[rr])[lane];
            const float4 v4 = reinterpret_cast<const float4*>(vt[rr])[lw];
            float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
            float gg[4] = {g4.x, g4.y, g4.z, g4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
            update(s, kk, gg, vv, 0, bt[rr]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                o = warp_sum(o);
                if (lane == 0) y_out[((size_t)(r0 + rr) * H + h) * DV + warp * 4 + j] = __float2bfloat16_rn(o);
            }
        }
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

// Block (row, head), 128 threads: the gated RMSNorm of the read-out.
__global__ void __launch_bounds__(128) out_kernel(
        int H, const __nv_bfloat16* __restrict__ y_in, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float rinv;
    const size_t base = ((size_t)r * H + h) * DV;
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float y = __bfloat162float(y_in[base + lane * 4 + i]); ss = ss + y * y; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = __bfloat162float(y_in[base + t]) * rinv;
    const float yw = __bfloat162float(norm_w[t]) * yn;
    const float gate = __bfloat162float(G[(size_t)r * g_stride + h * DV + t]);
    out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
}

}  // namespace

// the Python build's instantiations
namespace {
template __global__ void step_kernel<(int)4, (int)16>(int, const float *, const float *, const float *, const __nv_bfloat16 *, const float *, const float *, int, __nv_bfloat16 *, float *);
}  // namespace
