// Drama backbone glue in fp32, rounded to bf16 where SGLang v0.5.21, FlashInfer 0.6.18 and FLA store or cast.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace drama {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 tobf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float round_bf(float v) { return bf(tobf(v)); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, m));
    return x;
}

// Block-wide sum for blocks of `nwarps` full warps; every thread gets the total.
__device__ __forceinline__ float block_sum(float x, float* scratch, int nwarps) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    float s = lane < nwarps ? scratch[lane] : 0.f;
    s = warp_sum(s);
    __syncthreads();
    return s;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = bf(b[i]);
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float (&v)[8]) {
    uint4 u;
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i] = tobf(v[i]);
    *reinterpret_cast<uint4*>(p) = u;
}

// drama_act_quant's e4m3 byte of v under its group's multiplier inv = 448 / amax.
__device__ __forceinline__ uint32_t e4m3(float v, float inv) {
    const float c = fminf(fmaxf(v * inv, -448.0f), 448.0f);
    return static_cast<uint32_t>(__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3));
}

}  // namespace drama

using namespace drama;

// Semantic rows (ids in [lo, hi]): bf16(bf16(out + extra) * 0.70710677), as embed_backbone; other rows keep out.
extern "C" __global__ void __launch_bounds__(256) drama_semantic_mix(const int* ids, __nv_bfloat16* out,
                                                                    const __nv_bfloat16* extra, int width,
                                                                    int lo, int hi) {
    const int r = blockIdx.x;
    const int id = ids[r];
    if (id < lo || id > hi) return;
    const float scale = 0.70710678118654752f;
    __nv_bfloat16* o = out + static_cast<size_t>(r) * width;
    const __nv_bfloat16* e = extra + static_cast<size_t>(r) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float s = round_bf(bf(o[i]) + bf(e[i]));
        o[i] = tobf(s * scale);
    }
}

// GemmaRMSNorm in FlashInfer's CuTe order with one bf16 residual store; q / s (both or neither) add its FP8 groups.
extern "C" __global__ void __launch_bounds__(256) drama_add_norm(const __nv_bfloat16* x, __nv_bfloat16* res,
                                                                const __nv_bfloat16* w, __nv_bfloat16* y,
                                                                uint8_t* q, float* s, int width, float eps,
                                                                int first) {
    __shared__ float square_sum;
    const size_t base = static_cast<size_t>(blockIdx.x) * width;
    const int t = threadIdx.x;
    if (t < 32) {
        // Unrolled so every load is in flight before the FMA chain, which keeps the serial reference order.
        float sum = 0.f;
#pragma unroll 8
        for (int block = 0; block < width; block += 256) {
            float value[8], other[8];
            load8(x + base + block + 8 * t, value);
            if (!first) load8(res + base + block + 8 * t, other);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                if (!first) value[v] += other[v];
                sum = fmaf(value[v], value[v], sum);
            }
        }
        for (int offset = 1; offset < 32; offset <<= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if (t == 0) square_sum = sum;
    }
    __syncthreads();
    const float rcp = rsqrtf(square_sum / static_cast<float>(width) + eps);
    for (int c = 8 * t; c < width; c += 8 * blockDim.x) {
        float value[8], other[8], weight[8], out[8];
        load8(x + base + c, value);
        if (!first) load8(res + base + c, other);
        load8(w + c, weight);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (!first) value[i] += other[i];
            out[i] = value[i] * rcp * (1.0f + weight[i]);
        }
        store8(res + base + c, value);
        store8(y + base + c, out);
        if (q == nullptr) continue;
        // A 128-wide group is 16 lanes; the max is exact in any order.
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = round_bf(out[i]);
            amax = fmaxf(amax, fabsf(out[i]));
        }
#pragma unroll
        for (int m = 8; m; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        amax = fmaxf(amax, 1e-10f);
        const float inv = __fdividef(448.0f, amax);
        uint2 packed = {0u, 0u};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed.x |= e4m3(out[i], inv) << (8 * i);
            packed.y |= e4m3(out[4 + i], inv) << (8 * i);
        }
        *reinterpret_cast<uint2*>(q + base + c) = packed;
        if ((t & 15) == 0) s[static_cast<size_t>(blockIdx.x) * (width / 128) + c / 128] = amax * (1.0f / 448.0f);
    }
}

// Value-head gated RMSNorm in SGLang FLA's full-extend order (8 values a lane, 2 heads a warp) for every row count.
extern "C" __global__ void __launch_bounds__(256) drama_gated_rms_128(
    const __nv_bfloat16* y, long long y_ld,
    const __nv_bfloat16* z, long long z_ld,
    const __nv_bfloat16* w, __nv_bfloat16* out, float eps) {
    const int sub = threadIdx.x & 15;
    const int hd = (blockIdx.y * blockDim.x + threadIdx.x) / 16;
    const long long row = blockIdx.x;
    float v[8], gate[8], weight[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = sub * 8 + i;
        v[i] = __bfloat162float(y[row * y_ld + hd * 128 + col]);
        gate[i] = __bfloat162float(z[row * z_ld + hd * 128 + col]);
        weight[i] = __bfloat162float(w[col]);
    }
    float sum = __fmul_rn(v[1], v[1]);
    sum = __fmaf_rn(v[0], v[0], sum);
#pragma unroll
    for (int i = 2; i < 8; ++i) sum = __fmaf_rn(v[i], v[i], sum);
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, mask));
    float variance;
    asm("div.full.f32 %0, %1, %2;" : "=f"(variance) : "f"(sum), "f"(128.0f));
    variance = __fadd_rn(variance, eps);
    float inv;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(variance));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float exponent = __fmul_rn(__fsub_rn(0.0f, gate[i]), __int_as_float(0x3fb8aa3b));
        float e;
        asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(exponent));
        const float denom = __fadd_rn(e, 1.0f);
        float sigmoid;
        asm("div.full.f32 %0, %1, %2;" : "=f"(sigmoid) : "f"(1.0f), "f"(denom));
        const float norm = __fmul_rn(__fmul_rn(v[i], inv), weight[i]);
        const float result = __fmul_rn(norm, __fmul_rn(gate[i], sigmoid));
        out[(row * 32 + hd) * 128 + sub * 8 + i] = __float2bfloat16_rn(result);
    }
}

// Causal depthwise conv (kernel 4) and silu as causal_conv1d_update, fp32 products; state: the 3 previous inputs.
extern "C" __global__ void __launch_bounds__(256) drama_gdn_conv(const __nv_bfloat16* in, int ld_in,
                                                                const __nv_bfloat16* w, __nv_bfloat16* state,
                                                                __nv_bfloat16* out, int rows, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = bf(w[4 * c]), w1 = bf(w[4 * c + 1]), w2 = bf(w[4 * c + 2]), w3 = bf(w[4 * c + 3]);
    __nv_bfloat16 h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
    for (int t = 0; t < rows; ++t) {
        const __nv_bfloat16 xt = in[static_cast<size_t>(t) * ld_in + c];
        float acc = w0 * bf(h0);
        acc += w1 * bf(h1);
        acc += w2 * bf(h2);
        acc += w3 * bf(xt);
        out[static_cast<size_t>(t) * C + c] = tobf(silu(acc));
        h0 = h1;
        h1 = h2;
        h2 = xt;
    }
    state[c] = h0;
    state[C + c] = h1;
    state[2 * C + c] = h2;
}

// Prefill conv as the Triton kernel: bf16 products, fp32 accumulation, ex2.approx / div.full activation.
extern "C" __global__ void __launch_bounds__(256) drama_gdn_conv_prefill(
    const __nv_bfloat16* in, int ld_in, const __nv_bfloat16* w, __nv_bfloat16* state,
    __nv_bfloat16* out, int rows, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = bf(w[4 * c]), w1 = bf(w[4 * c + 1]), w2 = bf(w[4 * c + 2]), w3 = bf(w[4 * c + 3]);
    __nv_bfloat16 h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
    for (int t = 0; t < rows; ++t) {
        const __nv_bfloat16 xt = in[static_cast<size_t>(t) * ld_in + c];
        float acc = 0.0f + bf(tobf(w0 * bf(h0)));
        acc += bf(tobf(w1 * bf(h1)));
        acc += bf(tobf(w2 * bf(h2)));
        acc += bf(tobf(w3 * bf(xt)));
        const float exponent = __fmul_rn(__fsub_rn(0.0f, acc), __int_as_float(0x3fb8aa3b));
        float exponential;
        asm("ex2.approx.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
        const float denominator = __fadd_rn(1.0f, exponential);
        float y;
        asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(acc), "f"(denominator));
        out[static_cast<size_t>(t) * C + c] = tobf(y);
        h0 = h1;
        h1 = h2;
        h2 = xt;
    }
    state[c] = h0;
    state[C + c] = h1;
    state[2 * C + c] = h2;
}

// Packed gated-delta decode as FLA's fused_recurrent.py, q/k normalized in fp32; state fp32 [NV][128 v][128 k].
extern "C" __global__ void __launch_bounds__(128) drama_gdn_recur(const __nv_bfloat16* qkv, int ld_qkv,
                                                                 const __nv_bfloat16* ba, int ld_ba,
                                                                 const float* a_log, const float* dt_bias,
                                                                 float* state, __nv_bfloat16* y, int rows, int nk,
                                                                 int nv, float scale) {
    constexpr int DK = 128, DV = 128, R = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y;
    const int col0 = (blockIdx.x * 4 + warp) * R;
    const int kh = head / (nv / nk);
    float* s0 = state + static_cast<size_t>(head) * DV * DK;
    float s[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float4 v = *reinterpret_cast<const float4*>(s0 + static_cast<size_t>(col0 + r) * DK + lane * 4);
        s[r][0] = v.x;
        s[r][1] = v.y;
        s[r][2] = v.z;
        s[r][3] = v.w;
    }
    const float A = expf(a_log[head]);
    const float bias = dt_bias[head];
    for (int t = 0; t < rows; ++t) {
        const __nv_bfloat16* row = qkv + static_cast<size_t>(t) * ld_qkv;
        float q[4], k[4], v[R];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = bf(row[kh * DK + lane * 4 + i]);
            k[i] = bf(row[nk * DK + kh * DK + lane * 4 + i]);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = bf(row[2 * nk * DK + head * DV + col0 + r]);
        const float bb = bf(ba[static_cast<size_t>(t) * ld_ba + head]);
        const float aa = bf(ba[static_cast<size_t>(t) * ld_ba + nv + head]);
        const float x = aa + bias;
        const float sp = x <= 20.0f ? logf(1.0f + expf(x)) : x;
        const float decay = expf(-A * sp);
        const float beta = round_bf(1.0f / (1.0f + expf(-bb)));
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            qq += q[i] * q[i];
            kk += k[i] * k[i];
        }
        qq = warp_sum(qq);
        kk = warp_sum(kk);
        const float qn = sqrtf(qq + 1e-6f), kn = sqrtf(kk + 1e-6f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = (q[i] / qn) * scale;
            k[i] = k[i] / kn;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float mem = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[r][i] *= decay;
                mem += s[r][i] * k[i];
            }
            mem = warp_sum(mem);
            const float delta = (v[r] - mem) * beta;
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[r][i] += k[i] * delta;
                o += s[r][i] * q[i];
            }
            o = warp_sum(o);
            if (lane == r) y[(static_cast<size_t>(t) * nv + head) * DV + col0 + r] = tobf(o);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        *reinterpret_cast<float4*>(s0 + static_cast<size_t>(col0 + r) * DK + lane * 4) =
            make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
}

#include "drama_gdn_chunk.cuh"

#include "drama_backbone_attention.cuh"

// Router: softmax over 256 experts, top 8 (lower id on ties), renormalized to sum 1, in descending probability.
extern "C" __global__ void __launch_bounds__(256) drama_topk(const __nv_bfloat16* logits, int ld, int* ids, float* wts,
                                                            int top) {
    __shared__ float scratch[32];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ float picked_p[8];
    __shared__ int picked_i[8];
    const int r = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float x = bf(logits[static_cast<size_t>(r) * ld + e]);
    float mx = warp_max(x);
    if (lane == 0) scratch[warp] = mx;
    __syncthreads();
    mx = lane < 8 ? scratch[lane] : -INFINITY;
    mx = warp_max(mx);
    __syncthreads();
    const float ex = expf(x - mx);
    const float sum = block_sum(ex, scratch, 8);
    float p = ex / sum;
    bool taken = false;
    for (int k = 0; k < top; ++k) {
        float cv = taken ? -1.0f : p;
        int ci = e;
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, cv, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ci, m);
            if (ov > cv || (ov == cv && oi < ci)) {
                cv = ov;
                ci = oi;
            }
        }
        if (lane == 0) {
            bv[warp] = cv;
            bi[warp] = ci;
        }
        __syncthreads();
        if (e == 0) {
            float best = bv[0];
            int at = bi[0];
            for (int w = 1; w < 8; ++w)
                if (bv[w] > best || (bv[w] == best && bi[w] < at)) {
                    best = bv[w];
                    at = bi[w];
                }
            picked_p[k] = best;
            picked_i[k] = at;
        }
        __syncthreads();
        if (e == picked_i[k]) taken = true;
    }
    if (e == 0) {
        float s = 0.f;
        for (int k = 0; k < top; ++k) s += picked_p[k];
        for (int k = 0; k < top; ++k) {
            ids[r * top + k] = picked_i[k];
            wts[r * top + k] = picked_p[k] / s;
        }
    }
}

// Group-128 FP8 quantization; the pinned JIT's fast-math 448 / amax multiplier stays local, with __fdividef.
extern "C" __global__ void __launch_bounds__(128) drama_act_quant(const __nv_bfloat16* x, int ld_x, uint8_t* q,
                                                                 float* s, int K) {
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.y * 4 + warp;
    if (group * 128 >= K) return;
    const __nv_bfloat16* xr = x + static_cast<size_t>(r) * ld_x + group * 128 + lane * 4;
    float v[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = bf(xr[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(warp_max(amax), 1e-10f);
    const float sc = amax * (1.0f / 448.0f);
    const float inv = __fdividef(448.0f, amax);
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) packed |= e4m3(v[i], inv) << (8 * i);
    *reinterpret_cast<uint32_t*>(q + static_cast<size_t>(r) * K + group * 128 + lane * 4) = packed;
    if (lane == 0) s[static_cast<size_t>(r) * (K / 128) + group] = sc;
}

// SwiGLU in one launch: rows below routed FP8-quantized into hq / hs, the shared expert's rows stored bf16 to h_out.
extern "C" __global__ void __launch_bounds__(128) drama_silu_mul(const __nv_bfloat16* gu, int ld_gu, int routed,
                                                                uint8_t* hq, float* hs, const __nv_bfloat16* sgu,
                                                                int ld_sgu, __nv_bfloat16* h_out, int I) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.y * 4 + warp;
    if (group * 128 >= I) return;
    const bool quant = blockIdx.x < routed;
    const int r = quant ? blockIdx.x : blockIdx.x - routed;
    const int c0 = group * 128 + lane * 4;
    const __nv_bfloat16* row = quant ? gu + static_cast<size_t>(r) * ld_gu : sgu + static_cast<size_t>(r) * ld_sgu;
    float h[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = round_bf(silu(bf(row[c0 + i])) * bf(row[I + c0 + i]));
        amax = fmaxf(amax, fabsf(h[i]));
    }
    if (!quant) {
#pragma unroll
        for (int i = 0; i < 4; ++i) h_out[static_cast<size_t>(r) * I + c0 + i] = tobf(h[i]);
        return;
    }
    amax = fmaxf(warp_max(amax), 1e-10f);
    const float sc = amax * (1.0f / 448.0f);
    const float inv = __fdividef(448.0f, amax);
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) packed |= e4m3(h[i], inv) << (8 * i);
    *reinterpret_cast<uint32_t*>(hq + static_cast<size_t>(r) * I + c0) = packed;
    if (lane == 0) hs[static_cast<size_t>(r) * (I / 128) + group] = sc;
}

// MoE output: bf16 routed sum (moe_sum_reduce), then sigmoid(x . wg) * shared + routed (fused_gate_sigmoid_mul_add).
extern "C" __global__ void __launch_bounds__(256) drama_moe_combine(const __nv_bfloat16* down, int top,
                                                                   const __nv_bfloat16* x, const __nv_bfloat16* wg,
                                                                   const __nv_bfloat16* shared, __nv_bfloat16* out,
                                                                   int width) {
    __shared__ float scratch[32];
    const size_t r = blockIdx.x;
    float part = 0.f;
    for (int i = threadIdx.x; i < width; i += 256) part += bf(x[r * width + i]) * bf(wg[i]);
    const float gate = sigmoid(block_sum(part, scratch, 8));
    for (int i = threadIdx.x; i < width; i += 256) {
        float acc = 0.f;
        for (int s = 0; s < top; ++s) acc += bf(down[(r * top + s) * width + i]);
        const float routed = round_bf(acc);
        out[r * width + i] = tobf(gate * bf(shared[r * width + i]) + routed);
    }
}
