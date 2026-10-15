// Nemotron 3.5 Bolt's vision tower (C-RADIO ViT-H/16) and projector: glue around cuBLASLt's bf16 GEMMs, and attention.
// The residual stream stays fp32 (its rows grow to |x| ~1700); norms, softmax and epilogues compute in fp32, GEMM
// inputs are rounded to bf16 once. Oracle: tools/zig/bolt_vision_reference.py (fp32 PyTorch).
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// fp32 block sum over blockDim.x threads (a multiple of 32, at most 1024); every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float* part) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < warps; ++i) t += part[i];
    return t;
}

// x [prefix + n, D] fp32: rows below `prefix` the cls/register tokens, then acc (the patch Linear) plus the CPE
// position rows, bilinear from the 128x128 table: (p0 w0 + p1 w1) h0 + (p2 w0 + p3 w1) h1 as aten's interpolate.
extern "C" __global__ void __launch_bounds__(256) tf_rv_embed(const float* __restrict__ acc,
                                                              const __nv_bfloat16* __restrict__ pos,
                                                              const int* __restrict__ idx,
                                                              const float* __restrict__ wts,
                                                              const __nv_bfloat16* __restrict__ cls,
                                                              float* __restrict__ x, int prefix, int D) {
    const long long row = blockIdx.x;
    if (row < prefix) {
        for (int i = threadIdx.x; i < D; i += blockDim.x) x[row * D + i] = bf(cls[row * D + i]);
        return;
    }
    const long long p = row - prefix;
    const int* ri = idx + p * 4;
    const float* rw = wts + p * 4;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float a = bf(pos[(long long)ri[0] * D + i]) * rw[0] + bf(pos[(long long)ri[1] * D + i]) * rw[1];
        const float b = bf(pos[(long long)ri[2] * D + i]) * rw[0] + bf(pos[(long long)ri[3] * D + i]) * rw[1];
        x[row * D + i] = acc[p * D + i] + (a * rw[2] + b * rw[3]);
    }
}

// LayerNorm of fp32 rows (mean, then the biased variance about it), bf16 out for the next GEMM; a block a row.
extern "C" __global__ void __launch_bounds__(256) tf_rv_layernorm(const float* __restrict__ x,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  const __nv_bfloat16* __restrict__ b,
                                                                  __nv_bfloat16* __restrict__ y, int D, float eps) {
    __shared__ float part[32];
    const long long row = blockIdx.x;
    const float* xr = x + row * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, part) / D;
    float v = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float r = rsqrtf(block_sum(v, part) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        y[row * D + i] = __float2bfloat16_rn((xr[i] - mean) * r * bf(w[i]) + bf(b[i]));
}

// acc [rows, 3D] + bias split into q, k, v [heads][rows][hd] bf16 (the reference's view(T, 3, nh, hd)).
extern "C" __global__ void __launch_bounds__(256) tf_rv_qkv(const float* __restrict__ acc,
                                                            const __nv_bfloat16* __restrict__ bias,
                                                            __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
                                                            __nv_bfloat16* __restrict__ v, int rows, int heads, int hd) {
    const long long row = blockIdx.x;
    const int D = heads * hd;
    const float* a = acc + row * 3 * D;
    for (int c = threadIdx.x; c < 3 * D; c += blockDim.x) {
        const int s = c / D, h = (c % D) / hd, j = c % hd;
        const __nv_bfloat16 val = __float2bfloat16_rn(a[c] + bf(bias[c]));
        __nv_bfloat16* dst = s == 0 ? q : (s == 1 ? k : v);
        dst[((long long)h * rows + row) * hd + j] = val;
    }
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Full (bidirectional) attention over all `rows` tokens, heads of 80: flash style, one warp 16 queries, 64-key
// tiles in shared memory, bf16 mma.sync with fp32 scores, an online softmax in fp32 (exp2 of pre-scaled scores) and
// fp32 output accumulators; out [rows, heads * 80] bf16 (the reference's transpose(0, 1).reshape(T, C)).
#define RV_HD 80
#define RV_BK 64
extern "C" __global__ void __launch_bounds__(128) tf_rv_attention(const __nv_bfloat16* __restrict__ q,
                                                                  const __nv_bfloat16* __restrict__ k,
                                                                  const __nv_bfloat16* __restrict__ v,
                                                                  __nv_bfloat16* __restrict__ out, int rows, int heads,
                                                                  float scale_log2) {
    __shared__ __align__(16) __nv_bfloat16 ks[RV_BK][RV_HD + 8];
    __shared__ __align__(16) __nv_bfloat16 vt[RV_HD][RV_BK + 8];
    const int head = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const long long base = (long long)head * rows;
    const int q0 = blockIdx.x * 64 + warp * 16;
    uint32_t qa[RV_HD / 16][4];
    {
        const uint32_t* p0 = reinterpret_cast<const uint32_t*>(q + (base + min(q0 + g, rows - 1)) * RV_HD);
        const uint32_t* p1 = reinterpret_cast<const uint32_t*>(q + (base + min(q0 + g + 8, rows - 1)) * RV_HD);
#pragma unroll
        for (int kc = 0; kc < RV_HD / 16; ++kc) {
            qa[kc][0] = p0[kc * 8 + t];
            qa[kc][1] = p1[kc * 8 + t];
            qa[kc][2] = p0[kc * 8 + 4 + t];
            qa[kc][3] = p1[kc * 8 + 4 + t];
        }
    }
    float o[RV_HD / 8][4];
#pragma unroll
    for (int d = 0; d < RV_HD / 8; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    for (int k0 = 0; k0 < rows; k0 += RV_BK) {
        __syncthreads();
        for (int i = threadIdx.x; i < RV_BK * RV_HD / 2; i += blockDim.x) {
            const int r = i / (RV_HD / 2), c = i % (RV_HD / 2);
            uint32_t kv = 0, vv = 0;
            if (k0 + r < rows) {
                kv = reinterpret_cast<const uint32_t*>(k + (base + k0 + r) * RV_HD)[c];
                vv = reinterpret_cast<const uint32_t*>(v + (base + k0 + r) * RV_HD)[c];
            }
            *reinterpret_cast<uint32_t*>(&ks[r][2 * c]) = kv;
            const __nv_bfloat162 vh = *reinterpret_cast<__nv_bfloat162*>(&vv);
            vt[2 * c][r] = vh.x;
            vt[2 * c + 1][r] = vh.y;
        }
        __syncthreads();
        float s[RV_BK / 8][4];
#pragma unroll
        for (int nt = 0; nt < RV_BK / 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            const uint32_t* kr = reinterpret_cast<const uint32_t*>(&ks[nt * 8 + g][0]);
#pragma unroll
            for (int kc = 0; kc < RV_HD / 16; ++kc) mma16816(s[nt], qa[kc], kr[kc * 8 + t], kr[kc * 8 + 4 + t]);
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < RV_BK / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = k0 + nt * 8 + 2 * t + (e & 1);
                s[nt][e] = key < rows ? s[nt][e] * scale_log2 : -INFINITY;
            }
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
        m0 = n0;
        m1 = n1;
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < RV_BK / 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - n0);
            s[nt][1] = exp2f(s[nt][1] - n0);
            s[nt][2] = exp2f(s[nt][2] - n1);
            s[nt][3] = exp2f(s[nt][3] - n1);
            sum0 += s[nt][0] + s[nt][1];
            sum1 += s[nt][2] + s[nt][3];
        }
        l0 = l0 * c0 + sum0;
        l1 = l1 * c1 + sum1;
#pragma unroll
        for (int d = 0; d < RV_HD / 8; ++d) {
            o[d][0] *= c0;
            o[d][1] *= c0;
            o[d][2] *= c1;
            o[d][3] *= c1;
        }
#pragma unroll
        for (int kc = 0; kc < RV_BK / 16; ++kc) {
            uint32_t pa[4];
            pa[0] = pack2(s[2 * kc][0], s[2 * kc][1]);
            pa[1] = pack2(s[2 * kc][2], s[2 * kc][3]);
            pa[2] = pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
            pa[3] = pack2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
#pragma unroll
            for (int d = 0; d < RV_HD / 8; ++d) {
                const uint32_t* vr = reinterpret_cast<const uint32_t*>(&vt[d * 8 + g][0]);
                mma16816(o[d], pa, vr[kc * 8 + t], vr[kc * 8 + 4 + t]);
            }
        }
    }
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
    const int D = heads * RV_HD;
    const int r0 = q0 + g, r1 = q0 + g + 8;
#pragma unroll
    for (int d = 0; d < RV_HD / 8; ++d) {
        const int col = head * RV_HD + d * 8 + 2 * t;
        if (r0 < rows) *reinterpret_cast<uint32_t*>(out + (long long)r0 * D + col) = pack2(o[d][0] * i0, o[d][1] * i0);
        if (r1 < rows) *reinterpret_cast<uint32_t*>(out + (long long)r1 * D + col) = pack2(o[d][2] * i1, o[d][3] * i1);
    }
}

// y = gelu(acc + b), the exact erf form (nn.GELU), fp32, bf16 out.
extern "C" __global__ void __launch_bounds__(256) tf_rv_bias_gelu(const float* __restrict__ acc,
                                                                  const __nv_bfloat16* __restrict__ bias,
                                                                  __nv_bfloat16* __restrict__ y, int D) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float x = acc[row * D + i] + bf(bias[i]);
        y[row * D + i] = __float2bfloat16_rn(0.5f * x * (1.0f + erff(x * 0.7071067811865476f)));
    }
}

// x += acc + b, the residual in fp32.
extern "C" __global__ void __launch_bounds__(256) tf_rv_bias_residual(const float* __restrict__ acc,
                                                                      const __nv_bfloat16* __restrict__ bias,
                                                                      float* __restrict__ x, int D) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) x[row * D + i] += acc[row * D + i] + bf(bias[i]);
}

// Pixel shuffle v2 (2x2 patches to one row of 4D, channel (dy * 2 + dx) * D + c) of the patch rows behind `prefix`,
// then RMSNorm(4D, eps) with weight w in fp32; bf16 out. A block an output row (grid gh/2 x gw/2, row-major).
extern "C" __global__ void __launch_bounds__(256) tf_rv_shuffle_rms(const float* __restrict__ x,
                                                                    const __nv_bfloat16* __restrict__ w,
                                                                    __nv_bfloat16* __restrict__ y, int prefix, int gw,
                                                                    int D, float eps) {
    __shared__ float part[32];
    const int r = blockIdx.x;
    const int by = r / (gw / 2), bx = r % (gw / 2);
    const int W = 4 * D;
    float s = 0.0f;
    for (int ch = threadIdx.x; ch < W; ch += blockDim.x) {
        const int sub = ch / D, c = ch % D;
        const long long tok = prefix + (long long)(2 * by + sub / 2) * gw + 2 * bx + sub % 2;
        const float v = x[tok * D + c];
        s += v * v;
    }
    const float rs = rsqrtf(block_sum(s, part) / W + eps);
    for (int ch = threadIdx.x; ch < W; ch += blockDim.x) {
        const int sub = ch / D, c = ch % D;
        const long long tok = prefix + (long long)(2 * by + sub / 2) * gw + 2 * bx + sub % 2;
        y[(long long)r * W + ch] = __float2bfloat16_rn(x[tok * D + c] * rs * bf(w[ch]));
    }
}

// y = relu(acc)^2, bf16 out.
extern "C" __global__ void __launch_bounds__(256) tf_rv_relu2(const float* __restrict__ acc, __nv_bfloat16* __restrict__ y,
                                                              int D) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float a = fmaxf(acc[row * D + i], 0.0f);
        y[row * D + i] = __float2bfloat16_rn(a * a);
    }
}

// y = bf16(acc): the projector's last Linear.
extern "C" __global__ void __launch_bounds__(256) tf_rv_cast(const float* __restrict__ acc, __nv_bfloat16* __restrict__ y,
                                                             int D) {
    const long long row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) y[row * D + i] = __float2bfloat16_rn(acc[row * D + i]);
}
