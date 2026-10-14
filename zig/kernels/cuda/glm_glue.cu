// GLM glue kernels: bit-exact ports of the Triton kernels in families/glm5_next/cuda/glue.py (PTX-ordered).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

__device__ __forceinline__ float f32of(u16 v) { return __uint_as_float((u32)v << 16); }

__device__ __forceinline__ u16 bf16_rn(float v) { u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v)); return r; }

__device__ __forceinline__ float divf(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float rcpf(float x) {
    float r;
    asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float sqrtf_ftz(float x) {
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float ex2f(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

#define NEG_LOG2E __int_as_float((int)0xBFB8AA3Bu)
#define POS_LOG2E __int_as_float(0x3FB8AA3Bu)
#define TINY1E20 __int_as_float(0x1E3CE508u)

__device__ __forceinline__ float sigmoid(float x) { return divf(1.0f, 1.0f + ex2f(x * NEG_LOG2E)); }

__device__ __forceinline__ void load2bf(const u16* p, float* v) {
    u32 w = *(const u32*)p;
    v[0] = f32of((u16)w);
    v[1] = f32of((u16)(w >> 16));
}

__device__ __forceinline__ void load4bf(const u16* p, float* v) {
    u32 w0 = *(const u32*)p;
    u32 w1 = *(const u32*)(p + 2);
    v[0] = f32of((u16)w0);
    v[1] = f32of((u16)(w0 >> 16));
    v[2] = f32of((u16)w1);
    v[3] = f32of((u16)(w1 >> 16));
}

__device__ __forceinline__ void store4bf(u16* p, const u16* v) {
    *(u32*)p = (u32)v[0] | ((u32)v[1] << 16);
    *(u32*)(p + 2) = (u32)v[2] | ((u32)v[3] << 16);
}

// -- embedding (glue.py _embed_b16) ------------------------------------------------------------
__device__ __forceinline__ void embed_b16(const u16* ids, const u16* w, u16* out, int copies) {
    const int d = gridDim.y * 64;
    int r = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    u32 tok = ((const u32*)ids)[r];
    u32 v = *(const u32*)(w + (u64)tok * d + g * 64 + 2 * t);
    u16* dst = out + (u64)r * copies * d + g * 64;
    for (int c = 0; c < copies; ++c) ((u32*)(dst + (u64)c * d))[t] = v;
}

// -- RMSNorm (glue.py _rmsnorm, D power of two, 256 threads) ------------------------------------
template <int D>
__device__ __forceinline__ void rmsnorm(const u16* x, i64 xs, const u16* w, u16* out, i64 os, float* sums, float eps) {
    __shared__ float warp_sums[8];
    __shared__ float gs[64];
    int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const u16* xr = x + (i64)r * xs;
    float a[16];
    load4bf(xr + 8 * t, a);
    load4bf(xr + 8 * t + 4, a + 4);
    load4bf(xr + 2048 + 8 * t, a + 8);
    load4bf(xr + 2048 + 8 * t + 4, a + 12);
    float p02 = fmaf(a[0], a[0], a[2] * a[2]), p13 = fmaf(a[1], a[1], a[3] * a[3]);
    float p46 = fmaf(a[4], a[4], a[6] * a[6]), p57 = fmaf(a[5], a[5], a[7] * a[7]);
    float p8a = fmaf(a[8], a[8], a[10] * a[10]), p9b = fmaf(a[9], a[9], a[11] * a[11]);
    float pca = fmaf(a[12], a[12], a[14] * a[14]), pdb = fmaf(a[13], a[13], a[15] * a[15]);
    float ssum = (((p02 + p46) + (p8a + pca)) + ((p13 + p57) + (p9b + pdb)));
    for (int i = 16; i; i >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, i);
    if (lane == 0) warp_sums[warp] = ssum;
    __syncthreads();
    float v0 = warp_sums[0], v1 = warp_sums[1], v2 = warp_sums[2], v3 = warp_sums[3];
    float v4 = warp_sums[4], v5 = warp_sums[5], v6 = warp_sums[6], v7 = warp_sums[7];
    float total = ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7));
    float rinv = divf(1.0f, sqrtf_ftz(divf(total, (float)D) + eps));
    float ya[8], yb[8];
    u16 yva[8], yvb[8];
    const u16* wr = w + 8 * t;
    for (int i = 0; i < 8; ++i) {
        float wa = f32of(wr[i]), wb = f32of(wr[i + 2048]);
        yva[i] = bf16_rn(wa * f32of(bf16_rn(rinv * a[i])));
        yvb[i] = bf16_rn(wb * f32of(bf16_rn(rinv * a[i + 8])));
        ya[i] = f32of(yva[i]);
        yb[i] = f32of(yvb[i]);
    }
    store4bf(out + (i64)r * os + 8 * t, yva);
    store4bf(out + (i64)r * os + 8 * t + 4, yva + 4);
    store4bf(out + (i64)r * os + 2048 + 8 * t, yvb);
    store4bf(out + (i64)r * os + 2048 + 8 * t + 4, yvb + 4);
    if (sums) {
        float pa = ((ya[0] + ya[2]) + (ya[4] + ya[6])) + ((ya[1] + ya[3]) + (ya[5] + ya[7]));
        float pb = ((yb[0] + yb[2]) + (yb[4] + yb[6])) + ((yb[1] + yb[3]) + (yb[5] + yb[7]));
        for (int i = 4; i; i >>= 1) pa += __shfl_xor_sync(0xffffffffu, pa, i);
        for (int i = 4; i; i >>= 1) pb += __shfl_xor_sync(0xffffffffu, pb, i);
        if ((t & 7) == 0) {
            gs[t >> 3] = pa;
            gs[32 + (t >> 3)] = pb;
        }
        __syncthreads();
        if (t < 64) sums[r * (D / 64) + t] = gs[t];
    }
}

// -- hyper-connection partial dots (glue.py _hc_partial; WIDE x NB blocks of 1024) ---------------
template <int WIDE, int NB>
__device__ __forceinline__ void hc_partial(const u16* x, const float* fn, float* part) {
    __shared__ float rows[32];
    __shared__ float ssw[4];
    int r = blockIdx.x, b = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const u16* xr = x + (i64)r * WIDE;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float ss = 0.0f;
    for (int t = 0; t < 8; ++t) {
        int base = b * 1024 + t * 128;
        float xq[4];
        load2bf(xr + base + 4 * l, xq);
        load2bf(xr + base + 4 * l + 2, xq + 2);
        float xo = f32of(xr[base + threadIdx.x]);
        ss += xo * xo;
        for (int j = 0; j < 8; ++j) {
            int m = 4 * j + w;
            float wv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (m < 24) {
                const float* wp = fn + (i64)m * WIDE + base + 4 * l;
                wv[0] = wp[0];
                wv[1] = wp[1];
                wv[2] = wp[2];
                wv[3] = wp[3];
            }
            float p0 = xq[0] * wv[0], p1 = xq[1] * wv[1];
            float p2 = xq[2] * wv[2], p3 = xq[3] * wv[3];
            float q = (p0 + p2) + (p1 + p3);
            for (int i = 16; i; i >>= 1) q += __shfl_xor_sync(0xffffffffu, q, i);
            acc[j] += q;
        }
    }
    for (int i = 16; i; i >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, i);
    if (l == 0) {
        for (int j = 0; j < 8; ++j) rows[w * 8 + j] = acc[j];
        ssw[w] = ss;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        part[((i64)r * NB + b) * 32 + 24] = (ssw[0] + ssw[2]) + (ssw[1] + ssw[3]);
        for (int m = 25; m < 32; ++m) part[((i64)r * NB + b) * 32 + m] = 0.0f;
        for (int m = 0; m < 24; ++m) part[((i64)r * NB + b) * 32 + m] = rows[(m & 3) * 8 + (m >> 2)];
    }
}

// -- hyper-connection finish (glue.py _hc_finish; S = 4 streams) ---------------------------------
template <int D>
__device__ __forceinline__ void hc_finish(const u16* x, const float* part, const float* base, const float* scale, const u16* nw,
                          u16* out, float* xs, float* post, float* comb, float eps_norm, float hc_eps, int iters) {
    __shared__ float warp_sums[8];
    __shared__ float gs[64];
    int r = blockIdx.x, t = threadIdx.x, m = t & 31;
    float mix = 0.0f, ss = 0.0f;
    for (int b = 0; b < 16; ++b) mix += part[((i64)r * 16 + b) * 32 + m];
    for (int b = 0; b < 16; ++b) ss += part[((i64)r * 16 + b) * 32 + 24];
    float rinv = divf(1.0f, sqrtf_ftz(divf(ss, (float)(4 * D)) + eps_norm));
    mix *= rinv;
    float base_m = (m < 24) ? base[m] : 0.0f;
    float lpre = fmaf(mix, scale[0], base_m);
    float lpost = fmaf(mix, scale[1], base_m);
    float lcomb = fmaf(mix, scale[2], base_m);
    float pre = 0.0f, combv = 0.0f;
    if (m < 4) pre = sigmoid(lpre) + hc_eps;
    else if (m < 8) pre = sigmoid(lpost) + sigmoid(lpost);
    if (m >= 8 && m < 24) {
        const unsigned SM = 0x00ffff00u;
        int rr = (m - 8) >> 2, cc = (m - 8) & 3;
        float cmax = lcomb;
        cmax = fmaxf(cmax, __shfl_xor_sync(SM, cmax, 2));
        cmax = fmaxf(cmax, __shfl_xor_sync(SM, cmax, 1));
        float ce = ex2f((lcomb - cmax) * POS_LOG2E);
        // Row/col sums are (a+c)+(b+d) butterflies; divisions lower to fmaf(rcp(rs),ce,eps) then rcp(denom)*ce.
        float rs = (ce + __shfl_xor_sync(SM, ce, 2)) + (__shfl_xor_sync(SM, ce, 1) + __shfl_xor_sync(SM, ce, 3));
        ce = fmaf(rcpf(rs), ce, hc_eps);
        float b1 = __shfl_sync(SM, ce, 8 + ((rr ^ 1) << 2) + cc);
        float b2 = __shfl_sync(SM, ce, 8 + ((rr ^ 2) << 2) + cc);
        float b3 = __shfl_sync(SM, ce, 8 + ((rr ^ 3) << 2) + cc);
        ce = rcpf(((ce + b2) + (b1 + b3)) + hc_eps) * ce;
        for (int it = 1; it < iters; ++it) {
            rs = (ce + __shfl_xor_sync(SM, ce, 2)) + (__shfl_xor_sync(SM, ce, 1) + __shfl_xor_sync(SM, ce, 3));
            ce = rcpf(rs + hc_eps) * ce;
            b1 = __shfl_sync(SM, ce, 8 + ((rr ^ 1) << 2) + cc);
            b2 = __shfl_sync(SM, ce, 8 + ((rr ^ 2) << 2) + cc);
            b3 = __shfl_sync(SM, ce, 8 + ((rr ^ 3) << 2) + cc);
            ce = rcpf(((ce + b2) + (b1 + b3)) + hc_eps) * ce;
        }
        combv = ce;
    }
    if (t < 128) {
        if (m >= 4 && m < 8) post[(i64)r * 4 + (m - 4)] = pre;
        if (m >= 8 && m < 24) comb[(i64)r * 16 + (m - 8)] = combv;
    }
    float p[4];
    for (int sv = 0; sv < 4; ++sv) {
        float v = (m == sv) ? pre : 0.0f;
        for (int i = 16; i; i >>= 1) v += __shfl_xor_sync(0xffffffffu, v, i);
        p[sv] = v;
    }
    const u16* xr = x + (i64)r * (4 * D);
    float c[16];
    for (int half = 0; half < 2; ++half) {
        int off = half * 2048 + 8 * t;
        float x0[8], x1[8], x2[8], x3[8];
        load4bf(xr + off, x0);
        load4bf(xr + off + 4, x0 + 4);
        load4bf(xr + D + off, x1);
        load4bf(xr + D + off + 4, x1 + 4);
        load4bf(xr + 2 * D + off, x2);
        load4bf(xr + 2 * D + off + 4, x2 + 4);
        load4bf(xr + 3 * D + off, x3);
        load4bf(xr + 3 * D + off + 4, x3 + 4);
        for (int i = 0; i < 8; ++i) {
            // Triton's f32x2 pairs: even columns fma(p1, x1, p0 * x0), odd ones fma(p0, x0, p1 * x1) (probed)
            float v = (i & 1) ? fmaf(p[0], x0[i], p[1] * x1[i]) : fmaf(p[1], x1[i], p[0] * x0[i]);
            v = fmaf(p[2], x2[i], v);
            v = fmaf(p[3], x3[i], v);
            c[half * 8 + i] = f32of(bf16_rn(v));
        }
    }
    float q1o = fmaf(c[1], c[1], c[3] * c[3]), q1e = fmaf(c[0], c[0], c[2] * c[2]);
    float q2o = fmaf(c[5], c[5], c[7] * c[7]), q2e = fmaf(c[4], c[4], c[6] * c[6]);
    float q3o = fmaf(c[9], c[9], c[11] * c[11]), q3e = fmaf(c[8], c[8], c[10] * c[10]);
    float q4o = fmaf(c[13], c[13], c[15] * c[15]), q4e = fmaf(c[12], c[12], c[14] * c[14]);
    float ssum = ((q1o + q2o) + (q3o + q4o)) + ((q1e + q2e) + (q3e + q4e));
    for (int i = 16; i; i >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, i);
    if ((t & 31) == 0) warp_sums[t >> 5] = ssum;
    __syncthreads();
    float v0 = warp_sums[0], v1 = warp_sums[1], v2 = warp_sums[2], v3 = warp_sums[3];
    float v4 = warp_sums[4], v5 = warp_sums[5], v6 = warp_sums[6], v7 = warp_sums[7];
    float total = ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7));
    float rinv2 = divf(1.0f, sqrtf_ftz(divf(total, (float)D) + eps_norm));
    u16 yva[8], yvb[8];
    const u16* wr = nw + 8 * t;
    for (int i = 0; i < 8; ++i) {
        yva[i] = bf16_rn(f32of(wr[i]) * f32of(bf16_rn(rinv2 * c[i])));
        yvb[i] = bf16_rn(f32of(wr[i + 2048]) * f32of(bf16_rn(rinv2 * c[i + 8])));
    }
    store4bf(out + (i64)r * D + 8 * t, yva);
    store4bf(out + (i64)r * D + 8 * t + 4, yva + 4);
    store4bf(out + (i64)r * D + 2048 + 8 * t, yvb);
    store4bf(out + (i64)r * D + 2048 + 8 * t + 4, yvb + 4);
    float ya[8], yb[8];
    for (int i = 0; i < 8; ++i) {
        ya[i] = f32of(yva[i]);
        yb[i] = f32of(yvb[i]);
    }
    float pa = ((ya[0] + ya[2]) + (ya[4] + ya[6])) + ((ya[1] + ya[3]) + (ya[5] + ya[7]));
    float pb = ((yb[0] + yb[2]) + (yb[4] + yb[6])) + ((yb[1] + yb[3]) + (yb[5] + yb[7]));
    for (int i = 4; i; i >>= 1) pa += __shfl_xor_sync(0xffffffffu, pa, i);
    for (int i = 4; i; i >>= 1) pb += __shfl_xor_sync(0xffffffffu, pb, i);
    if ((t & 7) == 0) {
        gs[t >> 3] = pa;
        gs[32 + (t >> 3)] = pb;
    }
    __syncthreads();
    if (t < 64) xs[(i64)r * (D / 64) + t] = gs[t];
}

// -- hyper-connection post (glue.py _hc_post; world fp32 partials at element stride rs) -----------
__device__ __forceinline__ void hc_post(const u16* x, u16* xout, const float* g, const float* post, const float* comb, i64 rs,
                        int world, int s) {
    const int d = gridDim.y * 1024;
    int r = blockIdx.x, cb = blockIdx.y, t = threadIdx.x;
    int d0 = cb * 1024 + 8 * t;
    const float* gr = g + (i64)r * d + d0;
    float branch[8];
    for (int i = 0; i < 8; ++i) branch[i] = gr[i];
    for (int k = 1; k < world; ++k) {
        const float* gk = g + (i64)k * rs + (i64)r * d + d0;
        for (int i = 0; i < 8; ++i) branch[i] += gk[i];
    }
    const u16* xr = x + (i64)r * (s * d) + d0;
    float x0[8], x1[8], x2[8], x3[8];
    load4bf(xr, x0);
    load4bf(xr + 4, x0 + 4);
    load4bf(xr + d, x1);
    load4bf(xr + d + 4, x1 + 4);
    load4bf(xr + 2 * d, x2);
    load4bf(xr + 2 * d + 4, x2 + 4);
    load4bf(xr + 3 * d, x3);
    load4bf(xr + 3 * d + 4, x3 + 4);
    for (int sv = 0; sv < s; ++sv) {
        float c0v = comb[(i64)r * 16 + 0 * 4 + sv];
        float c1v = comb[(i64)r * 16 + 1 * 4 + sv];
        float c2v = comb[(i64)r * 16 + 2 * 4 + sv];
        float c3v = comb[(i64)r * 16 + 3 * 4 + sv];
        float psv = post[(i64)r * s + sv];
        u16 res[8];
        for (int i = 0; i < 8; ++i) {
            // Triton fuses the first product into the rounded second: fma(c0, x0, c1 * x1) (glm.layer.kda, 14 words)
            float mixed = fmaf(c0v, x0[i], c1v * x1[i]);
            mixed = fmaf(c2v, x2[i], mixed);
            mixed = fmaf(c3v, x3[i], mixed);
            res[i] = bf16_rn(fmaf(psv, f32of(bf16_rn(branch[i])), mixed));
        }
        store4bf(xout + (i64)r * (s * d) + (i64)sv * d + d0, res);
        store4bf(xout + (i64)r * (s * d) + (i64)sv * d + d0 + 4, res + 4);
    }
}

// -- residual add (glue.py _residual_add) --------------------------------------------------------
__device__ __forceinline__ void residual_add(const u16* x, u16* xout, const float* g, i64 rs, int world) {
    const int d = gridDim.y * 1024;
    int r = blockIdx.x, cb = blockIdx.y, t = threadIdx.x;
    int d0 = cb * 1024 + 8 * t;
    const float* gr = g + (i64)r * d + d0;
    float branch[8];
    for (int i = 0; i < 8; ++i) branch[i] = gr[i];
    for (int k = 1; k < world; ++k) {
        const float* gk = g + (i64)k * rs + (i64)r * d + d0;
        for (int i = 0; i < 8; ++i) branch[i] += gk[i];
    }
    const u16* xr = x + (i64)r * d + d0;
    u16* outr = xout + (i64)r * d + d0;
    for (int i = 0; i < 8; i += 4) {
        float xv[4];
        load4bf(xr + i, xv);
        u16 res[4];
        for (int j = 0; j < 4; ++j) res[j] = bf16_rn(xv[j] + f32of(bf16_rn(branch[i + j])));
        store4bf(outr + i, res);
    }
}

// -- stream mean (glue.py _stream_mean) ----------------------------------------------------------
__device__ __forceinline__ void stream_mean(const u16* x, u16* out, int s) {
    const int d = gridDim.y * 1024;
    int r = blockIdx.x, cb = blockIdx.y, t = threadIdx.x;
    int d0 = cb * 1024 + 8 * t;
    const u16* xr = x + (i64)r * (s * d) + d0;
    float acc[8];
    load4bf(xr, acc);
    load4bf(xr + 4, acc + 4);
    for (int k = 1; k < s; ++k) {
        float v[8];
        load4bf(xr + (i64)k * d, v);
        load4bf(xr + (i64)k * d + 4, v + 4);
        for (int i = 0; i < 8; ++i) acc[i] += v[i];
    }
    u16 res[8];
    for (int i = 0; i < 8; ++i) res[i] = bf16_rn(divf(acc[i], (float)s));
    store4bf(out + (i64)r * d + d0, res);
    store4bf(out + (i64)r * d + d0 + 4, res + 4);
}

// -- dense MLP swiglu (glue.py _swiglu; BLOCK 512, 64-group sums) ---------------------------------
__device__ __forceinline__ void swiglu(const u16* gu, u16* out, float* xs, float lim) {
    __shared__ float sm[8];
    int r = blockIdx.x, cb = blockIdx.y, t = threadIdx.x;
    int w = gridDim.y * 512;
    const u16* gur = gu + (i64)r * (2 * w) + cb * 512;
    float g[4], u[4];
    load4bf(gur + 4 * t, g);
    load4bf(gur + w + 4 * t, u);
    for (int i = 0; i < 4; ++i) {
        g[i] = fminf(g[i], lim);
        u[i] = fminf(fmaxf(u[i], -lim), lim);
        float d = ex2f(g[i] * NEG_LOG2E);
        float si = f32of(bf16_rn(divf(g[i], d + 1.0f)));
        g[i] = si * u[i];
    }
    u16 a[4];
    for (int i = 0; i < 4; ++i) a[i] = bf16_rn(g[i]);
    store4bf(out + (i64)r * w + cb * 512 + 4 * t, a);
    float p = (f32of(a[1]) + f32of(a[3])) + (f32of(a[0]) + f32of(a[2]));
    for (int i = 8; i; i >>= 1) p += __shfl_xor_sync(0xffffffffu, p, i);
    if ((t & 15) == 0) sm[t >> 4] = p;
    __syncthreads();
    if (t < 8) xs[(i64)r * (w / 64) + cb * 8 + t] = sm[t];
}

// -- MoE router top-k (glue.py _topk; BLOCK = next_pow2(ne+1) <= 512) -----------------------------
__device__ __forceinline__ void topk(const float* lg, const float* bias, int* pick, float* wts, float scale, int ne, int topk) {
    __shared__ float mxs[4];
    __shared__ float ivs[4];  // its own slots: a warp's index must not land in mxs while another still reads the max
    __shared__ float sks[4];
    int r = blockIdx.x, t = threadIdx.x, w = t >> 5, lane = t & 31;
    const int per = 4;  // BLOCK <= 512, 128 threads
    float choice[per], score[per];
    int eid[per];
    for (int i = 0; i < per; ++i) {
        int e = t * per + i;
        eid[i] = e;
        bool ok = e < ne;
        float s = ok ? sigmoid(lg[(i64)r * ne + e]) : 0.0f;
        score[i] = s;
        choice[i] = ok ? s + bias[e] : -__int_as_float(0x7f800000u);
    }
    float total = 0.0f, wk[16];  // every thread holds the picks' scores: no global read-back while others normalize
    for (int k = 0; k < topk; ++k) {
        float mx = choice[0];
        for (int i = 1; i < per; ++i) mx = fmaxf(mx, choice[i]);
        for (int i = 16; i; i >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, i));
        if (lane == 0) mxs[w] = mx;
        __syncthreads();
        mx = fmaxf(fmaxf(mxs[0], mxs[2]), fmaxf(mxs[1], mxs[3]));
        float iv = (float)(per * 128);
        for (int i = 0; i < per; ++i)
            if (choice[i] == mx) iv = fminf(iv, (float)eid[i]);
        for (int i = 16; i; i >>= 1) iv = fminf(iv, __shfl_xor_sync(0xffffffffu, iv, i));
        if (lane == 0) ivs[w] = iv;
        __syncthreads();
        int idx = (int)fminf(fminf(ivs[0], ivs[2]), fminf(ivs[1], ivs[3]));
        float sk = 0.0f;
        for (int i = 0; i < per; ++i)
            if (eid[i] == idx) sk += score[i];
        for (int i = 16; i; i >>= 1) sk += __shfl_xor_sync(0xffffffffu, sk, i);
        if (lane == 0) sks[w] = sk;
        __syncthreads();
        sk = (sks[0] + sks[2]) + (sks[1] + sks[3]);
        pick[(i64)r * (topk + 1) + k] = idx;
        wk[k] = sk;
        total += sk;
        for (int i = 0; i < per; ++i)
            if (eid[i] == idx) choice[i] = -__int_as_float(0x7f800000u);
        __syncthreads();
    }
    for (int k = 0; k < topk; ++k)
        if (t == k) wts[(i64)r * (topk + 1) + k] = scale * divf(wk[k], total + TINY1E20);
    if (t < topk + 1) {
        pick[(i64)r * (topk + 1) + topk] = ne;
        wts[(i64)r * (topk + 1) + topk] = 1.0f;
    }
}

// -- MoE combine (glue.py _combine; per-thread 8 consecutive d: pos 0,7 fma order, pos 1..6 fma slot 0, S-1, 1..S-2)
__device__ __forceinline__ void combine(const u16* y, const float* wts, float* out, int slots) {
    const int d = gridDim.y * 1024;
    int r = blockIdx.x, cb = blockIdx.y, t = threadIdx.x;
    int d0 = cb * 1024 + 8 * t;
    const u16* yrow = y + (i64)r * slots * d + d0;
    const float* wrow = wts + (i64)r * slots;
    float* orow = out + (i64)r * d + d0;
    for (int i = 0; i < 8; ++i) {
        bool edge = (i == 0 || i == 7);
        float acc = fmaf(f32of(yrow[i]), wrow[0], edge ? -0.0f : 0.0f);
        if (edge) {
            for (int k = 1; k < slots; ++k) acc = fmaf(f32of(yrow[(i64)k * d + i]), wrow[k], acc);
        } else {
            acc = fmaf(f32of(yrow[(i64)(slots - 1) * d + i]), wrow[slots - 1], acc);
            for (int k = 1; k < slots - 1; ++k) acc = fmaf(f32of(yrow[(i64)k * d + i]), wrow[k], acc);
        }
        orow[i] = acc;
    }
}

// -- greedy sampling over a rank's logits (decode.py sample_rows greedy path) ----------------------
__device__ __forceinline__ void greedy(const u16* lg, int* tok, float* packed, int n, int offset) {
    __shared__ float sv[8];
    __shared__ int si[8];
    int r = blockIdx.x, t = threadIdx.x;
    const u16* row = lg + (i64)r * n;
    float best = -__int_as_float(0x7f800000u);
    int bi = 0;
    for (int i = t; i < n; i += 256) {
        float v = f32of(row[i]);
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }
    for (int off = 16; off; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, off);
        int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    if ((t & 31) == 0) {
        sv[t >> 5] = best;
        si[t >> 5] = bi;
    }
    __syncthreads();
    if (t == 0) {
        float bv = sv[0];
        int bidx = si[0];
        for (int i = 1; i < 8; ++i)
            if (sv[i] > bv || (sv[i] == bv && si[i] < bidx)) {
                bv = sv[i];
                bidx = si[i];
            }
        if (tok) tok[r] = bidx;
        if (packed) {  // sample_rows' all-gather payload: the value, then the vocab id's bits
            packed[2 * r] = bv;
            packed[2 * r + 1] = __int_as_float(bidx + offset);
        }
    }
}

// -- entry kernels (zig launches these; grids in zig/src/cuda/glm_glue.zig) ----------------------
extern "C" __global__ void tf_glm_embed_b16(const u16* ids, const u16* w, u16* out, int copies) {
    embed_b16(ids, w, out, copies);
}

extern "C" __global__ void tf_glm_rmsnorm(const u16* x, i64 xs, const u16* w, u16* out, i64 os, float* sums,
                                          float eps) {
    rmsnorm<4096>(x, xs, w, out, os, sums, eps);
}

extern "C" __global__ void tf_glm_hc_partial(const u16* x, const float* fn, float* part) {
    hc_partial<16384, 16>(x, fn, part);
}

extern "C" __global__ void tf_glm_hc_finish(const u16* x, const float* part, const float* base, const float* scale,
                                            const u16* nw, u16* out, float* xs, float* post, float* comb,
                                            float eps_norm, float hc_eps, int iters) {
    hc_finish<4096>(x, part, base, scale, nw, out, xs, post, comb, eps_norm, hc_eps, iters);
}

extern "C" __global__ void tf_glm_hc_post(const u16* x, u16* xout, const float* g, const float* post,
                                          const float* comb, i64 rs, int world, int s) {
    hc_post(x, xout, g, post, comb, rs, world, s);
}

extern "C" __global__ void tf_glm_residual_add(const u16* x, u16* xout, const float* g, i64 rs, int world) {
    residual_add(x, xout, g, rs, world);
}

extern "C" __global__ void tf_glm_stream_mean(const u16* x, u16* out, int s) { stream_mean(x, out, s); }

extern "C" __global__ void tf_glm_swiglu(const u16* gu, u16* out, float* xs, float lim) { swiglu(gu, out, xs, lim); }

extern "C" __global__ void tf_glm_topk(const float* lg, const float* bias, int* pick, float* wts, float scale, int ne,
                                       int tk) {
    topk(lg, bias, pick, wts, scale, ne, tk);
}

extern "C" __global__ void tf_glm_combine(const u16* y, const float* wts, float* out, int slots) {
    combine(y, wts, out, slots);
}

extern "C" __global__ void tf_glm_greedy(const u16* lg, int* tok, int n) { greedy(lg, tok, nullptr, n, 0); }
extern "C" __global__ void tf_glm_greedy_pack(const u16* lg, float* packed, int n, int offset) {
    greedy(lg, nullptr, packed, n, offset);
}
