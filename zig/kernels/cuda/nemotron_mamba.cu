// Nemotron's Mamba-2 conv and scan: state lags a window, its kept rows replaying first in the same loop body.

#include <cuda_bf16.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

// out = bf16(silu(bf16(bias + w0 t0 + w1 t1 + w2 t2 + w3 cur))), the taps in that fma order.
__device__ __forceinline__ __nv_bfloat16 conv4(float bias, const float (&w)[4], float t0, float t1, float t2, float cur) {
    float acc = bias;
    acc = __fmaf_rn(w[0], t0, acc);
    acc = __fmaf_rn(w[1], t1, acc);
    acc = __fmaf_rn(w[2], t2, acc);
    acc = __fmaf_rn(w[3], cur, acc);
    return __float2bfloat16_rn(tf_silu(rbf(acc)));
}

// Thread a channel: BASE holds the last three committed inputs; RAW[1 - parity]'s kept rows replay, then the window's.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_conv(const __nv_bfloat16* __restrict__ P, __nv_bfloat16* __restrict__ BASE,
                                                                 __nv_bfloat16* __restrict__ RAW, __nv_bfloat16* __restrict__ XC,
                                                                 const float* __restrict__ CW, const float* __restrict__ CB,
                                                                 const int* __restrict__ META, int R, int PROJ, int XOFF, int CD, int RMAX) {
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= CD) return;
    const int parity = META[1], pk = META[2];
    float t0 = bf(BASE[ch]), t1 = bf(BASE[CD + ch]), t2 = bf(BASE[2 * CD + ch]);
    const float w[4] = {CW[ch], CW[CD + ch], CW[2 * CD + ch], CW[3 * CD + ch]};
    const float bias = CB[ch];
    for (int i = 0; i < pk + R; ++i) {
        const bool prev = i < pk;
        const float cur = prev ? bf(RAW[(static_cast<int64_t>(1 - parity) * RMAX + i) * CD + ch])
                               : bf(P[static_cast<int64_t>(i - pk) * PROJ + XOFF + ch]);
        if (!prev) {
            const int64_t at = (static_cast<int64_t>(parity) * RMAX + (i - pk)) * CD + ch;
            XC[at] = conv4(bias, w, t0, t1, t2, cur);
            RAW[at] = __float2bfloat16_rn(cur);
        }
        t0 = t1;
        t1 = t2;
        t2 = cur;
        if (i == pk - 1) {
            BASE[ch] = __float2bfloat16_rn(t0);
            BASE[CD + ch] = __float2bfloat16_rn(t1);
            BASE[2 * CD + ch] = __float2bfloat16_rn(t2);
        }
    }
}

// A prompt chunk's conv, thread (row, channel): taps from the chunk's rows, or BASE's last three before row 0.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_conv_rows(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ BASE,
                                                                      __nv_bfloat16* __restrict__ XC, const float* __restrict__ CW,
                                                                      const float* __restrict__ CB, int R, int PROJ, int XOFF, int CD) {
    const int ch = blockIdx.y * 256 + threadIdx.x;
    const int r = blockIdx.x;
    if (ch >= CD || r >= R) return;
    float t[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int src = r - 3 + j;
        t[j] = src >= 0 ? bf(P[static_cast<int64_t>(src) * PROJ + XOFF + ch]) : bf(BASE[(src + 3) * CD + ch]);
    }
    const float w[4] = {CW[ch], CW[CD + ch], CW[2 * CD + ch], CW[3 * CD + ch]};
    XC[static_cast<int64_t>(r) * CD + ch] = conv4(CB[ch], w, t[0], t[1], t[2], t[3]);
}

// BASE <- the last three raw inputs of [BASE; the chunk's R rows], launched after the chunk's conv has read BASE.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_conv_commit(const __nv_bfloat16* __restrict__ P, __nv_bfloat16* __restrict__ BASE,
                                                                        int R, int PROJ, int XOFF, int CD) {
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= CD) return;
    __nv_bfloat16 v[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = R - 3 + j;
        v[j] = src >= 0 ? P[static_cast<int64_t>(src) * PROJ + XOFF + ch] : BASE[(R + j) * CD + ch];
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) BASE[j * CD + ch] = v[j];
}

// Block (head, 32 value rows), states split as scan_rows.cu splits them; kept rows replay first, then the window's.
extern "C" __global__ void __launch_bounds__(128) tf_nemo_scan(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ XC,
                                                                 float* __restrict__ DT, float* __restrict__ S, const float* __restrict__ A,
                                                                 const float* __restrict__ DSK, const float* __restrict__ DTB,
                                                                 const int* __restrict__ META, __nv_bfloat16* __restrict__ Y, int R, float lo,
                                                                 float hi, int PROJ, int XD, int CD, int DTOFF, int H, int DH, int NG, int RMAX) {
    constexpr int DS = 128, TPR = 4, NJ = DS / TPR / 4;
    __shared__ float4 bs[DS / 4], cs[DS / 4];
    const int h = blockIdx.x, g = h / (H / NG);
    const int local = threadIdx.x / TPR, q = threadIdx.x % TPR, d = blockIdx.y * 32 + local;
    const int parity = META[1], pk = META[2];
    const float a = A[h], dsk = DSK[h], dtb = DTB[h];
    float* s0 = S + (static_cast<int64_t>(h) * DH + d) * DS;
    float s[4 * NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 4 * (TPR * j + q));
        s[4 * j] = t.x;
        s[4 * j + 1] = t.y;
        s[4 * j + 2] = t.z;
        s[4 * j + 3] = t.w;
    }
    for (int i = 0; i < pk + R; ++i) {
        const bool prev = i < pk;
        const int buf = prev ? 1 - parity : parity, row = prev ? i : i - pk;
        const int64_t base = (static_cast<int64_t>(buf) * RMAX + row) * CD;
        __syncthreads();
        if (threadIdx.x < DS / 4) {
            const int c = threadIdx.x;
            const __nv_bfloat16* b = XC + base + XD + g * DS + 4 * c;
            const __nv_bfloat16* cc = XC + base + XD + NG * DS + g * DS + 4 * c;
            bs[c] = make_float4(bf(b[0]), bf(b[1]), bf(b[2]), bf(b[3]));
            cs[c] = make_float4(bf(cc[0]), bf(cc[1]), bf(cc[2]), bf(cc[3]));
        }
        __syncthreads();
        const int64_t dt_at = (static_cast<int64_t>(buf) * RMAX + row) * H + h;
        float dt;
        if (prev) {
            dt = DT[dt_at];
        } else {
            const float v = __fadd_rn(bf(P[static_cast<int64_t>(row) * PROJ + DTOFF + h]), dtb);
            dt = fminf(fmaxf(tf_softplus(v), lo), hi);
            if (blockIdx.y == 0 && threadIdx.x == 0) DT[dt_at] = dt;
        }
        const float x = bf(XC[base + h * DH + d]);
        const float da = tf_exp(__fmul_rn(a, dt)), xdt = __fmul_rn(x, dt);
        float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const float4 b = bs[TPR * j + q], c = cs[TPR * j + q];
            s[4 * j] = __fmaf_rn(xdt, b.x, __fmul_rn(s[4 * j], da));
            s[4 * j + 1] = __fmaf_rn(xdt, b.y, __fmul_rn(s[4 * j + 1], da));
            s[4 * j + 2] = __fmaf_rn(xdt, b.z, __fmul_rn(s[4 * j + 2], da));
            s[4 * j + 3] = __fmaf_rn(xdt, b.w, __fmul_rn(s[4 * j + 3], da));
            m[0] = __fmaf_rn(s[4 * j], c.x, m[0]);
            m[1] = __fmaf_rn(s[4 * j + 1], c.y, m[1]);
            m[2] = __fmaf_rn(s[4 * j + 2], c.z, m[2]);
            m[3] = __fmaf_rn(s[4 * j + 3], c.w, m[3]);
        }
        if (i == pk - 1) {
#pragma unroll
            for (int j = 0; j < NJ; ++j)
                *reinterpret_cast<float4*>(s0 + 4 * (TPR * j + q)) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
        }
        float out = __fadd_rn(__fadd_rn(m[0], m[1]), __fadd_rn(m[2], m[3]));
        out = __fadd_rn(out, __shfl_xor_sync(0xffffffffu, out, 1));
        out = __fadd_rn(out, __shfl_xor_sync(0xffffffffu, out, 2));
        if (prev || q != 0) continue;
        const float y = rbf(__fmaf_rn(x, dsk, out));
        const float gz = rbf(tf_silu(bf(P[static_cast<int64_t>(row) * PROJ + h * DH + d])));
        Y[static_cast<int64_t>(row) * XD + h * DH + d] = __float2bfloat16_rn(__fmul_rn(gz, y));
    }
}
