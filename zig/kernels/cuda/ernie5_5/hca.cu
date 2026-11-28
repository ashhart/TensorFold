// ERNIE 5.5's HCA compressor: a block's 128 tokens pooled per channel, the RMSNorm, GPT-J's rope at the block's start.

#include <cuda_bf16.h>

namespace ernie55h {

constexpr int H = 512, R = 128, NOPE = 448, T = 256;
constexpr unsigned all = 0xffffffffu;

enum Fault { none, no_ape, last_position };

// sincosf out of line: one copy of its large-angle path.
static __device__ __noinline__ void turn(float a, float* s, float* c) { sincosf(a, s, c); }

// Row blockIdx.x (tokens 128 row + i, ts floats apart: kv [512] then score [512]), thread t's channels 2t and 2t + 1.
template <Fault F>
__device__ __forceinline__ void compress(const float* __restrict__ st, long long ts, const float* __restrict__ ape,
                                         const __nv_bfloat16* __restrict__ w, const float* __restrict__ freq,
                                         const int* __restrict__ pos, __nv_bfloat16* __restrict__ out, long long os,
                                         float eps) {
    __shared__ float sq[T / 32];
    const long long row = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float* base = st + row * R * ts + 2 * t;
    // s_i = score_i + ape_i (the control: score alone), its max, then e^(s_i - M) summed, kv_i e_i in fma, i in order
    float m0 = -INFINITY, m1 = -INFINITY;
    for (int i = 0; i < R; ++i) {
        const float2 sc = *reinterpret_cast<const float2*>(base + i * ts + H);
        const float2 ap = *reinterpret_cast<const float2*>(ape + i * H + 2 * t);
        m0 = fmaxf(m0, F == no_ape ? sc.x : __fadd_rn(sc.x, ap.x));
        m1 = fmaxf(m1, F == no_ape ? sc.y : __fadd_rn(sc.y, ap.y));
    }
    float s0 = 0.0f, s1 = 0.0f, p0 = 0.0f, p1 = 0.0f;
    for (int i = 0; i < R; ++i) {
        const float2 kv = *reinterpret_cast<const float2*>(base + i * ts);
        const float2 sc = *reinterpret_cast<const float2*>(base + i * ts + H);
        const float2 ap = *reinterpret_cast<const float2*>(ape + i * H + 2 * t);
        const float e0 = expf(__fsub_rn(F == no_ape ? sc.x : __fadd_rn(sc.x, ap.x), m0));
        const float e1 = expf(__fsub_rn(F == no_ape ? sc.y : __fadd_rn(sc.y, ap.y), m1));
        s0 = __fadd_rn(s0, e0);
        s1 = __fadd_rn(s1, e1);
        p0 = __fmaf_rn(kv.x, e0, p0);
        p1 = __fmaf_rn(kv.y, e1, p1);
    }
    float x0 = __fdiv_rn(p0, s0), x1 = __fdiv_rn(p1, s1);
    // RMSNorm 1 / sqrt(mean + eps): the thread's two squares, the warp's butterfly, the 8 warps in order
    float ss = __fmaf_rn(x1, x1, __fmul_rn(x0, x0));
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(all, ss, o));
    if (lane == 0) sq[warp] = ss;
    __syncthreads();
    float tot = sq[0];
    for (int i = 1; i < T / 32; ++i) tot = __fadd_rn(tot, sq[i]);
    const float r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fmul_rn(tot, 1.0f / H), eps)));
    const __nv_bfloat162 wv = *reinterpret_cast<const __nv_bfloat162*>(w + 2 * t);
    x0 = __fmul_rn(__fmul_rn(x0, r), __bfloat162float(wv.x));
    x1 = __fmul_rn(__fmul_rn(x1, r), __bfloat162float(wv.y));
    // pair l = t - 224 on dims 448 + 2l, 449 + 2l, at the block's first position (the control: its last)
    if (2 * t >= NOPE) {
        const long long at = static_cast<long long>(pos[0]) + row * R + (F == last_position ? R - 1 : 0);
        float sn, cs;
        turn(__fmul_rn(freq[t - NOPE / 2], static_cast<float>(at)), &sn, &cs);
        const float y0 = __fsub_rn(__fmul_rn(x0, cs), __fmul_rn(x1, sn));
        const float y1 = __fadd_rn(__fmul_rn(x1, cs), __fmul_rn(x0, sn));
        x0 = y0;
        x1 = y1;
    }
    *reinterpret_cast<__nv_bfloat162*>(out + row * os + 2 * t) = __floats2bfloat162_rn(x0, x1);
}

}  // namespace ernie55h

// 128-token blocks (kv then score, fp32, ts apart), ape, w, freq [32], row r at pos[0] + 128 r: bf16 rows os apart.
extern "C" __global__ void __launch_bounds__(256)
    tf_ernie55_hca_compress(const float* __restrict__ st, long long ts, const float* __restrict__ ape,
                            const __nv_bfloat16* __restrict__ w, const float* __restrict__ freq,
                            const int* __restrict__ pos, __nv_bfloat16* __restrict__ out, long long os, float eps) {
    ernie55h::compress<ernie55h::none>(st, ts, ape, w, freq, pos, out, os, eps);
}

// The contract's planted control: the pool's softmax over score alone, without ape.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_hca_compress_no_ape(const float* __restrict__ st, long long ts, const float* __restrict__ ape,
                                       const __nv_bfloat16* __restrict__ w, const float* __restrict__ freq,
                                       const int* __restrict__ pos, __nv_bfloat16* __restrict__ out, long long os,
                                       float eps) {
    ernie55h::compress<ernie55h::no_ape>(st, ts, ape, w, freq, pos, out, os, eps);
}

// The contract's planted control: the rope at the block's last position, 128 r + 127, in place of its first.
extern "C" __global__ void __launch_bounds__(256)
    tf_ctl_ernie55_hca_compress_last_position(const float* __restrict__ st, long long ts, const float* __restrict__ ape,
                                              const __nv_bfloat16* __restrict__ w, const float* __restrict__ freq,
                                              const int* __restrict__ pos, __nv_bfloat16* __restrict__ out,
                                              long long os, float eps) {
    ernie55h::compress<ernie55h::last_position>(st, ts, ape, w, freq, pos, out, os, eps);
}
