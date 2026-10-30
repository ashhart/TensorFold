#pragma once

// A prompt's gated delta rule in 32-token chunks (WY form) on the gfx11 matrix cores, numerics=inexact, Dk = Dv = 128.

// tf_gdn_wy_prep: every chunk's T = (I + A)^-1, P and f16 operands at once; tf_gdn_wy_scan: a value head walks chunks.

#include "common/arch.hpp"

#include <hip/hip_runtime.h>

namespace tf {
namespace rocm {

constexpr int kWyChunk = 32;              // tokens a chunk
constexpr int kWyDim = 128;               // Dk = Dv
constexpr int kWyPadD = kWyDim + 8;       // f16 row of a [.][dim] tile in LDS (16-byte aligned, banks spread)
constexpr int kWyPadC = kWyChunk + 8;     // f16 row of a [.][token] tile in LDS
constexpr int kWyPadA = kWyChunk + 4;     // fp32 row of A in LDS
constexpr int kWyPack = kWyDim * kWyChunk;  // f16 a packed (chunk, head) operand: k^T, q or v^T
constexpr int kWySquare = kWyChunk * kWyChunk;

typedef _Float16 wy_h16x16 __attribute__((ext_vector_type(16)));
typedef float wy_f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned wy_u32x8 __attribute__((ext_vector_type(8)));

#if TF_DEVICE_WMMA_GFX11
__device__ __forceinline__ wy_f32x8 wy_mma(wy_h16x16 a, wy_h16x16 b, wy_f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// 16 consecutive f16 (16-byte aligned) as one operand: row (A) or column (B) col of the 16 x 16 tile, k = 0..15
__device__ __forceinline__ wy_h16x16 wy_frag(const _Float16* p) {
    const uint4 lo = *reinterpret_cast<const uint4*>(p);
    const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);
    const wy_u32x8 w = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    return __builtin_bit_cast(wy_h16x16, w);
}

__device__ __forceinline__ unsigned wy_pair(float lo, float hi) {
    return static_cast<unsigned>(__builtin_bit_cast(unsigned short, static_cast<_Float16>(lo))) |
           (static_cast<unsigned>(__builtin_bit_cast(unsigned short, static_cast<_Float16>(hi))) << 16);
}

// An accumulator tile (lane: column col, rows 2 i + half) as its transpose's B operand (other half by permlanex16).
__device__ __forceinline__ wy_h16x16 wy_transposed(const wy_f32x8& x, int half) {
    wy_u32x8 w;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float own = x[i];
        const float other = __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(unsigned, own),
                                                                                  __builtin_bit_cast(unsigned, own),
                                                                                  0x76543210u, 0xfedcba98u, false, false));
        w[i] = half ? wy_pair(other, own) : wy_pair(own, other);
    }
    return __builtin_bit_cast(wy_h16x16, w);
}
#endif

}  // namespace rocm
}  // namespace tf

// Block (value head, chunk), 4 waves, zeros past `length`: the chunk's packed f16 k^T, q, v^T, t1, t2, p and g.
extern "C" __global__ void __launch_bounds__(128)
tf_gdn_wy_prep(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
               const float* __restrict__ gate, const float* __restrict__ beta, _Float16* __restrict__ kT16,
               _Float16* __restrict__ q16, _Float16* __restrict__ vT16, _Float16* __restrict__ t1,
               _Float16* __restrict__ t2, _Float16* __restrict__ p, float* __restrict__ g, int length, int key_heads,
               int value_heads) {
#if TF_DEVICE_WMMA_GFX11
    using namespace tf::rocm;
    constexpr int C = kWyChunk, D = kWyDim;
    const int head = blockIdx.x, chunk = blockIdx.y;
    const int t0 = chunk * C, n = min(C, length - t0);
    const int group = value_heads / key_heads, kh = head / group;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col = lane & 15, half = lane >> 4;
    __shared__ __attribute__((aligned(16))) _Float16 ks[C][kWyPadD];
    __shared__ __attribute__((aligned(16))) _Float16 qs[C][kWyPadD];
    __shared__ __attribute__((aligned(16))) _Float16 vs[C][kWyPadD];
    __shared__ float gs[C], bs[C];
    __shared__ __attribute__((aligned(16))) float a[C][kWyPadA];
    // every read first: 8 float4 of k, q and v a thread
    float4 kv[8], qv[8], vv[8];
#pragma unroll
    for (int it = 0; it < 8; ++it) {
        const int i = tid + it * 128, c = i / (D / 4), e = (i % (D / 4)) * 4;
        kv[it] = qv[it] = vv[it] = float4{0.f, 0.f, 0.f, 0.f};
        if (c < n) {
            const size_t at = (static_cast<size_t>(t0 + c) * key_heads + kh) * D + e;
            kv[it] = *reinterpret_cast<const float4*>(k + at);
            qv[it] = *reinterpret_cast<const float4*>(q + at);
            vv[it] = *reinterpret_cast<const float4*>(v + (static_cast<size_t>(t0 + c) * value_heads + head) * D + e);
        }
    }
    if (wave == 0) {
        float lg = 0.f, b = 0.f;
        if (lane < n) {
            const size_t at = static_cast<size_t>(t0 + lane) * value_heads + head;
            lg = fmaxf(logf(gate[at]), -80.f);
            b = beta[at];
        }
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float up = __shfl_up(lg, o, 32);
            if (lane >= o) lg += up;
        }
        gs[lane] = lg;
        bs[lane] = b;
        g[(static_cast<size_t>(chunk) * value_heads + head) * C + lane] = lg;
    }
#pragma unroll
    for (int it = 0; it < 8; ++it) {
        const int i = tid + it * 128, c = i / (D / 4), e = (i % (D / 4)) * 4;
        ks[c][e] = kv[it].x, ks[c][e + 1] = kv[it].y, ks[c][e + 2] = kv[it].z, ks[c][e + 3] = kv[it].w;
        qs[c][e] = qv[it].x, qs[c][e + 1] = qv[it].y, qs[c][e + 2] = qv[it].z, qs[c][e + 3] = qv[it].w;
        vs[c][e] = vv[it].x, vs[c][e + 1] = vv[it].y, vs[c][e + 2] = vv[it].z, vs[c][e + 3] = vv[it].w;
    }
    __syncthreads();
    {
        const size_t kat = (static_cast<size_t>(chunk) * key_heads + kh) * kWyPack;
        const size_t vat = (static_cast<size_t>(chunk) * value_heads + head) * kWyPack;
        const bool packs_k = head % group == 0;
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int u = tid + it * 128, row = u / 4, c0 = (u % 4) * 8;
            uint4 pk, pv;
            unsigned* wk = &pk.x;
            unsigned* wv = &pv.x;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                wk[e] = static_cast<unsigned>(__builtin_bit_cast(unsigned short, ks[c0 + 2 * e][row])) |
                        (static_cast<unsigned>(__builtin_bit_cast(unsigned short, ks[c0 + 2 * e + 1][row])) << 16);
                wv[e] = static_cast<unsigned>(__builtin_bit_cast(unsigned short, vs[c0 + 2 * e][row])) |
                        (static_cast<unsigned>(__builtin_bit_cast(unsigned short, vs[c0 + 2 * e + 1][row])) << 16);
            }
            *reinterpret_cast<uint4*>(vT16 + vat + row * C + c0) = pv;
            if (packs_k) {
                *reinterpret_cast<uint4*>(kT16 + kat + row * C + c0) = pk;
                const int c = u / 16, d0 = (u % 16) * 8;
                *reinterpret_cast<uint4*>(q16 + kat + c * D + d0) = *reinterpret_cast<const uint4*>(&qs[c][d0]);
            }
        }
    }
    // wave w: tile (w / 2, w % 2) of k k^T and q k^T
    const int ti = wave >> 1, tj = wave & 1;
    wy_f32x8 kk = {}, qk = {};
#pragma unroll
    for (int s = 0; s < D / 16; ++s) {
        const wy_h16x16 b = wy_frag(&ks[tj * 16 + col][s * 16]);
        kk = wy_mma(wy_frag(&ks[ti * 16 + col][s * 16]), b, kk);
        qk = wy_mma(wy_frag(&qs[ti * 16 + col][s * 16]), b, qk);
    }
    const size_t sq = (static_cast<size_t>(chunk) * value_heads + head) * kWySquare;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int i = ti * 16 + 2 * r + half, j = tj * 16 + col;
        const float d = i >= j ? expf(gs[i] - gs[j]) : 0.f;
        a[i][j] = i > j ? bs[i] * d * kk[r] : 0.f;
        p[sq + i * C + j] = static_cast<_Float16>(d * qk[r]);
    }
    __syncthreads();
    if (wave == 0) {
        // T = (I + A)^-1 by forward substitution, lane c its column c
        float tc[C];
#pragma unroll
        for (int i = 0; i < C; ++i) {
            float x = i == lane ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < i; ++j) x -= a[i][j] * tc[j];
            tc[i] = x;
        }
        const float s1 = bs[lane] * expf(gs[lane]), s2 = bs[lane];
#pragma unroll
        for (int i = 0; i < C; ++i) {
            t1[sq + i * C + lane] = static_cast<_Float16>(tc[i] * s1);
            t2[sq + i * C + lane] = static_cast<_Float16>(tc[i] * s2);
        }
    }
#else
    __builtin_trap();
#endif
}

// Block (value head), 8 waves of 16 value rows, the state transposed in registers; the next chunk loads meanwhile.
extern "C" __global__ void __launch_bounds__(256)
tf_gdn_wy_scan(const _Float16* __restrict__ kT16, const _Float16* __restrict__ q16, const _Float16* __restrict__ vT16,
               const _Float16* __restrict__ t1, const _Float16* __restrict__ t2, const _Float16* __restrict__ p,
               const float* __restrict__ g, float* __restrict__ state, float* __restrict__ y, int length,
               int key_heads, int value_heads) {
#if TF_DEVICE_WMMA_GFX11
    using namespace tf::rocm;
    constexpr int C = kWyChunk, D = kWyDim;
    const int head = blockIdx.x;
    const int kh = head / (value_heads / key_heads);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col = lane & 15, half = lane >> 4;
    const int r0 = wave * 16;
    __shared__ __attribute__((aligned(16))) _Float16 kT[D][kWyPadC];
    __shared__ __attribute__((aligned(16))) _Float16 qs[C][kWyPadD];
    __shared__ __attribute__((aligned(16))) _Float16 wn[C][kWyPadD];  // -W
    __shared__ __attribute__((aligned(16))) _Float16 vT[D][kWyPadC];
    __shared__ __attribute__((aligned(16))) _Float16 t1s[C][kWyPadC];
    __shared__ __attribute__((aligned(16))) _Float16 t2s[C][kWyPadC];
    __shared__ __attribute__((aligned(16))) _Float16 ps[C][kWyPadC];
    __shared__ __attribute__((aligned(16))) float gs[C];
    wy_f32x8 sa[D / 16];
    float* st = state + (static_cast<size_t>(head) * D + r0 + col) * D;
#pragma unroll
    for (int dt = 0; dt < D / 16; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) sa[dt][i] = st[dt * 16 + 2 * i + half];
    // a chunk's record in 16-byte slots, 8 a thread: 0-1 k^T, 2-3 q, 4-5 v^T, 6 t1 | t2, 7 p | g
    auto from = [&](int chunk, int s) -> const uint4* {
        const size_t kc = (static_cast<size_t>(chunk) * key_heads + kh) * kWyPack;
        const size_t vc = static_cast<size_t>(chunk) * value_heads + head;
        const int u = (s & 1) * 256 + tid;
        switch (s) {
            case 0: case 1: return reinterpret_cast<const uint4*>(kT16 + kc) + u;
            case 2: case 3: return reinterpret_cast<const uint4*>(q16 + kc) + u;
            case 4: case 5: return reinterpret_cast<const uint4*>(vT16 + vc * kWyPack) + u;
            case 6: return tid < 128 ? reinterpret_cast<const uint4*>(t1 + vc * kWySquare) + tid
                                     : reinterpret_cast<const uint4*>(t2 + vc * kWySquare) + (tid - 128);
            default: return tid < 128 ? reinterpret_cast<const uint4*>(p + vc * kWySquare) + tid
                                      : (tid < 136 ? reinterpret_cast<const uint4*>(g + vc * C) + (tid - 128) : nullptr);
        }
    };
    auto to = [&](int s) -> uint4* {
        const int u = (s & 1) * 256 + tid;
        switch (s) {
            case 0: case 1: return reinterpret_cast<uint4*>(&kT[u / 4][(u % 4) * 8]);
            case 2: case 3: return reinterpret_cast<uint4*>(&qs[u / 16][(u % 16) * 8]);
            case 4: case 5: return reinterpret_cast<uint4*>(&vT[u / 4][(u % 4) * 8]);
            case 6: return tid < 128 ? reinterpret_cast<uint4*>(&t1s[tid / 4][(tid % 4) * 8])
                                     : reinterpret_cast<uint4*>(&t2s[(tid - 128) / 4][((tid - 128) % 4) * 8]);
            default: return tid < 128 ? reinterpret_cast<uint4*>(&ps[tid / 4][(tid % 4) * 8])
                                      : (tid < 136 ? reinterpret_cast<uint4*>(&gs[(tid - 128) * 4]) : nullptr);
        }
    };
    const int chunks = (length + C - 1) / C;
    uint4 next[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const uint4* src = from(0, s);
        next[s] = src ? *src : uint4{0, 0, 0, 0};
    }
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int t0 = chunk * C, n = min(C, length - t0);
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint4* dst = to(s);
            if (dst) *dst = next[s];
        }
        if (chunk + 1 < chunks) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const uint4* src = from(chunk + 1, s);
                if (src) next[s] = *src;
            }
        }
        __syncthreads();
        {
            // W tiles (token tile ct, dim tiles dt0, dt0 + 1): -W into wn
            const int ct = wave & 1, dt0 = (wave >> 1) * 2;
            const wy_h16x16 a0 = wy_frag(&t1s[ct * 16 + col][0]), a1 = wy_frag(&t1s[ct * 16 + col][16]);
#pragma unroll
            for (int d = 0; d < 2; ++d) {
                const int dt = dt0 + d;
                wy_f32x8 w = {};
                w = wy_mma(a0, wy_frag(&kT[dt * 16 + col][0]), w);
                w = wy_mma(a1, wy_frag(&kT[dt * 16 + col][16]), w);
#pragma unroll
                for (int i = 0; i < 8; ++i) wn[ct * 16 + 2 * i + half][dt * 16 + col] = static_cast<_Float16>(-w[i]);
            }
        }
        wy_f32x8 u[2], o[2];
#pragma unroll
        for (int ct = 0; ct < 2; ++ct) {
            u[ct] = wy_f32x8{};
            o[ct] = wy_f32x8{};
#pragma unroll
            for (int s = 0; s < 2; ++s)
                u[ct] = wy_mma(wy_frag(&t2s[ct * 16 + col][s * 16]), wy_frag(&vT[r0 + col][s * 16]), u[ct]);
        }
        __syncthreads();
        // V' = U - W S0^T and Q S0^T, one S0^T operand at a time
#pragma unroll
        for (int s = 0; s < D / 16; ++s) {
            const wy_h16x16 sb = wy_transposed(sa[s], half);
#pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                u[ct] = wy_mma(wy_frag(&wn[ct * 16 + col][s * 16]), sb, u[ct]);
                o[ct] = wy_mma(wy_frag(&qs[ct * 16 + col][s * 16]), sb, o[ct]);
            }
        }
        const float last = gs[C - 1];
        wy_h16x16 vb[2], vd[2];
#pragma unroll
        for (int ct = 0; ct < 2; ++ct) {
            vb[ct] = wy_transposed(u[ct], half);
            wy_f32x8 dec;
#pragma unroll
            for (int i = 0; i < 8; ++i) dec[i] = u[ct][i] * expf(last - gs[ct * 16 + 2 * i + half]);
            vd[ct] = wy_transposed(dec, half);
        }
        // y = e^G (Q S0^T) + P V'
#pragma unroll
        for (int ct = 0; ct < 2; ++ct) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[ct][i] *= expf(gs[ct * 16 + 2 * i + half]);
#pragma unroll
            for (int s = 0; s < 2; ++s) o[ct] = wy_mma(wy_frag(&ps[ct * 16 + col][s * 16]), vb[s], o[ct]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int c = ct * 16 + 2 * i + half;
                if (c < n) y[(static_cast<size_t>(t0 + c) * value_heads + head) * D + r0 + col] = o[ct][i];
            }
        }
        // S^T = e^G_last S0^T + K^T (e^(G_last - G) V')
        const float decay = expf(last);
#pragma unroll
        for (int dt = 0; dt < D / 16; ++dt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) sa[dt][i] *= decay;
#pragma unroll
            for (int s = 0; s < 2; ++s) sa[dt] = wy_mma(wy_frag(&kT[dt * 16 + col][s * 16]), vd[s], sa[dt]);
        }
    }
#pragma unroll
    for (int dt = 0; dt < D / 16; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) st[dt * 16 + 2 * i + half] = sa[dt][i];
#else
    __builtin_trap();
#endif
}
