// Tensor-core MLA query absorption (`W_UK_T` + M-RoPE) and gated value expansion (`W_UV` + sigmoid gate).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../common/cp_async.cuh"
#include "../common/glue_math.cuh"
#include "../common/qmm_frag.cuh"

namespace hy4c2 {

typedef __nv_bfloat16 bf16;

constexpr int KL = 512;   // kv_lora_rank (latent)
constexpr int ND = 192;   // qk_nope_head_dim
constexpr int RD = 64;    // qk_rope_head_dim
constexpr int VD = 256;   // v_head_dim
constexpr int QW = 576;   // QA row: latent + rope
constexpr int QHS = 256;  // q_b columns a head (192 nope + 64 rope)
constexpr int KVB = 448;  // kv_b_proj rows a head (192 k_nope + 256 v)

__device__ __forceinline__ float bfv(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

// Decode streams weight bytes once (`L1::no_allocate`, `L2::evict_first`); prefill re-reads weights from L2.
__device__ __forceinline__ uint64_t evict_first_policy() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}

template <bool STREAM>
__device__ __forceinline__ uint4 ldw(const uint4* p, uint64_t pol) {
    (void)pol;
    if constexpr (STREAM) return tf_common::ld_nc(p);
    return __ldg(p);
}

__device__ __forceinline__ uint4 ld_stream(const void* p) {  // read-once activations (q_rope, gate)
    return tf_common::ld_nc(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    const uint32_t a[4] = {a0, a1, a2, a3};
    qmm_frag::mma(d, a, b0, b1);
}

__device__ __forceinline__ uint32_t word(const uint4& u, int i) { return i == 0 ? u.x : i == 1 ? u.y : i == 2 ? u.z : u.w; }

// Branchlessly zero any bf16 lane whose 8 exponent bits are all 1 (`Inf`/`NaN`) on mid/lo fp32 planes.
__device__ __forceinline__ uint32_t zero_nf_bf16x2(uint32_t v) {
    const uint32_t nf = (((v & 0x7F807F80u) + 0x00800080u) & 0x80008000u) >> 15;
    return v & ~(nf * 0xFFFFu);
}

// Split fp32 `x` into three exact bf16 planes `hi + mid + lo` (`Inf`/`NaN` carried in `hi` alone).
__device__ __forceinline__ void split3(float x, float& hi, float& mid, float& lo) {
    hi = rbf(x);
    const float r1 = __fsub_rn(x, hi);
    mid = rbf(r1);
    lo = rbf(__fsub_rn(r1, mid));
    if (!(fabsf(x) <= 3.0e38f)) {  // Inf / NaN: carry it in hi alone (x - x would turn Inf into NaN)
        hi = x;
        mid = 0.0f;
        lo = 0.0f;
    }
}

// Tile geometry: CTA owns `(h, [r0, r0+BN), [c0, c0+BM))`, staging `KC`-wide activation chunks in SMEM.
template <int K_, int MT_, int NT_, int WARPS_, int KC_, int STAGES_, int D_, int PLANES_, bool STREAM_, bool TILED_ = true>
struct Cfg {
    static constexpr int K = K_, MT = MT_, NT = NT_, WARPS = WARPS_, KC = KC_, PLANES = PLANES_;
    static constexpr bool STREAM = STREAM_, TILED = TILED_;
    static constexpr int BN = 8 * NT, CW = 16 * MT, BM = CW * WARPS, THREADS = 32 * WARPS;
    static constexpr int NKC = K / KC;                                 // activation chunks
    static constexpr int SLOTS = STAGES_ < NKC ? STAGES_ : NKC;        // ring slots actually used
    static constexpr int STAGES = SLOTS < 2 ? 2 : STAGES_;             // >= 2 for the wait arithmetic
    static constexpr int CH = K / 32;                                  // k32 blocks
    static constexpr int D = D_ < CH ? D_ : CH;                        // weight prefetch depth (k32 blocks)
    static constexpr int ROWB = KC * 2;                                // bytes of one activation row in a slot plane
    static constexpr int PLANE = BN * ROWB;
    static constexpr int SLOT = PLANES * PLANE;
    static constexpr int ACT = SLOTS * SLOT;
    static constexpr int EPS = CW + 4;                                 // epilogue staging row (floats)
    static constexpr int EPI = WARPS * 8 * EPS * 4;
    static constexpr int SMEM = ACT + EPI;
    static_assert(K % KC == 0 && KC % 64 == 0, "KC: a multiple of 64 dividing K");
    static_assert(K % 32 == 0, "K: k32 blocks");
};

// Bank-conflict-free byte offset of 16-byte unit `u` in activation row `n` within a slot plane.
template <class C>
__device__ __forceinline__ int act_off(int n, int u) {
    return n * C::ROWB + ((u ^ ((n & 1) << 2)) << 4);
}

// Stage activation chunk `kc` (`[kc*KC, +KC)`) of rows `r0 .. r0 + BN - 1` (zero-padded past `R`) into `slot`.
template <class C, typename T, class SRC>
__device__ __forceinline__ void stage_chunk(char* slot, int kc, int r0, int R, SRC src) {
    constexpr int UPR = C::KC / 8;  // 16-byte bf16 units a row
    if constexpr (C::PLANES == 1) {
        static_assert(sizeof(T) == 2, "single plane: bf16 activations");
        for (int e = threadIdx.x; e < C::BN * UPR; e += C::THREADS) {
            const int n = e / UPR, u = e % UPR, r = r0 + n;
            const bool ok = r < R;
            const T* p = src(ok ? r : r0) + kc * C::KC + u * 8;
            tf_common::cp16z(slot + act_off<C>(n, u), p, ok);
        }
    } else {
        static_assert(sizeof(T) == 4 && C::PLANES == 3, "three planes: fp32 activations split exactly");
        for (int e = threadIdx.x; e < C::BN * UPR; e += C::THREADS) {
            const int n = e / UPR, u = e % UPR, r = r0 + n;
            uint4 hi = make_uint4(0, 0, 0, 0), mid = hi, lo = hi;
            if (r < R) {
                const float* p = reinterpret_cast<const float*>(src(r)) + kc * C::KC + u * 8;
                const uint4 v0 = tf_common::ld_nc(reinterpret_cast<const uint4*>(p));
                const uint4 v1 = tf_common::ld_nc(reinterpret_cast<const uint4*>(p) + 1);
                float x[8] = {__uint_as_float(v0.x), __uint_as_float(v0.y), __uint_as_float(v0.z), __uint_as_float(v0.w),
                              __uint_as_float(v1.x), __uint_as_float(v1.y), __uint_as_float(v1.z), __uint_as_float(v1.w)};
                float h8[8], m8[8], l8[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) split3(x[i], h8[i], m8[i], l8[i]);
                hi = make_uint4(pack2(h8[0], h8[1]), pack2(h8[2], h8[3]), pack2(h8[4], h8[5]), pack2(h8[6], h8[7]));
                mid = make_uint4(pack2(m8[0], m8[1]), pack2(m8[2], m8[3]), pack2(m8[4], m8[5]), pack2(m8[6], m8[7]));
                lo = make_uint4(pack2(l8[0], l8[1]), pack2(l8[2], l8[3]), pack2(l8[4], l8[5]), pack2(l8[6], l8[7]));
            }
            const int off = act_off<C>(n, u);
            *reinterpret_cast<uint4*>(slot + off) = hi;
            *reinterpret_cast<uint4*>(slot + C::PLANE + off) = mid;
            *reinterpret_cast<uint4*>(slot + 2 * C::PLANE + off) = lo;
        }
    }
}

// Accumulate `W[col][k] * X[row][k]` over `K` in ascending `k32` order; `after_wait()` runs after PDL wait.
template <class C, typename T, class SRC, class HOOK>
__device__ __forceinline__ void chain(float (&acc)[C::MT][C::NT][4], const bf16* __restrict__ W, int c0w, int r0, int R,
                                      SRC src, char* act, HOOK after_wait) {
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    uint64_t pol = 0ull;
    if constexpr (C::STREAM) pol = evict_first_policy();
    // A rows `g` and `g + 8` of each `m16` tile (`TILED`: 64 consecutive `uint4` per `k32` block).
    constexpr int BSTEP = C::TILED ? 64 : 4;
    const uint4* wp[C::MT][2];
#pragma unroll
    for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wp[mt][j] = C::TILED ? reinterpret_cast<const uint4*>(W + static_cast<size_t>(c0w + mt * 16) * C::K) + j * 32 + lane
                                 : reinterpret_cast<const uint4*>(W + static_cast<size_t>(c0w + mt * 16 + g + 8 * j) * C::K) + q;

    uint4 a[C::D][C::MT][2];
#pragma unroll
    for (int b = 0; b < C::D; ++b)
#pragma unroll
        for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
            for (int j = 0; j < 2; ++j) a[b][mt][j] = ldw<C::STREAM>(wp[mt][j] + BSTEP * b, pol);

#pragma unroll
    for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < C::NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.0f;

    // Wait on previous stream kernel via PDL after weight prefetches are already in flight.
    qmm_frag::grid_wait();
    after_wait();

    // activation ring prologue
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; ++s) {
        if (s < C::NKC) stage_chunk<C, T>(act + (s % C::SLOTS) * C::SLOT, s, r0, R, src);
        tf_common::cp_commit();
    }

    constexpr int BPC = C::KC / 32;  // k32 blocks a chunk
#pragma unroll
    for (int kc = 0; kc < C::NKC; ++kc) {
        tf_common::cp_wait<C::STAGES - 2>();
        __syncthreads();
        {
            const int nx = kc + C::STAGES - 1;
            if (nx < C::NKC) stage_chunk<C, T>(act + (nx % C::SLOTS) * C::SLOT, nx, r0, R, src);
            tf_common::cp_commit();
        }
        const char* slot = act + (kc % C::SLOTS) * C::SLOT;
#pragma unroll
        for (int bb = 0; bb < BPC; ++bb) {
            const int b = kc * BPC + bb;  // global k32 block
            uint4 cur[C::MT][2];
#pragma unroll
            for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
                for (int j = 0; j < 2; ++j) cur[mt][j] = a[b % C::D][mt][j];
            if (b + C::D < C::CH) {
#pragma unroll
                for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
                    for (int j = 0; j < 2; ++j) a[b % C::D][mt][j] = ldw<C::STREAM>(wp[mt][j] + BSTEP * (b + C::D), pol);
            }
#pragma unroll
            for (int p = 0; p < C::PLANES; ++p) {
                if constexpr (C::PLANES == 3) {
                    if (p == 1) {
#pragma unroll
                        for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
                            for (int j = 0; j < 2; ++j) {
                                cur[mt][j].x = zero_nf_bf16x2(cur[mt][j].x);
                                cur[mt][j].y = zero_nf_bf16x2(cur[mt][j].y);
                                cur[mt][j].z = zero_nf_bf16x2(cur[mt][j].z);
                                cur[mt][j].w = zero_nf_bf16x2(cur[mt][j].w);
                            }
                    }
                }
#pragma unroll
                for (int nt = 0; nt < C::NT; ++nt) {
                    const uint4 x =
                        *reinterpret_cast<const uint4*>(slot + p * C::PLANE + act_off<C>(nt * 8 + g, bb * 4 + q));
#pragma unroll
                    for (int s = 0; s < 2; ++s)
#pragma unroll
                        for (int mt = 0; mt < C::MT; ++mt)
                            mma_bf16(acc[mt][nt], word(cur[mt][0], 2 * s), word(cur[mt][1], 2 * s),
                                     word(cur[mt][0], 2 * s + 1), word(cur[mt][1], 2 * s + 1), word(x, 2 * s),
                                     word(x, 2 * s + 1));
                }
            }
        }
    }
    tf_common::cp_wait<0>();
    qmm_frag::grid_launch();
}

// Stage the warp's fp32 results of n8 tile nt as [8 rows][CW cols] (row stride EPS) in its epilogue area.
template <class C>
__device__ __forceinline__ void stage_tile(float* ep, const float (&acc)[C::MT][C::NT][4], int nt) {
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int mt = 0; mt < C::MT; ++mt) {
        ep[(2 * q) * C::EPS + mt * 16 + g] = acc[mt][nt][0];
        ep[(2 * q + 1) * C::EPS + mt * 16 + g] = acc[mt][nt][1];
        ep[(2 * q) * C::EPS + mt * 16 + g + 8] = acc[mt][nt][2];
        ep[(2 * q + 1) * C::EPS + mt * 16 + g + 8] = acc[mt][nt][3];
    }
}

// GPT-J pair rotation `(q * cos) + (rotate_half(q) * sin)` matching `apply_rotary_pos_emb`.
__device__ __forceinline__ void rope_pair(float x0, float x1, float c, float s, float& y0, float& y1) {
    y0 = __fadd_rn(__fmul_rn(x0, c), __fmul_rn(-x1, s));
    y1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s));
}

// Absorb `QB [R, ldq]` through `W_UK_T [Hl, 512, 192]` into `QA [R, Hl, 576]` (plus optional `qlo` lo plane).
template <class C, bool OUT32, bool CS2 = false>
__device__ __forceinline__ void absorb_body(const bf16* __restrict__ QB, int ldq, int R, int Hl,
                                            const bf16* __restrict__ WUKT, const void* __restrict__ CSv,
                                            const int* __restrict__ CSI, void* __restrict__ QAv, long long qlo = 0) {
    static_assert(C::K == ND, "absorb reduces over the 192 nope columns");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    char* act = reinterpret_cast<char*>(smem_raw);
    float* epi = reinterpret_cast<float*>(smem_raw + C::ACT) + (threadIdx.x >> 5) * 8 * C::EPS;
    constexpr int NCB = KL / C::BM;
    const int cb = blockIdx.x % NCB, hr = blockIdx.x / NCB, h = hr % Hl, rb = hr / Hl;
    const int r0 = rb * C::BN, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0w = cb * C::BM + warp * C::CW;
    const bf16* W = WUKT + static_cast<size_t>(h) * KL * ND;
    auto src = [&](int r) { return QB + static_cast<size_t>(r) * ldq + h * QHS; };

    // Column block 0 writes the 64 roped columns (`qlo` zeroed when `CSv == nullptr`).
    auto rope = [&]() {
        if (cb == 0 && CSv == nullptr && qlo != 0 && !OUT32) {  // the lo plane's roped columns are 0 (in-tree rope is bf16)
            for (int e = threadIdx.x; e < C::BN * 8; e += C::THREADS) {
                const int rr = e >> 3, u = e & 7, r = r0 + rr;
                if (r >= R) continue;
                const size_t o = (static_cast<size_t>(r) * Hl + h) * QW + KL + 8 * u;
                *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(QAv) + qlo + o) = make_uint4(0, 0, 0, 0);
            }
        }
        if (cb == 0 && CSv != nullptr) {
            for (int e = threadIdx.x; e < C::BN * 8; e += C::THREADS) {
                const int rr = e >> 3, u = e & 7, r = r0 + rr;
                if (r >= R) continue;
                const uint4 xq = ld_stream(QB + static_cast<size_t>(r) * ldq + h * QHS + ND + 8 * u);
                const int ci = CSI ? CSI[r] : r;
                float cc[4], ss[4];
                if constexpr (CS2) {
                    const float2* cs2 = reinterpret_cast<const float2*>(CSv) + static_cast<size_t>(ci) * 32 + 4 * u;
                    const float4 ab0 = *reinterpret_cast<const float4*>(cs2);
                    const float4 ab1 = *reinterpret_cast<const float4*>(cs2 + 2);
                    cc[0] = ab0.x; ss[0] = ab0.y; cc[1] = ab0.z; ss[1] = ab0.w;
                    cc[2] = ab1.x; ss[2] = ab1.y; cc[3] = ab1.z; ss[3] = ab1.w;
                } else {
                    const float* CS = reinterpret_cast<const float*>(CSv);
                    const float4 c4 = *reinterpret_cast<const float4*>(CS + static_cast<size_t>(ci) * 64 + 4 * u);
                    const float4 s4 = *reinterpret_cast<const float4*>(CS + static_cast<size_t>(ci) * 64 + 32 + 4 * u);
                    cc[0] = c4.x; cc[1] = c4.y; cc[2] = c4.z; cc[3] = c4.w;
                    ss[0] = s4.x; ss[1] = s4.y; ss[2] = s4.z; ss[3] = s4.w;
                }
                float y[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t w = word(xq, i);
                    rope_pair(bf_lo(w), bf_hi(w), cc[i], ss[i], y[2 * i], y[2 * i + 1]);
                }
                const size_t o = (static_cast<size_t>(r) * Hl + h) * QW + KL + 8 * u;
                if constexpr (OUT32) {
                    float* d = reinterpret_cast<float*>(QAv) + o;
                    reinterpret_cast<float4*>(d)[0] = make_float4(rbf(y[0]), rbf(y[1]), rbf(y[2]), rbf(y[3]));
                    reinterpret_cast<float4*>(d)[1] = make_float4(rbf(y[4]), rbf(y[5]), rbf(y[6]), rbf(y[7]));
                } else {
                    *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(QAv) + o) =
                        make_uint4(pack2(y[0], y[1]), pack2(y[2], y[3]), pack2(y[4], y[5]), pack2(y[6], y[7]));
                }
            }
        }
    };
    float acc[C::MT][C::NT][4];
    chain<C, bf16>(acc, W, c0w, r0, R, src, act, rope);

    constexpr int UPR = OUT32 ? C::CW / 4 : C::CW / 8;  // 16-byte output units a row of the warp's columns
#pragma unroll
    for (int nt = 0; nt < C::NT; ++nt) {
        if (r0 + nt * 8 >= R) break;
        stage_tile<C>(epi, acc, nt);
        __syncwarp();
        for (int e = lane; e < 8 * UPR; e += 32) {
            const int rr = e / UPR, u = e % UPR, r = r0 + nt * 8 + rr;
            if (r >= R) continue;
            const size_t o = (static_cast<size_t>(r) * Hl + h) * QW + c0w;
            if constexpr (OUT32) {
                const float4 v = *reinterpret_cast<const float4*>(epi + rr * C::EPS + u * 4);
                reinterpret_cast<float4*>(reinterpret_cast<float*>(QAv) + o)[u] = v;
            } else {
                const float* s = epi + rr * C::EPS + u * 8;
                const uint4 v = make_uint4(pack2(s[0], s[1]), pack2(s[2], s[3]), pack2(s[4], s[5]), pack2(s[6], s[7]));
                reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(QAv) + o)[u] = v;
                if (qlo != 0) {  // c1's exact lo plane: what each bf16 round left out, rounded
                    float l8[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) l8[i] = __fsub_rn(s[i], rbf(s[i]));
                    const uint4 lo = make_uint4(pack2(l8[0], l8[1]), pack2(l8[2], l8[3]), pack2(l8[4], l8[5]), pack2(l8[6], l8[7]));
                    reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(QAv) + qlo + o)[u] = lo;
                }
            }
        }
        __syncwarp();
    }
}

// Expand `LAT [R, Hl, 512]` through `W_UV [Hl, 256, 512]` and apply `bf16(o * sigmoid(GATE))` into `ATT`.
template <class C, typename LT>
__device__ __forceinline__ void expand_body(const LT* __restrict__ LAT, int R, int Hl, const bf16* __restrict__ WUV,
                                            const bf16* __restrict__ GATE, int ldg, bf16* __restrict__ ATT) {
    static_assert(C::K == KL, "expand reduces over the 512 latent columns");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    char* act = reinterpret_cast<char*>(smem_raw);
    float* epi = reinterpret_cast<float*>(smem_raw + C::ACT) + (threadIdx.x >> 5) * 8 * C::EPS;
    constexpr int NCB = VD / C::BM;
    const int cb = blockIdx.x % NCB, hr = blockIdx.x / NCB, h = hr % Hl, rb = hr / Hl;
    const int r0 = rb * C::BN, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0w = cb * C::BM + warp * C::CW;
    const bf16* W = WUV + static_cast<size_t>(h) * VD * KL;
    auto src = [&](int r) { return LAT + (static_cast<size_t>(r) * Hl + h) * KL; };

    // Decode-sized tiles prefetch gate operands while weights stream in; larger tiles load in epilogue.
    constexpr int UPR = C::CW / 8;
    constexpr bool PREG = C::NT <= 2 && 8 * UPR <= 32;
    uint4 gpre[PREG ? C::NT : 1];
    auto gate_prefetch = [&]() {
        if constexpr (PREG) {
#pragma unroll
            for (int nt = 0; nt < C::NT; ++nt) {
                const int rr = lane / UPR, u = lane % UPR, r = r0 + nt * 8 + rr;
                gpre[nt] = make_uint4(0, 0, 0, 0);
                if (lane < 8 * UPR && r < R) gpre[nt] = ld_stream(GATE + static_cast<size_t>(r) * ldg + h * VD + c0w + 8 * u);
            }
        }
    };
    float acc[C::MT][C::NT][4];
    chain<C, LT>(acc, W, c0w, r0, R, src, act, gate_prefetch);

#pragma unroll
    for (int nt = 0; nt < C::NT; ++nt) {
        if (r0 + nt * 8 >= R) break;
        stage_tile<C>(epi, acc, nt);
        __syncwarp();
        for (int e = lane; e < 8 * UPR; e += 32) {
            const int rr = e / UPR, u = e % UPR, r = r0 + nt * 8 + rr;
            if (r >= R) continue;
            const int col = h * VD + c0w + 8 * u;
            uint4 g8;
            if constexpr (PREG) g8 = gpre[nt];
            else g8 = ld_stream(GATE + static_cast<size_t>(r) * ldg + col);
            const float* s = epi + rr * C::EPS + u * 8;
            float y[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t w = word(g8, i);
                // single rounding, as the in-tree tf_hy4_mla_value: bf16(o * sigmoid(g)) with o and sigmoid in fp32
                const float ga = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bf_lo(w))));
                const float gb = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bf_hi(w))));
                y[2 * i] = __fmul_rn(s[2 * i], ga);
                y[2 * i + 1] = __fmul_rn(s[2 * i + 1], gb);
            }
            *reinterpret_cast<uint4*>(ATT + static_cast<size_t>(r) * Hl * VD + col) =
                make_uint4(pack2(y[0], y[1]), pack2(y[2], y[3]), pack2(y[4], y[5]), pack2(y[6], y[7]));
        }
        __syncwarp();
    }
}

}  // namespace hy4c2
