// EXL3 projections for decode rows (CUDA's exl3_block/linear.cu): the input rotation, then the trellis linear (mul1).

#include "ops/common.hpp"

namespace {

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f32x8 __attribute__((__vector_size__(8 * sizeof(float))));

constexpr float kHadScale = 0.08838834764831845f;  // 1 / sqrt(128), CUDA's HAD_SCALE

// An fp32 value the compiler cannot fold into the next conversion (fp16(a * b) would otherwise round once, not twice).
__device__ __forceinline__ float opaque(float x) {
    asm volatile("" : "+v"(x));
    return x;
}

// A pointer in the global address space, so its loads are not flat loads that an LDS wait would also wait for.
template <typename T>
__device__ __forceinline__ const __attribute__((address_space(1))) T* gl(const T* p) {
    return (const __attribute__((address_space(1))) T*)p;
}

// The unscaled 128-point Walsh-Hadamard transform, 4 values a lane (4 lane + j), in CUDA's fwht128 butterfly order.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// E(p): one past the last stream bit of value p's 16-bit window (exl3_format.zig streamEnds).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// Lane l (column c = l % 16, k half h = l / 16) decodes rows 8 h + 2 q, + 1 of column c: its windows' words, shift.
template <int K2>
struct LaneWin {
    static constexpr int TW = 4 * K2;
    static constexpr int D = stream_end<K2>(1) - stream_end<K2>(0);
    int hi[4], lo[4], sh[4];
    __device__ __forceinline__ explicit LaneWin(int lane) {
        const int c = lane & 15, h = lane >> 4;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int p0 = 32 * (c & 7) + 8 * q + 4 * (c >> 3) + 2 * h;
            const int s = (stream_end<K2>(p0) - 16 + 128 * K2) % (128 * K2);
            const int o = s & 31;
            hi[q] = s >> 5;
            lo[q] = o ? (hi[q] + 1) % TW : hi[q];
            sh[q] = (32 - o) & 31;
        }
    }
};

// Two consecutive values from the 32 stream bits at the first window: fp16(0x6400 + bytesum(s * 0x83DCD12D)) fused.
template <int D>
__device__ __forceinline__ uint32_t decode_pair(uint32_t u) {
    const uint32_t w0 = u >> 16;
    const uint32_t w1 = __builtin_amdgcn_ubfe(u, 16 - D, 16);
    const uint32_t s0 = __builtin_amdgcn_sad_u8(w0 * 0x83DCD12Du, 0u, 0x64006400u);
    const uint32_t s = __builtin_amdgcn_sad_hi_u8(w1 * 0x83DCD12Du, 0u, s0);
    h2 hv = __builtin_bit_cast(h2, s);
    hv = __builtin_elementwise_fma(hv, __builtin_bit_cast(h2, 0x1eee1eeeu), __builtin_bit_cast(h2, 0xc931c931u));
    return __builtin_bit_cast(uint32_t, hv);
}

// This lane's 8 values of the tile at `tw`, as four fp16 pairs (rows 8 h + 2 q, 8 h + 2 q + 1 of column c).
template <int K2>
__device__ __forceinline__ void decode_own(const uint32_t* tw, const LaneWin<K2>& w, uint32_t (&own)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t u = __builtin_amdgcn_alignbit(tw[w.hi[q]], tw[w.lo[q]], w.sh[q]);
        own[q] = decode_pair<LaneWin<K2>::D>(u);
    }
}

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);
}

__device__ __forceinline__ float partner(float x) {
    const uint32_t u = __builtin_bit_cast(uint32_t, x);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xfedcba98u, false, false));
}

__device__ __forceinline__ void lds_fence() {
    __builtin_amdgcn_wave_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// One wave's k tiles [kt0, kt0 + n) of a 128-column strip against one row: a lane's dot2 chain, then the half-waves'.
template <int K2>
__device__ __forceinline__ void strip_1row(const uint32_t* tiles, long long step, int kt0, int n, const _Float16* xrow,
                                           uint32_t* st, int lane, f32x8 (&acc)[8]) {
    constexpr int TW = 4 * K2, LW = 8 * TW / 32;
    if (n <= 0) return;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) s[t] = 0.f;
    uint32_t pf[LW];
    uint4 xa;
    {
        const uint32_t* p = tiles + (long long)kt0 * step;
#pragma unroll
        for (int i = 0; i < LW; ++i) pf[i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
        xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * kt0 + 8 * h));
    }
    for (int j = 0; j < n; ++j) {
#pragma unroll
        for (int i = 0; i < LW; ++i) st[i * 32 + lane] = pf[i];
        const uint4 x = xa;
        if (j + 1 < n) {
            const uint32_t* p = tiles + (long long)(kt0 + j + 1) * step;
#pragma unroll
            for (int i = 0; i < LW; ++i) pf[i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
            xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * (kt0 + j + 1) + 8 * h));
        }
        lds_fence();
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t own[4];
            decode_own<K2>(st + t * TW, win, own);
            float a = dot2(own[0], x.x, s[t]);
            a = dot2(own[1], x.y, a);
            a = dot2(own[2], x.z, a);
            s[t] = dot2(own[3], x.w, a);
        }
        __builtin_amdgcn_wave_barrier();
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t][0] += s[t] + partner(s[t]);
}

// strip_1row's sums for up to RM rows, TT of the strip's tiles from t0, each tile decoded once for every row.
template <int K2, int RM, int TT, int t0>
__device__ __forceinline__ void strip_rows_part(const uint32_t* tiles, long long step, int kt0, int n,
                                                const _Float16* xlane, int R, uint32_t* st, uint4* xst, int lane,
                                                f32x8 (&acc)[8]) {
    constexpr int TW = 4 * K2, PW = TT * TW, RL = (PW + 31) / 32;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    const bool xlive = (lane >> 1) < R;
    const uint4* xl = reinterpret_cast<const uint4*>(xlane + 8 * (lane & 1));
    const uint32_t* tp = tiles + t0 * TW;
    float s[RM][TT];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int t = 0; t < TT; ++t) s[r][t] = 0.f;
    uint32_t pf[RL];
    uint4 xp = make_uint4(0u, 0u, 0u, 0u);
    auto fetch = [&](int kt) {
        const uint32_t* p = tp + (long long)kt * step;
#pragma unroll
        for (int i = 0; i < RL; ++i) {
            const int wi = i * 32 + lane;
            pf[i] = wi < PW ? __builtin_nontemporal_load(gl(p) + wi) : 0u;
        }
        if (xlive) xp = *gl(xl + 2 * kt);
    };
    fetch(kt0);
    for (int j = 0; j < n; ++j) {
#pragma unroll
        for (int i = 0; i < RL; ++i) {
            const int wi = i * 32 + lane;
            if (wi < PW) st[wi] = pf[i];
        }
        xst[lane] = xp;
        if (j + 1 < n) fetch(kt0 + j + 1);
        lds_fence();
        uint32_t own[TT][4];
#pragma unroll
        for (int t = 0; t < TT; ++t) decode_own<K2>(st + t * TW, win, own[t]);
#pragma unroll
        for (int r = 0; r < RM; ++r) {
            if (r < R) {
                const uint4 x = xst[2 * r + h];
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    float a = dot2(own[t][0], x.x, s[r][t]);
                    a = dot2(own[t][1], x.y, a);
                    a = dot2(own[t][2], x.z, a);
                    s[r][t] = dot2(own[t][3], x.w, a);
                }
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
    // lane l holds row 2 i + h of each pair as WMMA's C layout does: its half's sum plus its partner's, as strip_1row
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (2 * i + 1 < RM) {
                const float s0 = s[2 * i][t], s1 = s[2 * i + 1][t];
                const float o0 = partner(s0), o1 = partner(s1);
                v[i] = h ? s1 + o1 : s0 + o0;
            } else {
                v[i] = 0.f;
            }
        }
        acc[t0 + t] += (f32x8){v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    }
}

// Up to 8 rows take the strip's 8 tiles at once; 9 to 16 rows take them in two parts of 4 (the sums stay in registers).
template <int K2>
__device__ __forceinline__ void strip_rows(const uint32_t* tiles, long long step, int kt0, int n, const _Float16* xlane,
                                           int R, uint32_t* st, uint4* xst, int lane, f32x8 (&acc)[8]) {
    if (n <= 0) return;
    if (R <= 8) {
        strip_rows_part<K2, 8, 8, 0>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
    } else {
        strip_rows_part<K2, 16, 4, 0>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
        __builtin_amdgcn_wave_barrier();
        strip_rows_part<K2, 16, 4, 4>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
    }
}

// y = (xh @ W_q) H / sqrt(128) * svh for a 128-column block: wave w sums k tiles w K/16/WK on; rows in passes of 16.
template <int K2, int WK>
__device__ __forceinline__ void linear_body(const __half* xh_, const uint32_t* T, long long stride_k,
                                            long long stride_nb, const __half* svh, void* y, int y_kind, int M, int K,
                                            int N) {
    constexpr int TW = 4 * K2;
    extern __shared__ __attribute__((aligned(16))) float red[];  // WK * min(M, 8) * 128 floats
    __shared__ __attribute__((aligned(16))) uint32_t stage[WK][8 * TW];
    const _Float16* xh = reinterpret_cast<const _Float16*>(xh_);
    const int RH = M < 8 ? M : 8;
    const int nb = blockIdx.x, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per_wave = (K >> 4) / WK, kt0 = wave * per_wave;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col = nb * 128 + 4 * lane;
    for (int m0 = 0; m0 < M; m0 += 16) {
        const int R = min(16, M - m0);
        f32x8 acc[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[t] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        // a row's sums are the same bits whatever rows share its pass; the rows' inputs stage in the wave's red floats
        if (R == 1) {
            strip_1row<K2>(tiles, stride_k, kt0, per_wave, xh + (size_t)m0 * K, stage[wave], lane, acc);
        } else {
            const int ar = m0 + min(lane >> 1, R - 1);
            strip_rows<K2>(tiles, stride_k, kt0, per_wave, xh + (size_t)ar * K, R, stage[wave],
                           reinterpret_cast<uint4*>(red + wave * 128), lane, acc);
        }
#pragma unroll
        for (int hp = 0; hp < 2; ++hp) {
            const int rlo = 8 * hp;
            if (rlo >= R) break;
            const int rn = min(R - rlo, 8);
            __syncthreads();
#pragma unroll
            for (int t = 0; t < 8; ++t)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int rr = 2 * i + (lane >> 4);
                    if (rr < RH) red[(wave * RH + rr) * 128 + t * 16 + (lane & 15)] = acc[t][4 * hp + i];
                }
            __syncthreads();
            // the waves' sums in wave order, then the output rotation and svh as CUDA's finish does them
            for (int r = wave; r < rn; r += WK) {
                const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                float v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int w = 1; w < WK; ++w) {
                    const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                    v[0] += q.x;
                    v[1] += q.y;
                    v[2] += q.z;
                    v[3] += q.w;
                }
                fwht128(v, lane);
                const long long row = (long long)(m0 + rlo + r) * N + col;
#pragma unroll
                for (int j = 0; j < 4; ++j) store(y, y_kind, row + j, v[j] * kHadScale * __half2float(svh[col + j]));
            }
        }
        __syncthreads();
    }
}

}  // namespace

// xh = fp16((x * suh) H / sqrt(128)) a 128-wide block a wave, four a workgroup, a row a grid row (CUDA's rot_in).
extern "C" __global__ void __launch_bounds__(128)
tf_exl3_rot_in(const void* x, int x_kind, const __half* suh, __half* xh, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const long long k = blk * 128 + 4 * lane, at = (long long)blockIdx.y * K + k;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = load(x, x_kind, at + j) * __half2float(suh[k + j]);
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) xh[at + j] = __float2half_rn(opaque(v[j] * kHadScale));
}

#define TF_EXL3_LINEAR(K2_, WK_)                                                                                       \
    extern "C" __global__ void __launch_bounds__(WK_ * 32)                                                             \
    tf_exl3_linear_k##K2_##_w##WK_(const __half* xh, const uint32_t* T, long long stride_k, long long stride_nb,       \
                                   const __half* svh, void* y, int y_kind, int M, int K, int N) {                      \
        linear_body<K2_, WK_>(xh, T, stride_k, stride_nb, svh, y, y_kind, M, K, N);                                    \
    }
TF_EXL3_LINEAR(4, 4)
TF_EXL3_LINEAR(4, 8)
TF_EXL3_LINEAR(6, 4)
TF_EXL3_LINEAR(6, 8)
TF_EXL3_LINEAR(8, 4)
TF_EXL3_LINEAR(8, 8)
TF_EXL3_LINEAR(10, 4)
TF_EXL3_LINEAR(10, 8)
TF_EXL3_LINEAR(12, 4)
TF_EXL3_LINEAR(12, 8)
TF_EXL3_LINEAR(16, 4)
TF_EXL3_LINEAR(16, 8)
#undef TF_EXL3_LINEAR
