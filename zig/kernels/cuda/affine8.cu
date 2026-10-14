// Ours: MLX affine codes of 2, 3, 5 and 6 bits on tiles of their width, 7-8 on 8-bit tiles; same MMAs, no rounding.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace tf_affine8 {

using namespace qmm_frag;

// Code i of an MLX row packed `bits` wide, low code first, in little-endian words (a code may cross a word).
__device__ __forceinline__ uint32_t code_at(const uint32_t* __restrict__ row, long long i, int bits) {
    const long long bit = i * bits;
    const int sh = static_cast<int>(bit & 31);
    const long long wd = bit >> 5;
    uint32_t v = row[wd] >> sh;
    if (sh + bits > 32) v |= row[wd + 1] << (32 - sh);
    return v & ((1u << bits) - 1u);
}

// Bytes at s and s + 8 of w as a bf16 pair (the first low): exact, as every byte is a bf16 integer.
__device__ __forceinline__ uint32_t byte_pair(uint32_t w, int s) {
    const float a = __uint_as_float(0x4B000000u | ((w >> s) & 0xFFu)) - 8388608.0f;
    const float b = __uint_as_float(0x4B000000u | ((w >> (s + 8)) & 0xFFu)) - 8388608.0f;
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Codes c0 (low half) and c1 (high half) of up to 6 bits as a bf16 pair: 0x4300 | c is 128 + c, minus 128 is c.
__device__ __forceinline__ uint32_t code_pair(uint32_t r) {
    r |= 0x43004300u;
    const __nv_bfloat162 h = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&r), __floats2bfloat162_rn(128.0f, 128.0f));
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Pair i of a lane's group in a BITS-wide tile: low halves hold first codes, high halves second, 16 codes a stream.
template <int BITS>
__device__ __forceinline__ uint32_t pair_at(const uint32_t (&w)[BITS], int i) {
    constexpr uint32_t M = (1u << BITS) - 1u;
    const int o = i * BITS, h = o >> 4, sh = o & 15;
    if (sh + BITS <= 16) return code_pair((w[h] >> sh) & (M | M << 16));
    const uint32_t lo = __byte_perm(w[h], w[h + 1], 0x5410), hi = __byte_perm(w[h], w[h + 1], 0x7632);
    return code_pair(((lo >> sh) & M) | ((hi >> sh) & M) << 16);
}

struct Part {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    void* out;
    int n, npad, sk, tiles;
};

// lane_gemv's tile with BITS bits a weight: 16 rows, BN columns over four warps, one 64-group a stage.
template <int GS, int BN, int STAGES, int BITS = 8>
struct Tile {
    static constexpr int THREADS = 128;
    static constexpr int NT = BN / 4 / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = 16 * ROW;
    static constexpr int W = BN * GS * BITS / 8;
    static constexpr int S = BN * 2;
    static constexpr int XS = 16 * 4;
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int SMEM = STAGES * STAGE;
};

// lane_gemv's gemv_items on 8-bit or narrow tiles; every width runs acc = x_sum * b + (d * s + acc) a group.
template <int GS, int BN, int STAGES, int BITS, typename Map, typename After>
__device__ __forceinline__ void items(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M, int K,
                                      int ldx, int count, Map map, float (&acc)[Tile<GS, BN, STAGES, BITS>::NT][4], After after) {
    using T = Tile<GS, BN, STAGES, BITS>;
    static_assert(BITS == 8 || (GS == 64 && BN == 64 && BITS >= 2 && BITS <= 6), "narrow tiles are 64 by 64, 2 to 6 bits");
    constexpr int NT = T::NT;
    constexpr int KW = GS / 16;  // words a lane a group: one per k16 step
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(16, M);
    constexpr int TILE_BYTES = 64 * GS * BITS / 8;
    constexpr int XC = (16 * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    const __nv_bfloat16* xsrc[XC];
    size_t woff[WC];
    int soff[SC], xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC], sbias[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < 16 * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        woff[j] = static_cast<size_t>(t) * KG * TILE_BYTES + off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        sbias[j] = which;
        soff[j] = off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(min(tid, rows - 1)) * KG;
    auto load_x = [&](int s, int i) {
        const int g = map(i).x;
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int i) {
        const int2 at = map(i);
        const int g = at.x, n0 = at.y;
        unsigned char* p = stage(s);
        const unsigned char* wb = reinterpret_cast<const unsigned char*>(P.w) + static_cast<size_t>(n0 / 64) * KG * TILE_BYTES + (n0 % 64) * GS * BITS / 8;
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wb + woff[j] + static_cast<size_t>(g) * TILE_BYTES);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], (sbias[j] ? P.biases : P.scales) + n0 + soff[j] + static_cast<size_t>(g) * P.npad);
    };

    for (int c = tid; c < 16 * T::CHUNKS; c += T::THREADS)
        if (c / T::CHUNKS >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < count) load_w(s, s);
        commit();
    }
    grid_wait();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < count) load_x(s, s);
        commit();
    }
    grid_launch();
    for (int it = 0; it < count; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < count) {
            load_x(next % STAGES, next);
            load_w(next % STAGES, next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        constexpr int LW = BITS == 8 ? NT * KW : BITS;  // a lane's words a group
        uint32_t lw[LW];
        if constexpr (BITS == 8) {
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int v = 0; v < KW; ++v) lw[j * KW + v] = pw[((wn * NT + j) * 32 + lane) * KW + v];
        } else if constexpr (BITS % 2 == 0) {
#pragma unroll
            for (int v = 0; v < BITS; v += 2) {
                const uint2 two = *reinterpret_cast<const uint2*>(pw + (wn * 32 + lane) * BITS + v);
                lw[v] = two.x;
                lw[v + 1] = two.y;
            }
        } else {
#pragma unroll
            for (int v = 0; v < BITS; ++v) lw[v] = pw[(wn * 32 + lane) * BITS + v];
        }
        float d[NT][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[4];
            const int r = (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
            ldmatrix4(a, p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t b0, b1;
                if constexpr (BITS == 8) {
                    b0 = byte_pair(lw[j * KW + kt], 0);
                    b1 = byte_pair(lw[j * KW + kt], 16);
                } else {
                    b0 = pair_at<BITS>(lw, (j * 4 + kt) * 2);
                    b1 = pair_at<BITS>(lw, (j * 4 + kt) * 2 + 1);
                }
                if (kt == 0) mma0(d[j], a, b0, b1);
                else mma(d[j], a, b0, b1);
            }
        }
        const int row = lane >> 2;
        const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = wn * (BN / 4) + j * 8 + (lane & 3) * 2;
            const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
            const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
            const float sv[2] = {__low2float(s2), __high2float(s2)};
            const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[j][e], sv[e & 1], acc[j][e]));
        }
        after(it);
    }
}

// Rows below M of the BN columns at n0 as bf16, from fp32 pairs got(j, h, row, col) in the accumulator's layout.
template <int BN, int NT, typename Got>
__device__ __forceinline__ void store(const Part& P, int M, int n0, Got got) {
    const int lane = threadIdx.x & 31, wn = threadIdx.x >> 5, row = lane >> 2, N = P.n;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = n0 + wn * (BN / 4) + j * 8 + (lane & 3) * 2, r = row + h * 8;
            if (r >= M) continue;
            const float2 v = got(j, h, r, col - n0);
            auto* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(r) * N + col;
            if (col + 1 < N && (N & 1) == 0) {
                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v.x, v.y);
            } else {
                if (col < N) dst[0] = __float2bfloat16_rn(v.x);
                if (col + 1 < N) dst[1] = __float2bfloat16_rn(v.y);
            }
        }
}

// lane_gemv's gemv_kernel: CTAs loop over column tiles, each tile's K slices summed in slice order.
template <int GS, int BN, int STAGES, int BITS>
__device__ __forceinline__ void gemv(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M, int K, int ldx) {
    constexpr int NT = Tile<GS, BN, STAGES, BITS>::NT;
    const int KG = K / GS, per = KG / P.sk;
    const int mine = P.tiles > (int)blockIdx.x ? (P.tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    auto tile_of_item = [&](int i) { return (int)blockIdx.x + (i / KG) * (int)gridDim.x; };
    float acc[NT][4], tot[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = tot[j][e] = 0.0f;
    items<GS, BN, STAGES, BITS>(x, xs, P, M, K, ldx, mine * KG, [&](int i) { return make_int2(i % KG, tile_of_item(i) * BN); }, acc, [&](int it) {
        const int g = it % KG;
        if ((g + 1) % per != 0) return;
        const bool first = g + 1 == per;
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                tot[j][e] = first ? acc[j][e] : tot[j][e] + acc[j][e];
                acc[j][e] = 0.0f;
            }
        if (g != KG - 1) return;
        store<BN, NT>(P, M, tile_of_item(it) * BN, [&](int j, int h, int, int) { return make_float2(tot[j][2 * h], tot[j][2 * h + 1]); });
    });
}

// lane_gemv's gemv_split: a CTA a tile's K slice, the tile's last CTA summing the slices in their order from the first.
template <int GS, int BN, int STAGES, int BITS>
__device__ __forceinline__ void split(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M, int K, int ldx,
                                      float* __restrict__ work, unsigned* __restrict__ tickets) {
    constexpr int NT = Tile<GS, BN, STAGES, BITS>::NT;
    __shared__ unsigned ticket;
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int per = K / GS / P.sk;
    const int tile = (int)blockIdx.x / P.sk, slice = (int)blockIdx.x % P.sk, g0 = slice * per;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
    items<GS, BN, STAGES, BITS>(x, xs, P, M, K, ldx, per, [&](int i) { return make_int2(g0 + i, tile * BN); }, acc, [](int) {});
    const int row = lane >> 2;
    float* part = work + (static_cast<size_t>(tile) * P.sk + slice) * 16 * BN;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = wn * (BN / 4) + j * 8 + (lane & 3) * 2, r = row + h * 8;
            if (r >= M) continue;
            part[r * BN + c] = acc[j][2 * h];
            part[r * BN + c + 1] = acc[j][2 * h + 1];
        }
    __threadfence();
    __syncthreads();
    if (tid == 0) ticket = atomicAdd(tickets + tile, 1u);
    __syncthreads();
    if (ticket != static_cast<unsigned>(P.sk - 1)) return;
    __threadfence();
    const float* base = work + static_cast<size_t>(tile) * P.sk * 16 * BN;
    store<BN, NT>(P, M, tile * BN, [&](int, int, int r, int c) {
        float t0 = __ldcg(base + r * BN + c), t1 = __ldcg(base + r * BN + c + 1);
        for (int s = 1; s < P.sk; ++s) {
            t0 = t0 + __ldcg(base + (s * 16 + r) * BN + c);
            t1 = t1 + __ldcg(base + (s * 16 + r) * BN + c + 1);
        }
        return make_float2(t0, t1);
    });
    if (tid == 0) tickets[tile] = 0;
}

}  // namespace tf_affine8

// Shared bytes: 6 stages for gemv, 4 for split (cuda/qlinear.zig's affine8 constants).
static_assert(tf_affine8::Tile<64, 64, 6>::SMEM == 38784, "gemv8 smem");
static_assert(tf_affine8::Tile<64, 64, 4>::SMEM == 25856, "split8 smem");

extern "C" __global__ void __launch_bounds__(128) tf_affine8_gemv(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                                                  const __grid_constant__ tf_affine8::Part P, int M, int K, int ldx) {
    tf_affine8::gemv<64, 64, 6, 8>(x, xs, P, M, K, ldx);
}

extern "C" __global__ void __launch_bounds__(128) tf_affine8_split(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,
                                                                   const __grid_constant__ tf_affine8::Part P, int M, int K, int ldx,
                                                                   float* __restrict__ work, unsigned* __restrict__ tickets) {
    tf_affine8::split<64, 64, 4, 8>(x, xs, P, M, K, ldx, work, tickets);
}

// The narrow tiles' gemv and split, one pair a width; shared bytes 14208 + 3072 * BITS and 9472 + 2048 * BITS.
#define TF_AFFINE_NARROW(B)                                                                                                         \
    static_assert(tf_affine8::Tile<64, 64, 6, B>::SMEM == 14208 + 3072 * B, "narrow gemv smem");                                   \
    static_assert(tf_affine8::Tile<64, 64, 4, B>::SMEM == 9472 + 2048 * B, "narrow split smem");                                   \
    extern "C" __global__ void __launch_bounds__(128) tf_affine##B##_gemv(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, \
                                                                       const __grid_constant__ tf_affine8::Part P, int M, int K, int ldx) { \
        tf_affine8::gemv<64, 64, 6, B>(x, xs, P, M, K, ldx);                                                                        \
    }                                                                                                                               \
    extern "C" __global__ void __launch_bounds__(128) tf_affine##B##_split(const __nv_bfloat16* __restrict__ x,                     \
                                                                        const float* __restrict__ xs,                               \
                                                                        const __grid_constant__ tf_affine8::Part P, int M, int K,   \
                                                                        int ldx, float* __restrict__ work,                          \
                                                                        unsigned* __restrict__ tickets) {                           \
        tf_affine8::split<64, 64, 4, B>(x, xs, P, M, K, ldx, work, tickets);                                                        \
    }

TF_AFFINE_NARROW(2)
TF_AFFINE_NARROW(3)
TF_AFFINE_NARROW(5)
TF_AFFINE_NARROW(6)
#undef TF_AFFINE_NARROW

// MLX words of 2, 3, 5 or 6 bits -> narrow tiles [npad/64][kg][4][32][bits], word v bits 16v..16v+15 of two streams.
extern "C" __global__ void tf_affinen_pack(const uint32_t* __restrict__ w, int bits, int n, int kw, int kg, uint32_t* __restrict__ out,
                                           long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int v = static_cast<int>(idx % bits);
    const long long rest = idx / bits;
    const int lane = static_cast<int>(rest & 31), wn = static_cast<int>((rest >> 5) & 3);
    const long long tg = rest >> 7;
    const int g = static_cast<int>(tg % kg);
    const long long t = tg / kg;
    const int c = lane & 3;
    uint32_t word = 0;
    for (int b = 0; b < 16; ++b) {
        const int sb = 16 * v + b, i = sb / bits, at = sb % bits;
        const int j = i >> 3, kt = (i >> 1) & 3, h = i & 1;
        const long long row = t * 64 + (wn * 2 + j) * 8 + (lane >> 2);
        if (row >= n) continue;
        const uint32_t* src = w + row * kw;
        const long long k = static_cast<long long>(g) * 64 + 16 * kt + 2 * c + 8 * h;
        word |= ((tf_affine8::code_at(src, k, bits) >> at) & 1u) << b;
        word |= ((tf_affine8::code_at(src, k + 1, bits) >> at) & 1u) << (16 + b);
    }
    out[idx] = word;
}

// MLX words of 2 to 4 bits -> lane_gemv's 4-bit tiles [npad/64][kg][8][32][2], tf_pack_dense's layout.
extern "C" __global__ void tf_affine8_pack4(const uint32_t* __restrict__ w, int bits, int n, int kw, int kg, uint32_t* __restrict__ out,
                                           long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int v = idx & 1, lane = (idx >> 1) & 31, j = (idx >> 6) & 7;
    const long long tg = idx >> 9;
    const int g = static_cast<int>(tg % kg);
    const long long t = tg / kg;
    const long long row = t * 64 + j * 8 + (lane >> 2);
    const int c = lane & 3;
    const int offs[8] = {0, 8, 16, 24, 1, 9, 17, 25};
    uint32_t word = 0;
    if (row < n) {
        const uint32_t* src = w + row * kw;
#pragma unroll
        for (int p = 0; p < 8; ++p) word |= tf_affine8::code_at(src, g * 64 + 32 * v + 2 * c + offs[p], bits) << (4 * p);
    }
    out[idx] = word;
}

// MLX words of 5 to 8 bits -> 8-bit tiles [npad/64][kg][8][32][4], word kt holding inputs k0+2c, +1, +8, +9.
extern "C" __global__ void tf_affine8_pack8(const uint32_t* __restrict__ w, int bits, int n, int kw, int kg, uint32_t* __restrict__ out,
                                           long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int kt = idx & 3, lane = (idx >> 2) & 31, j = (idx >> 7) & 7;
    const long long tg = idx >> 10;
    const int g = static_cast<int>(tg % kg);
    const long long t = tg / kg;
    const long long row = t * 64 + j * 8 + (lane >> 2);
    const int k0 = g * 64 + 16 * kt + 2 * (lane & 3);
    const int offs[4] = {0, 1, 8, 9};
    uint32_t word = 0;
    if (row < n) {
        const uint32_t* src = w + row * kw;
#pragma unroll
        for (int p = 0; p < 4; ++p) word |= tf_affine8::code_at(src, k0 + offs[p], bits) << (8 * p);
    }
    out[idx] = word;
}

// MLX rows of `bits` codes -> MLX rows of `wide` (4 or 8) bit codes, the same values: total output words.
extern "C" __global__ void tf_affine8_widen(const uint32_t* __restrict__ w, int bits, int k, uint32_t* __restrict__ out, int wide,
                                           long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int per = 32 / wide;  // codes a word out
    const long long row_words = static_cast<long long>(k) / per;
    const long long row = idx / row_words, at = idx % row_words;
    const uint32_t* src = w + row * (static_cast<long long>(k) * bits / 32);
    uint32_t word = 0;
    for (int c = 0; c < per; ++c) word |= tf_affine8::code_at(src, at * per + c, bits) << (c * wide);
    out[idx] = word;
}

namespace tf_affine8 {

// qmm_prefill.cu's prompt GEMM over AffineN's tiles: a 128 by 128 block on 2 by 4 warps, three stages, each weight
// rounded once to bf16 (fma2: s * q + b) and one fp32 chain over K. Its operands are the affine-4 prompt GEMM's for
// any code, so a code of up to 4 bits gives the same bits here as there.
template <int BITS>
struct Prefill {
    static constexpr int GS = 64, BM = 128, BN = 128, WM = 2, WN = 4, STAGES = 3;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int TILE_BYTES = 64 * GS * BITS / 8;  // one stored 64-column tile's group block
    static constexpr int W = BN / 64 * TILE_BYTES;
    static constexpr int S = BN * 2;
    static constexpr int STAGE = X + W + 2 * S;
    static constexpr int SMEM = STAGES * STAGE;
};

__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

template <int BITS>
__device__ __forceinline__ void prefill(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,
                                        const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
                                        __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {
    using T = Prefill<BITS>;
    constexpr int GS = T::GS, BN = T::BN, BM = T::BM, WN = T::WN, WM = T::WM, NT = T::NT, MT = T::MT, STAGES = T::STAGES;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8, m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (T::TILE_BYTES / 16), off = c % (T::TILE_BYTES / 16);
            const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
            if (n0 + t * 64 < npad) cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * T::TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    // this warp's n8 tiles are 4 wn to 4 wn + 3 of the block: stored tile wn / 2, its n8 tiles (wn % 2) * 4 + j
    const int t64 = wn / 2, sub = (wn % 2) * 4;
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X) + t64 * (T::TILE_BYTES / 4);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
        constexpr int SETS = BITS == 8 ? NT : 2;  // a lane's word sets: one an n8 tile (8-bit), one a decode warp's pair
        constexpr int LW = BITS == 8 ? GS / 16 : BITS;
        uint32_t lw[SETS][LW], sv[NT], bv[NT];
#pragma unroll
        for (int q = 0; q < SETS; ++q)
#pragma unroll
            for (int v = 0; v < LW; ++v) lw[q][v] = pw[(((BITS == 8 ? sub : sub / 2) + q) * 32 + lane) * LW + v];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[BN + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t q0, q1;
                if constexpr (BITS == 8) {
                    q0 = byte_pair(lw[j][kt], 0);
                    q1 = byte_pair(lw[j][kt], 16);
                } else {
                    q0 = pair_at<BITS>(lw[j / 2], ((j % 2) * 4 + kt) * 2);
                    q1 = pair_at<BITS>(lw[j / 2], ((j % 2) * 4 + kt) * 2 + 1);
                }
                const uint32_t b0 = fma2(q0, sv[j], bv[j]);
                const uint32_t b1 = fma2(q1, sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                } else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                }
            }
        }
}

}  // namespace tf_affine8

// The prompt GEMM a width: shared bytes 50688 + 3072 * BITS.
#define TF_AFFINE_PREFILL(B)                                                                                                    \
    static_assert(tf_affine8::Prefill<B>::SMEM == 50688 + 3072 * B, "prefill smem");                                           \
    extern "C" __global__ void __launch_bounds__(256)                                                                           \
        tf_affine##B##_prefill(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,                             \
                               const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,              \
                               __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {            \
        tf_affine8::prefill<B>(x, w, scales, biases, out, M, N, K, npad, ldx, group);                                           \
    }

TF_AFFINE_PREFILL(2)
TF_AFFINE_PREFILL(3)
TF_AFFINE_PREFILL(5)
TF_AFFINE_PREFILL(6)
TF_AFFINE_PREFILL(8)
#undef TF_AFFINE_PREFILL
