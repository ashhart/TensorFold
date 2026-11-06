// One grid of prompt_lean's FP4 tiles over the NVFP4 projections of one bf16 input.
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/qmm_frag.cuh"
#include "common/fp_pairs.cuh"

namespace tf_prompt_many {

using namespace qmm_frag;

constexpr int GS = 64;
constexpr int PARTS = 8;

enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, BF16 = 4 };

struct Part {
    const unsigned char* w;
    const uint8_t* bs;
    float scale;
    void* out;
    int n, npad, first;
};

struct Parts {
    Part p[PARTS];
    int count;
};

using tf_common::fp4pair;

using tf_common::mul2;

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

template <int MODE, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == BF16 ? BN * GS * 2 : BN * GS;
    static constexpr int S = MODE == FP4 ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;
    static constexpr int SMEM = STAGES * STAGE;
};

__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem(b)), "r"(n) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(smem(b)) : "memory");
}

// arrives on b once this thread's earlier cp.async copies have landed
__device__ __forceinline__ void mbar_copies(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(smem(b)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
#if __CUDA_ARCH__ >= 900
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
#else
        "mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n"
#endif
        "@!p bra WAIT_%=;\n}\n" ::"r"(smem(b)),
        "r"(parity) : "memory");
}

#if TF_FP_NATIVE
// fp4pair(w, 8q), fp4pair(w, 8q + 4) bit for bit: byte q of x holds nibbles (2q, 2q + 4), of y (2q + 1, 2q + 5)
__device__ __forceinline__ void e2m1_pairs(uint32_t w, int q, uint32_t& b0, uint32_t& b1) {
    const uint32_t lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
    const uint32_t x = lo | (lo >> 12), y = hi | (hi >> 12);
    b0 = tf_common::e2m1x2(q ? x >> 8 : x);
    b1 = tf_common::e2m1x2(q ? y >> 8 : y);
}
#else
// fp4pair(w, 8q), fp4pair(w, 8q + 4) bit for bit: |e2m1| by bf16 byte tables (0 ... 6), sign or-ed in (-0 kept)
__device__ __forceinline__ void e2m1_pairs(uint32_t w, int q, uint32_t& b0, uint32_t& b1) {
    const uint32_t v = q ? w >> 8 : w;
    const uint32_t sel = (v & 0x77u) | ((v >> 8) & 0x7700u);  // magnitudes of nibbles 2q, 2q + 1, 2q + 4, 2q + 5
    const uint32_t lo = prmt(0xC0800000u, 0xC0804000u, sel), hi = prmt(0x3F3F3F00u, 0x40404040u, sel);
    b0 = prmt(lo, hi, 0x6240u) | ((v << 12) & 0x80008000u);
    b1 = prmt(lo, hi, 0x7351u) | ((v << 8) & 0x80008000u);
}
#endif

template <int BM, int BN, int WM, int WN, int STAGES, bool F32>
__device__ __forceinline__ void lean_many_body(
        const __nv_bfloat16* __restrict__ x, const Parts parts, int M, int K, int ldx, int group) {
    using T = Tile<FP4, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ Parts sp;
    if (tid == 0) sp = parts;
    __syncthreads();
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    constexpr int TILE_BYTES = 64 * GS / 2;
    constexpr int XC = BM * T::CHUNKS, WC = T::W / 16, SC = T::S / 16;
    constexpr int XN = (XC + T::THREADS - 1) / T::THREADS, WNC = (WC + T::THREADS - 1) / T::THREADS;
    constexpr int SN = SC > 0 ? (SC + T::THREADS - 1) / T::THREADS : 1;
    constexpr int PER = SC > 0 ? T::S / (BN / 64) : 16;
    int tiles = 0;
    for (int i = 0; i < sp.count; ++i)
        tiles += (M + BM - 1) / BM * ((sp.p[i].n + BN - 1) / BN);
    auto locate = [&](int tile) {
        int q = 0;
        for (int i = 1; i < sp.count; ++i)
            if (tile >= sp.p[i].first) q = i;
        return q;
    };
    const __nv_bfloat16* xs[XN];
    uint32_t xd[XN];
    bool xin[XN], xon[XN];
    const unsigned char* wsrc[WNC];
    bool win[WNC];
    const uint8_t* ssrc[SN];
    bool sin[SN];
    auto plan = [&](int tile) {  // the copies of tile's group: part q's local tile
        const int q = locate(tile);
        const Part P = sp.p[q];
        const int2 at = tile_of(tile - P.first, M, P.n, BM, BN, group);
#pragma unroll
        for (int i = 0; i < XN; ++i) {
            const int c = tid + i * T::THREADS, r = min(c / T::CHUNKS, BM - 1), ch = c % T::CHUNKS;
            xs[i] = x + static_cast<size_t>(min(at.x + r, M - 1)) * ldx + ch * 8;
            xd[i] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
            xin[i] = at.x + r < M;
            xon[i] = c < XC;
        }
#pragma unroll
        for (int i = 0; i < WNC; ++i) {
            const int c = tid + i * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            wsrc[i] = P.w + static_cast<size_t>(at.y / 64 + t) * KG * TILE_BYTES + off * 16;
            win[i] = c < WC && at.y + t * 64 < P.npad;
        }
#pragma unroll
        for (int i = 0; i < SN; ++i) {
            const int c = tid + i * T::THREADS, t = c / (PER / 16), off = c % (PER / 16);
            ssrc[i] = P.bs + static_cast<size_t>(at.y / 64 + t) * KG * PER + off * 16;
            sin[i] = c < SC && at.y + t * 64 < P.npad;
        }
    };
    uint64_t* full = reinterpret_cast<uint64_t*>(buf + STAGES * T::STAGE);
    uint64_t* empty = full + STAGES;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // the copy stream: its count of issued stages, and the tile and group it copies next
    uint32_t issued = 0;
    int lt = blockIdx.x, lg = 0;
    auto load = [&]() {
        const int s = issued % STAGES;
        if (issued >= STAGES) mbar_wait(&empty[s], (issued / STAGES - 1) & 1);
        unsigned char* p = stage(s);
#pragma unroll
        for (int i = 0; i < XN; ++i)
            if (xon[i]) cp16z(p + xd[i], xs[i] + lg * GS, xin[i]);
#pragma unroll
        for (int i = 0; i < WNC; ++i)
            if (win[i]) cp16(p + T::X + (tid + i * T::THREADS) * 16, wsrc[i] + static_cast<size_t>(lg) * TILE_BYTES);
        if constexpr (SC > 0) {
#pragma unroll
            for (int i = 0; i < SN; ++i)
                if (sin[i]) cp16(p + T::X + T::W + (tid + i * T::THREADS) * 16, ssrc[i] + static_cast<size_t>(lg) * PER);
        }
        mbar_copies(&full[s]);
        ++issued;
        if (++lg == KG) {
            lg = 0;
            lt += gridDim.x;
            if (lt < tiles) plan(lt);
        }
    };

    if (blockIdx.x >= tiles) return;
    if (tid == 0)
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], T::THREADS);
            mbar_init(&empty[s], T::THREADS / 32);
        }
    __syncthreads();
    plan(lt);
    for (int s = 0; s < STAGES - 1 && lt < tiles; ++s) load();
    uint32_t used = 0;
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int q = locate(tile);
        const Part P = sp.p[q];
        const int2 at = tile_of(tile - P.first, M, P.n, BM, BN, group);
        const int m0 = at.x, n0 = at.y;
        float acc[T::MT][T::NT][4];
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
        for (int g = 0; g < KG; ++g, ++used) {
            mbar_wait(&full[used % STAGES], (used / STAGES) & 1);
            const unsigned char* p = stage(used % STAGES);
            const unsigned char* pw = p + T::X;
            const uint8_t* ps = p + T::X + T::W;
            uint4 wq[T::NT];
            uint32_t sv[T::NT][4];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int jj = wn * T::NT + j;
                const int col = (jj & 7) * 8 + (lane >> 2);
                const uint2 u = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(u.x, u.y, 0u, 0u);
                const uint32_t b4 = *reinterpret_cast<const uint32_t*>(ps + (jj >> 3) * 256 + col * 4);
                // fp8pair is lane-wise: two blocks a call, each half then copied to both (sv as prompt_lean's)
#if TF_FP_NATIVE
                const uint32_t s01 = tf_common::e4m3x2(b4), s23 = tf_common::e4m3x2(b4 >> 16);
#else
                const uint32_t s01 = tf_common::fp8pair_unit(b4, 0x7B807B80u), s23 = tf_common::fp8pair_unit(b4 >> 16, 0x7B807B80u);
#endif
                sv[j][0] = prmt(s01, 0u, 0x1010u);
                sv[j][1] = prmt(s01, 0u, 0x3232u);
                sv[j][2] = prmt(s23, 0u, 0x1010u);
                sv[j][3] = prmt(s23, 0u, 0x3232u);
            }
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    uint32_t b0, b1;
                    e2m1_pairs(kt < 2 ? wq[j].x : wq[j].y, kt & 1, b0, b1);
                    b0 = mul2(b0, sv[j][kt]);
                    b1 = mul2(b1, sv[j][kt]);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty[used % STAGES]);
            if (lt < tiles) load();
        }
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                    if (row >= M) continue;
                    const float v0 = acc[i][j][2 * h] * P.scale, v1 = acc[i][j][2 * h + 1] * P.scale;
                    if (F32) {
                        float* dst = reinterpret_cast<float*>(P.out) + static_cast<size_t>(row) * P.n + col;
                        if (col < P.n) dst[0] = v0;
                        if (col + 1 < P.n) dst[1] = v1;
                    } else {
                        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(row) * P.n + col;
                        if (col + 1 < P.n && (P.n & 1) == 0)
                            *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                        else {
                            if (col < P.n) dst[0] = __float2bfloat16_rn(v0);
                            if (col + 1 < P.n) dst[1] = __float2bfloat16_rn(v1);
                        }
                    }
                }
            }
    }
}

template <int BM, int BN, int WM, int WN, int STAGES>
struct LeanTile {
    using T = Tile<FP4, BM, BN, WM, WN, STAGES>;
    static constexpr int SMEM = T::SMEM + 2 * STAGES * 8;  // the stages, then their full and empty barriers
};

// prompt_lean's bits, its tiles spread over the parts: the copies of one tile's groups, then its mma chain
template <bool F32>
__device__ __forceinline__ void lean_many(
        const __nv_bfloat16* __restrict__ x, const Parts parts, int M, int K, int ldx, int group) {
    lean_many_body<128, 256, 1, 8, 3, F32>(x, parts, M, K, ldx, group);
}

}  // namespace tf_prompt_many

// The instances the Zig launcher loads by name: one grid, every part's own prompt_lean bits.
extern "C" __global__ void __launch_bounds__(256, 1) tf_prompt16_many(
        const __nv_bfloat16* __restrict__ x, const __grid_constant__ tf_prompt_many::Parts parts, int M, int K, int ldx, int group) {
    tf_prompt_many::lean_many<false>(x, parts, M, K, ldx, group);
}

extern "C" __global__ void __launch_bounds__(256, 1) tf_prompt16_many_f32(
        const __nv_bfloat16* __restrict__ x, const __grid_constant__ tf_prompt_many::Parts parts, int M, int K, int ldx, int group) {
    tf_prompt_many::lean_many<true>(x, parts, M, K, ldx, group);
}
