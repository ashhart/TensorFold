// Copy of src/tensorfold/families/deepseek_v41/cuda/x3ld.cu (blob 5cd25f2cdf3c): sass_check.sh



// DeepSeek-V4.1-Flash routed experts (TF_DSV41_EXPERT_LOADS=1): grouped EXL3 expert GEMV with a faster load path.

// Same bits as grouped_kernel<CB, NT, 4, PF, LO, HI> at any row count; data movement changes (prologue, ring, PDL).

// PROBE 3 (timing only, wrong results): no decode and no mma, the load path alone.

#include "experts_grouped.cuh"

namespace dsv41_x3ld {

using tf_exl3x::decode_tile;
using tf_exl3x::Fmt;
using tf_exl3x::LaneMap;
using tf_exl3x::mma16816;

constexpr int W = 4;   // warps a CTA: upstream's default_config, which fixes the K ranges

__device__ __forceinline__ uint4 ldg_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// load_pair's value (ok ? *(const uint32_t*)x : 0), the load pinned where it stands
__device__ __forceinline__ uint32_t ldg_pair(const half* x, bool ok) {
    uint32_t v = 0u;
    if (ok) asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(x));
    return v;
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

template <int K2, int NT>
struct Step {
    static constexpr int TW = Fmt<K2>::TW;           // words a tile
    static constexpr int LW = Fmt<K2>::LW;           // words a lane a tile (load_words)
    static constexpr int WORDS = NT * TW;            // a warp's k step, contiguous
    static constexpr int NV = (WORDS + 127) / 128;   // 16-byte loads a lane a step
};

// One warp's k tiles [kt0, kt0 + nkt) into acc (warp_tiles' arithmetic, loads PD ahead); all CTA threads call it.
template <int CB, int K2, int NT, int PD, int PROBE>
__device__ __forceinline__ void ld_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                         const half* __restrict__ X, int K, const int* rows_sh, uint32_t* stage,
                                         int lane, float (&acc)[NT][2][4]) {
    using S = Step<K2, NT>;
    const int g = lane >> 2, t = lane & 3;
    const size_t kstride = (size_t)NTILES * S::TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * S::TW;

    uint4 ring[PD][S::NV];
    auto issue_w = [&](int d, int s) {
        const uint32_t* src = tp + (size_t)s * kstride;
#pragma unroll
        for (int v = 0; v < S::NV; ++v) {
            const int off = 128 * v + 4 * lane;
            if (off < S::WORDS) ring[d][v] = ldg_v4(src + off);
        }
    };
#pragma unroll
    for (int d = 0; d < PD; ++d) issue_w(d, d);                 // PD <= nkt (host)

    pdl_wait();                                                  // X (and Z) belong to the kernels before
    __syncthreads();                                             // rows_sh
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
    uint32_t a_ring[PD][4];
    auto issue_a = [&](int d, int s) {
        const int k = (kt0 + s) * 16;
        a_ring[d][0] = ldg_pair(x0 + k, r0 >= 0);
        a_ring[d][1] = ldg_pair(x1 + k, r1 >= 0);
        a_ring[d][2] = ldg_pair(x0 + k + 8, r0 >= 0);
        a_ring[d][3] = ldg_pair(x1 + k + 8, r1 >= 0);
    };
#pragma unroll
    for (int d = 0; d < PD; ++d) issue_a(d, d);

    const LaneMap<K2> map(lane);
    for (int kb = 0; kb < nkt; kb += PD) {                       // nkt % PD == 0 (host)
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int s = kb + d;
            const bool more = s + PD < nkt;
            __syncwarp();                                        // every lane has read step s - 1 back
#pragma unroll
            for (int v = 0; v < S::NV; ++v) {
                const int off = 128 * v + 4 * lane;
                if (off < S::WORDS) *reinterpret_cast<uint4*>(stage + off) = ring[d][v];
            }
            if (more) issue_w(d, s + PD);                        // the slot's registers are free again
            __syncwarp();
            uint32_t w[NT][S::LW];
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int l = 0; l < S::LW; ++l)
                    w[i][l] = (S::TW % 32 == 0 || l * 32 + lane < S::TW) ? stage[i * S::TW + l * 32 + lane] : 0u;
            uint32_t a[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) a[j] = a_ring[d][j];
            if (more) issue_a(d, s + PD);
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                if constexpr (PROBE == 3) {
                    uint32_t x = a[0] ^ a[1] ^ a[2] ^ a[3];
#pragma unroll
                    for (int l = 0; l < S::LW; ++l) x ^= w[i][l];
                    acc[i][0][0] = __uint_as_float(__float_as_uint(acc[i][0][0]) ^ x);
                } else {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
    __syncwarp();                                                // the staging area becomes red[warp] again
}

// Program (u, n block, (mat * SK + split) * MT + member tile): grouped_kernel<CB, NT, 4, *, LO, HI>'s work item.
template <int CB, int NT, int PD, int LO, int HI, int PROBE>
__global__ void __launch_bounds__(W * 32, NT == 8 ? 3 : 4) ld_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    static_assert(PD >= 1 && PD <= 4, "PD 1-4");
    const int u = blockIdx.x;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    // round trip 1: none of these depends on another (uids / members are sized by the grid: always in bounds)
    const int cnt = __ldcg(ucount);
    const int e = __ldcg(uids + u);
    const int m = mtile * 16 + threadIdx.x;
    const int code = threadIdx.x < 16 && m < maxm ? __ldcg(members + u * maxm + m) : -1;
    const int first = __ldcg(members + u * maxm + mtile * 16);
    if (u >= cnt || first < 0) return;                           // grouped_kernel's exits (CTA-uniform)

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[W][16][NT * 16];
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;

    // round trip 2: the expert's trellis and width
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const half* X = mat ? X1 : X0;
    const int KT = K >> 4, NTILES = N >> 4;
    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;
    uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (k2) {
#define DSV41_X3LD_CASE(K2_)                                                                                      \
    case K2_:                                                                                                    \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                    \
            ld_tiles<CB, K2_, NT, PD, PROBE>(T, NTILES, kt0, per_warp, nt0, X, K, rows_sh, stage, lane, acc);   \
        else                                                                                                     \
            __trap();                                                                                            \
        break;
        DSV41_X3LD_CASE(2)
        DSV41_X3LD_CASE(3)
        DSV41_X3LD_CASE(4)
        DSV41_X3LD_CASE(5)
        DSV41_X3LD_CASE(6)
        DSV41_X3LD_CASE(7)
        DSV41_X3LD_CASE(8)
        DSV41_X3LD_CASE(9)
        DSV41_X3LD_CASE(10)
        DSV41_X3LD_CASE(11)
        DSV41_X3LD_CASE(12)
        DSV41_X3LD_CASE(13)
        DSV41_X3LD_CASE(14)
        DSV41_X3LD_CASE(15)
        DSV41_X3LD_CASE(16)
#undef DSV41_X3LD_CASE
        default:
            __trap();
    }

    // grouped_kernel's reduction, verbatim: the warps' partial sums through shared memory, added in warp order
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
    }
}

}  // namespace dsv41_x3ld

// The instantiations dsv41_x3ld_grouped_cuda dispatches: mul1, (nt, pd) x probe x the (lo, hi) ranges.
#define DSV41_X3LD_INST(NT, PD, LO, HI, PROBE) template __global__ void dsv41_x3ld::ld_kernel<2, NT, PD, LO, HI, PROBE>( \
    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
    const int*, float*, int, int, int, int, int, int);
DSV41_X3LD_INST(8, 1, 8, 8, 0)
DSV41_X3LD_INST(8, 1, 2, 10, 0)
DSV41_X3LD_INST(8, 1, 2, 12, 0)
DSV41_X3LD_INST(8, 2, 8, 8, 0)
DSV41_X3LD_INST(8, 2, 2, 10, 0)
DSV41_X3LD_INST(8, 2, 2, 12, 0)
DSV41_X3LD_INST(4, 2, 8, 8, 0)
DSV41_X3LD_INST(4, 2, 2, 10, 0)
DSV41_X3LD_INST(4, 2, 2, 12, 0)
DSV41_X3LD_INST(8, 1, 8, 8, 3)
DSV41_X3LD_INST(8, 1, 2, 10, 3)
DSV41_X3LD_INST(8, 1, 2, 12, 3)
DSV41_X3LD_INST(8, 2, 8, 8, 3)
DSV41_X3LD_INST(8, 2, 2, 10, 3)
DSV41_X3LD_INST(8, 2, 2, 12, 3)
