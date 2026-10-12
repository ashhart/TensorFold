// x3ld with gateup_epilogue / down_combine fused in its tail (TF_DSV41_X3LD_EPI=1, default off): 2 launches, same bits.

// Preconditions (host binding checks): NT 8, gate/up mats 2, down mats 1, SK 1; down skips rows with no live slot.

#include "x3ld.cu"
#include "exl3_experts.cu"

namespace dsv41_x3ld_epi {

using dsv41_x3ld::ld_tiles;
using dsv41_x3ld::W;
using tf_exl3_experts::bf16r;
using tf_exl3_experts::fwht128;
using tf_exl3_experts::HAD_SCALE;
using tf_exl3_experts::store_out;

// The epilogue operands (one struct by value; the unused ones 0).
struct Epi {
    const int* pick;      // int32 [rows, slots]
    const half* sv0;      // gate/up: svh_g [E, N]; down: svh_d [E, N]
    const half* sv1;      // gate/up: svh_u [E, N]
    const half* sd;       // gate/up: suh_d [E, N]
    half* xd;             // gate/up: Xd fp16 [P, N]
    float* y;             // down: y fp32 [P, N]
    const float* wts;     // down: [rows, slots]
    void* out;            // down: [rows, N] fp32 or bf16
    int* ticket;          // int32, zero between launches: the last arriver resets its word (threadFenceReduction)
    int E;
    float limit;
    int act_mode;
};

// gateup_epilogue_kernel's body for program (p, blk) on one warp (Z read through L2: other CTAs wrote it).
__device__ __forceinline__ void gateup_block(const float* __restrict__ Z, int e, const half* __restrict__ svh_g,
                                             const half* __restrict__ svh_u, const half* __restrict__ suh_d,
                                             half* __restrict__ xd, int p, int blk, int lane, int P, int N, int SK,
                                             float limit, int act_mode) {
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
            su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// MODE 0: gate/up + gateup_epilogue; 1: down + down_combine (OutT the combine's output).
template <int CB, int NT, int PD, int LO, int HI, int MODE, typename OutT>
__device__ __forceinline__ void epi_body(const half* __restrict__ X0, const half* __restrict__ X1,
                                         const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
                                         const int* __restrict__ K2_0, const int* __restrict__ K2_1,
                                         const int* __restrict__ uids, const int* __restrict__ ucount,
                                         const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P,
                                         int SK, int maxm, int slots, const Epi& a) {
    static_assert(NT == 8, "a CTA's columns are one 128-column Hadamard block");
    // ---- ld_kernel<CB, NT, PD, LO, HI, 0>, verbatim but for the Z store in MODE 1 ----
    const int u = blockIdx.x;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    const int cnt = __ldcg(ucount);
    const int e = __ldcg(uids + u);
    const int m = mtile * 16 + threadIdx.x;
    const int code = threadIdx.x < 16 && m < maxm ? __ldcg(members + u * maxm + m) : -1;
    const int first = __ldcg(members + u * maxm + mtile * 16);
    if (u >= cnt || first < 0) return;

    __shared__ int rows_sh[16];
    __shared__ int last_sh[16];
    __shared__ __align__(16) float red[W][16][NT * 16];
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;

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
#define DSV41_X3LD_EPI_CASE(K2_)                                                                                  \
    case K2_:                                                                                                    \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                    \
            ld_tiles<CB, K2_, NT, PD, 0>(T, NTILES, kt0, per_warp, nt0, X, K, rows_sh, stage, lane, acc);       \
        else                                                                                                     \
            __trap();                                                                                            \
        break;
        DSV41_X3LD_EPI_CASE(2)
        DSV41_X3LD_EPI_CASE(3)
        DSV41_X3LD_EPI_CASE(4)
        DSV41_X3LD_EPI_CASE(5)
        DSV41_X3LD_EPI_CASE(6)
        DSV41_X3LD_EPI_CASE(7)
        DSV41_X3LD_EPI_CASE(8)
        DSV41_X3LD_EPI_CASE(9)
        DSV41_X3LD_EPI_CASE(10)
        DSV41_X3LD_EPI_CASE(11)
        DSV41_X3LD_EPI_CASE(12)
        DSV41_X3LD_EPI_CASE(13)
        DSV41_X3LD_EPI_CASE(14)
        DSV41_X3LD_EPI_CASE(15)
        DSV41_X3LD_EPI_CASE(16)
#undef DSV41_X3LD_EPI_CASE
        default:
            __trap();
    }

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

    if constexpr (MODE == 0) {
        // ---- Z as ld_kernel stores it ----
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        // ---- the last of the mats x SK CTAs of (u, tile, block): gateup_epilogue for the tile's rows ----
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            int* tk = a.ticket + ((size_t)u * MT + mtile) * gridDim.y + blockIdx.y;
            const int arrivals = gridDim.z / MT;                     // mats x SK
            const int last = atomicAdd(tk, 1) == arrivals - 1;
            if (last) *tk = 0;                                       // every arrival is in: back to zero
            last_sh[0] = last;
        }
        __syncthreads();
        if (!last_sh[0]) return;
        __threadfence();
        for (int row = warp; row < 16; row += W) {
            const int p = rows_sh[row];
            if (p < 0) continue;
            const int ep = a.pick[p];
            if (ep < 0 || ep >= a.E) continue;                       // gateup_epilogue_kernel's exit
            gateup_block(Z, ep, a.sv0, a.sv1, a.sd, a.xd, p, blockIdx.y, lane, P, N, SK, a.limit, a.act_mode);
        }
    } else {
        // ---- down_combine's per-slot part from the complete Z in shared memory (SK 1) ----
        const int n = blockIdx.y * 128 + 4 * lane;
        for (int row = warp; row < 16; row += W) {
            const int p = rows_sh[row];
            if (p < 0) continue;
            const int ep = a.pick[p];
            if (ep < 0 || ep >= a.E) continue;                       // never a member (down_combine reads its y)
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float z = red[0][row][4 * lane + j];                 // ld_kernel's Z value
#pragma unroll
                for (int w = 1; w < W; ++w) z += red[w][row][4 * lane + j];
                float s = 0.f;                                       // down_combine: the splits from 0.f (SK 1)
                s += z;
                v[j] = s;
            }
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a.y[(size_t)p * N + n + j] = v[j] * HAD_SCALE * __half2float(a.sv0[(size_t)ep * N + n + j]);
        }
        // ---- a ticket per (row, block): the arrival of the row's last live slot combines it ----
        __threadfence();
        __syncthreads();
        if (threadIdx.x < 16) {
            int last = 0;
            const int p = rows_sh[threadIdx.x];
            if (p >= 0 && a.pick[p] >= 0 && a.pick[p] < a.E) {
                const int r = p / slots;
                int live = 0;
                for (int q = 0; q < slots; ++q) {
                    const int eq = a.pick[r * slots + q];
                    live += eq >= 0 && eq < a.E;
                }
                int* tk = a.ticket + (size_t)r * gridDim.y + blockIdx.y;
                last = atomicAdd(tk, 1) == live - 1;
                if (last) *tk = 0;
            }
            last_sh[threadIdx.x] = last;
        }
        __syncthreads();
        bool fenced = false;
        OutT* out = reinterpret_cast<OutT*>(a.out);
        const int d = blockIdx.y * 128 + threadIdx.x;                // a column a thread
        for (int i = 0; i < 16; ++i) {
            if (!last_sh[i]) continue;                               // CTA-uniform
            if (!fenced) {
                __threadfence();
                fenced = true;
            }
            const int r = rows_sh[i] / slots;
            float acc1 = 0.f;
            for (int q = 0; q < slots; ++q)
                acc1 = fmaf(a.wts[r * slots + q], __ldcg(a.y + ((size_t)r * slots + q) * N + d), acc1);
            store_out(out + (size_t)r * N + d, acc1);
        }
    }
}

}  // namespace dsv41_x3ld_epi

// Instances the binding names (extern "C"): NT 8, PD 1 / 2, x3ld's (lo, hi) ranges; gate/up, down fp32 / bf16 out.
#define DSV41_X3LD_EPI_ARGS                                                                                        \
    const half *X0, const half *X1, const int64_t *TP0, const int64_t *TP1, const int *K2_0, const int *K2_1,     \
        const int *uids, const int *ucount, const int *members, float *Z, int K, int N, int P, int SK, int maxm,  \
        int slots, const dsv41_x3ld_epi::Epi a
#define DSV41_X3LD_EPI_PASS X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots, a
#define DSV41_X3LD_EPI_INST(PD, LO, HI)                                                                             \
    extern "C" __global__ void __launch_bounds__(128, 3) dsv41_x3ld_epi_gu_##PD##_##LO##_##HI(DSV41_X3LD_EPI_ARGS) { \
        dsv41_x3ld_epi::epi_body<2, 8, PD, LO, HI, 0, float>(DSV41_X3LD_EPI_PASS);                               \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128, 3) dsv41_x3ld_epi_dn_##PD##_##LO##_##HI(DSV41_X3LD_EPI_ARGS) { \
        dsv41_x3ld_epi::epi_body<2, 8, PD, LO, HI, 1, float>(DSV41_X3LD_EPI_PASS);                               \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128, 3) dsv41_x3ld_epi_dnb_##PD##_##LO##_##HI(DSV41_X3LD_EPI_ARGS) { \
        dsv41_x3ld_epi::epi_body<2, 8, PD, LO, HI, 1, __nv_bfloat16>(DSV41_X3LD_EPI_PASS);                       \
    }
DSV41_X3LD_EPI_INST(1, 8, 8)
DSV41_X3LD_EPI_INST(1, 2, 10)
DSV41_X3LD_EPI_INST(1, 2, 12)
DSV41_X3LD_EPI_INST(2, 8, 8)
DSV41_X3LD_EPI_INST(2, 2, 10)
DSV41_X3LD_EPI_INST(2, 2, 12)
