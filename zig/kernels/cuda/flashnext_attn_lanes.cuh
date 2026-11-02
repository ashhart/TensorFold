// One-launch lanes: LP [lane][4] cache addresses; LI positions less first rows, positions, rows, first rows, row lanes.
#pragma once

namespace fn_lanes {
constexpr int MAX = 32;
__device__ __forceinline__ const int* shift(const int* LI, int l) { return LI + l; }
__device__ __forceinline__ const int* pos(const int* LI, int l) { return LI + MAX + l; }
__device__ __forceinline__ int rows(const int* LI, int l) { return LI[2 * MAX + l]; }
__device__ __forceinline__ int row0(const int* LI, int l) { return LI[3 * MAX + l]; }
__device__ __forceinline__ int lane(const int* LI, int r) { return LI[4 * MAX + r]; }
__device__ __forceinline__ __nv_bfloat16* at(const unsigned long long* LP, int l, int k) {
    return reinterpret_cast<__nv_bfloat16*>(LP[4 * l + k]);
}
}  // namespace fn_lanes

// fn_attn_prep over a shared round's rows: each row's caches and position from its lane.
extern "C" __global__ void __launch_bounds__(256)
    fn_attn_prep_lanes(const __nv_bfloat16* __restrict__ P, const unsigned long long* __restrict__ LP,
                       const int* __restrict__ LI, const float* __restrict__ QW, const float* __restrict__ KW,
                       const float* __restrict__ IW, const float* __restrict__ INV, __nv_bfloat16* __restrict__ Q,
                       __nv_bfloat16* __restrict__ IQ, float eps, int PW, int NQ, int NKV, int NI, int IHD, int HALF) {
    const int l = fn_lanes::lane(LI, blockIdx.x);
    fn_attn_prep_body<false>(P, fn_lanes::shift(LI, l), QW, KW, IW, INV, Q, fn_lanes::at(LP, l, 0), fn_lanes::at(LP, l, 1),
                             IQ, fn_lanes::at(LP, l, 2), eps, PW, NQ, NKV, NI, IHD, HALF, nullptr, 0, 0, Pg{});
}

// fn_qsa_pool over a shared round's lanes, lane blockIdx.y: its rows' completed blocks of its own indexer keys.
extern "C" __global__ void __launch_bounds__(128)
    fn_qsa_pool_lanes(const unsigned long long* __restrict__ LP, const int* __restrict__ LI, const float* __restrict__ W,
                      const float* __restrict__ INV, float eps, int DI, int HALF, int RATIO) {
    const int l = blockIdx.y;
    fn_qsa_pool_body<false>(fn_lanes::at(LP, l, 2), fn_lanes::at(LP, l, 3), fn_lanes::pos(LI, l), W, INV, eps,
                            fn_lanes::rows(LI, l), DI, HALF, RATIO, nullptr, 0, 0, Pg{});
}

// fn_attn_chunks_ahead over a shared round's rows: each row's keys and values from its lane.
extern "C" __global__ void __launch_bounds__(128)
    fn_attn_chunks_lanes(const __nv_bfloat16* __restrict__ Q, const unsigned long long* __restrict__ LP,
                         const int* __restrict__ LI, float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                         const int* __restrict__ IDS, const int* __restrict__ NKR, const int* __restrict__ SPR, int H,
                         int HK, int G, int CH, int NCH, float SCALE, int IDW, int QSA) {
    const int l = fn_lanes::lane(LI, blockIdx.x);
    attn_chunks<2>(Q, fn_lanes::at(LP, l, 0), fn_lanes::at(LP, l, 1), fn_lanes::shift(LI, l), PO, PM, PL, IDS, NKR, SPR,
                   H, HK, G, CH, NCH, SCALE, IDW, QSA);
}

// fn_attn_merge over a shared round's rows: each row's position from its lane.
extern "C" __global__ void __launch_bounds__(256)
    fn_attn_merge_lanes(const float* __restrict__ PO, const float* __restrict__ PM, const float* __restrict__ PL,
                        const int* __restrict__ LI, __nv_bfloat16* __restrict__ OUT, const int* __restrict__ NKR,
                        const int* __restrict__ SPR, int H, int HK, int G, int CH, int NCH, int QSA) {
    fn_attn_merge_body(PO, PM, PL, fn_lanes::shift(LI, fn_lanes::lane(LI, blockIdx.x)), OUT, NKR, SPR, H, HK, G, CH, NCH, QSA);
}

// qsa_tile::scores<8, 4> over a shared round's lanes, lane blockIdx.z, on that lane's rows and pooled keys.
extern "C" __global__ void __launch_bounds__(256)
    fn_qsa_scores_lanes(const __nv_bfloat16* __restrict__ IQ, const unsigned long long* __restrict__ LP,
                        const int* __restrict__ LI, float* __restrict__ SC, int NB, int RATIO, int TOP) {
    const int l = blockIdx.z, rows = fn_lanes::rows(LI, l), r0 = fn_lanes::row0(LI, l);
    if (static_cast<int>(blockIdx.x) * 8 >= rows) return;
    qsa_tile::scores_body<8, 4, false>(IQ + static_cast<long long>(r0) * 4 * 128, fn_lanes::at(LP, l, 3),
                                       fn_lanes::pos(LI, l), SC + static_cast<long long>(r0) * NB, NB, RATIO, TOP, rows,
                                       Pg{});
}
