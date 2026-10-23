// TF_DSV41_LIN2: dense EXL3 linears at 17-32 rows in ONE weight pass, each decoded B fragment feeding two row tiles.

#include "linear.cu"
#include "dense3.cu"
#include <cooperative_groups.h>

namespace dsv41_lin2 {  // each row tile runs the original's k range, mma chain and finish: outputs are its bits

using namespace tf_exl3;

// -- dense3's lanes_linear_kernel, two row tiles a pass (M 17-32) ------------------------------------------------
template <int K2, int CB, int WK, int MINB>
__global__ void __launch_bounds__(WK * 32, MINB) lanes2_kernel(const dsv41_x3seg::LinTable tab, int y_dtype, int M) {
    using namespace dsv41_dense3;
    constexpr int G = group_steps(K2), GW = gwords<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats (the original's, RH = 8)
    __shared__ int last;
    const int RH = min(M, 8);

    dsv41_x3seg::LinSeg s = tab.s[0];
#pragma unroll
    for (int i = 1; i < dsv41_x3seg::MAXSEG; ++i)
        if (i < tab.n && (int)blockIdx.x >= tab.s[i].p0) s = tab.s[i];
    const half* __restrict__ xh = s.xh;
    const half* __restrict__ svh = s.svh;
    void* __restrict__ y = s.y;
    float* __restrict__ Z = s.Z;
    int* __restrict__ counters = s.counters;
    const int K = s.K, N = s.N, SK = s.SK, wk = s.wk, ld = s.ld;

    const int NB = N >> 7, q = (int)blockIdx.x - s.p0, nb = q % NB, split = q / NB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const bool on = warp < wk;
    const int per_warp = (K >> 4) / SK / wk;
    const int kt0 = split * (per_warp * wk) + warp * per_warp;
    const int steps = on ? per_warp : 0, groups = steps / G;
    const uint32_t* base = s.T + nb * s.stride_nb + (size_t)kt0 * s.stride_k + 4 * lane;
    const size_t gstride = (size_t)G * s.stride_k;
    const int col0 = nb * 128;

    float acc[2][8][2][4];
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[u][i][h][c] = 0.f;

    // each tile's walk rows (pass u: rows 16 u ..), clamped as the original's
    const half* x0[2];
    const half* x1[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int m0 = 16 * u, R = min(16, M - m0);
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        x0[u] = xh + (size_t)r0 * K;
        x1[u] = xh + (size_t)r1 * K;
    }
    uint32_t cur[GW], nxt[GW];
    if (groups > 0) load_group(base, cur);
    pdl_wait();                                               // weights above, xh / Z / counters below
    uint32_t a[2][4], an[2][4];
    if (steps > 0) {
        load_a(x0[0], x1[0], kt0, t, a[0]);
        load_a(x0[1], x1[1], kt0, t, a[1]);
    }
#pragma unroll 1
    for (int gi = 0; gi < groups; ++gi) {
        if (gi + 1 < groups) load_group(base + (size_t)(gi + 1) * gstride, nxt);
#pragma unroll
        for (int st = 0; st < G; ++st) {
            const int i = gi * G + st;
            if (i + 1 < steps) {
                load_a(x0[0], x1[0], kt0 + i + 1, t, an[0]);
                load_a(x0[1], x1[1], kt0 + i + 1, t, an[1]);
            }
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                uint32_t b[2][2][2];
                pair_frags<K2, CB>(cur, st, j, lane, b);
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    mma16816(acc[u][j][0], a[u], b[0][0]);
                    mma16816(acc[u][j][1], a[u], b[0][1]);
                    mma16816(acc[u][j + 1][0], a[u], b[1][0]);
                    mma16816(acc[u][j + 1][1], a[u], b[1][1]);
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u)
#pragma unroll
                for (int c = 0; c < 4; ++c) a[u][c] = an[u][c];
        }
#pragma unroll
        for (int c = 0; c < GW; ++c) cur[c] = nxt[c];
    }

    pdl_trigger();                                            // every weight is read

    // -- per tile, the original's statements after its k loop (its pass: m0 16 u, its counters) -----------------
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int m0 = 16 * pass, R = min(16, M - m0);
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();
            if (on && g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[pass][i][h][2], acc[pass][i][h][3])
                                : make_float2(acc[pass][i][h][0], acc[pass][i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
                    for (int w = 1; w < wk; ++w) {
                        const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += qv.x; v[1] += qv.y; v[2] += qv.z; v[3] += qv.w;
                    }
                    dsv41_x3seg::finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    dsv41_x3seg::store4(y, y_dtype, (size_t)(m0 + rlo + r) * ld + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 sm = *reinterpret_cast<const float4*>(red + r * 128 + c);
                    for (int w = 1; w < wk; ++w) {
                        const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        sm.x += qv.x; sm.y += qv.y; sm.z += qv.z; sm.w += qv.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = sm;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 sm = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int qq = 1; qq < SK; ++qq) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)qq * M * N + at));
                        sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w;
                    }
                    float v[4] = {sm.x, sm.y, sm.z, sm.w};
                    dsv41_x3seg::finish(v, lane, svh, nullptr, col0 + 4 * lane);
                    dsv41_x3seg::store4(y, y_dtype, (size_t)(m0 + r) * ld + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;
            }
        }
        __syncthreads();
    }
}

// -- linear.cu's linear_kernel (the head, the drafter's projections), two row tiles a pass (M 17-32) -------------
template <int K2, int CB, int WK, bool PF>  // PF: the original's prefetch; per-tile loads read the same words
__global__ void __launch_bounds__(WK * 32) linear2_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK, const int* __restrict__ skip) {
    using namespace tf_exl3_linear;
    if (skip != nullptr && *skip != 0) return;
    constexpr int TW = tile_words<K2>();
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];
    __shared__ int last;
    const int RH = min(M, 8);

    const int nb = blockIdx.x, split = blockIdx.y, NB = gridDim.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb;
    const int col0 = nb * 128;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    float acc[2][8][2][4];
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[u][i][h][c] = 0.f;

    const half* x0[2];
    const half* x1[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int m0 = 16 * u, R = min(16, M - m0);
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        x0[u] = xh + (size_t)r0 * K;
        x1[u] = xh + (size_t)r1 * K;
    }
    const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
    constexpr bool P = PF && K2 <= 12;
    constexpr int SR = step_regs<K2>();
    uint32_t cur[P ? SR : 1], nxt[P ? SR : 1];
    if constexpr (P) load_step<K2>(tile, lane, cur);
#pragma unroll 1
    for (int i = 0; i < per_warp; ++i) {
        const int kt = kt0 + i;
        uint32_t a[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            a[u][0] = __ldg(reinterpret_cast<const uint32_t*>(x0[u] + kt * 16 + 2 * t));
            a[u][1] = __ldg(reinterpret_cast<const uint32_t*>(x1[u] + kt * 16 + 2 * t));
            a[u][2] = __ldg(reinterpret_cast<const uint32_t*>(x0[u] + kt * 16 + 2 * t + 8));
            a[u][3] = __ldg(reinterpret_cast<const uint32_t*>(x1[u] + kt * 16 + 2 * t + 8));
        }
        if constexpr (P) {
            if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t w[LW];
            if constexpr (P) step_lane_words<K2>(cur, j, lane, prev, w);
            else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
            uint32_t b0[2], b1[2];
            decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                mma16816(acc[u][j][0], a[u], b0);
                mma16816(acc[u][j][1], a[u], b1);
            }
        }
        if constexpr (P) {
#pragma unroll
            for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
        }
    }

#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int m0 = 16 * pass, R = min(16, M - m0);
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[pass][i][h][2], acc[pass][i][h][3])
                                : make_float2(acc[pass][i][h][0], acc[pass][i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += qv.x; v[1] += qv.y; v[2] += qv.z; v[3] += qv.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 sm = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        sm.x += qv.x; sm.y += qv.y; sm.z += qv.z; sm.w += qv.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = sm;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 sm = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int qq = 1; qq < SK; ++qq) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)qq * M * N + at));
                        sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w;
                    }
                    float v[4] = {sm.x, sm.y, sm.z, sm.w};
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;
            }
        }
        __syncthreads();
    }
}


// -- TF_DSV41_LIN2_CLUSTER: dense3's split-K, partials reduced in cluster shared memory, no Z ---------------------
template <int K2, int CB, int WK, int MINB, int T>  // a cluster = a column block's SK splits, summed in split order
__global__ void __launch_bounds__(WK * 32, MINB) lanesc_kernel(const dsv41_x3seg::LinTable tab, int y_dtype, int M) {
    using namespace dsv41_dense3;  // T row tiles of 16; every segment of a launch has the same SK > 1 (checked)
    namespace cg = cooperative_groups;
    constexpr int G = group_steps(K2), GW = gwords<K2>();
    extern __shared__ __align__(16) float smem[];
    const int RH = min(M, 8);
    float* red = smem;                                        // WK * RH * 128 floats (the original's)
    float* part = smem + WK * RH * 128;                       // [16 T, 128] this split's warp-order sums

    dsv41_x3seg::LinSeg s = tab.s[0];
#pragma unroll
    for (int i = 1; i < dsv41_x3seg::MAXSEG; ++i)
        if (i < tab.n && (int)blockIdx.x >= tab.s[i].p0) s = tab.s[i];
    const half* __restrict__ xh = s.xh;
    const half* __restrict__ svh = s.svh;
    void* __restrict__ y = s.y;
    const int K = s.K, SK = s.SK, wk = s.wk, ld = s.ld;

    const int q = (int)blockIdx.x - s.p0, nb = q / SK, split = q % SK;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const bool on = warp < wk;
    const int per_warp = (K >> 4) / SK / wk;
    const int kt0 = split * (per_warp * wk) + warp * per_warp;
    const int steps = on ? per_warp : 0, groups = steps / G;
    const uint32_t* base = s.T + nb * s.stride_nb + (size_t)kt0 * s.stride_k + 4 * lane;
    const size_t gstride = (size_t)G * s.stride_k;
    const int col0 = nb * 128;

    float acc[T][8][2][4];
#pragma unroll
    for (int u = 0; u < T; ++u)
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[u][i][h][c] = 0.f;
    const half* x0[T];
    const half* x1[T];
#pragma unroll
    for (int u = 0; u < T; ++u) {
        const int m0 = 16 * u, R = max(1, min(16, M - m0));
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        x0[u] = xh + (size_t)r0 * K;
        x1[u] = xh + (size_t)r1 * K;
    }
    uint32_t cur[GW], nxt[GW];
    if (groups > 0) load_group(base, cur);
    pdl_wait();
    uint32_t a[T][4], an[T][4];
    if (steps > 0)
#pragma unroll
        for (int u = 0; u < T; ++u) load_a(x0[u], x1[u], kt0, t, a[u]);
#pragma unroll 1
    for (int gi = 0; gi < groups; ++gi) {
        if (gi + 1 < groups) load_group(base + (size_t)(gi + 1) * gstride, nxt);
#pragma unroll
        for (int st = 0; st < G; ++st) {
            const int i = gi * G + st;
            if (i + 1 < steps)
#pragma unroll
                for (int u = 0; u < T; ++u) load_a(x0[u], x1[u], kt0 + i + 1, t, an[u]);
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                uint32_t b[2][2][2];
                pair_frags<K2, CB>(cur, st, j, lane, b);
#pragma unroll
                for (int u = 0; u < T; ++u) {
                    mma16816(acc[u][j][0], a[u], b[0][0]);
                    mma16816(acc[u][j][1], a[u], b[0][1]);
                    mma16816(acc[u][j + 1][0], a[u], b[1][0]);
                    mma16816(acc[u][j + 1][1], a[u], b[1][1]);
                }
            }
#pragma unroll
            for (int u = 0; u < T; ++u)
#pragma unroll
                for (int c = 0; c < 4; ++c) a[u][c] = an[u][c];
        }
#pragma unroll
        for (int c = 0; c < GW; ++c) cur[c] = nxt[c];
    }
    pdl_trigger();                                            // every weight is read

    // this split's partials: the original's SK > 1 statements, into `part` instead of Z
#pragma unroll
    for (int pass = 0; pass < T; ++pass) {
        const int m0 = 16 * pass, R = min(16, M - m0);
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();
            if (on && g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[pass][i][h][2], acc[pass][i][h][3])
                                : make_float2(acc[pass][i][h][0], acc[pass][i][h][1]);
                    }
            }
            __syncthreads();
            for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                const int r = idx >> 5, c = 4 * (idx & 31);
                float4 sm = *reinterpret_cast<const float4*>(red + r * 128 + c);
                for (int w = 1; w < wk; ++w) {
                    const float4 qv = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                    sm.x += qv.x; sm.y += qv.y; sm.z += qv.z; sm.w += qv.w;
                }
                *reinterpret_cast<float4*>(part + (m0 + rlo + r) * 128 + c) = sm;
            }
        }
    }

    // the splits' sums in split order (Z's reader: Z[0], then + Z[1], + Z[2], ...), a row a warp over the cluster
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();                                                // every rank's partials written (release / acquire)
    for (int r = split * WK + warp; r < M; r += SK * WK) {
        float4 sm = *reinterpret_cast<const float4*>(cl.map_shared_rank(part, 0) + r * 128 + 4 * lane);
        for (int qq = 1; qq < SK; ++qq) {
            const float4 u = *reinterpret_cast<const float4*>(cl.map_shared_rank(part, qq) + r * 128 + 4 * lane);
            sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w;
        }
        float v[4] = {sm.x, sm.y, sm.z, sm.w};
        dsv41_x3seg::finish(v, lane, svh, nullptr, col0 + 4 * lane);
        dsv41_x3seg::store4(y, y_dtype, (size_t)r * ld + col0 + 4 * lane, v);
    }
    cl.sync();                                                // no rank leaves while another still reads its partials
}

}  // namespace dsv41_lin2

// The variants kernels_dense.zig resolves (TF_DSV41_LIN2): lanes2 at dense3's widths, linear2 at linear.cu's, PF as it
#define LIN2_LANES(K2)                                                                                              \
    template __global__ void dsv41_lin2::lanes2_kernel<K2, 2, 4, 2>(const dsv41_x3seg::LinTable, int, int);      \
    template __global__ void dsv41_lin2::lanes2_kernel<K2, 2, 8, 1>(const dsv41_x3seg::LinTable, int, int);
LIN2_LANES(4) LIN2_LANES(6) LIN2_LANES(8) LIN2_LANES(10) LIN2_LANES(12)
#define LIN2_LINEAR(K2, WK)                                                                                         \
    template __global__ void dsv41_lin2::linear2_kernel<K2, 2, WK, true>(const half* __restrict__, \
        const uint32_t* __restrict__, long long, long long, const half* __restrict__, const half* __restrict__, \
        void* __restrict__, int, float* __restrict__, int* __restrict__, int, int, int, int, const int* __restrict__); \
    template __global__ void dsv41_lin2::linear2_kernel<K2, 2, WK, false>(const half* __restrict__, \
        const uint32_t* __restrict__, long long, long long, const half* __restrict__, const half* __restrict__, \
        void* __restrict__, int, float* __restrict__, int* __restrict__, int, int, int, int, const int* __restrict__);
#define LIN2_LINEAR_W(K2) LIN2_LINEAR(K2, 2) LIN2_LINEAR(K2, 4) LIN2_LINEAR(K2, 8)
LIN2_LINEAR_W(2) LIN2_LINEAR_W(3) LIN2_LINEAR_W(4) LIN2_LINEAR_W(5) LIN2_LINEAR_W(6) LIN2_LINEAR_W(7)
LIN2_LINEAR_W(8) LIN2_LINEAR_W(10) LIN2_LINEAR_W(12) LIN2_LINEAR_W(14) LIN2_LINEAR_W(16)
// TF_DSV41_LIN2_CLUSTER: 4- and 8-warp CTAs at T = 1 (MINB 3 / 1) and T = 2 (2 / 1); MINB is a launch bound only
#define LIN2_CL(K2, WK, MB, T) \
    template __global__ void dsv41_lin2::lanesc_kernel<K2, 2, WK, MB, T>(const dsv41_x3seg::LinTable, int, int);
#define LIN2_CL_W(K2) LIN2_CL(K2, 4, 3, 1) LIN2_CL(K2, 8, 1, 1) LIN2_CL(K2, 4, 2, 2) LIN2_CL(K2, 8, 1, 2)
LIN2_CL_W(4) LIN2_CL_W(6) LIN2_CL_W(8) LIN2_CL_W(10) LIN2_CL_W(12)
