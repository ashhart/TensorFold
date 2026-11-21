// Inkling 2's grouped expert projections: exl3_grouped.cuh's arithmetic per output, launched without empty tiles.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_decode.cuh"
#include "common/fp_pairs.cuh"

namespace ink2x {
using namespace tf_exl3;

constexpr int ITEM_THREADS = 1024;

// tf_ink2x_group's arguments: tf_exl3_group's lists, plus items (u, first, count), runs of at most `per` members.
struct GroupArgs {
    const int* pick; // int32 [rows * slots]
    int* uids; // int32 [nexp_max]
    int* ucount; // int32 [1]
    int* members; // int32 [nexp_max, maxm]
    int* items; // int32 [3 * max items] (per > 0)
    int* nitems; // int32 [1] (per > 0)
    int rows, slots, experts, maxm, per, pad; // pad: -1 after each expert's members up to maxm
};

// One projection; z, k, n, p, sk, maxm and slots as GroupedArgs has them.
struct WideArgs {
    const half* x0; // fp16 [P, K], the rotated gate (or down) inputs
    const half* x1; // fp16 [P, K], the rotated up inputs
    const long long* tp0; // int64 [E], one trellis pointer per expert
    const long long* tp1;
    const int* k2_0; // int32 [E], half-bits a value
    const int* k2_1;
    const int* uids; // int32 [nexp_max], distinct routed experts in id order
    const int* items; // int32 [3 * max items], (u, first member, count); null: item u is expert u, maxm <= 16 MT
    const int* nitems; // int32 [1]: the items, or the group's count when items is null
    const int* members; // int32 [nexp_max, maxm], row * 32 + slot
    float* z; // fp32 partials, [(mat * sk + split) * P + p] * N + col
    int k, n, p, sk, maxm, slots;
};

// A warp's k tiles for MT member tiles in grouped_body's order per output; words and inputs travel a tile ahead.
template <int CB, int K2, int NT, int MT>
struct Tiles {
    static constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    uint32_t nx[NT][LW];

    __device__ __forceinline__ void first(const uint32_t* tb, int nkt, int lane) {
        if (nkt > 0)
#pragma unroll
            for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tb + j * TW, lane, nx[j]);
    }

    __device__ __forceinline__ static void xload(const half* (&xr)[MT][2], const bool (&ok)[MT][2], int mts, int k,
                                                 uint32_t (&a)[MT][4]) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const bool o0 = m < mts && ok[m][0], o1 = m < mts && ok[m][1];
            a[m][0] = o0 ? *reinterpret_cast<const uint32_t*>(xr[m][0] + k) : 0u;
            a[m][1] = o1 ? *reinterpret_cast<const uint32_t*>(xr[m][1] + k) : 0u;
            a[m][2] = o0 ? *reinterpret_cast<const uint32_t*>(xr[m][0] + k + 8) : 0u;
            a[m][3] = o1 ? *reinterpret_cast<const uint32_t*>(xr[m][1] + k + 8) : 0u;
        }
    }

    __device__ __forceinline__ void run(const uint32_t* tb, size_t kstride, int kt0, int nkt, const half* (&xr)[MT][2],
                                        const bool (&ok)[MT][2], int mts, int lane, float (&acc)[MT][NT][2][4]) {
        uint32_t xa[MT][4];
        if (nkt > 0) xload(xr, ok, mts, kt0 * 16, xa);
        for (int i = 0; i < nkt; ++i) {
            uint32_t w[NT][LW], a[MT][4];
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int l = 0; l < LW; ++l) w[j][l] = nx[j][l];
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int c = 0; c < 4; ++c) a[m][c] = xa[m][c];
            if (i + 1 < nkt) {
#pragma unroll
                for (int j = 0; j < NT; ++j) ldg_lane_words<K2>(tb + (size_t)(i + 1) * kstride + j * TW, lane, nx[j]);
                xload(xr, ok, mts, (kt0 + i + 1) * 16, xa);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w[j], lane, b0, b1);
#pragma unroll
                for (int m = 0; m < MT; ++m)
                    if (m < mts) {
                        mma16816(acc[m][j][0], a[m], b0);
                        mma16816(acc[m][j][1], a[m], b1);
                    }
            }
        }
    }
};

// This lane's member-row input pointers (rows_sh filled and synced).
template <int MT>
__device__ __forceinline__ void member_rows(const half* X, int k, const int* rows_sh, int mts, int g, int t,
                                            const half* (&xr)[MT][2], bool (&ok)[MT][2]) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = m < mts ? rows_sh[16 * m + g] : -1, r1 = m < mts ? rows_sh[16 * m + g + 8] : -1;
        xr[m][0] = X + (size_t)(r0 < 0 ? 0 : r0) * k + 2 * t;
        xr[m][1] = X + (size_t)(r1 < 0 ? 0 : r1) * k + 2 * t;
        ok[m][0] = r0 >= 0;
        ok[m][1] = r1 >= 0;
    }
}

// Program (n block, item, mat * sk + split): 16 MT members by W_q over the split's K; SEQ: warps add in turn.
template <int CB, int NT, int MT, int W, bool SEQ>
__device__ __forceinline__ void wide_body(const WideArgs a) {
    const int nblock = blockIdx.x, item = blockIdx.y;
    if (item >= a.nitems[0]) return;
    int u = item, first = 0, cnt = a.maxm;
    if (a.items) {
        u = a.items[3 * item];
        first = a.items[3 * item + 1];
        cnt = a.items[3 * item + 2];
    }
    const int split = blockIdx.z % a.sk, mat = blockIdx.z / a.sk;
    const half* X = mat ? a.x1 : a.x0;
    const int e = a.uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.tp1[e] : a.tp0[e]);
    const int k2 = mat ? a.k2_1[e] : a.k2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int ntiles = a.n >> 4;
    const int per_warp = (a.k >> 4) / a.sk / W;
    const int kt0 = split * (per_warp * W) + warp * per_warp;
    const int nt0 = nblock * NT;

    __shared__ int rows_sh[16 * MT];
    for (int m = threadIdx.x; m < 16 * MT; m += W * 32) {
        const int code = m < cnt ? a.members[(size_t)u * a.maxm + first + m] : -1;
        rows_sh[m] = code >= 0 ? (code >> 5) * a.slots + (code & 31) : -1;
    }
    float acc[MT][NT][2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    int mts = 0;
    switch (k2) {  // the first k tile's words are in flight before the members' rows are read
#define TF_INK2X_CASE(K2_)                                                                               \
    case K2_: {                                                                                          \
        Tiles<CB, K2_, NT, MT> tl;                                                                       \
        const size_t kstride = (size_t)ntiles * tile_words<K2_>();                                       \
        const uint32_t* tb = T + (size_t)kt0 * kstride + (size_t)nt0 * tile_words<K2_>();                \
        tl.first(tb, per_warp, lane);                                                                    \
        __syncthreads();                                                                                 \
        if (!a.items)                                                                                    \
            while (cnt > 0 && rows_sh[cnt - 1] < 0) --cnt;                                               \
        mts = (cnt + 15) >> 4;                                                                           \
        if (mts == 0) return;                                                                            \
        const half* xr[MT][2];                                                                           \
        bool ok[MT][2];                                                                                  \
        member_rows<MT>(X, a.k, rows_sh, mts, g, t, xr, ok);                                             \
        tl.run(tb, kstride, kt0, per_warp, xr, ok, mts, lane, acc);                                      \
        break;                                                                                           \
    }
        TF_INK2X_CASE(2)
        TF_INK2X_CASE(3)
        TF_INK2X_CASE(4)
        TF_INK2X_CASE(5)
        TF_INK2X_CASE(6)
        TF_INK2X_CASE(7)
        TF_INK2X_CASE(8)
#undef TF_INK2X_CASE
        default:
            __trap();
    }

    __shared__ __align__(16) float red[SEQ ? 1 : W][16][NT * 16];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m >= mts) break;
        if (m) __syncthreads();  // the last tile's sums are read
        for (int w = SEQ ? 0 : warp; w < (SEQ ? W : warp + 1); ++w) {
            if (w == warp)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        float* p0 = &red[SEQ ? 0 : warp][g][col];
                        float* p1 = &red[SEQ ? 0 : warp][g + 8][col];
                        if (!SEQ || w == 0) {
                            p0[0] = acc[m][i][h][0];
                            p0[1] = acc[m][i][h][1];
                            p1[0] = acc[m][i][h][2];
                            p1[1] = acc[m][i][h][3];
                        } else {
                            p0[0] += acc[m][i][h][0];
                            p0[1] += acc[m][i][h][1];
                            p1[0] += acc[m][i][h][2];
                            p1[1] += acc[m][i][h][3];
                        }
                    }
            if (SEQ) __syncthreads();
        }
        if (!SEQ) __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[16 * m + row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < (SEQ ? 1 : W); ++w) s += red[w][row][col];
            a.z[(((size_t)mat * a.sk + split) * a.p + r) * a.n + nt0 * 16 + col] = s;
        }
    }
}

// Exclusive sum over the block of one value a thread (ITEM_THREADS threads); `tot` gets the total.
__device__ __forceinline__ int block_scan(int v, int* part, int& tot) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int x = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += x;
    }
    if (lane == 31) part[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int w = part[lane];
        int sum = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, sum, o);
            if (lane >= o) sum += x;
        }
        part[lane] = sum - w;
        if (lane == 31) part[32] = sum;
    }
    __syncthreads();
    tot = part[32];
    const int r = part[warp] + inc - v;
    __syncthreads();
    return r;
}

}  // namespace ink2x

// tf_exl3_group's lists and the item list in O(pairs): warp w counts, then places, its 32nd of the pairs in order.
extern "C" __global__ void __launch_bounds__(ink2x::ITEM_THREADS) tf_ink2x_group(const ink2x::GroupArgs a) {
    extern __shared__ int sh[];
    __shared__ int part[33];
    const int E = a.experts, n = a.rows * a.slots;
    uint16_t* wc = reinterpret_cast<uint16_t*>(sh);  // [32][E]: a warp's pairs of each expert, then its first rank
    int* cnt = sh + 16 * E;
    int* uidx = cnt + E;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int span = (n + 31) / 32, lo = warp * span, hi = min(n, lo + span);
    const unsigned below = (1u << lane) - 1u;
    for (int i = threadIdx.x; i < 32 * E; i += ink2x::ITEM_THREADS) wc[i] = 0;
    __syncthreads();
    for (int i0 = lo; i0 < hi; i0 += 32) {
        const int i = i0 + lane;
        const int e = i < hi ? a.pick[i] : -1;
        const unsigned m = __match_any_sync(0xffffffffu, e);
        if (e >= 0 && e < E && (m & below) == 0) wc[warp * E + e] += __popc(m);
    }
    __syncthreads();
    int has = 0, k = 0, c = 0;
    const int e = threadIdx.x;
    if (e < E) {
        for (int w = 0; w < 32; ++w) {
            const int t = wc[w * E + e];
            wc[w * E + e] = c;
            c += t;
        }
        cnt[e] = c;
        has = c > 0;
        k = a.per > 0 ? (c + a.per - 1) / a.per : 0;
    }
    int nu, ni;
    const int u = ink2x::block_scan(has, part, nu);
    int it = ink2x::block_scan(k, part, ni);
    if (e < E) {
        uidx[e] = u;
        if (has) {
            a.uids[u] = e;
            for (int j = 0; j < k; ++j, ++it) {
                a.items[3 * it] = u;
                a.items[3 * it + 1] = j * a.per;
                a.items[3 * it + 2] = min(a.per, c - j * a.per);
            }
        }
    }
    if (threadIdx.x == 0) {
        a.ucount[0] = nu;
        if (a.per > 0) a.nitems[0] = ni;
    }
    __syncthreads();
    for (int i0 = lo; i0 < hi; i0 += 32) {
        const int i = i0 + lane;
        const int pe = i < hi ? a.pick[i] : -1;
        const unsigned m = __match_any_sync(0xffffffffu, pe);
        if (pe >= 0 && pe < E) {
            const int rank = wc[warp * E + pe] + __popc(m & below);
            a.members[(size_t)uidx[pe] * a.maxm + rank] = (i / a.slots) * 32 + i % a.slots;
        }
        __syncwarp();
        if (pe >= 0 && pe < E && (m & below) == 0) wc[warp * E + pe] += __popc(m);
        __syncwarp();
    }
    if (a.pad)  // -1 after each expert's members, as tf_exl3_group leaves them
        for (int x = threadIdx.x; x < E; x += ink2x::ITEM_THREADS)
            if (cnt[x] > 0)
                for (int j = cnt[x]; j < a.maxm; ++j) a.members[(size_t)uidx[x] * a.maxm + j] = -1;
}

// Decode rows and short windows: 128 columns and 16 members a block (grouped_body's tile), warp sums apart.
extern "C" __global__ void __launch_bounds__(128) tf_ink2x_rows(const ink2x::WideArgs a) {
    ink2x::wide_body<tf_exl3::CB_MUL1, 8, 1, 4, false>(a);
}

// Prompt chunks: 64 columns and 32 members a block, one sum buffer.
extern "C" __global__ void __launch_bounds__(128) tf_ink2x_wide(const ink2x::WideArgs a) {
    ink2x::wide_body<tf_exl3::CB_MUL1, 4, 2, 4, true>(a);
}

namespace ink2m {

__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }
__device__ __forceinline__ uint16_t tobf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfr(float f) { return bf(tobf(f)); }
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <int N>
__device__ __forceinline__ void blockSums(float (&v)[N], float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = 0; i < N; ++i) v[i] = warpSum(v[i]);
    __syncthreads();
    if (lane == 0)
        for (int i = 0; i < N; ++i) sh[i * 32 + w] = v[i];
    __syncthreads();
    for (int i = 0; i < N; ++i) {
        float t = lane < nw ? sh[i * 32 + lane] : 0.0f;
        v[i] = __shfl_sync(0xffffffffu, warpSum(t), 0);
    }
}

struct GateUpArgs {
    const uint16_t* x;  // bf16 [rows, x_stride], pair p reads row p / slots
    const uint8_t *codes, *scales;
    const float *amax, *denominator;
    const int* pick;  // int32 [P]
    uint16_t* out;  // bf16 [P, N], the rounded softcapped SwiGLU
    int x_stride, p, slots, n, k, experts, format;
    float gate_cap, up_cap;
};

struct DownArgs {
    const uint16_t* x;  // bf16 [P, K], gateup's output
    const uint8_t *codes, *scales;
    const float *amax, *denominator;
    const int* pick;
    uint16_t* out;  // bf16 [P, N], each selected expert's rounded output
    int p, n, k, experts, format;
};

struct CombineArgs {
    const uint16_t* experts;  // bf16 [rows * slots, H]
    const int* pick;
    const float* weights;
    uint16_t* out;  // bf16 [rows, H]
    int rows, h, slots;
};

struct TensorActArgs {
    const uint16_t* gateup;  // bf16 [P, 2 * M], adjacent gate and up projections
    uint16_t* out;  // bf16 [P, M]
    int n;
};

__device__ __forceinline__ float e2m1(unsigned code) {
    const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = values[code & 7u];
    return code & 8u ? -v : v;
}

// The checkpoint's stored codes and scales become one bf16 weight, with no activation quantization.
__device__ __forceinline__ float weight(const uint8_t* codes, const uint8_t* scales, const float* amax,
                                        const float* denominator, int e, int r, int c, int n, int k, int format) {
    const int er = e * n + r;
    if (format == 0) {
        const uint8_t packed = codes[(size_t)er * (k / 2) + c / 2];
        const unsigned code = c & 1 ? packed >> 4 : packed & 15u;
        const float scale = tf_common::e4m3f(scales[(size_t)er * (k / 16) + c / 16]);
        const float phi = __fdiv_rn(bfr(amax[e]), bfr(denominator[0]));
        return bfr(__fmul_rn(__fmul_rn(e2m1(code), scale), phi));
    }
    const float value = tf_common::e4m3f(codes[(size_t)er * k + c]);
    const float scale = ldexpf(1.0f, static_cast<int>(scales[(size_t)er * (k / 32) + c / 32]) - 127);
    return bfr(__fmul_rn(value, scale));
}

__device__ __forceinline__ float cap(float x, float c) { return c * tanhf(x / c); }

}  // namespace ink2m

// One (routed pair, intermediate unit): two stored NVFP4/MXFP8 rows and the rounded SwiGLU epilogue.
extern "C" __global__ void __launch_bounds__(128) tf_ink2moe_gateup(const ink2m::GateUpArgs a) {
    __shared__ float sh[64];
    const int p = blockIdx.y, unit = blockIdx.x, e = a.pick[p];
    const uint16_t* x = a.x + static_cast<size_t>(p / a.slots) * a.x_stride;
    float v[2] = {0.0f, 0.0f};
    for (int c = threadIdx.x; c < a.k; c += blockDim.x) {
        const float xv = ink2m::bf(x[c]);
        const float g = ink2m::weight(a.codes, a.scales, a.amax, a.denominator, e, 2 * unit, c, 2 * a.n, a.k, a.format);
        const float u = ink2m::weight(a.codes, a.scales, a.amax, a.denominator, e, 2 * unit + 1, c, 2 * a.n, a.k, a.format);
        v[0] = __fmaf_rn(xv, g, v[0]);
        v[1] = __fmaf_rn(xv, u, v[1]);
    }
    ink2m::blockSums<2>(v, sh);
    if (threadIdx.x == 0) {
        const float g = ink2m::cap(ink2m::bfr(v[0]), a.gate_cap);
        const float u = ink2m::cap(ink2m::bfr(v[1]), a.up_cap);
        a.out[(size_t)p * a.n + unit] = ink2m::tobf(g * (1.0f / (1.0f + expf(-ink2m::bfr(v[0])))) * u);
    }
}

// One (routed pair, hidden output): the down projection, rounded to bf16 before routing weights are applied.
extern "C" __global__ void __launch_bounds__(128) tf_ink2moe_down(const ink2m::DownArgs a) {
    __shared__ float sh[32];
    const int p = blockIdx.y, row = blockIdx.x, e = a.pick[p];
    float v[1] = {0.0f};
    for (int c = threadIdx.x; c < a.k; c += blockDim.x) {
        const float w = ink2m::weight(a.codes, a.scales, a.amax, a.denominator, e, row, c, a.n, a.k, a.format);
        v[0] = __fmaf_rn(ink2m::bf(a.x[(size_t)p * a.k + c]), w, v[0]);
    }
    ink2m::blockSums<1>(v, sh);
    if (threadIdx.x == 0) a.out[(size_t)p * a.n + row] = ink2m::tobf(v[0]);
}

// Tensor-core gate/up rows are interleaved; keep the scalar baseline's softcapped SwiGLU arithmetic.
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_tensor_act(const ink2m::TensorActArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n) return;
    const float gate = ink2m::bf(a.gateup[2 * (size_t)i]);
    const float up = ink2m::bf(a.gateup[2 * (size_t)i + 1]);
    const float g = ink2m::cap(gate, 4.0f);
    const float u = ink2m::cap(up, 25.0f);
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    const float activated = g * sigmoid * u;
    a.out[i] = ink2m::tobf(activated);
}

// One output element: route in ascending expert id, independent of the router's selection order.
extern "C" __global__ void __launch_bounds__(256) tf_ink2moe_combine(const ink2m::CombineArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.rows * a.h) return;
    const int row = i / a.h, col = i % a.h;
    float total = 0.0f;
    uint32_t seen = 0;
    for (int n = 0; n < a.slots; ++n) {
        int best = -1, id = 0;
        for (int s = 0; s < a.slots; ++s) {
            if (seen & (1u << s)) continue;
            const int e = a.pick[row * a.slots + s];
            if (best < 0 || e < id) {
                best = s;
                id = e;
            }
        }
        seen |= 1u << best;
        const float out = ink2m::bf(a.experts[(size_t)(row * a.slots + best) * a.h + col]);
        total = __fadd_rn(total, __fmul_rn(a.weights[row * a.slots + best], out));
    }
    a.out[i] = ink2m::tobf(total);
}
