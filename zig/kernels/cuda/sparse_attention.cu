// Sparse gather attention for every CUDA family (float64 reference: core/attention_ref.zig): a row's listed keys.

// Block (row, key head): its G <= 16 query heads a warp's mma rows; the list in 4 warps' slices, folded in warp order.

// A row's bits depend on its query, its list and its count alone: not on the other rows, their number or lists.

#include "attn_fold.cuh"

using namespace attn_fold;

namespace tf_sparse {

constexpr int W = 4;               // warps a block, each a slice of the list
constexpr int TILE = 2 * KT * C;   // a warp's K and V tile, in uint4
constexpr int ROWS = 16;           // mma rows: query heads a key head at most

// A warp's partial: its rows' running max and sum, and o [16][128], in shared memory after the folds.
struct Part {
    float o[ROWS][D];
    float m[ROWS], l[ROWS];
};

}  // namespace tf_sparse

using namespace tf_sparse;

// out, q [rows, h, 128]: key head g over cache rows list[row][g][0 .. counts[row][g]) (rows `stride` apart, at g 128).

// A cache row listed twice weighs twice; a list of no keys gives NaN, as float64's 0 / 0.

// A count is taken in [0, cap]; an entry outside [0, rows_kv) names no key and weighs nothing (nothing is read).

// Numerators are fp32 as block_attention's fold: past its range (128 equal keys of V 2^122) a head is +-inf there too.
extern "C" __global__ void __launch_bounds__(128) tf_sparse_attention(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
    const int* __restrict__ list, const int* __restrict__ counts, __nv_bfloat16* __restrict__ out, float scale, int h,
    int hk, int cap, int rows_kv, long long stride) {
    extern __shared__ uint4 smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g8 = lane / 4, qd = lane % 4;
    const int64_t row = blockIdx.x;
    const int g = blockIdx.y, G = h / hk;
    const int n = min(max(counts[row * hk + g], 0), cap);
    const int* keys = list + (row * hk + g) * (int64_t)cap;
    const auto named = [&](int k) { return (unsigned)keys[k] < (unsigned)rows_kv; };
    const __nv_bfloat16* kh = kc + (int64_t)g * D;
    const __nv_bfloat16* vh = vc + (int64_t)g * D;
    const auto krow = [&](int k) { return kh + (int64_t)keys[k] * stride; };
    const auto vrow = [&](int k) { return vh + (int64_t)keys[k] * stride; };
    const auto qrow = [&](int r) { return q + (row * h + (int64_t)g * G + r) * D; };
    uint32_t qa[K16][4];
    {
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(qrow(min(g8, G - 1)));
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(qrow(min(g8 + 8, G - 1)));
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = g8 < G ? q0[8 * k + qd] : 0u;
            qa[k][1] = g8 + 8 < G ? q1[8 * k + qd] : 0u;
            qa[k][2] = g8 < G ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = g8 + 8 < G ? q1[8 * k + qd + 4] : 0u;
        }
    }
    uint32_t live = 0;  // the staged tile's keys a warp sees: bit j for key k0 + j, listed and naming a cache row
    const auto vis = [&](int r, int key) { return r < G && (live >> (key & (KT - 1)) & 1u); };
    const auto seen = [&](int r, int key) { return r < G && key < n && named(key); };
    uint4* ks = smem + warp * TILE;
    uint4* vs = ks + KT * C;
    // a warp's 32 listed keys from k0 into its tile, past the count or unnamed zeros; odd: V's NaN and infinities as 0
    const auto stage = [&](int k0, bool odd) {
        live = __ballot_sync(0xffffffffu, k0 + lane < n && named(k0 + lane));
        for (int i = lane; i < KT * C; i += 32) {
            const int key = k0 + i / C;
            const bool in = live >> (i / C) & 1u;
            cp16z(ks + sw(i / C, i % C), (in ? krow(key) : kh) + (i % C) * 8, in);
            cp16z(vs + sw(i / C, i % C), (in ? vrow(key) : vh) + (i % C) * 8, in);
        }
        commit();
        cp_wait<0>();
        if (odd) attn_odd::clean_rows<D, KT>(vs, lane, 32);
        __syncwarp();
    };
    const int T = n / KT + (n % KT != 0);  // not (n + KT - 1) / KT: n + KT overflows at max_cap
    Fold f;
    for (int t = T * warp / W; t < T * (warp + 1) / W; ++t) {
        stage(t * KT, false);
        f.tile(qa, ks, vs, scale, lane, t * KT, vis);
        __syncwarp();
    }
    const bool odd = attn_odd::nonfinite_out<NB>(f.o, g8 < G, g8 + 8 < G);
    if (!__syncthreads_or(odd)) {
        Part* part = reinterpret_cast<Part*>(smem);
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            part[warp].o[g8][8 * i + 2 * qd] = f.o[i][0];
            part[warp].o[g8][8 * i + 2 * qd + 1] = f.o[i][1];
            part[warp].o[g8 + 8][8 * i + 2 * qd] = f.o[i][2];
            part[warp].o[g8 + 8][8 * i + 2 * qd + 1] = f.o[i][3];
        }
        if (qd == 0) {
            part[warp].m[g8] = f.m0;
            part[warp].l[g8] = f.l0;
            part[warp].m[g8 + 8] = f.m1;
            part[warp].l[g8 + 8] = f.l1;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
            const int r = i / D, e = i % D;
            float mx = -INFINITY;
#pragma unroll
            for (int w = 0; w < W; ++w) mx = fmaxf(mx, part[w].m[r]);
            float o = 0.f, l = 0.f;
#pragma unroll
            for (int w = 0; w < W; ++w) { // warp order; a slice without keys (max -inf) adds nothing
                const float a = part[w].m[r] == -INFINITY ? 0.f : texp(__fsub_rn(part[w].m[r], mx));
                o = __fmaf_rn(part[w].o[r][e], a, o);
                l = __fmaf_rn(part[w].l[r], a, l);
            }
            out[(row * h + (int64_t)g * G + r) * D + e] = __float2bfloat16_rn(__fdiv_rn(o, l));
        }
        return;
    }
    // a NaN or infinite output: warp 0 folds the whole list, V's odd values out of the mma and back by attn_odd
    if (warp != 0) return;
    f = Fold();
    for (int t = 0; t < T; ++t) {
        stage(t * KT, true);
        f.tile(qa, ks, vs, scale, lane, t * KT, vis);
        __syncwarp();
    }
    const auto each = [&](int r, auto fn) {
        for (int k = 0; k < n; ++k)
            if (seen(r, k)) fn(krow(k));
    };
    auto w = attn_odd::weights<D>(qrow, each, scale, lane);
    attn_odd::add_back<D>(f.o, vrow, krow, 0, n, lane, seen, f.l0, f.l1, w);
    __nv_bfloat16* o0 = out + (row * h + (int64_t)g * G + g8) * D + 2 * qd;
    __nv_bfloat16* o1 = o0 + 8 * D;
#pragma unroll
    for (int i = 0; i < NB; ++i)
        for (int x = 0; x < 2; ++x) {
            if (g8 < G) o0[8 * i + x] = __float2bfloat16_rn(__fdiv_rn(f.o[i][x], f.l0));
            if (g8 + 8 < G) o1[8 * i + x] = __float2bfloat16_rn(__fdiv_rn(f.o[i][2 + x], f.l1));
        }
}
