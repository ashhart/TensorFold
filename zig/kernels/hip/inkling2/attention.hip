// Inkling 2 attention on HIP: q/k/v/rel prep, the relative-bias table, chunked attention and the ordered merge.
#include "inkling2/tile.hpp"

using ink2::Seg;
using ink2::Tile;

namespace ink2 {

// L2 clip of one head vector held 4 values a lane: scale down to `thr` when longer, as Inkling2L2Clip.
__device__ __forceinline__ void clip(float* x, float thr) {
    const float norm = sqrtf(waveSum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]));
    const float f = fminf(thr / (norm + 1e-6f), 1.0f);
    for (int i = 0; i < 4; ++i) x[i] *= f;
}

}  // namespace ink2

// Per row and head: q = clip(q * q_scale) (* tau), rel = clip(rel * rel_scale) (* tau), k = clip(k), v = rms(v).
extern "C" __global__ void __launch_bounds__(32)
    tf_ink2_prep(const uint16_t* __restrict__ q_raw, const uint16_t* __restrict__ k_raw,
                 const uint16_t* __restrict__ v_raw, const uint16_t* __restrict__ rel_raw,
                 const uint16_t* __restrict__ q_scale, const uint16_t* __restrict__ rel_scale,
                 const int* __restrict__ rpos, const int* __restrict__ rseg, const Seg* __restrict__ segs,
                 uint16_t* __restrict__ q_out, uint16_t* __restrict__ rel_out, uint16_t* __restrict__ kc,
                 uint16_t* __restrict__ vc, int heads, int kv_heads, int ring, int global, float qk_clip,
                 float rel_clip, float floor_pos, float alpha, float eps) {
    using namespace ink2;
    const int row = blockIdx.x, y = blockIdx.y, lane = threadIdx.x;
    const int pos = rpos[row];
    const float tau = global ? 1.0f + alpha * logf(fmaxf(static_cast<float>(pos + 1) / floor_pos, 1.0f)) : 1.0f;
    float x[4];
    if (y < heads) {
        const size_t rh = static_cast<size_t>(row) * heads + y;
        for (int i = 0; i < 4; ++i) x[i] = bf(q_raw[rh * kDim + lane * 4 + i]) * bf(q_scale[y * kDim + lane * 4 + i]);
        clip(x, qk_clip);
        for (int i = 0; i < 4; ++i) q_out[rh * kDim + lane * 4 + i] = global ? tobf(bf(tobf(x[i])) * tau) : tobf(x[i]);
        float r = lane < kRel ? bf(rel_raw[rh * kRel + lane]) * bf(rel_scale[y * kRel + lane]) : 0.0f;
        const float norm = sqrtf(waveSum(r * r));
        r *= fminf(rel_clip / (norm + 1e-6f), 1.0f);
        if (lane < kRel) rel_out[rh * kRel + lane] = global ? tobf(bf(tobf(r)) * tau) : tobf(r);
        return;
    }
    const bool key = y < heads + kv_heads;
    const int h = key ? y - heads : y - heads - kv_heads;
    const uint16_t* src = (key ? k_raw : v_raw) + (static_cast<size_t>(row) * kv_heads + h) * kDim;
    for (int i = 0; i < 4; ++i) x[i] = bf(src[lane * 4 + i]);
    if (key) {
        clip(x, qk_clip);
    } else {
        const float r = rsqrtf(waveSum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / kDim + eps);
        for (int i = 0; i < 4; ++i) x[i] *= r;
    }
    const Seg s = segs[rseg[row]];
    uint16_t* dst = (key ? kc : vc) + s.off + (static_cast<size_t>(h) * ring + pos % ring) * kDim;
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = tobf(x[i]);
}

// tab[row, head, d] = bf16(rel[row, head, :] . proj[:, d]) for d < 1024, fp32 sums in tap order (rel_logits_proj).
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_rel_table(const uint16_t* __restrict__ rel, const uint16_t* __restrict__ proj,
                      uint16_t* __restrict__ tab) {
    using namespace ink2;
    const size_t rh = blockIdx.x;
    float r[kRel];
    for (int j = 0; j < kRel; ++j) r[j] = bf(rel[rh * kRel + j]);
    for (int d = threadIdx.x; d < kExtent; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < kRel; ++j) acc += r[j] * bf(proj[j * kExtent + d]);
        tab[rh * kExtent + d] = tobf(acc);
    }
}

// Sliding layers (a chunk a block) and split global calls (a sub-chunk a block, `chunks` partials a row and head).
extern "C" __global__ void __launch_bounds__(ink2::kWaves * 32)
    tf_ink2_attn(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                 const uint16_t* __restrict__ tab, const Seg* __restrict__ segs, const Tile* __restrict__ tiles,
                 float* __restrict__ part_o, float2* __restrict__ part_ml, int heads, int kv_heads, int ring,
                 int window, int chunk, int c0, int chunks, float scale, int split) {
    using namespace ink2;
    __shared__ uint32_t ks[kKeys * kKRow];
    __shared__ uint4 vs[kKeys * kVRow / 4];
    __shared__ float qs[kWaves * kDim];
    const Tile t = tiles[blockIdx.x];
    attnTile<kWaves, false, false>({q, kc, vc, tab, part_o, part_ml}, segs[t.seg], t.m0, blockIdx.y, blockIdx.z,
                                   {heads, kv_heads, ring, window, chunk, c0, chunks, split}, scale, ks,
                                   reinterpret_cast<uint32_t*>(vs), qs);
}

// Global layers a chunk a block, its sub-chunks folded in order (several streams' rows: no partial a sub-chunk).
extern "C" __global__ void __launch_bounds__(ink2::kWaves * 32)
    tf_ink2_attn_fold(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                      const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab, const Seg* __restrict__ segs,
                      const Tile* __restrict__ tiles, float* __restrict__ part_o, float2* __restrict__ part_ml,
                      int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale) {
    using namespace ink2;
    __shared__ uint32_t ks[kKeys * kKRow];
    __shared__ uint4 vs[kKeys * kVRow / 4];
    __shared__ float qs[kWaves * kDim];
    const Tile t = tiles[blockIdx.x];
    attnTile<kWaves, true, false>({q, kc, vc, tab, part_o, part_ml}, segs[t.seg], t.m0, blockIdx.y, blockIdx.z,
                                  {heads, kv_heads, ring, 0, chunk, c0, chunks, 0}, scale, ks,
                                  reinterpret_cast<uint32_t*>(vs), qs);
}

// One stream's prompt rows on a global layer: 8 pairs a block share each K/V tile, the next one loading meanwhile.
extern "C" __global__ void __launch_bounds__(ink2::kPromptWaves * 32)
    tf_ink2_attn_prompt(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                        const uint16_t* __restrict__ vc, const uint16_t* __restrict__ tab,
                        const Seg* __restrict__ segs, float* __restrict__ part_o, float2* __restrict__ part_ml,
                        int heads, int kv_heads, int ring, int chunk, int c0, int chunks, float scale) {
    using namespace ink2;
    __shared__ uint32_t ks[kKeys * kKRow];
    __shared__ uint4 vs[kKeys * kVRow / 4];
    __shared__ float qs[kPromptWaves * kDim];
    const Seg s = segs[0];
    const int m0 = static_cast<int>(blockIdx.x) * kPromptWaves;
    if (m0 >= s.n * (heads / kv_heads)) return;
    attnTile<kPromptWaves, true, true>({q, kc, vc, tab, part_o, part_ml}, s, m0, blockIdx.y, blockIdx.z,
                                       {heads, kv_heads, ring, 0, chunk, c0, chunks, 0}, scale, ks,
                                       reinterpret_cast<uint32_t*>(vs), qs);
}

// Per (row, head): chunks folded in order (subs > 1: sub-chunks first), then the RAW sink logit and RAW gate's sigmoid.
extern "C" __global__ void __launch_bounds__(ink2::kDim)
    tf_ink2_merge(const float* __restrict__ part_o, const float2* __restrict__ part_ml,
                  const uint16_t* __restrict__ sink, const uint16_t* __restrict__ gate, const int* __restrict__ rpos,
                  uint16_t* __restrict__ out, int heads, int window, int chunk, int c0, int chunks, int subs) {
    using namespace ink2;
    const size_t rh = blockIdx.x;
    const int row = static_cast<int>(rh / heads), d = threadIdx.x;
    const int pos = rpos[row];
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    float mm = -INFINITY, ll = 0.0f, o = 0.0f;
    for (int c = lo / chunk; c <= pos / chunk; ++c) {
        float2 ml;
        float po;
        if (subs > 1) {
            float M = -INFINITY, L = 0.0f, O = 0.0f;
            const int last = min(subs - 1, (pos - c * chunk) / kSub);
            for (int u = 0; u <= last; ++u) {
                const size_t at = rh * chunks + (c - c0) * subs + u;
                const float2 sm = part_ml[at];
                float a, b;
                fold(M, L, sm.x, sm.y, a, b);
                O = O * a + part_o[at * kDim + d] * b;
            }
            ml = make_float2(M, L);
            po = O;
        } else {
            const size_t at = rh * chunks + (c - (window > 0 ? lo / chunk : c0));
            ml = part_ml[at];
            po = part_o[at * kDim + d];
        }
        float a, b;
        fold(mm, ll, ml.x, ml.y, a, b);
        o = o * a + po * b;
    }
    const float sk = bf(sink[rh]) * kLog2e;
    const float mf = fmaxf(mm, sk);
    const float a = exp2f(mm - mf);
    const float y = o * a / (ll * a + exp2f(sk - mf));
    const float gv = bf(gate[rh * kDim + d]);
    out[rh * kDim + d] = tobf(y * (1.0f / (1.0f + expf(-gv))));
}
