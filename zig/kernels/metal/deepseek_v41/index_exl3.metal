// Source is compiled after kv.metal; no query-by-key-by-head score tensor is materialized.
#include "../nax.h"
using namespace tfp;
struct Dsv41Index {
    uint rows, heads, width, q_stride, weight_stride, position, ratio, keys, key_stride;
    uint tile_start, tile_width, topk, candidate_topk, block_size, use_candidates, make_candidates;
    // `layer`/`skip_mask` sit before `scale`; leaving them out made `scale` read `layer`'s bits and every score 0.0.
    uint layer, skip_mask;
    // 0 = the heap path's per-row score slots (row * tile_width), 1 = the decode select's tile-major ones.
    uint score_layout;
    float scale;
};
static_assert(sizeof(Dsv41Index) == 80, "Dsv41Index must mirror csa2_index.zig's Params");

// A key's score slot: the heap path keeps one tile a row, the decode select keeps every tile.
inline device float *dsv41_score_slot(device float *scores, constant Dsv41Index &p, uint row, uint tile) {
    if (p.score_layout == 0u) return scores + size_t(row) * p.tile_width;
    return scores + (size_t(tile) * p.rows + row) * p.tile_width;
}

inline const device float *dsv41_score_slot_ro(const device float *scores, constant Dsv41Index &p, uint row, uint tile) {
    if (p.score_layout == 0u) return scores + size_t(row) * p.tile_width;
    return scores + (size_t(tile) * p.rows + row) * p.tile_width;
}

// The fp32 score's monotone 32-bit image: float order == uint order.
inline uint dsv41_index_mono(float s) {
    const uint b = as_type<uint>(s == 0.0f ? 0.0f : s);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
struct Dsv41Pick { float score; int position; };
static_assert(sizeof(Dsv41Pick) == 8, "Dsv41Pick must keep its 8-byte (float score, int position) record");
inline bool dsv41_less(Dsv41Pick a, Dsv41Pick b) {
    return a.score < b.score || (a.score == b.score && a.position > b.position);
}
inline void dsv41_heap(device Dsv41Pick *h, uint size, Dsv41Pick v) {
    if (!dsv41_less(h[0], v)) return;
    uint at = 0;
    while (2 * at + 1 < size) {
        uint child = 2 * at + 1;
        if (child + 1 < size && dsv41_less(h[child + 1], h[child])) ++child;
        if (!dsv41_less(h[child], v)) break;
        h[at] = h[child]; at = child;
    }
    h[at] = v;
}
inline int dsv41_index_position(uint local, uint row, device const Dsv41Pick *candidates,
                                constant Dsv41Index &p) {
    uint slot = p.tile_start + local;
    if (!p.use_candidates) return int(slot);
    uint block = slot / p.block_size;
    if (block >= p.candidate_topk) return -1;
    int id = candidates[row * p.candidate_topk + block].position;
    return id < 0 ? -1 : id * int(p.block_size) + int(slot % p.block_size);
}
kernel void dsv41_index_score(device const ushort *q [[buffer(0)]], device const ushort *weight [[buffer(1)]],
    device const uchar *key [[buffer(2)]], device const Dsv41Pick *candidates [[buffer(3)]],
    device float *scores [[buffer(4)]], constant Dsv41Index &p [[buffer(5)]],
    uint3 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
    const uint warps = tptg.x / 32u, lane = tid % 32u;
    // The query's scaled per-head weights do not depend on the key.
    threadgroup float scaled[64];
    const uint row = g.y;
    if (tid < min(p.heads, 64u))
        scaled[tid] = dsv41_bf(dsv41_round(dsv41_bf(weight[row * p.weight_stride + tid]) * p.scale));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint local = g.x * warps + tid / 32u;
    if (local >= p.tile_width || row >= p.rows) return;
    int pos = dsv41_index_position(local, row, candidates, p);
    uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    if (pos < 0 || uint(pos) >= visible) {
        if (lane == 0) *(dsv41_score_slot(scores, p, row, p.tile_start / p.tile_width) + local) = -INFINITY;
        return;
    }
    device const uchar *k = key + uint(pos) * p.key_stride;
    float key_values[4];
    for (uint d = 0; d < p.width / 32; ++d) key_values[d] = dsv41_unpack(k, lane + 32 * d, p.width, 32, 2);
    float total = 0;
    for (uint head = 0; head < p.heads; ++head) {
        float dot = 0;
        for (uint d = 0; d < p.width / 32; ++d)
            dot += dsv41_bf(q[row * p.q_stride + head * p.width + lane + 32 * d]) * key_values[d];
        dot = dsv41_bf(dsv41_round(simd_sum(dot)));
        total += dsv41_bf(dsv41_round(max(dot, 0.0f) * scaled[head]));
    }
    if (lane == 0) *(dsv41_score_slot(scores, p, row, p.tile_start / p.tile_width) + local) = dsv41_bf(dsv41_round(total));
}
// The index keys a layer scores, decoded from the packed fp4 cache once into bf16 rows of p.width.
kernel void dsv41_index_keys_dense(device const uchar *key [[buffer(0)]], device ushort *dense [[buffer(1)]],
                                   constant Dsv41Index &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint total = p.keys * (p.width / 2u);
  if (i >= total) return;
  const uint k = i / (p.width / 2u), n = i % (p.width / 2u); // a thread a 2-value byte pair
  const device uchar *row = key + uint(k) * p.key_stride;
  device ushort *out = dense + uint(k) * p.width + 2u * n;
  out[0] = ushort(as_type<ushort>(bfloat(dsv41_unpack(row, 2u * n, p.width, 32, 2))));
  out[1] = ushort(as_type<ushort>(bfloat(dsv41_unpack(row, 2u * n + 1u, p.width, 32, 2))));
}

// Every visible key is selected: no more compressed keys than the selection keeps.
kernel void dsv41_index_identity(device int *positions [[buffer(0)]], constant Dsv41Index &p [[buffer(1)]],
    uint2 i [[thread_position_in_grid]]) {
    if (i.y >= p.rows || i.x >= p.topk) return;
    const uint visible = min(p.keys, (p.position + i.y + 1) / p.ratio);
    positions[i.y * p.topk + i.x] = i.x < visible ? int(i.x) : -1;
}
kernel void dsv41_index_clear(device Dsv41Pick *top [[buffer(0)]], device Dsv41Pick *candidate [[buffer(1)]],
    constant Dsv41Index &p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
    if (i.y >= p.rows) return;
    if (i.x < p.topk) top[i.y * p.topk + i.x] = {-INFINITY, -1};
    if (p.make_candidates && i.x < p.candidate_topk) candidate[i.y * p.candidate_topk + i.x] = {-INFINITY, -1};
}
kernel void dsv41_index_merge(device const float *scores [[buffer(0)]], device Dsv41Pick *top [[buffer(1)]],
    device Dsv41Pick *candidate [[buffer(2)]], constant Dsv41Index &p [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device Dsv41Pick *heap = top + row * p.topk;
    // Only the tile's visible keys exist; the rest of the width holds -INFINITY and the heap ignores those anyway.
    const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    const uint have = visible > p.tile_start ? min(p.tile_width, visible - p.tile_start) : 0u;
    for (uint i = 0; i < have; ++i) {
        float score = scores[row * p.tile_width + i];
        int pos = dsv41_index_position(i, row, candidate, p);
        if (isfinite(score)) dsv41_heap(heap, p.topk, {score, pos});
    }
    if (!p.make_candidates) return;
    device Dsv41Pick *blocks = candidate + row * p.candidate_topk;
    for (uint base = 0; base < have; base += p.block_size) {
        float score = -INFINITY;
        for (uint j = 0; j < p.block_size; ++j) score = max(score, scores[row * p.tile_width + base + j]);
        uint block = (p.tile_start + base) / p.block_size;
        if (visible && block == (visible - 1) / p.block_size) score = INFINITY;
        if (score > -INFINITY) dsv41_heap(blocks, p.candidate_topk, {score, int(block)});
    }
}
// One thread a row: fill an empty heap from the tile's first keys and publish where to resume.
kernel void dsv41_index_seed(device const float *scores [[buffer(0)]], device Dsv41Pick *top [[buffer(1)]],
    device Dsv41Pick *candidate [[buffer(2)]], device uint *seed_end [[buffer(3)]],
    constant Dsv41Index &p [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device Dsv41Pick *heap = top + row * p.topk;
    const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    const uint have = visible > p.tile_start ? min(p.tile_width, visible - p.tile_start) : 0u;
    uint at = 0;
    if (heap[0].position < 0) { // the clear kernel's empty state: fill it
        uint filled = 0;
        while (at < have && filled < p.topk) {
            const float score = scores[row * p.tile_width + at];
            const int pos = dsv41_index_position(at, row, candidate, p);
            if (isfinite(score) && pos >= 0) {
                dsv41_heap(heap, p.topk, {score, pos});
                ++filled;
            }
            ++at;
        }
    }
    seed_end[row] = at;
}

// One thread a key: the merge's scan, in parallel.
kernel void dsv41_index_filter(device const float *scores [[buffer(0)]], device const Dsv41Pick *top [[buffer(1)]],
    device const Dsv41Pick *candidate [[buffer(2)]], device Dsv41Pick *picks [[buffer(3)]],
    device const uint *seed_end [[buffer(4)]], device atomic_uint *count [[buffer(5)]],
    constant Dsv41Index &p [[buffer(6)]],
    uint3 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
    const uint row = g.y;
    if (row >= p.rows) return; // uniform across the threadgroup
    const uint local = g.x * tptg.x + tid;
    const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    Dsv41Pick v = { -INFINITY, -1 };
    bool keeps = false;
    // Every lane runs the ballot below, so no condition may return early: they set the predicate.
    if (local < p.tile_width && local >= seed_end[row] && visible > p.tile_start + local) {
        const float score = scores[row * p.tile_width + local];
        if (isfinite(score)) {
            const int pos = dsv41_index_position(local, row, candidate, p);
            if (pos >= 0) {
                v = { score, pos };
                keeps = dsv41_less(top[row * p.topk], v);
            }
        }
    }
    if (!keeps) return;
    const uint at = atomic_fetch_add_explicit(&count[row], 1u, memory_order_relaxed);
    if (at < p.tile_width) picks[row * p.tile_width + at] = v;
}

// The merge's serial part, now only the keys the filter kept.
kernel void dsv41_index_insert(device Dsv41Pick *top [[buffer(0)]], device const Dsv41Pick *picks [[buffer(1)]],
    device atomic_uint *count [[buffer(2)]], device Dsv41Pick *candidate [[buffer(3)]],
    device const float *scores [[buffer(4)]], constant Dsv41Index &p [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device Dsv41Pick *heap = top + row * p.topk;
    const uint kept = atomic_load_explicit(&count[row], memory_order_relaxed);
    for (uint i = 0; i < kept; ++i) dsv41_heap(heap, p.topk, picks[row * p.tile_width + i]);
    atomic_store_explicit(&count[row], 0u, memory_order_relaxed);
    if (!p.make_candidates) return;
    const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    const uint have = visible > p.tile_start ? min(p.tile_width, visible - p.tile_start) : 0u;
    device Dsv41Pick *blocks = candidate + row * p.candidate_topk;
    for (uint base = 0; base < have; base += p.block_size) {
        float score = -INFINITY;
        for (uint j = 0; j < p.block_size; ++j) score = max(score, scores[row * p.tile_width + base + j]);
        uint block = (p.tile_start + base) / p.block_size;
        if (visible && block == (visible - 1) / p.block_size) score = INFINITY;
        if (score > -INFINITY) dsv41_heap(blocks, p.candidate_topk, {score, int(block)});
    }
}

inline void dsv41_sort_positions(device Dsv41Pick *p, uint size) {
    // Selection is score-ordered; attention sees the reference's ascending position order.
    uint valid = 0;
    for (uint i = 0; i < size; ++i) if (p[i].position >= 0) p[valid++] = p[i];
    for (uint i = 1; i < valid; ++i) {
        const Dsv41Pick value = p[i]; uint j = i;
        while (j > 0 && p[j - 1].position > value.position) { p[j] = p[j - 1]; --j; }
        p[j] = value;
    }
    for (uint i = valid; i < size; ++i) p[i] = { -INFINITY, -1 };
}
kernel void dsv41_index_finish(device Dsv41Pick *top [[buffer(0)]], device Dsv41Pick *candidate [[buffer(1)]],
    device int *positions [[buffer(2)]], constant Dsv41Index &p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device Dsv41Pick *heap = top + row * p.topk;
    dsv41_sort_positions(heap, p.topk);
    for (uint i = 0; i < p.topk; ++i) positions[row * p.topk + i] = heap[i].position;
    if (p.make_candidates) dsv41_sort_positions(candidate + row * p.candidate_topk, p.candidate_topk);
}

// The finish's PARALLEL form -- a threadgroup a row instead of one THREAD a row.
kernel void dsv41_index_finish_par(device Dsv41Pick *top [[buffer(0)]], device Dsv41Pick *candidate [[buffer(1)]],
    device int *positions [[buffer(2)]], constant Dsv41Index &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    const uint TG = 256;
    threadgroup uint sh[512];
    device Dsv41Pick *heap = top + row * p.topk;
    for (uint i = tid; i < p.topk; i += TG) {
        const int pos = heap[i].position;
        sh[i] = pos < 0 ? 0xFFFFFFFFu : uint(pos);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A bitonic sort of the positions, ascending. Each compare-exchange pair is owned by one thread.
    for (uint k = 2; k <= p.topk; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < p.topk; i += TG) {
                const uint ixj = i ^ j;
                if (ixj > i) {
                    const bool asc = (i & k) == 0;
                    const uint a = sh[i], b = sh[ixj];
                    if (asc ? (a > b) : (a < b)) { sh[i] = b; sh[ixj] = a; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = tid; i < p.topk; i += TG) {
        const uint key = sh[i];
        positions[row * p.topk + i] = key == 0xFFFFFFFFu ? -1 : int(key);
    }
}
