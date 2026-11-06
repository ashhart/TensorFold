// Source is compiled after kv.metal; no query-by-key-by-head score tensor is materialized.
#include "../nax.h"
using namespace tfp;
struct Dsv41Index {
    uint rows, heads, width, q_stride, weight_stride, position, ratio, keys, key_stride;
    uint tile_start, tile_width, topk, candidate_topk, block_size, use_candidates, make_candidates;
    // ROUND-468: `layer` and `skip_mask` are the Zig Params' own fields (ROUND-411's removal arms). They are
    // not read here, but they sit BEFORE `scale`, and leaving them out made `scale` read `layer`'s bits --
    // a denormal near zero, so every scaled weight and every score was 0.0 and the heap's ties kept the
    // first 512 positions. The assert keeps the two declarations in step.
    uint layer, skip_mask;
    // ROUND-469: 0 = the heap path's per-row score slots (row * tile_width), 1 = the decode select's
    // tile-major ones ((tile * rows + row) * tile_width, every tile resident).
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

// The fp32 score's monotone 32-bit image: float order == uint order, so a radix select on it is exact.
// -0.0f is folded to +0.0f so the two zeros cannot split a tie.
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
    // A warp a key: the reduction is across its 32 lanes, so a threadgroup of W warps scores W adjacent keys and the
    // dispatch needs W times fewer threadgroups (one a key cost 4.8e9 of them at a 15k prompt). Four keys a warp with
    // the query hoisted measured slower: the state it keeps costs more than the loads it saves.
    const uint warps = tptg.x / 32u, lane = tid % 32u;
    // The query's scaled per-head weights do not depend on the key, and every warp of a threadgroup scores a
    // different key: compute them once here instead of once for every key a warp handles.
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
// The index keys a layer scores, decoded from the packed fp4 cache once into bf16 rows of p.width: the
// table a tensor-unit score tile stages in threadgroup memory. Same rounding as the scorer's own
// dsv41_unpack, so a table read and a packed read give the same values.
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

// Every visible key is selected: no more compressed keys than the selection keeps, so the picks are the
// identity in ascending position order -- the same set, in the same order, that the score/heap/sort path
// returns for those keys. The identity's row bound is the same visible formula the merge uses.
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
// One thread a row: fill an empty heap from the tile's first keys and publish where to resume, so the
// parallel filter never starts against a root of -infinity. A layer's first tile used to insert all 4096
// of its keys on one thread; it now inserts topk and hands the rest to the filter.
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

// One thread a key: the merge's scan, in parallel. The comparator is a total order (score, then
// position), so which keys reach the heap does not depend on the order they arrive in and the compacted
// list needs no ordering of its own. Keys the running worst already beats are dropped here; the insert
// re-checks anyway, so a permissive filter changes nothing.
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
    // A per-key atomic: warp aggregation measured wrong here (the suite's index case rejects it), so the
    // counter stays one increment a kept key until that is understood.
    const uint at = atomic_fetch_add_explicit(&count[row], 1u, memory_order_relaxed);
    if (at < p.tile_width) picks[row * p.tile_width + at] = v;
}

// The merge's serial part, now only the keys the filter kept (and the candidate-block highs, which are
// a block a pass and stay here because they are one a tile).
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
    // Selection is score-ordered; attention sees the reference's ascending position order. Only the valid picks
    // matter — attention skips an invalid one — so they are compacted first and the sort then costs what the live
    // keys cost, not what the slots cost.
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

// ROUND-411: the finish's PARALLEL form -- a threadgroup a row instead of one THREAD a row. The shipped
// finish sorts the row's topk picks with an insertion sort (~65k moves a row at topk 512), one thread
// a row: the removal arms put ~46 ms of the block's 69 ms a prompt in that sort. This form stages the
// picks' positions in threadgroup memory and sorts them with a bitonic sort: the keys are the
// positions (the invalid ones share the last key, so they land last, where the insertion sort's
// compaction leaves them), the valid ones are distinct, so the ascending order is the same array the
// insertion sort produced -- BIT-IDENTICAL, and nothing downstream moves.
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


// ---------------------------------------------------------------- prompt scores on the tensor units
// The scalar scorer's arithmetic, a tile at a time: 64 keys by 64 rows, four SIMDgroups of 32x32, the
// 64x128 key tile staged once and read by every row group; one 128-dim transposed MMA a head instead of a
// warp a key with sixty-four simd_sums. The epilogue is the scalar kernel's own term -- bf16 dot, relu, times
// the bf16-rounded scaled weight, summed in fp32 in head order -- so only the dot's summation order differs.
constant constexpr int INAX_D = 128, INAX_PAD = INAX_D + 8;

inline frag<bfloat> inax_frag(const device bfloat* r0, const device bfloat* r1, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  return frag<bfloat>(*(const device bfloat4*)(r0 + c + home.x), *(const device bfloat4*)(r1 + c + home.x));
#else
  frag<bfloat> f;
  TF_UNROLL
  for (short e = 0; e < 8; e++) f[e] = (e < 4 ? r0 : r1)[c + home.x + TF_COL(e)];
  return f;
#endif
}

kernel void dsv41_index_scores_nax(device const ushort *qp [[buffer(0)]], device const ushort *wp [[buffer(1)]],
                                   device const ushort *kp [[buffer(2)]], device float *scores [[buffer(3)]],
                                   constant Dsv41Index &p [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const device bfloat* Q = reinterpret_cast<const device bfloat*>(qp);
  const device bfloat* W = reinterpret_cast<const device bfloat*>(wp);
  const device bfloat* K = reinterpret_cast<const device bfloat*>(kp);
  threadgroup bfloat tile[64 * INAX_PAD];
  const int t0 = int(p.tile_start) + int(tg.x) * 64, r0 = int(tg.y) * 64;
  const int rows = min(64, int(p.rows) - r0);
  const uint t = sg * 32u + lane;
  if (p.keys == 0 or p.ratio == 0) return;
  // A tile past the last row's visible keys scores -inf and is not staged.
  if (t0 >= int((p.position + uint(r0) + uint(rows)) / p.ratio)) {
    for (uint i = t; i < 64u * 64u; i += 128u) {
      const int r = int(i) / 64, k = t0 + int(i) % 64;
      if (r < rows && k < int(p.keys))
        *(dsv41_score_slot(scores, p, r0 + r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) = -INFINITY;
    }
    return;
  }
  for (uint i = t; i < 64u * (INAX_D / 8u); i += 128u) {
    const uint rr = i / (INAX_D / 8u), cc = (i % (INAX_D / 8u)) * 8u;
    const int k = min(t0 + int(rr), int(p.keys) - 1);
    *(threadgroup uint4*)(tile + rr * INAX_PAD + cc) = *(const device uint4*)(K + size_t(k) * INAX_D + cc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 home = frag_home(ushort(lane));
  const int tm = 32 * int(sg / 2), tn = 32 * int(sg % 2);
  const device bfloat* qr[2][2];
  const device bfloat* wr[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = tm + 16 * m + home.y + 8 * h;
      const int at = r0 + (r < rows ? r : 0); // a dead row reads a live one; its scores are never stored
      qr[m][h] = Q + size_t(at) * p.q_stride;
      wr[m][h] = W + size_t(at) * p.weight_stride;
    }
  }
  frag<float> total[2][2];
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    total[m][0] = frag<float>(0);
    total[m][1] = frag<float>(0);
  }
  for (int hd = 0; hd < int(p.heads); hd++) {
    frag<float> s[2][2];
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      s[m][0] = frag<float>(0);
      s[m][1] = frag<float>(0);
    }
    TF_UNROLL
    for (short q = 0; q < INAX_D / 16; q++) {
      frag<bfloat> k0, k1;
      frag_get_t(k0, (const threadgroup bfloat*)tile, INAX_PAD, tn, 16 * q, home);
      frag_get_t(k1, (const threadgroup bfloat*)tile, INAX_PAD, tn + 16, 16 * q, home);
      TF_UNROLL
      for (short m = 0; m < 2; m++) {
        const frag<bfloat> qa = inax_frag(qr[m][0] + hd * INAX_D, qr[m][1] + hd * INAX_D, 16 * q, home);
        mma_16x32<false, true>(s[m][0], s[m][1], qa, k0, k1);
      }
    }
    // the head's term as the scalar makes it: bf16 dot, relu, times the bf16-rounded scaled weight, summed in fp32
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      const float w0 = dsv41_bf(dsv41_round(float(wr[m][0][hd]) * p.scale));
      const float w1 = dsv41_bf(dsv41_round(float(wr[m][1][hd]) * p.scale));
      TF_UNROLL
      for (short j = 0; j < 2; j++) {
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const float dot = dsv41_bf(dsv41_round(s[m][j][e]));
          total[m][j][e] += dsv41_bf(dsv41_round(max(dot, 0.0f) * (e < 4 ? w0 : w1)));
        }
      }
    }
  }
  TF_UNROLL
  for (short m = 0; m < 2; m++) {
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int r = tm + 16 * m + home.y + (e >> 2) * 8;
        const int k = t0 + tn + 16 * j + home.x + TF_COL(e);
        if (r < rows && k < int(p.keys))
          *(dsv41_score_slot(scores, p, r0 + r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) =
              dsv41_bf(dsv41_round(total[m][j][e]));
      }
    }
  }
}

// ---------------------------------------------------------------- the decode-shaped score tile (ROUND-470)
// The prompt tile above scores a 64-row by 64-key tile, so ONE decode row pays 64 rows of tensor-unit work
// (its dispatch count and its bytes are small; 98.4 % of the MMAs are dead). This is that kernel's own body
// with the tile's row extent cut to sixteen -- the MMA's M granularity, so nothing narrower can help -- and
// the same operand roles, the same 128-dim k steps in the same order, the same epilogue term by term and
// the same store expression, so a live row's score is BIT-IDENTICAL to dsv41_index_scores_nax's. A
// threadgroup keeps its 64 keys and two SIMDgroups take 32 keys each (tn), the other two stage and stop:
// the tile's MMA work is a quarter of the prompt tile's at one row whatever the window's row count.
kernel void dsv41_index_scores_row(device const ushort *qp [[buffer(0)]], device const ushort *wp [[buffer(1)]],
                                   device const ushort *kp [[buffer(2)]], device float *scores [[buffer(3)]],
                                   constant Dsv41Index &p [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const device bfloat* Q = reinterpret_cast<const device bfloat*>(qp);
  const device bfloat* W = reinterpret_cast<const device bfloat*>(wp);
  const device bfloat* K = reinterpret_cast<const device bfloat*>(kp);
  threadgroup bfloat tile[64 * INAX_PAD];
  const int t0 = int(p.tile_start) + int(tg.x) * 64, r0 = int(tg.y) * 16;
  const int rows = min(16, int(p.rows) - r0);
  const uint t = sg * 32u + lane;
  if (p.keys == 0 or p.ratio == 0) return;
  if (t0 >= int((p.position + uint(r0) + uint(rows)) / p.ratio)) {
    for (uint i = t; i < 16u * 64u; i += 128u) {
      const int r = int(i) / 64, k = t0 + int(i) % 64;
      if (r < rows && k < int(p.keys))
        *(dsv41_score_slot(scores, p, r0 + r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) = -INFINITY;
    }
    return;
  }
  for (uint i = t; i < 64u * (INAX_D / 8u); i += 128u) {
    const uint rr = i / (INAX_D / 8u), cc = (i % (INAX_D / 8u)) * 8u;
    const int k = min(t0 + int(rr), int(p.keys) - 1);
    *(threadgroup uint4*)(tile + rr * INAX_PAD + cc) = *(const device uint4*)(K + size_t(k) * INAX_D + cc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(sg) >= 2) return;  // sixteen rows by sixty-four keys is two 16x32 column blocks
  const short2 home = frag_home(ushort(lane));
  const int tn = 32 * int(sg);
  const int tm = 0;
  const device bfloat* qr[2];
  const device bfloat* wr[2];
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const int r = tm + home.y + 8 * h;
    const int at = r0 + (r < rows ? r : 0);
    qr[h] = Q + size_t(at) * p.q_stride;
    wr[h] = W + size_t(at) * p.weight_stride;
  }
  frag<float> total0 = frag<float>(0), total1 = frag<float>(0);
  for (int hd = 0; hd < int(p.heads); hd++) {
    frag<float> s0 = frag<float>(0), s1 = frag<float>(0);
    TF_UNROLL
    for (short q = 0; q < INAX_D / 16; q++) {
      frag<bfloat> k0, k1;
      frag_get_t(k0, (const threadgroup bfloat*)tile, INAX_PAD, tn, 16 * q, home);
      frag_get_t(k1, (const threadgroup bfloat*)tile, INAX_PAD, tn + 16, 16 * q, home);
      const frag<bfloat> qa = inax_frag(qr[0] + hd * INAX_D, qr[1] + hd * INAX_D, 16 * q, home);
      mma_16x32<false, true>(s0, s1, qa, k0, k1);
    }
    // the head's term as the scalar makes it: bf16 dot, relu, times the bf16-rounded scaled weight, fp32 sum
    const float w0 = dsv41_bf(dsv41_round(float(wr[0][hd]) * p.scale));
    const float w1 = dsv41_bf(dsv41_round(float(wr[1][hd]) * p.scale));
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const float dot = dsv41_bf(dsv41_round((j == 0 ? s0 : s1)[e]));
        const float add = dsv41_bf(dsv41_round(max(dot, 0.0f) * (e < 4 ? w0 : w1)));
        if (j == 0) total0[e] += add; else total1[e] += add;
      }
    }
  }
  TF_UNROLL
  for (short j = 0; j < 2; j++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int r = tm + home.y + (e >> 2) * 8;
      const int k = t0 + tn + 16 * j + home.x + TF_COL(e);
      if (r < rows && k < int(p.keys))
        *(dsv41_score_slot(scores, p, r0 + r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) =
            dsv41_bf(dsv41_round((j == 0 ? total0 : total1)[e]));
    }
  }
}

// ---------------------------------------------------------------- the keys-on-M score tile (ROUND-481)
// The row tile above cuts the prompt tile's row extent to the MMA's sixteen, but a one-row step still runs a
// sixteen-row MMA: fifteen of sixteen M rows are dead. This form swaps the operand roles -- the KEYS are the
// M operand (four 16-key blocks, one a SIMDgroup) and the head queries are the N operand (32 heads) -- so a
// threadgroup scores 64 keys x 32 heads with eight MMAs a SIMDgroup instead of a quarter of 1024. The k
// chain (128 dims in sixteen-wide steps, ascending), the per-head term (bf16 dot, relu, the bf16-rounded
// scaled weight) and the head-ordered fp32 sum are the scalar's own: the head terms are staged as bf16 (the
// value is already bf16) and summed 0..31 in order by one lane a key, so a score is BIT-IDENTICAL to
// dsv41_index_scores_row's. The key tile is staged once and reused by every row of the window.
kernel void dsv41_index_scores_keys(device const ushort *qp [[buffer(0)]], device const ushort *wp [[buffer(1)]],
                                    device const ushort *kp [[buffer(2)]], device float *scores [[buffer(3)]],
                                    constant Dsv41Index &p [[buffer(4)]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const device bfloat* Q = reinterpret_cast<const device bfloat*>(qp);
  const device bfloat* W = reinterpret_cast<const device bfloat*>(wp);
  const device bfloat* K = reinterpret_cast<const device bfloat*>(kp);
  threadgroup bfloat tile[64 * INAX_PAD];
  threadgroup bfloat terms[64 * 32];
  const int t0 = int(p.tile_start) + int(tg.x) * 64;
  const uint t = sg * 32u + lane;
  if (p.keys == 0 or p.ratio == 0) return;
  const int rows = int(p.rows);
  // A tile past the last row's visible keys scores -inf and is not staged.
  if (t0 >= int((p.position + uint(rows)) / p.ratio)) {
    for (uint r = 0; r < uint(rows); r++)
      for (uint i = t; i < 64u; i += 128u) {
        const int k = t0 + int(i);
        if (k < int(p.keys))
          *(dsv41_score_slot(scores, p, int(r), p.tile_start / p.tile_width) + (k - int(p.tile_start))) = -INFINITY;
      }
    return;
  }
  for (uint i = t; i < 64u * (INAX_D / 8u); i += 128u) {
    const uint rr = i / (INAX_D / 8u), cc = (i % (INAX_D / 8u)) * 8u;
    const int k = min(t0 + int(rr), int(p.keys) - 1);
    *(threadgroup uint4*)(tile + rr * INAX_PAD + cc) = *(const device uint4*)(K + size_t(k) * INAX_D + cc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 home = frag_home(ushort(lane));
  const int m0 = 16 * int(sg);
  for (int r = 0; r < rows; r++) {
    const device bfloat* qrow = Q + size_t(r) * p.q_stride;
    const device bfloat* wrow = W + size_t(r) * p.weight_stride;
    frag<float> s0 = frag<float>(0), s1 = frag<float>(0);
    TF_UNROLL
    for (short q = 0; q < INAX_D / 16; q++) {
      frag<bfloat> ka, b0, b1;
      frag_get(ka, (const threadgroup bfloat*)tile, INAX_PAD, m0, 16 * q, home);
      frag_get_t(b0, qrow, INAX_D, 0, 16 * q, home);
      frag_get_t(b1, qrow, INAX_D, 16, 16 * q, home);
      mma_16x32<false, true>(s0, s1, ka, b0, b1);
    }
    TF_UNROLL
    for (short j = 0; j < 2; j++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int key = m0 + home.y + (e >> 2) * 8;
        const int h = 16 * j + home.x + TF_COL(e);
        const float w = dsv41_bf(dsv41_round(float(wrow[h]) * p.scale));
        const float dot = dsv41_bf(dsv41_round((j == 0 ? s0 : s1)[e]));
        terms[key * 32 + h] = bfloat(dsv41_bf(dsv41_round(max(dot, 0.0f) * w)));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 64u) {
      const int k = t0 + int(t);
      if (k < int(p.keys)) {
        float total = 0.0f;
        TF_UNROLL
        for (short h = 0; h < 32; h++) total += float(terms[t * 32u + uint(h)]);
        *(dsv41_score_slot(scores, p, r, p.tile_start / p.tile_width) + (k - int(p.tile_start))) =
            dsv41_bf(dsv41_round(total));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ---------------------------------------------------------------- the decode selection, one dispatch
// ROUND-469: at decode rows the heap path costs 27 dispatches and ~1,000 one-thread pushes a source. This
// does the same selection in ONE threadgroup a row: a radix select on the score's monotone image (four
// 8-bit passes over the resident scores) fixes the k-th largest exactly, a threshold filter keeps the k
// keys above it (plus the lowest-positioned ties), and the finish's bitonic sort writes the ascending
// positions. The heap's total order is (score desc, position asc), so the SET and the array are the same.
inline void dsv41_ixsort(threadgroup uint *sh, uint n, uint tid) {
    const uint TG = 256u;
    for (uint k = 2; k <= n; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < n; i += TG) {
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
}

kernel void dsv41_index_select(device const float *scores [[buffer(0)]],
                               device int *positions [[buffer(1)]],
                               constant Dsv41Index &p [[buffer(2)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const uint TG = 256u, topk = 512u;
    if (row >= p.rows) return;
    threadgroup uint hist[256];
    threadgroup uint sel[512];
    threadgroup uint ties[512];
    threadgroup atomic_uint counter;
    const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    if (visible <= topk) { // every visible key is kept: the identity branch's array
        for (uint i = tid; i < topk; i += TG) positions[row * topk + i] = i < visible ? int(i) : -1;
        return;
    }
    // ---- the radix select: `prefix` converges to the k-th largest's image, `above` is the count above it
    uint prefix = 0u, above = 0u, need = topk;
    for (uint pass = 0; pass < 4u; ++pass) {
        const uint shift = 24u - 8u * pass;
        for (uint i = tid; i < 256u; i += TG) hist[i] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
            const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
            const uint n = min(p.tile_width, visible - base);
            for (uint i = tid; i < n; i += TG) {
                const uint u = dsv41_index_mono(slot[i]);
                if (pass == 0u || (u >> (shift + 8u)) == prefix) {
                    atomic_fetch_add_explicit((threadgroup atomic_uint *)&hist[(u >> shift) & 0xFFu], 1u,
                                              memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint digit, ran;
        // ROUND-526: digit/ran are threadgroup memory. Initialising them here was a race: every
        // thread wrote 0 to them and tid 0 wrote the chosen digit, with no ordering between the two
        // writes before the barrier, so a late thread's zero could land on top of the digit. A zero
        // prefix then made the next pass's histogram empty and the filter count every visible key as
        // above the threshold, which the first 512 atomic arrivals truncated -- a nondeterministic
        // key SET from unchanged scores. Only tid 0 writes them now, and it writes them once.
        if (tid == 0u) {
            uint cum = 0u;
            digit = 0u; ran = 0u;
            for (int d = 255; d >= 0; --d) {
                const uint c = hist[uint(d)];
                if (cum + c >= need) { digit = uint(d); ran = cum; break; }
                cum += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = (pass == 0u) ? digit : ((prefix << 8u) | digit);
        above += ran;
        need -= ran;
    }
    // ---- the filter: every key above the threshold (exactly `above` of them), compacted in arrival order
    if (tid == 0u) atomic_store_explicit(&counter, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
        const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
        const uint n = min(p.tile_width, visible - base);
        for (uint i = tid; i < n; i += TG) {
            if (dsv41_index_mono(slot[i]) > prefix) {
                const uint at = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                if (at < topk) sel[at] = base + i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- the ties at the threshold, the lowest `need` positions of them
    const uint have = min(atomic_load_explicit(&counter, memory_order_relaxed), topk);
    // ROUND-526: the ties pass below reuses this counter, so it has to be zeroed AFTER every thread
    // has read `have`. Without the barrier the zero could land first and leave `have` divergent
    // across the threadgroup -- the ties then got written at different slots (and the sentinel fill
    // started at a different index) per thread, which is a second nondeterministic key SET.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) atomic_store_explicit(&counter, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
        const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
        const uint n = min(p.tile_width, visible - base);
        for (uint i = tid; i < n; i += TG) {
            if (dsv41_index_mono(slot[i]) == prefix) {
                const uint at = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                if (at < 512u) ties[at] = base + i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nties = atomic_load_explicit(&counter, memory_order_relaxed);
    if (need > 0u) {
        if (nties <= 512u) { // sort the ties ascending and take the first `need`
            for (uint i = nties + tid; i < 512u; i += TG) ties[i] = 0xFFFFFFFFu;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv41_ixsort(ties, 512u, tid);
            for (uint i = tid; i < need && have + i < topk; i += TG) sel[have + i] = ties[i];
        } else { // more ties than a threadgroup array: the lowest positions, scanned in order
            if (tid == 0u) {
                uint got = 0u;
                for (uint t = 0, base = 0; base < visible && got < need; ++t, base += p.tile_width) {
                    const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
                    const uint n = min(p.tile_width, visible - base);
                    for (uint i = 0; i < n && got < need; ++i)
                        if (dsv41_index_mono(slot[i]) == prefix && have + got < topk) { sel[have + got] = base + i; ++got; }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- the ascending positions array the finish writes: the same array, sorted the same way
    for (uint i = have + need + tid; i < topk; i += TG) sel[i] = 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dsv41_ixsort(sel, 512u, tid);
    for (uint i = tid; i < topk; i += TG)
        positions[row * topk + i] = sel[i] == 0xFFFFFFFFu ? -1 : int(sel[i]);
}



// ROUND-483: the one-dispatch select with vector reads, a two-pass radix and the ties folded into the
// filter. Exactness: every score this path ranks is bf16-valued (the scorer stores dsv41_bf(x)), so
// mono(score)'s low 16 bits are zero and its top 16 bits fix the order; two 8-bit passes converge on
// the k-th image. The filter/ties compaction order differs from the shipped kernel's, but the array
// is sorted before it is written, so the output is the same array.
kernel void dsv41_index_select_v(device const float *scores [[buffer(0)]],
                                 device int *positions [[buffer(1)]],
                                 constant Dsv41Index &p [[buffer(2)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint TG = 256u, topk = 512u;
    if (row >= p.rows) return;
    threadgroup uint hist[256];
    threadgroup uint sel[512];
    threadgroup uint ties[512];
    threadgroup atomic_uint counter;
    threadgroup atomic_uint tcounter;
    const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    if (visible <= topk) { // every visible key is kept: the identity branch's array
        for (uint i = tid; i < topk; i += TG) positions[row * topk + i] = i < visible ? int(i) : -1;
        return;
    }
    // ---- the two-pass radix: bits 31..24 then 23..16; the low 16 are zero by the bf16 invariant
    uint prefix = 0u, above = 0u, need = topk;
    for (uint pass = 0; pass < 2u; ++pass) {
        const uint shift = 24u - 8u * pass;
        for (uint i = tid; i < 256u; i += TG) hist[i] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
            const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
            const uint n = min(p.tile_width, visible - base);
            const uint n4 = n & ~3u;
            const device float4 *s4 = (const device float4 *)slot;
            for (uint i = tid; i < n4 / 4u; i += TG) {
                const float4 v = s4[i];
                const uint u[4] = {dsv41_index_mono(v.x), dsv41_index_mono(v.y),
                                   dsv41_index_mono(v.z), dsv41_index_mono(v.w)};
                for (uint j = 0; j < 4u; ++j)
                    if (pass == 0u || (u[j] >> (shift + 8u)) == prefix)
                        atomic_fetch_add_explicit((threadgroup atomic_uint *)&hist[(u[j] >> shift) & 0xFFu],
                                                  1u, memory_order_relaxed);
            }
            for (uint i = n4 + tid; i < n; i += TG) {
                const uint u = dsv41_index_mono(slot[i]);
                if (pass == 0u || (u >> (shift + 8u)) == prefix)
                    atomic_fetch_add_explicit((threadgroup atomic_uint *)&hist[(u >> shift) & 0xFFu],
                                              1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint digit, ran;
        // ROUND-526: digit/ran are threadgroup memory. Initialising them here was a race: every
        // thread wrote 0 to them and tid 0 wrote the chosen digit, with no ordering between the two
        // writes before the barrier, so a late thread's zero could land on top of the digit. A zero
        // prefix then made the next pass's histogram empty and the filter count every visible key as
        // above the threshold, which the first 512 atomic arrivals truncated -- a nondeterministic
        // key SET from unchanged scores. Only tid 0 writes them now, and it writes them once.
        if (tid == 0u) {
            uint cum = 0u;
            digit = 0u; ran = 0u;
            for (int d = 255; d >= 0; --d) {
                const uint c = hist[uint(d)];
                if (cum + c >= need) { digit = uint(d); ran = cum; break; }
                cum += c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = (pass == 0u) ? digit : ((prefix << 8u) | digit);
        above += ran;
        need -= ran;
    }
    // The two passes leave the top sixteen bits in the low half. mono()'s low sixteen are 0 for a
    // positive bf16 score and 0xFFFF for a negative one (mono is b's complement there), so the full
    // image is (prefix << 16) | (negative ? 0xFFFF : 0).
    prefix = (prefix << 16u) | (((prefix & 0x8000u) == 0u) ? 0xFFFFu : 0u);
    // ---- the filter and the ties in one pass: every key above the threshold (exactly `above` of them)
    if (tid == 0u) {
        atomic_store_explicit(&counter, 0u, memory_order_relaxed);
        atomic_store_explicit(&tcounter, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0, base = 0; base < visible; ++t, base += p.tile_width) {
        const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
        const uint n = min(p.tile_width, visible - base);
        const uint n4 = n & ~3u;
        const device float4 *s4 = (const device float4 *)slot;
        for (uint i = tid; i < n4 / 4u; i += TG) {
            const float4 v = s4[i];
            const uint u[4] = {dsv41_index_mono(v.x), dsv41_index_mono(v.y),
                               dsv41_index_mono(v.z), dsv41_index_mono(v.w)};
            for (uint j = 0; j < 4u; ++j) {
                const uint at = base + i * 4u + j;
                if (u[j] > prefix) {
                    const uint k = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                    if (k < topk) sel[k] = at;
                } else if (u[j] == prefix && need > 0u) {
                    const uint k = atomic_fetch_add_explicit(&tcounter, 1u, memory_order_relaxed);
                    if (k < 512u) ties[k] = at;
                }
            }
        }
        for (uint i = n4 + tid; i < n; i += TG) {
            const uint u = dsv41_index_mono(slot[i]);
            const uint at = base + i;
            if (u > prefix) {
                const uint k = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                if (k < topk) sel[k] = at;
            } else if (u == prefix && need > 0u) {
                const uint k = atomic_fetch_add_explicit(&tcounter, 1u, memory_order_relaxed);
                if (k < 512u) ties[k] = at;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint have = min(atomic_load_explicit(&counter, memory_order_relaxed), topk);
    const uint nties = atomic_load_explicit(&tcounter, memory_order_relaxed);
    if (need > 0u) {
        if (nties <= 512u) { // sort the ties ascending and take the first `need`
            for (uint i = nties + tid; i < 512u; i += TG) ties[i] = 0xFFFFFFFFu;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv41_ixsort(ties, 512u, tid);
            for (uint i = tid; i < need && have + i < topk; i += TG) sel[have + i] = ties[i];
        } else { // more ties than a threadgroup array: the lowest positions, scanned in order
            if (tid == 0u) {
                uint got = 0u;
                for (uint t = 0, base = 0; base < visible && got < need; ++t, base += p.tile_width) {
                    const device float *slot = dsv41_score_slot_ro(scores, p, row, t);
                    const uint n = min(p.tile_width, visible - base);
                    for (uint i = 0; i < n && got < need; ++i)
                        if (dsv41_index_mono(slot[i]) == prefix && have + got < topk) { sel[have + got] = base + i; ++got; }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // ---- the ascending positions array the finish writes: the same array, sorted the same way
    for (uint i = have + need + tid; i < topk; i += TG) sel[i] = 0xFFFFFFFFu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dsv41_ixsort(sel, 512u, tid);
    for (uint i = tid; i < topk; i += TG)
        positions[row * topk + i] = sel[i] == 0xFFFFFFFFu ? -1 : int(sel[i]);
}
