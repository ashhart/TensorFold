// Source is compiled after kv.metal; no query-by-key-by-head score tensor is materialized.
struct Dsv41Index {
    uint rows, heads, width, q_stride, weight_stride, position, ratio, keys, key_stride;
    uint tile_start, tile_width, topk, candidate_topk, block_size, use_candidates, make_candidates;
    float scale;
};
struct Dsv41Pick { float score; int position; };
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
    uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint local = g.x, row = g.y;
    if (local >= p.tile_width || row >= p.rows) return;
    int pos = dsv41_index_position(local, row, candidates, p);
    uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    if (pos < 0 || uint(pos) >= visible) {
        if (lane == 0) scores[row * p.tile_width + local] = -INFINITY;
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
        const float scaled_weight = dsv41_bf(dsv41_round(dsv41_bf(weight[row * p.weight_stride + head]) * p.scale));
        total += dsv41_bf(dsv41_round(max(dot, 0.0f) * scaled_weight));
    }
    if (lane == 0) scores[row * p.tile_width + local] = dsv41_bf(dsv41_round(total));
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
    for (uint i = 0; i < p.tile_width; ++i) {
        float score = scores[row * p.tile_width + i];
        int pos = dsv41_index_position(i, row, candidate, p);
        if (isfinite(score)) dsv41_heap(heap, p.topk, {score, pos});
    }
    if (!p.make_candidates) return;
    uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    device Dsv41Pick *blocks = candidate + row * p.candidate_topk;
    for (uint base = 0; base < p.tile_width; base += p.block_size) {
        float score = -INFINITY;
        for (uint j = 0; j < p.block_size; ++j) score = max(score, scores[row * p.tile_width + base + j]);
        uint block = (p.tile_start + base) / p.block_size;
        if (visible && block == (visible - 1) / p.block_size) score = INFINITY;
        if (score > -INFINITY) dsv41_heap(blocks, p.candidate_topk, {score, int(block)});
    }
}
inline void dsv41_sort_positions(device Dsv41Pick *p, uint size) {
    // Selection is score-ordered; attention sees the reference's ascending position order.
    for (uint i = 1; i < size; ++i) {
        Dsv41Pick value = p[i]; uint j = i;
        while (j > 0 && (p[j - 1].position < 0 ||
               (value.position >= 0 && p[j - 1].position > value.position))) {
            if (value.position < 0) break;
            p[j] = p[j - 1]; --j;
        }
        p[j] = value;
    }
}
kernel void dsv41_index_finish(device Dsv41Pick *top [[buffer(0)]], device Dsv41Pick *candidate [[buffer(1)]],
    device int *positions [[buffer(2)]], constant Dsv41Index &p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device Dsv41Pick *heap = top + row * p.topk;
    dsv41_sort_positions(heap, p.topk);
    for (uint i = 0; i < p.topk; ++i) positions[row * p.topk + i] = heap[i].position;
    if (p.make_candidates) dsv41_sort_positions(candidate + row * p.candidate_topk, p.candidate_topk);
}
