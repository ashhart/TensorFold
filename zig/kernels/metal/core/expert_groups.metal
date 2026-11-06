// Group routed slots into padded expert tiles (sixty-four rows, or the tile's own width -- ROUND-348);
// scatter preserves original token and output IDs.
#include <metal_stdlib>
using namespace metal;
struct TfExpertGroups { uint rows, experts, repeat, padded, block, tile; };
#define TF_GROUP_ARGS \
    const device uint* ids [[buffer(0)]], device atomic_uint* counts [[buffer(1)]], \
    device uint* offsets [[buffer(2)]], device uint* ranks [[buffer(3)]], \
    device uint* expert_ids [[buffer(4)]], device uint* token_ids [[buffer(5)]], \
    device uint* slot_ids [[buffer(6)]], device uint* destinations [[buffer(7)]], \
    constant TfExpertGroups& d [[buffer(8)]], uint i [[thread_position_in_grid]]

kernel void tf_expert_group_clear(TF_GROUP_ARGS) {
    if (i < d.experts) atomic_store_explicit(counts + i, 0u, memory_order_relaxed);
    if (i < d.padded) expert_ids[i] = 0xffffffffu;
}
kernel void tf_expert_group_count(TF_GROUP_ARGS) {
    if (i >= d.rows || ids[i] >= d.experts) return;
    ranks[i] = atomic_fetch_add_explicit(counts + ids[i], 1u, memory_order_relaxed);
}
kernel void tf_expert_group_prefix(TF_GROUP_ARGS) {
    if (i != 0) return;
    uint offset = 0;
    // ROUND-348: each expert's block is padded to a multiple of the tile's rows (the tile reads one expert's
    // weights, so a tile may not straddle two experts); the tile's width is the plan's own.
    const uint tile = d.tile == 0u ? 64u : d.tile;
    for (uint expert = 0; expert < d.experts; ++expert) {
        offsets[expert] = offset;
        const uint count = atomic_load_explicit(counts + expert, memory_order_relaxed);
        offset += (count + tile - 1u) & ~(tile - 1u);
    }
}
kernel void tf_expert_group_scatter(TF_GROUP_ARGS) {
    if (i >= d.rows || ids[i] >= d.experts) return;
    const uint destination = offsets[ids[i]] + ranks[i];
    if (destination >= d.padded) return;
    expert_ids[destination] = ids[i];
    // A fused gate+up table is two blocks of `repeat` slots a token (gate picks, then up picks one table entry
    // further), so a slot's token is its position inside its block, not a running quotient.
    token_ids[destination] = d.block == 0 ? i / d.repeat : (i % d.block) / d.repeat;
    slot_ids[destination] = i;
    // The inverse map, token-major slot to expert-tile destination: the fused SwiGLU+rotate pass writes a row's
    // rotated down input where the tile GEMM reads it, so the down call skips its rotate_in.
    destinations[i] = destination;
}
