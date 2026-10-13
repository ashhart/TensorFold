// Group routed slots into padded eight-row expert tiles; scatter preserves original token and output IDs.
#include <metal_stdlib>
using namespace metal;
struct TfExpertGroups { uint rows, experts, repeat, padded; };
#define TF_GROUP_ARGS \
    const device uint* ids [[buffer(0)]], device atomic_uint* counts [[buffer(1)]], \
    device uint* offsets [[buffer(2)]], device uint* ranks [[buffer(3)]], \
    device uint* expert_ids [[buffer(4)]], device uint* token_ids [[buffer(5)]], \
    device uint* slot_ids [[buffer(6)]], constant TfExpertGroups& d [[buffer(7)]], \
    uint i [[thread_position_in_grid]]

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
    for (uint expert = 0; expert < d.experts; ++expert) {
        offsets[expert] = offset;
        const uint count = atomic_load_explicit(counts + expert, memory_order_relaxed);
        offset += (count + 7u) & ~7u;
    }
}
kernel void tf_expert_group_scatter(TF_GROUP_ARGS) {
    if (i >= d.rows || ids[i] >= d.experts) return;
    const uint destination = offsets[ids[i]] + ranks[i];
    if (destination >= d.padded) return;
    expert_ids[destination] = ids[i];
    token_ids[destination] = i / d.repeat;
    slot_ids[destination] = i;
}
