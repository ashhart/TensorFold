//! One stream's device bytes at a rank's heads: DeltaNet conv windows and states, attention caches, the n-gram tail.
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const pages = @import("cuda_pages.zig");
const Heads = @import("cuda_heads.zig").Heads;

const linear_n = wt.linear_n;
const attn_n = wt.attn_n;
const index_dim = wt.index_dim;
const max_tokens = wt.window;

/// One DeltaNet layer's conv window: the last taps - 1 rows of the heads' conv channels, bf16.
pub fn convBytes(h: Heads) usize {
    return (gdn.taps - 1) * h.convDim() * 2;
}

/// One DeltaNet layer's recurrent state: a dk x dv fp32 matrix for each of the heads' value heads.
pub fn stateBytes(h: Heads) usize {
    return h.nv * gdn.dk * gdn.dv * 4;
}

/// One attention layer's keys (or values) for `cap` positions of the heads' KV heads.
pub fn kvBytes(cap: usize, h: Heads) usize {
    return cap * h.kvRow();
}

/// One attention layer's indexer keys for `cap` positions: whole on every rank.
pub fn indexBytes(cap: usize) usize {
    return cap * index_dim * 2;
}

/// One attention layer's pooled indexer keys for a cache of `cap` tokens: a bf16 row of 128 a block of 4.
pub fn pooledBytes(cap: usize) usize {
    return (cap + wt.index_ratio - 1) / wt.index_ratio * index_dim * 2;
}

/// Every attention layer's contiguous caches for `cap` positions: keys, values, indexer keys and pooled blocks.
fn attnBytes(cap: usize, h: Heads) usize {
    return attn_n * (2 * kvBytes(cap, h) + indexBytes(cap) + pooledBytes(cap));
}

/// Device bytes `Engine.newSeq(cap)` allocates: conv tails, recurrent states, KV with indexer keys, the n-gram tail.
pub fn seqBytes(cap: usize, h: Heads) usize {
    const linear = linear_n * (convBytes(h) + stateBytes(h));
    return linear + attnBytes(cap, h) + wt.ple_tail_n * wt.wide * 2;
}

/// Contiguous storage or the page diagnostic's rounded arena and table, with every live allocation priced.
pub fn seqBytesPaged(cap: usize, tokens: usize, h: Heads) !usize {
    if (cap == 0 or cap > max_tokens) return error.BadFlashNextPageTest;
    if (tokens == 0) return seqBytes(cap, h);
    // the pages hold the whole layer's KV rows (cuda_pages.zig): a rank's half heads take none
    if (!h.eql(.full)) return error.FlashNextPagesUnsupported;
    const p = try pages.checkedLayout(cap, tokens);
    return std.math.add(usize, seqBytes(cap, h) - attnBytes(cap, h), p.total) catch error.BadFlashNextPageTest;
}

test "a sequence's device bytes grow by the twelve attention layers' keys, values, indexer keys and pooled blocks" {
    for ([_]Heads{ .full, .half }) |h| {
        const per_token = attn_n * (2 * h.kv * wt.head_dim * 2 + index_dim * 2);
        try std.testing.expectEqual(per_token, seqBytes(6, h) - seqBytes(5, h));
        try std.testing.expectEqual(per_token + attn_n * index_dim * 2, seqBytes(5, h) - seqBytes(4, h));
        try std.testing.expect(seqBytes(max_tokens, h) > max_tokens * per_token);
    }
}

test "a stream at 65,536 tokens: 1.84 GiB with the whole layer, 1.04 GiB with a rank's half of the heads" {
    const cap = 65536;
    // 36 DeltaNet layers: 3 conv rows of 10,240 or 5,120 bf16 channels, 48 or 24 fp32 states of 128 x 128
    const full_linear: usize = 36 * (3 * 10240 * 2 + 48 * 128 * 128 * 4);
    const half_linear: usize = 36 * (3 * 5120 * 2 + 24 * 128 * 128 * 4);
    // 12 attention layers: 2 or 1 KV heads of 256 bf16 keys and values, 128 indexer keys, a pooled row a block of 4
    const full_attn: usize = 12 * (cap * (2 * 2 * 256 * 2 + 128 * 2) + cap / 4 * 128 * 2);
    const half_attn: usize = 12 * (cap * (2 * 1 * 256 * 2 + 128 * 2) + cap / 4 * 128 * 2);
    const tail: usize = 9 * 4 * 2560 * 2;
    try std.testing.expectEqual(full_linear + full_attn + tail, seqBytes(cap, .full));
    try std.testing.expectEqual(half_linear + half_attn + tail, seqBytes(cap, .half));
    try std.testing.expectEqual(@as(usize, 1_977_913_344), seqBytes(cap, .full));
    try std.testing.expectEqual(@as(usize, 1_114_877_952), seqBytes(cap, .half));
    try std.testing.expectEqual(@as(usize, 30720), convBytes(.half));
    try std.testing.expectEqual(@as(usize, 1_572_864), stateBytes(.half));
    try std.testing.expectEqual(@as(usize, cap * 512), kvBytes(cap, .half));
}

test "the page diagnostic prices the whole layer's pages and refuses a rank's half of the heads" {
    try std.testing.expectEqual(seqBytes(64, .half), try seqBytesPaged(64, 0, .half));
    try std.testing.expectError(error.FlashNextPagesUnsupported, seqBytesPaged(64, 16, .half));
    try std.testing.expect(try seqBytesPaged(64, 16, .full) > 0);
    try std.testing.expectError(error.BadFlashNextPageTest, seqBytesPaged(0, 0, .half));
}
