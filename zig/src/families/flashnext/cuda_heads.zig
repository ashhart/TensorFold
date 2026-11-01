//! A layer's head geometry: its DeltaNet key and value heads, attention query and KV heads, and the widths they set.

const std = @import("std");
const gdn = @import("cuda_gdn.zig");
const wt = @import("cuda_prompt_wt.zig");

pub const Heads = struct {
    nk: u32,
    nv: u32,
    q: u32,
    kv: u32,

    /// The checkpoint's whole layer: what one rank holds today, and the MTP head always.
    pub const full: Heads = .{ .nk = 16, .nv = 48, .q = 24, .kv = 2 };
    /// One rank's half under the head split: key and value heads halved (1:3 kept), one KV head and its 12 queries.
    pub const half: Heads = .{ .nk = 8, .nv = 24, .q = 12, .kv = 1 };

    pub fn eql(a: Heads, b: Heads) bool {
        return a.nk == b.nk and a.nv == b.nv and a.q == b.q and a.kv == b.kv;
    }

    /// The DeltaNet conv channels: q and k of every key head, v of every value head.
    pub fn convDim(h: Heads) usize {
        return 2 * h.nk * gdn.dk + h.nv * gdn.dv;
    }

    /// A DeltaNet projection row: the conv channels, then z, b and a.
    pub fn projWidth(h: Heads) usize {
        return h.convDim() + h.nv * gdn.dv + 2 * h.nv;
    }

    /// A DeltaNet output row, the out projection's input (an attention row's q * head_dim is the same width).
    pub fn valueDim(h: Heads) usize {
        return h.nv * gdn.dv;
    }

    /// An attention projection row: each query head's q and gate, k and v of each KV head, then the whole indexer.
    pub fn projN(h: Heads) usize {
        return (2 * h.q + 2 * h.kv) * wt.head_dim + wt.index_heads * wt.index_dim + wt.index_dim;
    }

    /// Bytes of one position's keys (or values) in the bf16 attention cache.
    pub fn kvRow(h: Heads) usize {
        return h.kv * wt.head_dim * 2;
    }

    /// What each of `world` ranks computes of a layer the head split cuts: the whole layer at one rank, half at two.
    pub fn split(world: usize) error{UnevenHeadSplit}!Heads {
        return switch (world) {
            1 => .full,
            2 => .half,
            else => error.UnevenHeadSplit,
        };
    }
};

comptime {
    const f = Heads.full;
    std.debug.assert(f.nk == gdn.nk and f.nv == gdn.nv and f.q == wt.q_heads and f.kv == wt.kv_heads);
    std.debug.assert(f.convDim() == gdn.conv_dim and f.projWidth() == gdn.proj_width and f.valueDim() == gdn.value_dim);
    std.debug.assert(f.projN() == wt.proj_n and f.kvRow() == wt.kv_heads * wt.head_dim * 2);
    for ([_]Heads{ f, Heads.half }) |h| std.debug.assert(h.q * wt.head_dim == h.valueDim() and h.nv == 3 * h.nk);
}

test "the full heads' widths are the checkpoint constants they replace" {
    const f = Heads.full;
    try std.testing.expectEqual(gdn.conv_dim, f.convDim());
    try std.testing.expectEqual(gdn.proj_width, f.projWidth());
    try std.testing.expectEqual(gdn.value_dim, f.valueDim());
    try std.testing.expectEqual(wt.proj_n, f.projN());
    try std.testing.expectEqual(wt.out_k, f.q * wt.head_dim);
    try std.testing.expectEqual(@as(usize, wt.kv_heads * wt.head_dim * 2), f.kvRow());
    try std.testing.expectEqual(@as(usize, 1024), f.kvRow());
}

test "half the heads: 5,120 conv channels, 8,240-wide DeltaNet rows, 3,072 values, 7,296-wide attention rows" {
    const h = Heads.half;
    try std.testing.expectEqual(@as(usize, 5120), h.convDim());
    try std.testing.expectEqual(@as(usize, 8240), h.projWidth());
    try std.testing.expectEqual(@as(usize, 3072), h.valueDim());
    try std.testing.expectEqual(@as(usize, 7296), h.projN());
    try std.testing.expectEqual(@as(usize, 512), h.kvRow());
    try std.testing.expectEqual(h.q / h.kv, Heads.full.q / Heads.full.kv);
}

test "one rank holds the whole layer, each of two half of it, and no other world splits" {
    try std.testing.expect((try Heads.split(1)).eql(.full));
    try std.testing.expect((try Heads.split(2)).eql(.half));
    for ([_]usize{ 0, 3, 4 }) |w| try std.testing.expectError(error.UnevenHeadSplit, Heads.split(w));
    try std.testing.expect(!Heads.half.eql(.full));
}
