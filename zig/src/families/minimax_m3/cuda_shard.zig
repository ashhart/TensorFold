//! A TP rank's share of MiniMax-M3.1-Flash on CUDA: whole key heads, widths in 128s (EXL3 tiles and rotations).

const std = @import("std");
const cfg = @import("config.zig");

/// Elements of an MLP width or the vocabulary a rank takes whole.
pub const unit = 128;

pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

pub const Share = struct {
    rank: usize,
    world: usize,
    kv: Span, // key heads; a key head's query heads (heads / kv_heads of them) and its index head come with it
    dense: Span, // the dense MLP's intermediate rows
    moe: Span, // each routed expert's intermediate rows
    shared: Span, // the shared expert's intermediate rows
    vocab: Span, // the head's rows

    /// Rank `rank` of `world`; refused where a rank would be left empty or a cut would split a head or a 128-block.
    pub fn of(c: *const cfg.Config, rank: usize, world: usize) !Share {
        if (world == 0 or rank >= world) return error.BadRank;
        return .{
            .rank = rank,
            .world = world,
            .kv = try span(c.kv_heads, 1, rank, world),
            .dense = try span(c.dense_intermediate, unit, rank, world),
            .moe = try span(c.moe_intermediate, unit, rank, world),
            .shared = try span(c.shared_intermediate, unit, rank, world),
            .vocab = try span(c.vocab, unit, rank, world),
        };
    }

    /// This rank's query heads: the groups of its key heads.
    pub fn queries(s: Share, c: *const cfg.Config) Span {
        const g = c.heads / c.kv_heads;
        return .{ .lo = s.kv.lo * g, .hi = s.kv.hi * g };
    }

    /// This rank's largest fraction of an axis as (part, whole): its weights are at most that much of the checkpoint.
    pub fn part(s: Share, c: *const cfg.Config) [2]u64 {
        const axes = [_][2]u64{
            .{ s.kv.len(), c.kv_heads },
            .{ s.dense.len(), c.dense_intermediate },
            .{ s.moe.len(), c.moe_intermediate },
            .{ s.shared.len(), c.shared_intermediate },
            .{ s.vocab.len(), c.vocab },
        };
        var best: [2]u64 = .{ 0, 1 };
        for (axes) |x| if (x[0] * best[1] > best[0] * x[1]) {
            best = .{ x[0], x[1] };
        };
        return best;
    }
};

/// Rank `rank`'s [lo, hi) of `total` elements taken in whole `per`s, the first ranks one more when uneven.
fn span(total: usize, per: usize, rank: usize, world: usize) !Span {
    if (total % per != 0) return error.BadSplit;
    const units = total / per;
    if (units < world) return error.EmptyShare;
    const base = units / world;
    const extra = units % world;
    const lo = rank * base + @min(rank, extra);
    const hi = lo + base + @intFromBool(rank < extra);
    return .{ .lo = lo * per, .hi = hi * per };
}

test "TP=2 halves every axis of the release's shape (the vocabulary in 128s); TP=4 takes a key head each; TP=5 is refused" {
    const c = cfg.Config{};
    const r1 = try Share.of(&c, 1, 2);
    try std.testing.expectEqual(Span{ .lo = 2, .hi = 4 }, r1.kv);
    try std.testing.expectEqual(Span{ .lo = 32, .hi = 64 }, r1.queries(&c));
    try std.testing.expectEqual(Span{ .lo = 6144, .hi = 12288 }, r1.dense);
    try std.testing.expectEqual(Span{ .lo = 1536, .hi = 3072 }, r1.moe);
    try std.testing.expectEqual(Span{ .lo = 100096, .hi = 200064 }, r1.vocab); // 1563 blocks of 128: 782 and 781
    try std.testing.expectEqual(Span{ .lo = 3, .hi = 4 }, (try Share.of(&c, 3, 4)).kv);
    try std.testing.expectError(error.EmptyShare, Share.of(&c, 0, 5));
    try std.testing.expectError(error.BadRank, Share.of(&c, 2, 2));
    // rank 0's largest fraction is its 782 of the vocabulary's 1563 blocks
    try std.testing.expectEqual([2]u64{ 100096, 200064 }, (try Share.of(&c, 0, 2)).part(&c));
}
