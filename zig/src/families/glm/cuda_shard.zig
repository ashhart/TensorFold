//! A TP rank's share of GLM-5.3-Flash on CUDA (the Python family's ShardPlan): heads one by one, widths in 64s.

const std = @import("std");
const canon = @import("canon");
const Config = @import("config.zig").Config;

/// Elements of an MLP width or the vocabulary a rank takes whole (split.UNIT).
pub const unit = 64;

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
    kda: Span, // KDA heads
    mla: Span, // DSA heads
    dense: Span, // the dense MLP's intermediate rows
    moe: Span, // each routed expert's intermediate rows
    shared: Span, // the shared expert's intermediate rows
    vocab: Span, // the head's rows

    /// Rank `rank` of `world`; refused where a rank would be left empty (the Python family's split_units refuses it).
    pub fn of(c: *const Config, rank: usize, world: usize) !Share {
        if (world == 0 or rank >= world) return error.BadRank;
        return .{
            .rank = rank,
            .world = world,
            .kda = try span(c.kda_heads, 1, rank, world),
            .mla = try span(c.mla_heads, 1, rank, world),
            .dense = try span(c.dense_inter, unit, rank, world),
            .moe = try span(c.moe_inter, unit, rank, world),
            .shared = try span(c.shared_inter, unit, rank, world),
            .vocab = try span(c.vocab, c.head_unit, rank, world),
        };
    }

    /// This rank's largest fraction of an axis as (part, whole): its weights are at most that much of the checkpoint.
    pub fn part(s: Share, c: *const Config) [2]u64 {
        const axes = [_][2]u64{ .{ s.kda.len(), c.kda_heads }, .{ s.mla.len(), c.mla_heads }, .{ s.dense.len(), c.dense_inter }, .{ s.moe.len(), c.moe_inter }, .{ s.shared.len(), c.shared_inter }, .{ s.vocab.len(), c.vocab } };
        var best: [2]u64 = .{ 0, 1 };
        for (axes) |x| if (x[0] * best[1] > best[0] * x[1]) {
            best = .{ x[0], x[1] };
        };
        return best;
    }
};

/// Rank `rank`'s [lo, hi) of `total` elements taken in whole `per`s.
fn span(total: usize, per: usize, rank: usize, world: usize) !Span {
    if (total % per != 0 or world > 256) return error.BadSplit;
    const ones: [256]u64 = @splat(1);
    var out: [256]canon.Range = undefined;
    canon.partition(@intCast(total / per), ones[0..world], out[0..world]);
    const r = out[rank];
    if (r.len() == 0) return error.EmptyShare;
    return .{ .lo = r.begin * per, .hi = r.end * per };
}

test "a rank's part is its largest axis: TP=3 rank 0's 22 of 64 KDA heads, rank 2's 4,096 of 12,288 dense rows" {
    const c = Config{};
    const p0 = (try Share.of(&c, 0, 3)).part(&c);
    try std.testing.expectEqual(@as(u64, 11) * p0[1], 32 * p0[0]);
    const p2 = (try Share.of(&c, 2, 3)).part(&c);
    try std.testing.expectEqual(p2[1], 3 * p2[0]);
    const h = (try Share.of(&c, 1, 2)).part(&c);
    try std.testing.expectEqual(h[1], 2 * h[0]);
}

test "TP=2 halves every axis; TP=3 takes the Python family's uneven shares (22/21/21 heads, 704/704/640 expert rows)" {
    const c = Config{};
    const r1 = try Share.of(&c, 1, 2);
    try std.testing.expectEqual(Span{ .lo = 32, .hi = 64 }, r1.kda);
    try std.testing.expectEqual(Span{ .lo = 6144, .hi = 12288 }, r1.dense);
    try std.testing.expectEqual(Span{ .lo = 1024, .hi = 2048 }, r1.moe);
    try std.testing.expectEqual(Span{ .lo = 1024, .hi = 2048 }, r1.shared);
    try std.testing.expectEqual(Span{ .lo = 77440, .hi = 154880 }, r1.vocab);
    const t0 = try Share.of(&c, 0, 3);
    const t2 = try Share.of(&c, 2, 3);
    try std.testing.expectEqual(@as(usize, 22), t0.mla.len());
    try std.testing.expectEqual(Span{ .lo = 43, .hi = 64 }, t2.kda);
    try std.testing.expectEqual(@as(usize, 704), t0.moe.len());
    try std.testing.expectEqual(@as(usize, 640), t2.moe.len());
    try std.testing.expectEqual(Span{ .lo = 103296, .hi = 154880 }, t2.vocab);
    try std.testing.expectError(error.BadRank, Share.of(&c, 2, 2));
}

test "an EXL3 checkpoint cuts the vocabulary at 128s: TP=4 takes 303, 303, 302 and 302 blocks" {
    var c = Config{};
    c.head_unit = 128;
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 38784 }, (try Share.of(&c, 0, 4)).vocab);
    try std.testing.expectEqual(Span{ .lo = 116224, .hi = 154880 }, (try Share.of(&c, 3, 4)).vocab);
    try std.testing.expectEqual(Span{ .lo = 77440, .hi = 154880 }, (try Share.of(&c, 1, 2)).vocab);
}
