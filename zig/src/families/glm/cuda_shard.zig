//! A TP rank's share of GLM-5.3-Flash on CUDA (the Python family's ShardPlan): heads one by one, widths in 64s.

const std = @import("std");
const canon = @import("canon");
const mesh = @import("mesh");
const shard = mesh.shard;
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
        const d = decls(c);
        return .{
            .rank = rank,
            .world = world,
            .kda = try span(d.kda, c.kda_heads, rank, world),
            .mla = try span(d.mla, c.mla_heads, rank, world),
            .dense = try span(d.dense, c.dense_inter, rank, world),
            .moe = try span(d.moe, c.moe_inter, rank, world),
            .shared = try span(d.shared, c.shared_inter, rank, world),
            .vocab = try span(d.vocab, c.vocab, rank, world),
        };
    }

    /// This rank's largest fraction of an axis as (part, whole): its weights are at most that much of the checkpoint.
    pub fn part(s: Share, c: *const Config) [2]u64 {
        const axes = [_][2]u64{
            .{ s.kda.len(), c.kda_heads },
            .{ s.mla.len(), c.mla_heads },
            .{ s.dense.len(), c.dense_inter },
            .{ s.moe.len(), c.moe_inter },
            .{ s.shared.len(), c.shared_inter },
            .{ s.vocab.len(), c.vocab },
        };
        var best: [2]u64 = .{ 0, 1 };
        for (axes) |x| if (x[0] * best[1] > best[0] * x[1]) {
            best = .{ x[0], x[1] };
        };
        return best;
    }
};

/// GLM's shares, declared: heads one by one, widths in 64s (EXL3 experts in 128s), the vocabulary in head_unit.
pub const Decls = struct { kda: shard.Decl, mla: shard.Decl, dense: shard.Decl, moe: shard.Decl, shared: shard.Decl, vocab: shard.Decl };

pub fn decls(c: *const Config) Decls {
    return .{
        .kda = .{ .kind = .column, .unit = 1 },
        .mla = .{ .kind = .column, .unit = 1 },
        .dense = .{ .kind = .column, .unit = unit },
        .moe = .{ .kind = .column, .unit = if (c.exl3) 128 else unit }, // EXL3 tiles and rotations: 128
        .shared = .{ .kind = .column, .unit = unit },
        .vocab = .{ .kind = .vocab, .unit = c.head_unit },
    };
}

/// Rank `rank`'s [lo, hi) of `total` elements under `d`, refused as GLM always has: BadSplit, else EmptyShare.
fn span(d: shard.Decl, total: usize, rank: usize, world: usize) !Span {
    if (world > shard.max_world) return error.BadSplit;
    const r = shard.range(d, total, mesh.tensorGroup(@intCast(rank), @intCast(world))) catch |e| return switch (e) {
        error.EmptyShare => error.EmptyShare,
        error.BadGroup, error.NotOnUnits, error.Uneven, error.ReplicaMismatch => error.BadSplit,
    };
    return .{ .lo = @intCast(r.lo), .hi = @intCast(r.hi) };
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

/// The cut `Share.of` made before its shares were declared, kept as the oracle the declarations must equal.
fn legacySpan(total: usize, per: usize, rank: usize, world: usize) !Span {
    if (total % per != 0 or world > 256) return error.BadSplit;
    const ones: [256]u64 = @splat(1);
    var out: [256]canon.Range = undefined;
    canon.partition(@intCast(total / per), ones[0..world], out[0..world]);
    const r = out[rank];
    if (r.len() == 0) return error.EmptyShare;
    return .{ .lo = r.begin * per, .hi = r.end * per };
}

fn legacyOf(c: *const Config, rank: usize, world: usize) !Share {
    if (world == 0 or rank >= world) return error.BadRank;
    return .{
        .rank = rank,
        .world = world,
        .kda = try legacySpan(c.kda_heads, 1, rank, world),
        .mla = try legacySpan(c.mla_heads, 1, rank, world),
        .dense = try legacySpan(c.dense_inter, unit, rank, world),
        .moe = try legacySpan(c.moe_inter, if (c.exl3) 128 else unit, rank, world),
        .shared = try legacySpan(c.shared_inter, unit, rank, world),
        .vocab = try legacySpan(c.vocab, c.head_unit, rank, world),
    };
}

fn expectSame(c: *const Config, rank: usize, world: usize) !void {
    const want = legacyOf(c, rank, world);
    const got = Share.of(c, rank, world);
    if (want) |w| try std.testing.expectEqual(w, try got) else |e| try std.testing.expectError(e, got);
}

test "the declared shares equal the cut before them at every world to 16, refusals and their order included" {
    var configs: [5]Config = .{ .{}, .{}, .{}, .{}, .{} };
    configs[1].head_unit = 128; // EXL3: the vocabulary and the routed experts in 128s
    configs[1].exl3 = true;
    configs[2].kda_heads = 5; // too few heads for wide worlds: EmptyShare on the last ranks only
    configs[3].dense_inter = 100; // not on 64s: BadSplit, after the head axes
    configs[3].kda_heads = 2; // and an EmptyShare on the KDA axis first, past world 2
    configs[4].mla_heads = 3;
    configs[4].vocab = 200; // two axes wrong at once: the first in Share.of's order decides
    for (&configs) |*c| {
        for (1..17) |world| for (0..world) |rank| try expectSame(c, rank, world);
        try expectSame(c, 0, 0);
        try expectSame(c, 2, 2);
        try expectSame(c, 0, 257);
    }
}
