//! A TP rank's share of Inkling 2: whole key heads (their query groups with them), 128-wide expert and vocab blocks.
const std = @import("std");
const cfg = @import("shape.zig");

/// [lo, hi) of something a rank holds.
pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }

    /// Part `rank` of `world` of [0, n) in whole `unit`s, the earlier ranks taking any extra unit.
    pub fn part(n: usize, unit: usize, rank: usize, world: usize) !Span {
        if (n % unit != 0 or world == 0 or rank >= world) return error.BadSplit;
        const units = n / unit;
        if (units < world) return error.BadSplit;
        const base = units / world;
        const extra = units % world;
        const lo = rank * base + @min(rank, extra);
        return .{ .lo = lo * unit, .hi = (lo + base + @intFromBool(rank < extra)) * unit };
    }
};

pub const Share = struct {
    rank: usize,
    world: usize,
    kv: Span, // key heads
    heads: Span, // query heads: the key heads' groups
    inter: Span, // expert intermediate columns
    vocab: Span, // padded vocabulary rows of the head

    /// The rank's share; key heads must split evenly (each rank holds whole query groups and the same cache shape).
    pub fn of(c: *const cfg.Config, padded_vocab: usize, rank: usize, world: usize) !Share {
        if (c.kv_heads % world != 0) return error.BadSplit;
        const kv = try Span.part(c.kv_heads, 1, rank, world);
        const g = c.group();
        return .{
            .rank = rank,
            .world = world,
            .kv = kv,
            .heads = .{ .lo = kv.lo * g, .hi = kv.hi * g },
            .inter = try Span.part(c.intermediate, 128, rank, world),
            .vocab = try Span.part(padded_vocab, 128, rank, world),
        };
    }

    pub fn whole(s: Share) bool {
        return s.world == 1;
    }
};

test "shard: two ranks take whole key heads with their query groups, 128-wide expert and vocab blocks" {
    const c: cfg.Config = .{};
    const r0 = try Share.of(&c, 201088, 0, 2);
    const r1 = try Share.of(&c, 201088, 1, 2);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 4 }, r0.kv);
    try std.testing.expectEqual(Span{ .lo = 24, .hi = 48 }, r1.heads);
    try std.testing.expectEqual(Span{ .lo = 1152, .hi = 2304 }, r1.inter);
    // 201,088 = 1,571 blocks of 128: rank 0 takes the odd one
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 786 * 128 }, r0.vocab);
    try std.testing.expectEqual(@as(usize, 201088), r1.vocab.hi);
    const one = try Share.of(&c, 201088, 0, 1);
    try std.testing.expect(one.whole() and one.heads.len() == 48 and one.inter.len() == 2304);
    try std.testing.expectError(error.BadSplit, Share.of(&c, 201088, 0, 3));
}
