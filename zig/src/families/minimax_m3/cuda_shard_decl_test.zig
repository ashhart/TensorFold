//! MiniMax-M3.1-Flash's TP shares as mesh declarations: whole key heads, 128s, every rank refused below world.
const std = @import("std");
const mesh = @import("mesh");
const shard = mesh.shard;
const cfg = @import("config.zig");
const ms = @import("cuda_shard.zig");

/// MiniMax's cut declared (worlds to shard.max_world, 256): every rank refused once units are below world (`.any`).
fn span(total: usize, per: usize, rank: usize, world: usize) !ms.Span {
    const d: shard.Decl = .{ .kind = .column, .unit = per, .empty = .any };
    const r = shard.range(d, total, mesh.tensorGroup(@intCast(rank), @intCast(world))) catch |e| return switch (e) {
        error.EmptyShare => error.EmptyShare,
        error.BadGroup, error.NotOnUnits, error.Uneven, error.ReplicaMismatch => error.BadSplit,
    };
    return .{ .lo = @intCast(r.lo), .hi = @intCast(r.hi) };
}

fn declaredOf(c: *const cfg.Config, rank: usize, world: usize) !ms.Share {
    if (world == 0 or rank >= world) return error.BadRank;
    return .{
        .rank = rank,
        .world = world,
        .kv = try span(c.kv_heads, 1, rank, world),
        .dense = try span(c.dense_intermediate, ms.unit, rank, world),
        .moe = try span(c.moe_intermediate, ms.unit, rank, world),
        .shared = try span(c.shared_intermediate, ms.unit, rank, world),
        .vocab = try span(c.vocab, ms.unit, rank, world),
    };
}

fn expectSame(c: *const cfg.Config, rank: usize, world: usize) !void {
    const want = ms.Share.of(c, rank, world);
    const got = declaredOf(c, rank, world);
    if (want) |w| {
        const g = try got;
        try std.testing.expectEqual(w, g);
        const kv: shard.Range = .{ .lo = g.kv.lo, .hi = g.kv.hi };
        const q = shard.derived(kv, c.heads, c.kv_heads);
        try std.testing.expectEqual(w.queries(c), ms.Span{ .lo = @intCast(q.lo), .hi = @intCast(q.hi) });
    } else |e| try std.testing.expectError(e, got);
}

test "the declared shares equal Share.of at every world to 16, its refusals and their order included" {
    var configs: [6]cfg.Config = .{ .{}, .{}, .{}, .{}, .{}, .{} };
    configs[1].kv_heads = 2; // fewer key heads than ranks past world 2: every rank refused
    configs[1].heads = 32;
    configs[2].dense_intermediate = 100; // not on 128s: BadSplit
    configs[3].kv_heads = 3; // two axes wrong at once: the key heads refuse first past world 3
    configs[3].heads = 48;
    configs[3].vocab = 1000;
    configs[4].heads = configs[4].kv_heads; // one query head a key head
    configs[5].heads = 8 * configs[5].kv_heads; // eight
    for (&configs) |*c| {
        for (1..17) |world| for (0..world) |rank| try expectSame(c, rank, world);
        try expectSame(c, 0, 0);
        try expectSame(c, 2, 2);
    }
}
