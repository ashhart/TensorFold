//! The dense Qwen3.5 CUDA family's TP shares as mesh declarations: whole heads, packed segments, 128s, every refusal.
const std = @import("std");
const mesh = @import("mesh");
const shard = mesh.shard;
const cs = @import("cuda_shard.zig");

fn group(rank: usize, world: usize) mesh.Group {
    return mesh.tensorGroup(@intCast(rank), @intCast(world));
}

/// The refusal `Share.of` names for a mesh refusal; `uneven` is the name of an even split that does not divide.
fn named(e: shard.Error, uneven: cs.Error) cs.Error {
    return switch (e) {
        error.BadGroup => error.BadRank,
        error.NotOnUnits => error.NotOnTiles,
        error.EmptyShare => error.EmptyShare,
        error.Uneven => uneven,
        error.ReplicaMismatch => error.KvHeads,
    };
}

fn span(d: shard.Decl, total: usize, rank: usize, world: usize, uneven: cs.Error) cs.Error!cs.Span {
    const r = shard.range(d, total, group(rank, world)) catch |e| return named(e, uneven);
    return .{ .lo = @intCast(r.lo), .hi = @intCast(r.hi) };
}

fn segment(r: shard.Range) cs.Span {
    return .{ .lo = @intCast(r.lo), .hi = @intCast(r.hi) };
}

/// Every weight's share declared as the mesh declares it, in the order `Share.of` reads the widths.
fn declaredOf(s: cs.Shape, rank: usize, world: usize) anyerror!cs.Share {
    const hidden: shard.Decl = .{ .kind = .replicated, .unit = cs.unit };
    _ = try span(hidden, s.hidden, rank, world, error.NotOnTiles);
    const heads: shard.Decl = .{ .kind = .column, .unit = 1, .rule = .even };
    const q_rows: shard.Decl = .{ .kind = .column, .unit = 2 * cs.head_dim, .rule = .even };
    const q = try span(q_rows, s.q_heads * 2 * cs.head_dim, rank, world, error.UnevenSplit);
    const kv_rows: shard.Decl = .{ .kind = .column, .unit = cs.head_dim, .rule = .even, .replicate = true };
    const kv = try span(kv_rows, s.kv_heads * cs.head_dim, rank, world, error.KvHeads);
    const o_cols: shard.Decl = .{ .kind = .row, .unit = cs.head_dim, .rule = .even };
    const o = try span(o_cols, s.q_heads * cs.head_dim, rank, world, error.UnevenSplit);
    const segments = [3]u64{ s.k_heads * cs.dk, s.k_heads * cs.dk, s.v_heads * cs.dk };
    const qkv_rows: shard.Decl = .{ .kind = .column, .unit = cs.dk, .rule = .even, .segments = &segments };
    var cut: [3]shard.Range = undefined;
    const qkv = shard.ranges(qkv_rows, segments[0] + segments[1] + segments[2], group(rank, world), &cut) catch |e| return named(e, error.UnevenSplit);
    const z_rows: shard.Decl = .{ .kind = .column, .unit = cs.dk, .rule = .even };
    const z = try span(z_rows, s.v_heads * cs.dk, rank, world, error.UnevenSplit);
    const heads_rows = try span(heads, s.v_heads, rank, world, error.UnevenSplit);
    const out_cols: shard.Decl = .{ .kind = .row, .unit = cs.dk, .rule = .even };
    const out = try span(out_cols, s.v_heads * cs.dk, rank, world, error.UnevenSplit);
    const tile_rows: shard.Decl = .{ .kind = .column, .unit = cs.unit, .rule = .even };
    const tile_cols: shard.Decl = .{ .kind = .row, .unit = cs.unit, .rule = .even };
    const gate = try span(tile_rows, s.intermediate, rank, world, error.NotOnTiles);
    const down = try span(tile_cols, s.intermediate, rank, world, error.NotOnTiles);
    try std.testing.expectEqual(gate, down); // gate_proj, up_proj and down_proj cut one range
    const head: shard.Decl = .{ .kind = .vocab, .unit = cs.unit, .rule = .even };
    const vocab = try span(head, s.vocab, rank, world, error.NotOnTiles);
    return .{
        .rank = rank,
        .world = world,
        .q = q,
        .kv = kv,
        .o = o,
        .qkv = .{ segment(qkv[0]), segment(qkv[1]), segment(qkv[2]) },
        .z = z,
        .heads = heads_rows,
        .out = out,
        .mlp = gate,
        .vocab = vocab,
    };
}

fn expectSame(s: cs.Shape, rank: usize, world: usize) !void {
    const want = cs.Share.of(s, rank, world);
    const got = declaredOf(s, rank, world);
    if (want) |w| try std.testing.expectEqual(w, try got) else |e| try std.testing.expectError(e, got);
}

/// The 27B's widths (cuda_geometry.zig's test of them).
const qwen27: cs.Shape = .{ .hidden = 5120, .intermediate = 17408, .vocab = 248320, .q_heads = 24, .kv_heads = 4, .k_heads = 16, .v_heads = 48 };

test "the declared shares equal Share.of at every world to 16, its refusals and their order included" {
    var shapes: [8]cs.Shape = @splat(qwen27);
    shapes[1].hidden = 5120 + 64; // off the tiles: refused first, at every world
    shapes[2].kv_heads = 2; // fewer KV heads than ranks past world 2: kept by their query ranks, or KvHeads
    shapes[3].kv_heads = 3; // 3 does not split over 2 or 4 ranks
    shapes[4].intermediate = 17408 + 64; // the width is off the tiles
    shapes[5].k_heads = 12; // key heads that do not split by 8
    shapes[6].q_heads = 20; // two axes wrong at once: the query heads refuse first, where 20 does not split
    shapes[6].vocab = 248320 + 64;
    shapes[7].v_heads = 0; // nothing to cut: EmptyShare
    for (&shapes) |*s| {
        for (1..17) |world| for (0..world) |rank| try expectSame(s.*, rank, world);
        try expectSame(s.*, 0, 0);
        try expectSame(s.*, 2, 2);
        try expectSame(s.*, 0, cs.max_world + 1);
    }
}

test "the declared 27B shares at worlds 1, 2, 4 and 8 are the ones Share.of pins" {
    for ([_]usize{ 1, 2, 4, 8 }) |world| for (0..world) |rank| {
        const got = declaredOf(qwen27, rank, world);
        // 1,940 vocabulary tiles do not split by 8: the declaration refuses the head as Share.of does
        if (world == 8) try std.testing.expectError(error.NotOnTiles, got) else try std.testing.expectEqual(try cs.Share.of(qwen27, rank, world), try got);
    };
}
