//! Qwen3.5's TP cuts as mesh declarations: packed segments, kv heads kept by their query ranks, whole head counts.
const std = @import("std");
const mesh = @import("mesh");
const shard = mesh.shard;
const slicing = @import("weights/slicing.zig");
const Spec = @import("config.zig").Spec;
const quant = @import("core").quant;

const Span = quant.Span;
const worlds = [_]usize{ 1, 2, 4, 8 };

fn group(rank: usize, world: usize) mesh.Group {
    return mesh.tensorGroup(@intCast(rank), @intCast(world));
}

fn toSpan(r: shard.Range) Span {
    return .{ .from = @intCast(r.lo), .to = @intCast(r.hi) };
}

/// Whether `even` (core/quant/slice.zig) refuses one of `widths` at `world`; it logs at err, so tests skip it then.
fn uneven(widths: []const usize, world: usize) bool {
    for (widths) |w| if (w % world != 0) return true;
    return false;
}

test "packed outputs: each segment cut evenly at its offset, as segments cuts q | k | v and in_proj_qkv" {
    const packs = [_][]const usize{ &.{ 2048, 2048, 4096 }, &.{ 4096, 4096, 8192 }, &.{ 1536, 1536, 3072 }, &.{4096 * 2}, &.{ 6, 6, 12 } };
    for (packs) |widths| for (worlds) |world| {
        var segs: [3]u64 = undefined;
        var total: u64 = 0;
        for (widths, 0..) |w, i| {
            segs[i] = w;
            total += w;
        }
        const d: shard.Decl = .{ .kind = .column, .unit = 1, .rule = .even, .segments = segs[0..widths.len] };
        for (0..world) |rank| {
            var out: [3]shard.Range = undefined;
            const got = shard.ranges(d, total, group(rank, world), &out);
            if (uneven(widths, world)) {
                try std.testing.expectError(error.Uneven, got);
                continue;
            }
            var buf: [3]Span = undefined;
            const want = try slicing.segments(widths, .{ .rank = rank, .world = world }, &buf, "q_proj");
            const g = try got;
            for (want, g) |w, x| try std.testing.expectEqual(w, toSpan(x));
        }
    };
}

test "gate and up are two projections that share one range of the intermediate width" {
    for ([_]usize{ 12288, 17408, 6 }) |width| for (worlds) |world| for (0..world) |rank| {
        const got = shard.range(.{ .kind = .column, .unit = 1, .rule = .even }, width, group(rank, world));
        if (uneven(&.{width}, world)) {
            try std.testing.expectError(error.Uneven, got);
            continue;
        }
        var buf: [1]Span = undefined;
        const want = try slicing.segments(&.{width}, .{ .rank = rank, .world = world }, &buf, "mlp");
        try std.testing.expectEqual(want[0], toSpan(try got));
    };
}

test "kv heads: whole heads a rank, or one head kept by its query ranks, refused as kvSpan refuses" {
    const head_dim = 128;
    for ([_]usize{ 1, 2, 3, 4, 6, 8 }) |kv_heads| for (worlds ++ [_]usize{ 3, 6 }) |world| for (0..world) |rank| {
        var buf: [1]Span = undefined;
        const want = slicing.kvSpan(kv_heads * head_dim, kv_heads, .{ .rank = rank, .world = world }, &buf);
        const d: shard.Decl = .{ .kind = .column, .unit = head_dim, .rule = .even, .replicate = true };
        const got = shard.range(d, kv_heads * head_dim, group(rank, world)) catch |e| switch (e) {
            error.Uneven, error.ReplicaMismatch => {
                try std.testing.expectError(error.KvHeads, want);
                continue;
            },
            else => return e,
        };
        try std.testing.expectEqual((try want)[0], toSpan(got));
    };
}

test "localSpec's head counts are whole even shares of heads, key heads and value heads; the kv count stays its own" {
    var s: Spec = .{ .hidden = 4096, .intermediate = 12288, .n_layers = 4, .heads = 32, .kv_heads = 2, .head_dim = 256, .key_heads = 16, .value_heads = 32, .key_dim = 128, .value_dim = 128, .conv = 4, .vocab = 248320, .eps = 1e-6, .rope_theta = 1e7, .rotary_dim = 64, .bits = 4, .group = 64, .experts = 0, .top_k = 0, .moe_width = 0, .shared_width = 0, .mtp_layers = 0 };
    // heads, key heads, value heads, vocabulary: each fixture past the first breaks one count at world 8
    const fixtures = [_][4]usize{ .{ 32, 16, 32, 248320 }, .{ 36, 16, 32, 248320 }, .{ 32, 12, 32, 248320 }, .{ 32, 16, 36, 248320 }, .{ 32, 16, 32, 248324 } };
    for (fixtures) |h| for (worlds) |world| {
        s.heads = h[0];
        s.key_heads = h[1];
        s.value_heads = h[2];
        s.vocab = h[3];
        const even: shard.Decl = .{ .kind = .column, .unit = 1, .rule = .even };
        const counts = [_]usize{ s.heads, s.key_heads, s.value_heads, s.vocab };
        for (counts) |count| for (0..world) |rank| {
            const got = shard.range(even, count, group(rank, world));
            if (uneven(&.{count}, world)) try std.testing.expectError(error.Uneven, got) else try std.testing.expectEqual(count / world, (try got).len());
        };
        if (uneven(&counts, world)) continue; // localSpec refuses here, logging at err
        for (0..world) |rank| {
            const local = try slicing.localSpec(s, .{ .rank = rank, .world = world });
            try std.testing.expectEqual(local.heads, (try shard.range(even, s.heads, group(rank, world))).len());
            try std.testing.expectEqual(local.key_heads, (try shard.range(even, s.key_heads, group(rank, world))).len());
            try std.testing.expectEqual(local.value_heads, (try shard.range(even, s.value_heads, group(rank, world))).len());
            try std.testing.expectEqual(@max(1, s.kv_heads / world), local.kv_heads); // the count rule, not a share
        }
    };
}
