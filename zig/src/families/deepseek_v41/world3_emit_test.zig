//! Three ranks' emitted calls take each rank's own heads and output groups; serving still refuses three ranks.

const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const calls = @import("block_calls.zig");
const ds = @import("dspark_emit.zig");
const Config = @import("config.zig").Config;
const widths = @import("prefill_heads_test.zig").widths;

const heads = [_]u64{ 24, 24, 16 };
const groups = [_]u64{ 3, 3, 2 };

/// Layer `L`'s attention in `cs` holds rank `rank`'s heads and groups, and reads only that rank's `wo_a` groups.
fn expectOwnAttention(a: std.mem.Allocator, cfg: *const Config, cs: []const calls.Call, L: u32, rank: usize, rows: u64) !void {
    var plan: buffers.Plan = .{ .a = a };
    try plan.add(cs);
    const columns = 2 * rows * heads[rank] * cfg.head_dim;
    try t.expectEqual(@as(?u64, columns), plan.sizes.get("L.q"));
    try t.expectEqual(@as(?u64, columns), plan.sizes.get("L.o"));
    try t.expectEqual(@as(?u64, 2 * rows * groups[rank] * cfg.o_lora), plan.sizes.get("L.oa"));
    try t.expect(plan.weights.contains(try std.fmt.allocPrint(a, "L{d}.attn.wo_a.{d}.suh", .{ L, groups[rank] - 1 })));
    try t.expect(!plan.weights.contains(try std.fmt.allocPrint(a, "L{d}.attn.wo_a.{d}.suh", .{ L, groups[rank] })));
}

test "world three: decode calls take each rank's own heads and output groups" {
    const cfg: Config = .{};
    for (0..3) |rank| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        const o: block.Options = .{ .world = 3, .rank = @intCast(rank) };
        try expectOwnAttention(a, &cfg, try block.emit(a, &cfg, &w, o, &.{0}, 17, 2048, true), 0, rank, 17);
    }
}

test "world three: DSpark calls take each rank's own heads and output groups" {
    const cfg: Config = .{};
    for (0..3) |rank| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        const o: block.Options = .{ .world = 3, .rank = @intCast(rank) };
        const cs = try ds.emitPass(a, &cfg, &w, o, cfg.dspark_block);
        try expectOwnAttention(a, &cfg, cs, cfg.layers, rank, @intCast(cfg.dspark_block));
    }
}

fn argOf(c: calls.Call, name: []const u8) calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    unreachable;
}

test "world three: every rank's prefill pads its heads to the 32-head core and copies out its own groups" {
    const cfg: Config = .{};
    for (0..3) |rank| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        const cs = try bp.emitPrefill(a, &cfg, &w, .{ .world = 3, .rank = @intCast(rank) }, &.{ 0, 40 }, 17, 0, true);
        var plan: buffers.Plan = .{ .a = a };
        try plan.add(cs);
        // the rank's own heads in q and in the core output it copies back, its own groups into z
        const columns = 2 * 17 * heads[rank] * cfg.head_dim;
        try t.expectEqual(@as(?u64, columns), plan.sizes.get("L.q"));
        try t.expectEqual(@as(?u64, columns), plan.sizes.get("L.o"));
        try t.expectEqual(@as(?u64, 2 * 17 * groups[rank] * cfg.o_lora), plan.sizes.get("L.z"));
        try t.expectEqual(@as(?u64, 557056), plan.sizes.get("L.pf.q32"));
        try t.expectEqual(@as(?u64, 128), plan.sizes.get("L.pf.sink32"));
        try t.expectEqual(@as(?u64, 557056), plan.sizes.get("L.pf.o32"));
        const real: i64 = @intCast(heads[rank] * cfg.head_dim);
        var fused: usize = 0;
        var copies: [3]usize = @splat(0);
        for (cs) |c| {
            if (std.mem.eql(u8, c.name, "_fused")) {
                fused += 1;
                try t.expectEqual(@as(i64, 32), argOf(c, "H").i);
                try t.expectEqual(@as(i64, 32), argOf(c, "BMQ").i);
                try t.expectEqual(@as(i64, 8), argOf(c, "HG").i);
                try t.expectEqualSlices(i64, &.{ 17, 32, 512 }, argOf(c, "Q").t.shape);
                try t.expectEqualSlices(i64, &.{32}, argOf(c, "SINK").t.shape);
                try t.expectEqualSlices(i64, &.{ 4, 17, 4096 }, argOf(c, "OUT").t.shape);
                try t.expectEqual([3]i64{ 17, 1, 1 }, c.grid);
            } else if (std.mem.eql(u8, c.name, "glue.pf_heads_q")) {
                copies[0] += 1;
                try t.expectEqualSlices(i64, &.{ 17, real }, c.args[0].arg.t.shape);
                try t.expectEqualSlices(i64, &.{ 17, 32 * 512 }, c.args[1].arg.t.shape);
            } else if (std.mem.eql(u8, c.name, "glue.pf_heads_sink")) {
                copies[1] += 1;
                try t.expectEqualSlices(i64, &.{@intCast(heads[rank])}, c.args[0].arg.t.shape);
            } else if (std.mem.eql(u8, c.name, "glue.pf_heads_out")) {
                copies[2] += 1;
                try t.expectEqualSlices(i64, &.{ 4, 17, 4096 }, c.args[0].arg.t.shape);
                try t.expectEqualSlices(i64, &.{ @intCast(groups[rank]), 17, 4096 }, c.args[1].arg.t.shape);
            }
        }
        try t.expectEqual(@as(usize, 2), fused);
        try t.expectEqual([3]usize{ 2, 2, 2 }, copies);
    }
}

test "world three: prefetch sources read each rank's own wq_b and wo_a groups" {
    const cfg: Config = .{};
    for (0..3) |rank| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try widths(a);
        var e: block.Emitter = .{ .a = a, .cfg = &cfg, .w = &w, .o = .{ .world = 3, .rank = @intCast(rank) }, .n = 1, .start = 0 };
        const q = try e.qGroup(0);
        try t.expectEqual(@as(i64, @intCast(heads[rank] * cfg.head_dim)), q.ns[0]);
        // each projection is a suh, a svh and its trellis words: the rank's wo_a groups, then wo_b
        try t.expectEqual(3 * (groups[rank] + 1), (try e.l2pfSources(0, .o)).len);
    }
}

test "prefill refuses a padded share whose group width does not tile the 32-head core" {
    // 48 heads in 4 output groups at two ranks: 24 heads a rank in groups of 12, which 32 is not a multiple of
    const cfg: Config = .{ .heads = 48, .o_groups = 4 };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    try t.expectError(error.Unsupported, bp.emitPrefill(a, &cfg, &w, .{ .world = 2, .rank = 0 }, &.{0}, 17, 0, true));
}
