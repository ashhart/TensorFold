//! Three ranks' emitted calls take each rank's own heads and output groups; no kernel or serving path runs them yet.

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

test "world three: prefill pads a rank of sixteen heads as world four does, and refuses twenty-four" {
    const cfg: Config = .{};
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    for ([_]u32{ 0, 1 }) |rank| {
        try t.expectError(error.Unsupported, bp.emitPrefill(a, &cfg, &w, .{ .world = 3, .rank = rank }, &.{ 0, 40 }, 17, 0, true));
    }
    var plan: buffers.Plan = .{ .a = a };
    try plan.add(try bp.emitPrefill(a, &cfg, &w, .{ .world = 3, .rank = 2 }, &.{ 0, 40 }, 17, 0, true));
    try t.expectEqual(@as(?u64, 557056), plan.sizes.get("L.pf.q32"));
    try t.expectEqual(@as(?u64, 128), plan.sizes.get("L.pf.sink32"));
    try t.expectEqual(@as(?u64, 557056), plan.sizes.get("L.pf.o32"));
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
