const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const calls = @import("block_calls.zig");
const buffers = @import("buffers.zig");
const Config = @import("config.zig").Config;

pub fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try @import("widths_listing.zig").parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try @import("widths_listing.zig").parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..43) |layer| try w.gm.put(a, @intCast(layer), .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    for (0..43) |layer| for (4..8) |g| {
        const first = try std.fmt.allocPrint(a, "L{d}.attn.wo_a.0", .{layer});
        try w.dense.put(a, try std.fmt.allocPrint(a, "L{d}.attn.wo_a.{d}", .{ layer, g }), try w.k2(first));
    };
    return w;
}

fn arg(c: calls.Call, name: []const u8) calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    unreachable;
}

test "prefill heads: world four emits and plans the padded 32-head operation" {
    const cfg: Config = .{};
    try @import("block_admission.zig").nativePrefill(&cfg, 4);
    for (0..4) |rank| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var w = try widths(a);
        const opts: block.Options = .{ .world = 4, .rank = @intCast(rank) };
        var layers: [43]u32 = undefined;
        for (&layers, 0..) |*layer, i| layer.* = @intCast(i);
        const cs = try bp.emitPrefill(a, &cfg, &w, opts, &layers, 17, 0, true);
        var plan: buffers.Plan = .{ .a = a };
        try plan.add(cs);
        try t.expectEqual(@as(u64, 557056), plan.sizes.get("L.pf.q32").?);
        try t.expectEqual(@as(u64, 128), plan.sizes.get("L.pf.sink32").?);
        try t.expectEqual(@as(u64, 557056), plan.sizes.get("L.pf.o32").?);
        var seen: usize = 0;
        for (cs) |c| if (std.mem.eql(u8, c.name, "_fused")) {
            seen += 1;
            try t.expectEqual(@as(i64, 32), arg(c, "H").i);
            try t.expectEqual(@as(i64, 32), arg(c, "BMQ").i);
            try t.expectEqual(@as(i64, 8), arg(c, "HG").i);
            try t.expectEqualSlices(i64, &.{ 17, 32, 512 }, arg(c, "Q").t.shape);
            try t.expectEqualSlices(i64, &.{32}, arg(c, "SINK").t.shape);
            try t.expectEqualSlices(i64, &.{ 4, 17, 4096 }, arg(c, "OUT").t.shape);
            try t.expectEqual([3]i64{ 17, 1, 1 }, c.grid);
        };
        try t.expectEqual(@as(usize, 43), seen);
    }
}

test {
    _ = @import("prefill_heads_copy_test.zig");
}

test "prefill heads: several segments preserve row0 and group strides before the output projection" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(a);
    const cs = try bp.emitMulti(a, &cfg, &w, .{ .world = 4 }, &.{0}, &.{ .{ .slot = 0, .start = 0, .n = 33 }, .{ .slot = 1, .start = 128, .n = 17 } }, .whole);
    var seen: usize = 0;
    var copied: usize = 0;
    var query_at: usize = 0;
    for (cs, 0..) |c, i| {
        if (std.mem.eql(u8, c.name, "glue.pf_heads_q")) {
            query_at = i;
            try t.expectEqualSlices(i64, &.{ 50, 8192 }, c.args[0].arg.t.shape);
            try t.expectEqualSlices(i64, &.{ 50, 16384 }, c.args[1].arg.t.shape);
            var rope_at: ?usize = null;
            for (cs[0..i], 0..) |old, j| if (std.mem.eql(u8, old.name, "_rope")) {
                rope_at = j;
            };
            try t.expect(rope_at != null and rope_at.? < i);
        }
        if (std.mem.eql(u8, c.name, "_fused")) {
            try t.expect(query_at < i);
            const offset: i64 = if (seen == 0) 0 else 33 * 4096 * 2;
            const rows: i64 = if (seen == 0) 33 else 17;
            const out = arg(c, "OUT").t;
            try t.expectEqualSlices(i64, &.{ 4, rows, 4096 }, out.shape);
            try t.expectEqualSlices(i64, &.{ 50 * 4096, 4096, 1 }, out.stride);
            try t.expectEqual(offset, out.offset);
            try t.expectEqual(offset * 4, arg(c, "Q").t.offset);
            seen += 1;
        }
        if (std.mem.eql(u8, c.name, "glue.pf_heads_out")) {
            try t.expectEqual(seen, copied + 1);
            const src = c.args[0].arg.t;
            const dst = c.args[1].arg.t;
            try t.expectEqualSlices(i64, &.{ 2, src.shape[1], 4096 }, dst.shape);
            try t.expectEqualSlices(i64, src.stride, dst.stride);
            try t.expectEqual(src.offset, dst.offset);
            copied += 1;
        }
        if (std.mem.eql(u8, c.name, "_gemm") and arg(c, "OUT") == .t and std.mem.eql(u8, arg(c, "OUT").t.role.buf, "L.z")) try t.expectEqual(@as(usize, 2), copied);
    }
    try t.expectEqual(@as(usize, 2), seen);
    try t.expectEqual(seen, copied);
}
