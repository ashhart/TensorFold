//! The lane products' widths on the host: K slices as recorded, every recorded text patched, columns put back.
const std = @import("std");
const ks = @import("kernel_sources");
const lane_w = @import("lane_w.zig");
const mtl = @import("metal");
const replay = @import("replay.zig");
const variants = @import("variants.zig");
const pio = @import("pack_io.zig");

test "split_k gives every recorded lane kernel's SK from its shape alone" {
    const recorded = [_][3]usize{ .{ 13952, 2560, 4 }, .{ 16480, 2560, 2 }, .{ 2560, 6144, 8 }, .{ 248320, 2560, 1 }, .{ 12800, 2560, 4 }, .{ 2560, 2560, 4 }, .{ 79616, 2560, 1 } };
    for (recorded) |c| try std.testing.expectEqual(c[2], lane_w.splitK(c[0], c[1]));
}

test "every recorded lane text takes any width, a row stride and a column map, and keeps its arithmetic" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var seen: usize = 0;
    for (ks.flashnext_gen.sources) |src| {
        if (std.mem.indexOf(u8, src.name, "lane_qmm_bytes_grouped") == null) continue;
        seen += 1;
        for ([_]bool{ false, true }) |mapped| {
            const out = try lane_w.source(a, src.text, .{ .n = 512, .k = 6144, .bits = 3, .group = 64, .tmr = 2, .ld = 13952, .mapped = mapped });
            try std.testing.expect(std.mem.indexOf(u8, out, "constexpr int BITS = 3;") != null);
            try std.testing.expect(std.mem.indexOf(u8, out, "constexpr int SK = 8;") != null);
            try std.testing.expect(std.mem.indexOf(u8, out, "constexpr int TMR = 2;") != null);
            try std.testing.expect(std.mem.indexOf(u8, out, "#define FZ_GS 64\n#define FZ_LD 13952\n") != null);
            try std.testing.expectEqual(@intFromBool(mapped), std.mem.count(u8, out, "FZ_MAP [[buffer(6)]]"));
            try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, out, "g * 32 + i"));
            // the sums: one fma per scale and bias, unchanged
            try std.testing.expectEqual(std.mem.count(u8, src.text, "fma("), std.mem.count(u8, out, "fma("));
        }
    }
    try std.testing.expectEqual(@as(usize, 16), seen);
}

test "a run's columns land at their members' fused columns, adjacent or not" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    // q 4 rows and idx 2 rows in run 0, v 3 rows in run 1: fused q | k(=run 0 too, 1 row) | v | idx
    const members = [_]u32{ 0, 0, 0, 0, 4, 4, 1, 0, 5, 0, 5, 8 };
    const maps = try lane_w.columnMaps(arena.allocator(), &members, &.{ 7, 3 });
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 8, 9 }, maps[0]);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, maps[1]);
    try std.testing.expectError(error.BadLaneMembers, lane_w.columnMaps(arena.allocator(), &.{ 0, 0, 0, 2, 0, 7 }, &.{ 7, 3 }));
}

/// A pack of a lane at two widths (8/64, 3/64), a 6/32 lane, and a hyper-connection with a widths table.
fn widePack(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    const synthetic = @import("pack_widen_test.zig").synthetic;
    var out = pio.Out.init(gpa);
    defer out.deinit();
    try @import("pack_lane.zig").putLane(&out, gpa, "L0.att.proj", &.{ try synthetic(a, 1, 64, 256, 8, 64), try synthetic(a, 2, 32, 256, 3, 64) });
    try @import("pack_lane.zig").putLane(&out, gpa, "L0.att.o", &.{try synthetic(a, 3, 32, 256, 6, 32)});
    for ([_][]const u8{ "scale", "down.w", "down.s", "down.b", "up.w", "up.s", "up.b" }) |part| {
        const bytes = try gpa.alloc(u8, 64);
        @memset(bytes, 0);
        try out.put(try std.fmt.allocPrint(a, "L0.ahc.{s}", .{part}), "U8", &.{64}, bytes);
    }
    const w = try gpa.alloc(u32, 4);
    w[0..4].* = .{ 4, 64, 4, 64 };
    try out.put("L0.ahc.widths", "U32", &.{4}, w);
    try out.write(gpa, io, path, "test");
}

test "a lane's and a hyper-connection's width tables are read on the host: only weights reach the device" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/pack.safetensors", .{tmp.sub_path}, 0);
    try widePack(std.testing.allocator, a, std.testing.io, path);
    var r = replay.Run{ .arena = a, .device = device, .queue = try device.queue() };
    try r.indexFile(path);
    var weights: usize = 0;
    for ([_][]const u8{ "L0.att.proj.r0.wq", "L0.att.proj.r0.sbt", "L0.att.proj.r1.wq", "L0.att.proj.r1.sbt" }) |n| weights += (try r.entry(n)).len;
    const w = (try lane_w.of(&r, "L0.att.proj")).?;
    try std.testing.expectEqual(@as(usize, 96), w.n);
    try std.testing.expectEqual(weights, r.loaded);
    const hc = try variants.hcOf(&r, "L0.ahc", .{});
    try std.testing.expectEqual(weights + 7 * 64, r.loaded);
    try std.testing.expectEqual([2][2]usize{ .{ 4, 64 }, .{ 4, 64 } }, hc.w.?);
}

test "a lane at other widths gives a 6-bit kernel nothing to read; a 6/32 lane gives its recorded pair" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/pack.safetensors", .{tmp.sub_path}, 0);
    try widePack(std.testing.allocator, a, std.testing.io, path);
    var r = replay.Run{ .arena = a, .device = device, .queue = try device.queue() };
    try r.indexFile(path);
    const wide = try lane_w.laneOf(&r, "L{d}.att.proj", .{0});
    try std.testing.expectEqual(@as(usize, 2), wide.w.?.runs.len);
    try std.testing.expectError(error.LaneNot6Bit32, wide.recorded());
    const six = try lane_w.laneOf(&r, "L{d}.att.o", .{0});
    try std.testing.expect(six.w == null);
    try std.testing.expectEqual((try r.entry("L0.att.o.wq")).len, (try six.recorded()).wq.b.length());
}
