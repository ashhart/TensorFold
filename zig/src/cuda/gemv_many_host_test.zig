//! Exercise the real GB10 launch, including PDL and member-local geometry, with no device.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const ql = @import("qlinear.zig");
const A = ql.Affine4Kernels;
const Mock = struct {
    var rows: usize = 0;
    var calls: usize = 0;
    var failure = false;
    var wide = false;
    fn read(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(cfg: *const abi.LaunchConfig, _: abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        const t = std.testing;
        const narrow = rows <= 16;
        t.expectEqual(@as(u32, if (narrow) 36 else if (wide) 480 else 528), cfg.grid_x) catch unreachable;
        t.expectEqual(@as(u32, 1), cfg.grid_y) catch unreachable;
        t.expectEqual(@as(u32, 1), cfg.grid_z) catch unreachable;
        t.expectEqual(@as(u32, 128), cfg.block_x) catch unreachable;
        t.expectEqual(@as(u32, if (narrow) 35328 else if (rows <= 32) 32640 else if (rows <= 64) 43008 else 57600), cfg.shared_bytes) catch unreachable;
        t.expectEqual(@as(u32, if (narrow) 1 else 2), cfg.num_attrs) catch unreachable;
        t.expectEqual(abi.LaunchAttributeId.programmatic_stream_serialization, cfg.attrs.?[0].id) catch unreachable;
        t.expectEqual(@as(c_int, 1), cfg.attrs.?[0].value.int) catch unreachable;
        if (!narrow) {
            t.expectEqual(abi.LaunchAttributeId.cluster_dimension, cfg.attrs.?[1].id) catch unreachable;
            t.expectEqual(@as(u32, 8), cfg.attrs.?[1].value.cluster_dim.x) catch unreachable;
        }
        t.expectEqual(@as(u64, 10), read(u64, args, 0)) catch unreachable;
        t.expectEqual(@as(u64, 20), read(u64, args, 1)) catch unreachable;
        if (narrow) {
            const p = read(ql.GemvManyParts, args, 2);
            t.expectEqual(@as(c_int, 4), p.count) catch unreachable;
            for (p.p, [_]c_int{ 0, 17, 34, 35 }, [_]c_int{ 17, 17, 1, 1 }, 0..) |m, first, blocks, i| {
                t.expectEqual(first, m.first) catch unreachable;
                t.expectEqual(blocks, m.blocks) catch unreachable;
                t.expectEqual(@as(c_int, if (i < 2) (if (wide) 1 else 2) else 8), m.p.sk) catch unreachable;
                t.expectEqual(@as(u64, 100 + i * 10), m.p.w) catch unreachable;
                t.expectEqual(@as(u64, 200 + i * 10), m.p.out) catch unreachable;
            }
        } else {
            const p = read(ql.Parts, args, 2);
            const firsts = if (wide) [_]c_int{ 0, 34, 58, 59 } else [_]c_int{ 0, 40, 64, 65 };
            t.expectEqual(@as(c_int, 4), p.count) catch unreachable;
            for (p.p, firsts, 0..) |m, first, i| {
                t.expectEqual(first, m.first) catch unreachable;
                t.expectEqual(@as(c_int, if (i < 2) (if (wide) 1 else 2) else 8), m.sk) catch unreachable;
                t.expectEqual(@as(u64, 100 + i * 10), m.w) catch unreachable;
                t.expectEqual(@as(u64, 200 + i * 10), m.out) catch unreachable;
            }
            t.expectEqual(@as(c_int, 8), read(c_int, args, 7)) catch unreachable;
        }
        t.expectEqual(@as(c_int, @intCast(rows)), read(c_int, args, 3)) catch unreachable;
        t.expectEqual(@as(c_int, 5120), read(c_int, args, 4)) catch unreachable;
        calls += 1;
        return if (failure) abi.error_not_ready else 0;
    }
};
test "GB10 grouping: every row count keeps PDL, independent splits and local persistent grids" {
    var d: Driver = undefined;
    d.api.cuLaunchKernelEx = Mock.launch;
    var a: A = undefined;
    a.pdl = true;
    a.gemv_blocks = 17;
    a.many_fn = .{ .d = &d, .handle = @ptrFromInt(1) };
    a.verify_fn = .{ .d = &d, .handle = @ptrFromInt(2) };
    a.verify64_fn = .{ .d = &d, .handle = @ptrFromInt(3) };
    a.verify128_fn = .{ .d = &d, .handle = @ptrFromInt(4) };
    const stream: @import("stream.zig").Stream = .{ .d = &d, .handle = null };
    var outs: [4]A.Out = undefined;
    Mock.calls = 0;
    Mock.failure = false;
    for ([_]bool{ false, true }) |wide| {
        Mock.wide = wide;
        for ([_]usize{ if (wide) 17408 else 10240, if (wide) 12288 else 6144, 48, 48 }, 0..) |n, i| outs[i] = .{ .q = .{ .w = 100 + i * 10, .s = 101 + i * 10, .b = 102 + i * 10, .n = n, .npad = (n + 63) / 64 * 64, .k = 5120 }, .out = 200 + i * 10 };
        for (1..129) |rows| {
            Mock.rows = rows;
            if (rows <= 16) try a.decodeMany(stream, 10, 20, &outs, rows) else try a.verifyMany(stream, 10, 20, &outs, rows);
        }
    }
    try std.testing.expectEqual(@as(usize, 256), Mock.calls);
    try std.testing.expectError(error.WindowTooWide, a.verifyMany(stream, 10, 20, &outs, 129));
    try std.testing.expectError(error.WindowTooWide, a.verifyMany(stream, 10, 20, &outs, 0));
    outs[1].out = outs[0].out;
    try std.testing.expectError(error.InputMismatch, a.verifyMany(stream, 10, 20, &outs, 17));
    outs[1].out = 210;
    outs[1].q.k = 4096;
    try std.testing.expectError(error.InputMismatch, a.verifyMany(stream, 10, 20, &outs, 17));
    outs[1].q.k = 5120;
    Mock.rows = 17;
    Mock.failure = true;
    try std.testing.expectError(error.NotReady, a.verifyMany(stream, 10, 20, &outs, 17));
    Mock.failure = false;
}
