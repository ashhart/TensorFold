//! CPU tests of Rig.close: only stream synchronization and readback are replaced; the comparison is the GPU gate's.
const std = @import("std");
const cuda = @import("cuda");
const Rig = @import("quant_ref_tests.zig").RigT;

const Readback = struct {
    var syncs: usize = 0;
    var copies: usize = 0;

    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        return cuda.abi.success;
    }

    fn copy(dst: ?*anyopaque, src: cuda.abi.DevicePtr, n: usize) callconv(.c) cuda.abi.Result {
        const out: [*]u8 = @ptrCast(dst.?);
        const in: [*]const u8 = @ptrFromInt(src);
        @memcpy(out[0..n], in[0..n]);
        copies += 1;
        return cuda.abi.success;
    }
};

const Closed = struct { failed: bool, last: Rig.Last };

/// One quiet close against zero within one: whether it failed its gate, and the element it reported.
fn close(comptime T: type, got: []const T) !Closed {
    var d: cuda.Driver = undefined;
    d.api.cuStreamSynchronize = Readback.sync;
    d.api.cuMemcpyDtoH_v2 = Readback.copy;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var rig: Rig = .{
        .gpu = .{ .d = &d, .ctx = undefined, .gpa = std.testing.allocator, .io = std.testing.io },
        .a = arena.allocator(),
        .s = .{ .d = &d, .handle = null },
        .rng = undefined,
        .quiet = true,
    };
    Readback.syncs = 0;
    Readback.copies = 0;
    const result = rig.close("CPU reference comparison", @intFromPtr(got.ptr), T == f32, &zero, &tolerance);
    try std.testing.expectEqual(@as(usize, 1), Readback.syncs);
    try std.testing.expectEqual(@as(usize, 1), Readback.copies);
    result catch |err| {
        if (err != error.TestFailed) return err;
        return .{ .failed = true, .last = rig.last };
    };
    return .{ .failed = false, .last = rig.last };
}

fn passes(comptime T: type, got: []const T, most: f64, at: usize) !void {
    try std.testing.expectEqual(Closed{ .failed = false, .last = .{ .most = most, .at = at } }, try close(T, got));
}

/// The close fails at element `at`: beyond the bound by `most`, or non-finite there when `most` is null.
fn fails(comptime T: type, got: []const T, most: ?f64, at: usize) !void {
    const c = try close(T, got);
    try std.testing.expect(c.failed);
    try std.testing.expectEqual(at, c.last.at);
    if (most) |m| return std.testing.expectEqual(m, c.last.most);
    try std.testing.expect(!std.math.isFinite(c.last.most));
}

const zero = [_]f64{ 0, 0, 0 };
const tolerance = [_]f64{ 1, 1, 1 };

test "finite FP32 and BF16 outputs pass at the bound and fail beyond it" {
    try passes(f32, &.{ 0.5, 1, 0 }, 1, 1);
    try passes(u16, &.{ 0x3f00, 0x3f80, 0 }, 1, 1);
    try fails(f32, &.{ 2, 0, 0 }, 2, 0);
    try fails(u16, &.{ 0x4000, 0, 0 }, 2, 0);
}

test "FP32 NaN cannot disappear before a later finite result" {
    for ([_]u32{ 0x7fc00000, 0xffffffff }) |bits| {
        for (0..3) |i| {
            var got = [_]f32{ 0, 0, 0 };
            got[i] = @bitCast(bits);
            try fails(f32, &got, null, i);
        }
    }
}

test "BF16 NaN cannot disappear before a later finite result" {
    for ([_]u16{ 0x7fc0, 0xffff }) |bits| {
        for (0..3) |i| {
            var got = [_]u16{ 0, 0, 0 };
            got[i] = bits;
            try fails(u16, &got, null, i);
        }
    }
}

test "NaN cannot erase an earlier out-of-bound or infinite result" {
    try fails(f32, &.{ 0, std.math.inf(f32), 0 }, null, 1);
    try fails(f32, &.{ 0, -std.math.inf(f32), 0 }, null, 1);
    try fails(u16, &.{ 0, 0x7f80, 0 }, null, 1);
    try fails(u16, &.{ 0, 0xff80, 0 }, null, 1);
    try fails(f32, &.{ 2, std.math.nan(f32), 0 }, null, 1);
    try fails(f32, &.{ std.math.inf(f32), std.math.nan(f32), 0 }, null, 0);
    try fails(u16, &.{ 0x4000, 0xffff, 0 }, null, 1);
    try fails(u16, &.{ 0x7f80, 0xffff, 0 }, null, 0);
}
