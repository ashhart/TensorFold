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

fn close(comptime T: type, got: []const T, want: []const f64, tol: []const f64) !void {
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
    };
    Readback.syncs = 0;
    Readback.copies = 0;
    const result = rig.close("CPU reference comparison", @intFromPtr(got.ptr), T == f32, want, tol);
    try std.testing.expectEqual(@as(usize, 1), Readback.syncs);
    try std.testing.expectEqual(@as(usize, 1), Readback.copies);
    return result;
}

const zero = [_]f64{ 0, 0, 0 };
const tolerance = [_]f64{ 1, 1, 1 };

test "finite FP32 and BF16 outputs pass at the bound and fail beyond it" {
    try close(f32, &.{ 0.5, 1, 0 }, &zero, &tolerance);
    try close(u16, &.{ 0x3f00, 0x3f80, 0 }, &zero, &tolerance);
    try std.testing.expectError(error.TestFailed, close(f32, &.{ 2, 0, 0 }, &zero, &tolerance));
    try std.testing.expectError(error.TestFailed, close(u16, &.{ 0x4000, 0, 0 }, &zero, &tolerance));
}

test "FP32 NaN cannot disappear before a later finite result" {
    for ([_]u32{ 0x7fc00000, 0xffffffff }) |bits| {
        for (0..3) |i| {
            var got = [_]f32{ 0, 0, 0 };
            got[i] = @bitCast(bits);
            try std.testing.expectError(error.TestFailed, close(f32, &got, &zero, &tolerance));
        }
    }
}

test "BF16 NaN cannot disappear before a later finite result" {
    for ([_]u16{ 0x7fc0, 0xffff }) |bits| {
        for (0..3) |i| {
            var got = [_]u16{ 0, 0, 0 };
            got[i] = bits;
            try std.testing.expectError(error.TestFailed, close(u16, &got, &zero, &tolerance));
        }
    }
}

test "NaN cannot erase an earlier out-of-bound or infinite result" {
    try std.testing.expectError(error.TestFailed, close(f32, &.{ 0, std.math.inf(f32), 0 }, &zero, &tolerance));
    try std.testing.expectError(error.TestFailed, close(f32, &.{ 0, -std.math.inf(f32), 0 }, &zero, &tolerance));
    try std.testing.expectError(error.TestFailed, close(u16, &.{ 0, 0x7f80, 0 }, &zero, &tolerance));
    try std.testing.expectError(error.TestFailed, close(u16, &.{ 0, 0xff80, 0 }, &zero, &tolerance));
    try std.testing.expectError(error.TestFailed, close(f32, &.{ 2, std.math.nan(f32), 0 }, &zero, &tolerance));
    try std.testing.expectError(error.TestFailed, close(f32, &.{ std.math.inf(f32), std.math.nan(f32), 0 }, &zero, &tolerance));
    try std.testing.expectError(error.TestFailed, close(u16, &.{ 0x4000, 0xffff, 0 }, &zero, &tolerance));
    try std.testing.expectError(error.TestFailed, close(u16, &.{ 0x7f80, 0xffff, 0 }, &zero, &tolerance));
}
