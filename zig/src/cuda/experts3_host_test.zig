//! The real pack wrapper's complete launch ABI, with failure propagation; no GPU arithmetic is simulated.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const A3 = @import("experts3.zig").Affine3Experts;
const A4 = @import("experts.zig").Affine4Experts;
const Mock = struct {
    var calls: usize = 0;
    var fail: usize = 0;
    var cfg: [3][7]u32 = undefined;
    var values: [3][10]u64 = undefined;
    fn read(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: abi.Stream, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
        std.debug.assert(stream == null and extra == null and calls < 3);
        const tag = @intFromPtr(f);
        cfg[calls] = .{ gx, gy, gz, bx, by, bz, shared };
        @memset(&values[calls], 0);
        const ptrs: usize = if (tag == 2) 4 else 2;
        for (0..ptrs) |i| values[calls][i] = read(u64, args, i);
        const n: usize = if (tag == 1) 3 else if (tag == 2) 10 else 5;
        for (ptrs..n) |i| values[calls][i] = if (tag != 2 and i == 2) @intCast(read(c_long, args, i)) else @intCast(read(c_int, args, i));
        calls += 1;
        return if (fail == tag) abi.error_not_ready else 0;
    }
};
fn run(failure: usize) !void {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    var a3: A3 = undefined;
    a3.widen_fn = .{ .d = &d, .handle = @ptrFromInt(1) };
    a3.planes_fn = .{ .d = &d, .handle = @ptrFromInt(3) };
    var a4: A4 = undefined;
    a4.pack_fn = .{ .d = &d, .handle = @ptrFromInt(2) };
    Mock.calls = 0;
    Mock.fail = failure;
    const result = a3.packInto(&a4, .{ .d = &d, .handle = null }, 1000, 2000, 3000, .{ 4000, 5000 }, 6000, 2, 64, 128, 2, 1);
    if (failure != 0) {
        try std.testing.expectError(error.NotReady, result);
        try std.testing.expectEqual(failure, Mock.calls);
        return;
    }
    try result;
    try std.testing.expectEqual(@as(usize, 3), Mock.calls);
    try std.testing.expectEqual([7]u32{ 8, 1, 1, 256, 1, 1, 0 }, Mock.cfg[0]);
    try std.testing.expectEqual([7]u32{ 2, 2, 2, 288, 1, 1, 0 }, Mock.cfg[1]);
    try std.testing.expectEqual([7]u32{ 1, 1, 1, 256, 1, 1, 0 }, Mock.cfg[2]);
    try std.testing.expectEqualSlices(u64, &.{ 1000, 4000, 2048 }, Mock.values[0][0..3]);
    try std.testing.expectEqualSlices(u64, &.{ 4000, 2000, 3000, 5000, 64, 16, 2, 2, 1, 0 }, &Mock.values[1]);
    try std.testing.expectEqualSlices(u64, &.{ 5000, 6000, 8, 448, 224 }, Mock.values[2][0..5]);
    try std.testing.expectEqual(@as(usize, 2 * 8 * 224 * 4), A3.pairBytes(2, 64, 128));
}
test "affine-3 packInto launches with every grid, block and shared size, pointer and scalar" {
    try run(0);
}
test "affine-3 packing stops at each injected launch failure" {
    for (1..4) |i| try run(i);
}
