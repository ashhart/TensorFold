//! Record the production ABI, ragged column grid and failure path; no driver or GPU is opened.
const std = @import("std");
const t = std.testing;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const tile = @import("experts_tile.zig");
const Mock = struct {
    var calls: usize = 0;
    var failed = false;
    var cfg: [7]u32 = undefined;
    var values: [14]u64 = undefined;
    fn read(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(_: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: abi.Stream, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
        std.debug.assert(stream == null and extra == null);
        cfg = .{ gx, gy, gz, bx, by, bz, shared };
        for (0..14) |i| values[i] = switch (i) {
            0, 3, 4, 7, 8, 9, 10 => read(u64, args, i),
            12 => @as(u32, @bitCast(read(f32, args, i))),
            13 => @as(u32, @bitCast(read(i32, args, i))),
            else => @intCast(read(i32, args, i)),
        };
        calls += 1;
        return if (failed) abi.error_not_ready else 0;
    }
};
test "K32 expert tile: explicit opt-in modes and invalid settings" {
    try t.expectEqual(tile.Mode.off, try tile.mode(null));
    try t.expectEqual(tile.Mode.off, try tile.mode("0"));
    try t.expectEqual(tile.Mode.up, try tile.mode("1"));
    try t.expectEqual(tile.Mode.both, try tile.mode("2"));
    for ([_][]const u8{ "", "true", "3", "2,0,1" }) |v| try t.expectError(error.BadPromptExpertTile, tile.mode(v));
}
test "K32 expert tile: actual launch preserves every pointer, scalar and ragged block" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    const k: tile.Tile = .{ .mod = undefined, .fns = .{ .{ .d = &d, .handle = @ptrFromInt(1) }, .{ .d = &d, .handle = @ptrFromInt(2) } }, .resident = .{ 17, 11 }, .selection = .both };
    const layer: @import("experts.zig").Layer = .{ .format = .nvfp4, .up = 20, .down = 30, .up_scale = 40, .down_scale = 50, .width = 320, .dims = 2560, .experts = 128, .limit = 3 };
    const plan: @import("grouped.zig").Plan = .{ .items = 60, .counts = 70, .members = 80, .rank = 90, .hist = 100 };
    const stream: @import("stream.zig").Stream = .{ .d = &d, .handle = null };
    Mock.calls = 0;
    Mock.failed = false;
    try k.run(stream, .up, .{ .x = 10, .stride = 2560, .slots = 9 }, layer, plan, 110, 5, 128);
    try t.expectEqual([7]u32{ 15, 1, 1, 128, 1, 1, 34816 }, Mock.cfg);
    try t.expectEqual([14]u64{ 10, 2560, 9, 20, 40, 80, 10, 60, 70, 80, 110, 320, @as(u32, @bitCast(@as(f32, 3))), 128 }, Mock.values);
    try k.run(stream, .down_f32, .{ .x = 120, .stride = 320, .slots = 9 }, layer, plan, 130, 5, -1);
    try t.expectEqual([7]u32{ 11, 1, 1, 128, 1, 1, 19200 }, Mock.cfg);
    try t.expectEqual([14]u64{ 120, 320, 0, 30, 50, 10, 80, 60, 70, 80, 130, 2560, 0, 0xffffffff }, Mock.values);
    const calls = Mock.calls;
    try k.run(stream, .up, .{ .x = 10, .stride = 2560 }, layer, plan, 110, 0, -1);
    try t.expectEqual(calls, Mock.calls);
    Mock.failed = true;
    try t.expectError(error.NotReady, k.run(stream, .up, .{ .x = 10, .stride = 2560 }, layer, plan, 110, 1, -1));
    try t.expectEqual(calls + 1, Mock.calls);
    Mock.failed = false;
    var bad = layer;
    bad.width = 33;
    try t.expectError(error.BadExpertShape, k.run(stream, .up, .{ .x = 10, .stride = 2560 }, bad, plan, 110, 1, -1));
    try t.expectError(error.EpilogueNotBuilt, k.run(stream, .down_bf16, .{ .x = 10, .stride = 2560 }, layer, plan, 110, 1, -1));
    bad = layer;
    bad.relu2 = true;
    try t.expectError(error.EpilogueNotBuilt, k.run(stream, .up, .{ .x = 10, .stride = 2560 }, bad, plan, 110, 1, -1));
    try t.expectEqual(calls + 1, Mock.calls);
}
