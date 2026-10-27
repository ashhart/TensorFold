//! Actual family dispatch keeps the baseline function and ABI unless the prompt opt-in selects that epilogue.
const std = @import("std");
const cuda = @import("cuda");
const tile = @import("cuda_nvfp4_tile.zig");
const Mock = struct {
    var selected: usize = 0;
    var threads: u32 = 0;
    var shared: u32 = 0;
    var args: [14]u64 = undefined;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, x: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        selected = @intFromPtr(f.?);
        threads = x;
        shared = sm;
        for (0..14) |i| args[i] = switch (i) {
            0, 3, 4, 7, 8, 9, 10 => @as(*const u64, @ptrCast(@alignCast(params.?[i].?))).*,
            else => @as(*const u32, @ptrCast(@alignCast(params.?[i].?))).*,
        };
        return 0;
    }
    fn function(d: *const cuda.Driver, id: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
};
test "K32 family dispatch: flag off, decode and up-only fallback keep the actual baseline ABI" {
    const t = std.testing;
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Mock.launch;
    var k: struct { nvfp4: cuda.experts.Nvfp4Experts, expert_tile: ?cuda.experts_tile.Tile } = .{ .nvfp4 = undefined, .expert_tile = null };
    k.nvfp4.warps = 4;
    for (&k.nvfp4.fns, 0..) |*f, i| f.* = Mock.function(&driver, i + 1);
    for (&k.nvfp4.fact_fns, 0..) |*f, i| f.* = Mock.function(&driver, i + 21);
    k.nvfp4.resident = @splat(64);
    const s: cuda.Stream = .{ .d = &driver, .handle = null };
    const layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 20, .down = 30, .up_scale = 40, .down_scale = 50, .width = 160, .dims = 256, .experts = 6, .limit = 3 };
    const plan: cuda.grouped.Plan = .{ .items = 60, .counts = 70, .members = 80, .rank = 90, .hist = 100 };
    try tile.run(&k, 64, s, .up, .{ .x = 10, .stride = 256, .slots = 3 }, layer, plan, 110, 3, 5);
    try t.expectEqual(@as(usize, 21), Mock.selected);
    const baseline_up = Mock.args;
    try tile.run(&k, 64, s, .down_f32, .{ .x = 10, .stride = 160 }, layer, plan, 110, 3, 5);
    try t.expectEqual(@as(usize, 22), Mock.selected);
    const baseline_down = Mock.args;
    k.expert_tile = .{ .mod = undefined, .fns = .{ Mock.function(&driver, 31), Mock.function(&driver, 32) }, .resident = .{ 64, 64 }, .selection = .up };
    try tile.run(&k, 64, s, .up, .{ .x = 10, .stride = 256, .slots = 3 }, layer, plan, 110, 3, 5);
    try t.expectEqual(@as(usize, 31), Mock.selected);
    try t.expectEqualSlices(u64, &baseline_up, &Mock.args);
    try t.expectEqual(@as(u32, 128), Mock.threads);
    try t.expectEqual(cuda.experts_tile.Tile.shared[0], Mock.shared);
    try tile.run(&k, 64, s, .down_f32, .{ .x = 10, .stride = 160 }, layer, plan, 110, 3, 5);
    try t.expectEqual(@as(usize, 22), Mock.selected);
    try t.expectEqualSlices(u64, &baseline_down, &Mock.args);
    k.expert_tile.?.selection = .both;
    try tile.run(&k, 64, s, .down_f32, .{ .x = 10, .stride = 160 }, layer, plan, 110, 3, 5);
    try t.expectEqual(@as(usize, 32), Mock.selected);
    try t.expectEqualSlices(u64, &baseline_down, &Mock.args);
    try t.expectEqual(cuda.experts_tile.Tile.shared[1], Mock.shared);
    // decode, 5 blocks on 256 warps: half (fns[6]) at 3 items, classic at 30 (one wave), waves (fns[4]) at 60
    for ([_]usize{ 3, 30, 60 }, [_]usize{ 7, 1, 5 }) |items, id| {
        try tile.run(&k, 16, s, .up, .{ .x = 10, .stride = 256, .slots = 3 }, layer, plan, 110, items, 5);
        try t.expectEqual(id, Mock.selected);
        try t.expectEqualSlices(u64, &baseline_up, &Mock.args);
        try t.expectEqual(@as(u32, 128), Mock.threads);
        try t.expectEqual(@as(u32, 0), Mock.shared);
    }
}
