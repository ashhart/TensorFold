//! Recorded routed plans keep interleaved picks while excluding the shared-only item on one rank.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_moe_native.zig");
const Kernels = @import("cuda_batch_kernels.zig").Kernels;

const route_id = 1;
const Call = struct { id: usize, grid: u32, args: [14]u64 };
const Rec = struct {
    var calls: [3]Call = undefined;
    var n: usize = 0;

    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f.?);
        var args: [14]u64 = @splat(0);
        for (0..if (id == route_id) 7 else 14) |i| args[i] = if (id == route_id) switch (i) {
            0, 5, 6 => @as(*const u64, @ptrCast(@alignCast(params.?[i].?))).*,
            else => @as(*const u32, @ptrCast(@alignCast(params.?[i].?))).*,
        } else switch (i) {
            0, 3, 4, 7, 8, 9, 10 => @as(*const u64, @ptrCast(@alignCast(params.?[i].?))).*,
            else => @as(*const u32, @ptrCast(@alignCast(params.?[i].?))).*,
        };
        calls[n] = .{ .id = id, .grid = gx, .args = args };
        n += 1;
        return 0;
    }
};

test "CUDA routed plan: one-rank opt-in removes only the shared placeholder's grids" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Rec.launch;
    var k: Kernels = undefined;
    k.forms = .{};
    k.router = undefined;
    k.router.plan_small = .{ .d = &driver, .handle = @ptrFromInt(route_id) };
    for (&k.fp8.fns, 0..) |*set, i| {
        for (set, 0..) |*f, j| f.* = .{ .d = &driver, .handle = @ptrFromInt(10 + i * 3 + j) };
    }
    k.fp8.resident = @splat(@splat(std.math.maxInt(usize)));
    const layer: cuda.experts.Layer = .{ .format = .fp8g, .up = 20, .down = 30, .up_scale = 40, .down_scale = 50, .width = 640, .dims = 2560, .experts = 512, .limit = 3 };
    const plan: cuda.grouped.Plan = .{ .items = 60, .counts = 70, .members = 80, .rank = 90, .hist = 100 };
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var baseline: [2][14]u64 = undefined;
    for ([_]bool{ false, true }) |enabled| {
        k.forms.routed_plan = enabled;
        for ([_]usize{ 1, 2 }) |world| for ([_]usize{ 2, 4, 8 }) |rows| {
            const routed = enabled and world == 1;
            const pairs = rows * 11;
            Rec.n = 0;
            try native.fp8(&k, stream, 10, 110, layer, plan, pairs, rows, 120, 130, 11, world);
            try std.testing.expectEqual(@as(usize, 3), Rec.n);
            try std.testing.expectEqual(@as(usize, route_id), Rec.calls[0].id);
            try std.testing.expectEqual(@as(usize, 10), Rec.calls[1].id);
            try std.testing.expectEqual(@as(usize, 11), Rec.calls[2].id);
            try std.testing.expectEqual(@as(u32, @intCast(pairs)), Rec.calls[0].args[1]);
            try std.testing.expectEqual(@as(u32, if (routed) 512 else 513), Rec.calls[0].args[2]);
            try std.testing.expectEqual(@as(u32, 16), Rec.calls[0].args[3]);
            const items = cuda.grouped.maxItems(pairs - @as(usize, @intFromBool(routed)) * rows, if (routed) 512 else 513, 16);
            try std.testing.expectEqual(@as(u32, @intCast((items * 20 + 3) / 4)), Rec.calls[1].grid);
            try std.testing.expectEqual(@as(u32, @intCast((items * 80 + 3) / 4)), Rec.calls[2].grid);
            if (!enabled and world == 1 and rows == 2) {
                baseline = .{ Rec.calls[1].args, Rec.calls[2].args };
            } else if (enabled and rows == 2) {
                try std.testing.expectEqualSlices(u64, &baseline[0], &Rec.calls[1].args);
                try std.testing.expectEqualSlices(u64, &baseline[1], &Rec.calls[2].args);
            }
        };
    }
}
