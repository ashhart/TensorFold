//! NVFP4 same-input decode fusion, with the normal per-format decoder as the incompatible-member fallback.
const std = @import("std");
const linear = @import("qlinear.zig");
const many = @import("qmmf_many.zig");
const Stream = @import("stream.zig").Stream;

/// Decode rows of projections sharing `in`: one grouped launch if the lane takes every member, else decode() each.
pub fn decodeMany(l: linear.Linear, s: Stream, weights: []const linear.Weight, in: linear.Rows, outs: []const u64, rows: usize) !void {
    if (weights.len == 0 or weights.len != outs.len) return error.ShapeMismatch;
    if (l.lane) |lane| if (weights.len >= 2 and weights.len <= 4) grouped: {
        var parts: [4]many.Many = undefined;
        for (weights, outs, 0..) |w, out, i| {
            if (w != .nvfp4 or w.nvfp4.mode != .fp4) break :grouped;
            parts[i] = .{ .w = w.nvfp4, .out = out };
        }
        if (!many.eligible(lane.*, rows, parts[0..weights.len])) break :grouped;
        return lane.matmulMany(s, in.x, if (in.ldx == 0) weights[0].inputs() else in.ldx, rows, parts[0..weights.len], in.part);
    };
    for (weights, outs) |w, out| try l.decode(s, w, in, out, rows);
}

test "qmmf many: members the lane cannot group decode as separate calls do, each at its own row width" {
    const abi = @import("abi.zig");
    const Mock = struct {
        var ldx: [4]i32 = undefined;
        var calls: usize = 0;
        fn launch(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            ldx[calls] = @as(*const i32, @ptrCast(@alignCast(args.?[11].?))).*; // matmulOut's row stride
            calls += 1;
            return 0;
        }
    };
    var driver: @import("driver.zig").Driver = undefined;
    driver.api.cuLaunchKernel = Mock.launch;
    const f: @import("module.zig").Function = .{ .d = &driver, .handle = @ptrFromInt(7) };
    var lane: @import("qmmf.zig").Lane = undefined;
    lane.clusters = false;
    lane.half_below = 0;
    for (&lane.fns) |*set| set.* = @splat(f);
    var narrow = @import("nvfp4.zig").weight(1, 2, 1, 129, 512);
    narrow.sk = 1;
    var wide = narrow;
    wide.k = 1024;
    const l: linear.Linear = .{ .lane = &lane };
    const stream: Stream = .{ .d = &driver, .handle = null };
    try decodeMany(l, stream, &.{ .{ .nvfp4 = narrow }, .{ .nvfp4 = wide } }, .{ .x = 0x1000 }, &.{ 0x10000, 0x20000 }, 2);
    try std.testing.expectEqualSlices(i32, &.{ 512, 1024 }, Mock.ldx[0..Mock.calls]);
}
