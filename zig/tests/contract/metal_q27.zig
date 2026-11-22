const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;

pub fn collectMixed(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
    var q27_qk: @import("metal_q27_qk.zig").Suite = .{ .a = a, .device = device, .queue = queue, .kernels = kernels };
    defer q27_qk.deinit();
    try q27_qk.collect(only);
    var q27_mixed: @import("metal_q27_mixed.zig").Suite = .{ .a = a, .device = device, .queue = queue, .kernels = kernels };
    defer q27_mixed.deinit();
    try q27_mixed.collect(only);
    var lfm3_suite: @import("metal_lfm3_suite.zig").Suite = .{ .a = a, .device = device, .queue = queue, .kernels = kernels };
    defer lfm3_suite.deinit();
    try lfm3_suite.collect(only);
}

pub fn collectStack(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
    var picks: @import("metal_tree_round_picks.zig").Suite = .{};
    defer picks.deinit(a);
    try picks.collect(a, device, queue, only, kernels);
    var q27_stack: @import("metal_q27_stack.zig").Suite = .{};
    defer q27_stack.deinit(a);
    try q27_stack.collect(a, device, queue, only, kernels);
    var lfm2: @import("metal_lfm2_suite.zig").Suite = .{ .a = a, .device = device, .queue = queue, .kernels = kernels };
    defer lfm2.deinit();
    try lfm2.collect(only);
}
