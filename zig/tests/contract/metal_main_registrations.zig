//! Registration suites kept alive by metal_main while their kernels run.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;

pub const QkMixedLfm3 = struct {
    q27_cache: @import("metal_q27_qk_cache.zig").Suite,
    q27_qk: @import("metal_q27_qk.zig").Suite,
    q27_mixed: @import("metal_q27_mixed.zig").Suite,
    lfm3: @import("metal_lfm3_suite.zig").Suite,

    pub fn collect(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !QkMixedLfm3 {
        var s = QkMixedLfm3{
            .q27_cache = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .q27_qk = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .q27_mixed = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .lfm3 = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
        };
        errdefer s.deinit();
        try s.q27_cache.collect(only);
        try s.q27_qk.collect(only);
        try s.q27_mixed.collect(only);
        try s.lfm3.collect(only);
        return s;
    }

    pub fn deinit(s: *QkMixedLfm3) void {
        s.lfm3.deinit();
        s.q27_mixed.deinit();
        s.q27_qk.deinit();
        s.q27_cache.deinit();
    }
};

pub const PicksStackLfm2 = struct {
    a: std.mem.Allocator,
    picks: @import("metal_tree_round_picks.zig").Suite = .{},
    q27_stack: @import("metal_q27_stack.zig").Suite = .{},
    lfm2: @import("metal_lfm2_suite.zig").Suite,

    pub fn collect(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !PicksStackLfm2 {
        var s = PicksStackLfm2{ .a = a, .lfm2 = .{ .a = a, .device = device, .queue = queue, .kernels = kernels } };
        errdefer s.deinit();
        try s.picks.collect(a, device, queue, only, kernels);
        try s.q27_stack.collect(a, device, queue, only, kernels);
        try s.lfm2.collect(only);
        return s;
    }

    pub fn deinit(s: *PicksStackLfm2) void {
        s.lfm2.deinit();
        s.q27_stack.deinit(s.a);
        s.picks.deinit(s.a);
    }
};
