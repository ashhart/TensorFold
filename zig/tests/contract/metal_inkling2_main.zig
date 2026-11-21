//! Inkling 2 Mini's contract suites for tf-kernel-contract: ops, loop, experts, media, kept alive by the caller.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;
const Ops = @import("metal_inkling2_suite.zig").Suite;
const Loop = @import("metal_inkling2_loop.zig").Suite;
const Experts = @import("metal_inkling2_experts.zig").Suite;
const Media = @import("metal_inkling2_media_suite.zig").Suite;

pub const Suites = struct {
    ops: Ops,
    loop: Loop,
    experts: Experts,
    media: Media,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel)) Suites {
        return .{
            .ops = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .loop = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .experts = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .media = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
        };
    }

    /// Each suite's gates and controls, filtered by `only`; the suites stay alive until deinit.
    pub fn collect(s: *Suites, only: ?[]const u8) !void {
        try s.ops.collect(only);
        try s.loop.collect(only);
        try s.experts.collect(only);
        try s.media.collect(only);
    }

    pub fn deinit(s: *Suites) void {
        s.ops.deinit();
        s.loop.deinit();
        s.experts.deinit();
        s.media.deinit();
    }
};
