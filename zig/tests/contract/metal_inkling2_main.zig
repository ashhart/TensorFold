//! Inkling 2 Mini Metal suites and the lifetimes of their contract probes.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;

pub const Suites = struct {
    attention: @import("metal_inkling2_attention.zig").Suite,
    glue: @import("metal_inkling2_glue.zig").Suite,
    experts: @import("metal_inkling2_experts.zig").Suite,
    loop: @import("metal_inkling2_loop.zig").Suite,
    runtime: @import("metal_inkling2_runtime.zig").Suite,
    dspark: @import("metal_inkling2_dspark.zig").Suite,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        var s = Suites{
            .attention = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .glue = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .experts = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .loop = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .runtime = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dspark = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
        };
        errdefer s.deinit();
        try s.attention.collect(only);
        try s.glue.collect(only);
        try s.experts.collect(only);
        try s.loop.collect(only);
        try s.runtime.collect(only);
        try s.dspark.collect(only);
        return s;
    }

    pub fn deinit(s: *Suites) void {
        s.attention.deinit();
        s.glue.deinit();
        s.experts.deinit();
        s.loop.deinit();
        s.runtime.deinit();
        s.dspark.deinit();
    }
};
