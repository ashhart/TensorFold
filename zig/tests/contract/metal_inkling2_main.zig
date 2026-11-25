//! Inkling 2 Mini Metal suites and the lifetimes of their contract probes.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;

pub const Suites = struct {
    a: std.mem.Allocator,
    inner: *Inner,

    const Inner = struct {
        attention: @import("metal_inkling2_attention.zig").Suite,
        glue: @import("metal_inkling2_glue.zig").Suite,
        experts: @import("metal_inkling2_experts.zig").Suite,
        loop: @import("metal_inkling2_loop.zig").Suite,
        runtime: @import("metal_inkling2_runtime.zig").Suite,
        dspark: @import("metal_inkling2_dspark.zig").Suite,
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !Suites {
        const inner = try a.create(Inner);
        errdefer a.destroy(inner);
        inner.* = .{
            .attention = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .glue = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .experts = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .loop = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .runtime = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dspark = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
        };
        errdefer deinitInner(inner);
        try inner.attention.collect(only);
        try inner.glue.collect(only);
        try inner.experts.collect(only);
        try inner.loop.collect(only);
        try inner.runtime.collect(only);
        try inner.dspark.collect(only);
        return .{ .a = a, .inner = inner };
    }

    fn deinitInner(s: *Inner) void {
        s.attention.deinit();
        s.glue.deinit();
        s.experts.deinit();
        s.loop.deinit();
        s.runtime.deinit();
        s.dspark.deinit();
    }

    pub fn deinit(s: *Suites) void {
        deinitInner(s.inner);
        s.a.destroy(s.inner);
    }
};
