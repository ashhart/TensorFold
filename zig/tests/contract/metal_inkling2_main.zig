//! Inkling 2 Mini Metal suites and the lifetimes of their contract probes.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;

pub const Suites = struct {
    a: std.mem.Allocator,
    attention: @import("metal_inkling2_attention.zig").Suite,
    dense: @import("metal_inkling2_dense.zig").Suite,
    glue: @import("metal_inkling2_glue.zig").Suite,
    experts: @import("metal_inkling2_experts.zig").Suite,
    units: @import("metal_inkling2_units.zig").Suite,
    dense_units: @import("metal_inkling2_dense_units.zig").Suite,
    loop: @import("metal_inkling2_loop.zig").Suite,
    runtime: @import("metal_inkling2_runtime.zig").Suite,
    dspark: @import("metal_inkling2_dspark.zig").Suite,
    chain: @import("metal_inkling2_dspark_chain.zig").Suite,
    dspark_screen: @import("metal_inkling2_dspark_screen.zig").Suite,

    /// Heap-allocated so inline probe arrays inside each suite keep stable addresses across return.
    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suites {
        const s = try a.create(Suites);
        s.* = .{
            .a = a,
            .attention = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dense = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .glue = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .experts = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .units = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dense_units = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .loop = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .runtime = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dspark = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .chain = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dspark_screen = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
        };
        errdefer s.destroy();
        try s.attention.collect(only);
        try s.dense.collect(only);
        try s.glue.collect(only);
        try s.experts.collect(only);
        try s.units.collect(only);
        try s.dense_units.collect(only);
        try s.loop.collect(only);
        try s.runtime.collect(only);
        try s.dspark.collect(only);
        try s.chain.collect(only);
        try s.dspark_screen.collect(only);
        return s;
    }

    pub fn destroy(s: *Suites) void {
        s.attention.deinit();
        s.dense.deinit();
        s.glue.deinit();
        s.experts.deinit();
        s.units.deinit();
        s.dense_units.deinit();
        s.loop.deinit();
        s.runtime.deinit();
        s.dspark.deinit();
        s.chain.deinit();
        s.dspark_screen.deinit();
        s.a.destroy(s);
    }
};
