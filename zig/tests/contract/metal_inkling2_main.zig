//! Inkling 2 Mini Metal suites and the lifetimes of their contract probes.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;

pub const Suites = struct {
    attention: @import("metal_inkling2_attention.zig").Suite,
    glue: @import("metal_inkling2_glue.zig").Suite,
    experts: @import("metal_inkling2_experts.zig").Suite,
    units: @import("metal_inkling2_units.zig").Suite,
    logits_units: @import("metal_inkling2_logits_units.zig").Suite,
    dense_units: @import("metal_inkling2_dense_units.zig").Suite,
    loop: @import("metal_inkling2_loop.zig").Suite,
    runtime: @import("metal_inkling2_runtime.zig").Suite,
    dspark: @import("metal_inkling2_dspark.zig").Suite,
    chain: @import("metal_inkling2_dspark_chain.zig").Suite,
    dspark_units: @import("metal_inkling2_dspark_units.zig").Suite,

    /// Built where it will live: each kernel's context points at a probe inside these suites.
    pub fn init(s: *Suites, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        s.* = .{
            .attention = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .glue = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .experts = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .units = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .logits_units = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dense_units = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .loop = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .runtime = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dspark = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .chain = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dspark_units = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
        };
        errdefer s.deinit();
        try s.attention.collect(only);
        try s.glue.collect(only);
        try s.experts.collect(only);
        try s.units.collect(only);
        try s.logits_units.collect(only);
        try s.dense_units.collect(only);
        try s.loop.collect(only);
        try s.runtime.collect(only);
        try s.dspark.collect(only);
        try s.chain.collect(only);
        try s.dspark_units.collect(only);
    }

    pub fn deinit(s: *Suites) void {
        s.attention.deinit();
        s.glue.deinit();
        s.experts.deinit();
        s.units.deinit();
        s.logits_units.deinit();
        s.dense_units.deinit();
        s.loop.deinit();
        s.runtime.deinit();
        s.dspark.deinit();
        s.chain.deinit();
        s.dspark_units.deinit();
    }
};
