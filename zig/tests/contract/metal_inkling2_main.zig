//! Inkling 2 Mini Metal suites and the lifetimes of their contract probes.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;

pub const Suites = struct {
    attention: @import("metal_inkling2_attention.zig").Suite,
    attention_gqa: @import("metal_inkling2_attention_gqa.zig").Suite,
    glue: @import("metal_inkling2_glue.zig").Suite,
    logits: @import("metal_inkling2_logits.zig").Suite,
    experts: @import("metal_inkling2_experts.zig").Suite,
    mma: @import("metal_inkling2_mma.zig").Suite,
    dense_mma: @import("metal_inkling2_dense_mma.zig").Suite,
    dense_tiled: @import("metal_inkling2_dense_tiled.zig").Suite,
    gates_mma: @import("metal_inkling2_gates_mma.zig").Suite,
    loop: @import("metal_inkling2_loop.zig").Suite,
    runtime: @import("metal_inkling2_runtime.zig").Suite,
    dspark: @import("metal_inkling2_dspark.zig").Suite,
    draft_head_q4: @import("metal_inkling2_draft_head_q4.zig").Suite,
    vision: @import("metal_inkling2_vision.zig").Suite,
    dense: @import("metal_inkling2_dense.zig").Suite,
    tree: @import("metal_inkling2_tree.zig").Suite,
    dflash2: @import("metal_inkling2_dflash2.zig").Suite,
    sample: @import("metal_inkling2_sample.zig").Suite,

    /// The probes' contexts point into the suites, so the set is built where it stays.
    pub fn init(s: *Suites, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        s.* = .{
            .attention = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .attention_gqa = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .glue = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .logits = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .experts = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .mma = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dense_mma = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dense_tiled = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .gates_mma = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .loop = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .runtime = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dspark = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .draft_head_q4 = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .vision = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dense = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .tree = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .dflash2 = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .sample = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
        };
        errdefer s.deinit();
        try s.attention.collect(only);
        try s.attention_gqa.collect(only);
        try s.glue.collect(only);
        try s.logits.collect(only);
        try s.experts.collect(only);
        try s.mma.collect(only);
        try s.dense_mma.collect(only);
        try s.dense_tiled.collect(only);
        try s.gates_mma.collect(only);
        try s.loop.collect(only);
        try s.runtime.collect(only);
        try s.dspark.collect(only);
        try s.draft_head_q4.collect(only);
        try s.vision.collect(only);
        try s.dense.collect(only);
        try s.tree.collect(only);
        try s.dflash2.collect(only);
        try s.sample.collect(only);
    }

    pub fn deinit(s: *Suites) void {
        s.attention.deinit();
        s.attention_gqa.deinit();
        s.glue.deinit();
        s.logits.deinit();
        s.experts.deinit();
        s.mma.deinit();
        s.dense_mma.deinit();
        s.dense_tiled.deinit();
        s.gates_mma.deinit();
        s.loop.deinit();
        s.runtime.deinit();
        s.dspark.deinit();
        s.draft_head_q4.deinit();
        s.vision.deinit();
        s.dense.deinit();
        s.tree.deinit();
        s.dflash2.deinit();
        s.sample.deinit();
    }

    pub fn slidingZeroUlp(s: *const Suites, seed: u64, only: ?[]const u8) !void {
        if (only) |needle| {
            if (std.mem.indexOf(u8, "metal inkling2 attention sliding prompt tf_ink2_attn_sliding_prompt", needle) == null) return;
        }
        try @import("metal_inkling2_attention_sliding.zig").run(s.attention.a, s.attention.device, s.attention.queue, seed);
    }
};
