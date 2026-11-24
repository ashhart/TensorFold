//! SAM3 and SAM3.1's Metal suites, collected in one place so the runner's wiring stays one line.
const std = @import("std");
const mtl = @import("metal");
const g = @import("contract").gates;

pub const Suites = struct {
    a: std.mem.Allocator,
    mma: @import("metal_sam3_mma.zig").Suite,
    deconv: @import("metal_sam3_deconv.zig").Suite,
    gemm: @import("metal_sam3_gemm.zig").Suite,
    small: @import("metal_sam3_dense_small.zig").Suite,
    planes: @import("metal_sam3_planes.zig").Suite,
    pair: @import("metal_sam3_pair.zig").Suite,
    attention: @import("metal_sam3_attention.zig").Suite,
    norm: @import("metal_sam3_norm.zig").Suite,
    resize: @import("metal_sam3_resize.zig").Suite,
    sparse_mask: @import("metal_sam3_sparse_mask.zig").Suite,
    layout: @import("metal_sam3_layout.zig").Suite,
    arithmetic: @import("metal_sam3_arithmetic.zig").Suite,
    rotary: @import("metal_sam3_rotary.zig").Suite,
    rows: @import("metal_sam3_rows.zig").Suite,
    flash: @import("metal_sam3_flash.zig").Suite,
    trunk: @import("metal_sam3_trunk.zig").Suite,
    tensor_ops: @import("metal_sam3_tensor_ops.zig").Suite,
    core_attention: @import("metal_sam3_core_attention.zig").Suite,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suites {
        const s = try a.create(Suites);
        s.* = .{
            .a = a,
            .mma = .{ .a = a, .device = device, .queue = queue, .kernels = kernels },
            .deconv = .{ .a = a },
            .gemm = .{ .a = a },
            .small = .{ .allocator = a },
            .planes = .{ .a = a },
            .pair = .{ .a = a },
            .attention = .{ .a = a },
            .norm = .{ .a = a },
            .resize = .{ .a = a },
            .sparse_mask = .{ .a = a },
            .layout = .{ .a = a },
            .arithmetic = .{ .a = a },
            .rotary = .{ .a = a },
            .rows = .{ .a = a },
            .flash = .{ .a = a },
            .trunk = .{ .a = a },
            .tensor_ops = .{ .a = a },
            .core_attention = .{ .a = a },
        };
        errdefer s.destroy();
        try s.mma.collect(only);
        inline for (.{ "deconv", "gemm", "small", "planes", "pair", "attention", "norm", "resize", "sparse_mask", "layout", "arithmetic", "rotary", "rows", "flash", "trunk", "tensor_ops", "core_attention" }) |field|
            try @field(s, field).collect(device, queue, kernels, only);
        return s;
    }

    pub fn destroy(s: *Suites) void {
        s.mma.deinit();
        inline for (.{ "deconv", "gemm", "small", "planes", "pair", "attention", "norm", "resize", "sparse_mask", "layout", "arithmetic", "rotary", "rows", "flash", "trunk", "tensor_ops", "core_attention" }) |field|
            @field(s, field).deinit();
        s.a.destroy(s);
    }
};
