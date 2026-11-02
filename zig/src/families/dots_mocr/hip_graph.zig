//! Native dots.mocr operators composed from the shared HIP projections, RMS norm and column operations.
const std = @import("std");
const hip = @import("hip");
const cfg = @import("config.zig");
const weights = @import("hip_weights.zig");
const Tensor = hip.ops.Tensor;

pub const max_patches = 57600;

pub const Patch = struct { projection: Tensor, normalized: Tensor };

pub fn patchBytes(c: cfg.Config, rows: usize) !usize {
    if (rows == 0 or rows > max_patches) return error.DotsImageBudget;
    const patch = @as(usize, 3) * c.vision.patch * c.vision.patch;
    const x = try std.math.mul(usize, rows, patch * 2);
    const y = try std.math.mul(usize, rows, @as(usize, c.vision.hidden) * 2);
    return try std.math.add(usize, try std.math.add(usize, x, 3 * y), 4 * 256);
}

pub const Graph = struct {
    c: cfg.Config,
    w: *const weights.Weights,
    ops: hip.ops.Ops,
    blas: *const hip.blas.Blas,

    pub fn tensor(g: Graph, elements: usize) !Tensor {
        return .{ .ptr = try g.ops.arena.take(try std.math.mul(usize, elements, 2)), .kind = .bf16 };
    }

    pub const visionBlock = @import("hip_vision.zig").block;
    pub const vision = @import("hip_vision.zig").forward;

    /// A bias is copied before GEMM so beta 1 adds it in FP32 before the one BF16 output rounding.
    pub fn linear(g: Graph, x: Tensor, weight: u64, bias: ?u64, rows: usize, n: usize, k: usize) !Tensor {
        if (x.kind != .bf16) return error.BadShape;
        _ = try hip.blas.Shape.of(rows, n, k);
        const out = try g.tensor(try std.math.mul(usize, rows, n));
        if (bias) |b| {
            try hip.columns.fill(g.ops, b, out, rows, n);
        }
        // A lane round uses the same M=1 BLAS call for plain decode and every row of a draft window.
        const batch = if (g.ops.window) 1 else rows;
        var row: usize = 0;
        while (row < rows) : (row += batch) {
            if (bias != null) try g.blas.projectAdd(x.ptr + row * k * 2, weight, out.ptr + row * n * 2, batch, n, k) else try g.blas.project(x.ptr + row * k * 2, weight, out.ptr + row * n * 2, batch, n, k);
        }
        return out;
    }

    pub fn norm(g: Graph, x: Tensor, weight: u64, rows: usize, width: usize, eps: f32) !Tensor {
        const out = try g.tensor(try std.math.mul(usize, rows, width));
        try hip.columns.rmsTorch(g.ops, x, weight, out, rows, width, eps);
        return out;
    }

    /// The patch Conv2d at kernel/stride 14 is the same stored [out, channel, y, x] BF16 matrix product.
    pub fn patch(g: Graph, pixels: Tensor, rows: usize) !Patch {
        _ = try patchBytes(g.c, rows);
        if (pixels.kind != .f32) return error.BadShape;
        const width = @as(usize, 3) * g.c.vision.patch * g.c.vision.patch;
        const x = try g.tensor(rows * width);
        try g.ops.cast(pixels, x, rows * width);
        const prefix = "vision_tower.patch_embed.patchifier.";
        const projected = try g.linear(x, try g.w.get(prefix ++ "proj.weight"), null, rows, g.c.vision.hidden, width);
        const bias = try g.tensor(rows * g.c.vision.hidden);
        try hip.columns.fill(g.ops, try g.w.get(prefix ++ "proj.bias"), bias, rows, g.c.vision.hidden);
        try g.ops.add(projected, bias, projected, rows * g.c.vision.hidden);
        const normalized = try g.norm(projected, try g.w.get(prefix ++ "norm.weight"), rows, g.c.vision.hidden, g.c.vision.eps);
        return .{ .projection = projected, .normalized = normalized };
    }
};

test "dots HIP graph: patch scratch covers all legal image rows and refuses overflow inputs" {
    const cases = @import("config_test.zig");
    const c = try cfg.Config.parse(std.testing.allocator, cases.sample, cases.generation);
    try std.testing.expectEqual(@as(usize, 598580224), try patchBytes(c, max_patches));
    try std.testing.expectError(error.DotsImageBudget, patchBytes(c, 0));
    try std.testing.expectError(error.DotsImageBudget, patchBytes(c, max_patches + 1));
    std.testing.refAllDecls(Graph);
}
