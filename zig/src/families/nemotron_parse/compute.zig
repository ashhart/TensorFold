//! The family graph over shared HIP operators; scratch is reused in stream order, never read back for inference.
const std = @import("std");
const hip = @import("hip");
const Weights = @import("weights.zig").Weights;
const Tensor = hip.ops.Tensor;

pub const Tap = struct {
    ctx: *anyopaque,
    record: *const fn (*anyopaque, []const u8, Tensor, usize, usize) anyerror!void,
};

pub const Compute = struct {
    ops: hip.ops.Ops,
    kernels: hip.vision.Kernels,
    stream: hip.Stream,
    weights: *const Weights,
    tap: ?Tap = null,
    cancel: ?struct { ptr: *anyopaque, check: *const fn (*anyopaque) bool } = null,

    pub fn poll(c: Compute) !void {
        if (c.cancel) |x| {
            try c.stream.synchronize();
            if (x.check(x.ptr)) return error.Cancelled;
        }
    }

    pub fn alloc(c: Compute, count: usize, kind: hip.ops.Kind) !Tensor {
        return .{ .ptr = try c.ops.arena.take(count * kind.size()), .kind = kind };
    }

    pub fn named(c: Compute, stem: []const u8, suffix: []const u8) !Tensor {
        var buf: [256]u8 = undefined;
        return c.weights.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, suffix }));
    }

    pub fn linear(c: Compute, x: Tensor, stem: []const u8, rows: usize, n: usize, k: usize, bias: bool) !Tensor {
        const out = try c.alloc(rows * n, .bf16);
        try c.kernels.linear(c.stream, x, (try c.named(stem, "weight")).ptr, if (bias) (try c.named(stem, "bias")).ptr else null, out, rows, n, k);
        return out;
    }

    pub fn norm(c: Compute, x: Tensor, stem: []const u8, rows: usize, width: usize, eps: f32) !Tensor {
        const out = try c.alloc(rows * width, .bf16);
        try c.kernels.norm(c.stream, x, (try c.named(stem, "weight")).ptr, (try c.named(stem, "bias")).ptr, out, rows, width, eps);
        return out;
    }

    pub fn residual(c: Compute, x: Tensor, y: Tensor, count: usize) !void {
        try c.kernels.point(c.stream, x, y, x, count, .add, 1);
    }

    pub fn gelu(c: Compute, x: Tensor, count: usize) !void {
        try c.kernels.point(c.stream, x, null, x, count, .gelu, 1);
    }

    pub fn record(c: Compute, name: []const u8, x: Tensor, rows: usize, width: usize) !void {
        if (c.tap) |tap| try tap.record(tap.ctx, name, x, rows, width);
    }

    /// Copy interleaved projection rows into [head,row,dim], without applying text rotary positions.
    pub fn heads(c: Compute, x: Tensor, rows: usize, count: usize, dim: usize, wide: bool) !Tensor {
        const out = try c.alloc(rows * count * dim, if (wide) .f32 else .bf16);
        try c.ops.ropePrefill(x, out, rows * dim, dim, rows, count, dim, 0, 0, 1);
        return out;
    }

    /// Global attention is causal attention whose first query can already see the complete span.
    pub fn global(c: Compute, qkv: Tensor, rows: usize, count: usize, dim: usize) !Tensor {
        const width = count * dim;
        var parts: [3]Tensor = undefined;
        for (&parts, 0..) |*p, i| {
            const slice = try c.alloc(rows * width, .bf16);
            try c.ops.copyCols(qkv, 3 * width, i * width, slice.ptr, rows, width);
            p.* = try c.heads(slice, rows, count, dim, i == 0);
        }
        const att = try c.alloc(rows * width, .f32);
        try c.ops.causalPrefill(parts[0].ptr, .{ .k = parts[1].ptr, .v = parts[2].ptr, .kind = .bf16, .kv_heads = count, .total = rows, .d = dim }, att.ptr, rows, rows, count, 1 / @sqrt(@as(f32, @floatFromInt(dim))), rows - 1);
        const out = try c.alloc(rows * width, .bf16);
        try c.kernels.layout(c.stream, att, out, rows, count, dim, .heads_to_rows);
        return out;
    }
};
