//! Owned rank-two factors and allocation-free host evaluation of y += (x A) B * alpha/r.
const std = @import("std");
const rt = @import("runtime_lora_target.zig");
const lm = @import("lora_merge.zig");
const lq = @import("lora_quant.zig");

pub const Overlay = struct {
    module: []const u8,
    n: usize,
    k: usize,
    rank: usize,
    scale: f64,
    storage: rt.Storage,
    a: []const f32,
    b: []const f32,

    fn check(o: *const Overlay, x: []const f64, x_stride: usize, y: []const f64, y_stride: usize, rows: usize, work: []const f64) !void {
        if (rows == 0) return;
        if (x_stride < o.k or y_stride < o.n) return error.InvalidBatchStride;
        const x_offset = std.math.mul(usize, rows - 1, x_stride) catch return error.InvalidBatchStorage;
        const y_offset = std.math.mul(usize, rows - 1, y_stride) catch return error.InvalidBatchStorage;
        const x_span = std.math.add(usize, x_offset, o.k) catch return error.InvalidBatchStorage;
        const y_span = std.math.add(usize, y_offset, o.n) catch return error.InvalidBatchStorage;
        const work_len = std.math.mul(usize, rows, o.rank) catch return error.InvalidBatchStorage;
        if (x.len < x_span or y.len < y_span or work.len < work_len) return error.InvalidBatchStorage;
    }

    pub fn apply(o: *const Overlay, x: []const f64, x_stride: usize, y: []f64, y_stride: usize, rows: usize, work: []f64) !void {
        try o.check(x, x_stride, y, y_stride, rows, work);
        if (o.scale == 0) return;
        for (0..rows) |row| {
            const input = x[row * x_stride ..][0..o.k];
            const latent = work[row * o.rank ..][0..o.rank];
            for (latent, 0..) |*sum, j| {
                sum.* = 0;
                for (input, o.a[j * o.k ..][0..o.k]) |value, factor| sum.* += value * @as(f64, factor);
            }
            const output = y[row * y_stride ..][0..o.n];
            for (output, 0..) |*value, column| {
                var sum: f64 = 0;
                for (latent, o.b[column * o.rank ..][0..o.rank]) |left, factor| sum += left * @as(f64, factor);
                const delta = sum * o.scale;
                if (delta != 0) value.* += delta;
            }
        }
    }
};

pub fn validateFactor(t: @import("safetensors.zig").Tensor, rows: usize, columns: usize) !void {
    if (t.rank != 2 or t.dim(0) != rows or t.dim(1) != columns) return error.MismatchedTensorShape;
    if (t.dtype != .f32 and t.dtype != .bf16 and t.dtype != .f16) return error.UnsupportedAdapterDType;
    const count = std.math.mul(usize, rows, columns) catch return error.MismatchedTensorShape;
    const bytes = std.math.mul(usize, count, t.dtype.size()) catch return error.MismatchedTensorShape;
    if (t.bytes.len != bytes) return error.MismatchedTensorShape;
}

pub fn convertFactor(t: @import("safetensors.zig").Tensor, rows: usize, columns: usize, out: []f32) !void {
    try validateFactor(t, rows, columns);
    if (out.len != rows * columns) return error.MismatchedTensorShape;
    try lq.toF32(t.dtype, t.bytes, out);
    for (out) |value| if (!std.math.isFinite(value)) return error.NonFiniteAdapter;
}

pub fn parseName(name: []const u8) ?lm.AdapterKey {
    return lm.parseAdapterKey(name);
}
