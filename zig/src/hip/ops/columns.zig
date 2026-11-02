//! BF16 column operations and the two-rounding RMS graph used by Qwen2 and dots.mocr.
const std = @import("std");
const Ops = @import("ops.zig").Ops;
const t = @import("types.zig");
const raw = @import("../launches/columns.zig");

fn spans(x: ?u64, weight: u64, out: u64, rows: usize, width: usize) t.Error!void {
    const n = try raw.count(rows, width);
    const bytes = @as(u64, n) * 2;
    const wbytes = @as(u64, width) * 2;
    if (weight == 0 or out == 0 or weight % 2 != 0 or out % 2 != 0) return error.BadShape;
    const wend = std.math.add(u64, weight, wbytes) catch return error.BadShape;
    const oend = std.math.add(u64, out, bytes) catch return error.BadShape;
    if (weight < oend and out < wend) return error.BadShape;
    if (x) |src| {
        if (src == 0 or src % 2 != 0) return error.BadShape;
        const xend = std.math.add(u64, src, bytes) catch return error.BadShape;
        if (src != out and src < oend and out < xend) return error.BadShape;
    }
}

/// Elementwise BF16 multiplication by a BF16 column vector, rounded once; x and out may be identical.
pub fn multiply(o: Ops, x: t.Tensor, weight: u64, out: t.Tensor, rows: usize, width: usize) t.Error!void {
    if (x.kind != .bf16 or out.kind != .bf16) return error.BadShape;
    try spans(x.ptr, weight, out.ptr, rows, width);
    try raw.multiply(o.l, x.ptr, weight, out.ptr, rows, width, o.stream);
}

/// Copy the BF16 column vector to every row, preserving its stored bits.
pub fn fill(o: Ops, weight: u64, out: t.Tensor, rows: usize, width: usize) t.Error!void {
    if (out.kind != .bf16) return error.BadShape;
    try spans(null, weight, out.ptr, rows, width);
    try raw.fill(o.l, weight, out.ptr, rows, width, o.stream);
}

/// FP32 RMS, BF16 normalization, then BF16 weight multiplication and a second BF16 rounding.
pub fn rmsTorch(o: Ops, x: t.Tensor, weight: u64, out: t.Tensor, rows: usize, width: usize, eps: f32) t.Error!void {
    if (x.kind != .bf16 or out.kind != .bf16 or x.ptr == out.ptr or !std.math.isFinite(eps) or eps <= 0) return error.BadShape;
    try spans(x.ptr, weight, out.ptr, rows, width);
    try o.rms(x, null, out, rows, width, eps);
    try multiply(o, out, weight, out, rows, width);
}

test "dots HIP columns: malformed geometry and aliases fail before a launch" {
    var o: Ops = undefined;
    o.l = undefined;
    try std.testing.expectError(error.BadShape, multiply(o, .{ .ptr = 100, .kind = .f32 }, 200, .{ .ptr = 300, .kind = .bf16 }, 1, 1));
    try std.testing.expectError(error.BadShape, spans(100, 200, 102, 2, 2));
    try std.testing.expectError(error.BadShape, spans(null, 200, 202, 2, 2));
    try std.testing.expectError(error.Invalid, spans(null, 200, 300, 0, 1));
    try std.testing.expectError(error.Invalid, spans(null, 200, 300, std.math.maxInt(usize), 2));
    try std.testing.expectError(error.BadShape, spans(null, 200, std.math.maxInt(u64) - 1, 2, 2));
    try std.testing.expectError(error.BadShape, rmsTorch(o, .{ .ptr = 100, .kind = .bf16 }, 200, .{ .ptr = 100, .kind = .bf16 }, 1, 1, 1e-6));
}
