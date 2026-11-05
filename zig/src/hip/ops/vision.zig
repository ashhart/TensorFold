//! Shape and address admission for shared BF16 vision operations.
const std = @import("std");
const Ops = @import("ops.zig").Ops;
const t = @import("types.zig");
const raw = @import("../launches/vision.zig");
const count = @import("../launches/columns.zig").count;

const Span = struct { at: u64, bytes: usize };

fn disjoint(spans: []const Span) t.Error!void {
    for (spans, 0..) |a, i| {
        if (a.at == 0 or a.at % 2 != 0) return error.BadShape;
        const end = std.math.add(u64, a.at, a.bytes) catch return error.BadShape;
        for (spans[0..i]) |b| {
            const bend = std.math.add(u64, b.at, b.bytes) catch return error.BadShape;
            if (a.at < bend and b.at < end) return error.BadShape;
        }
    }
}

pub fn layerNorm(o: Ops, x: t.Tensor, w: u64, b: u64, out: t.Tensor, rows: usize, width: usize, eps: f32) t.Error!void {
    if (x.kind != .bf16 or out.kind != .bf16 or !std.math.isFinite(eps) or eps <= 0) return error.BadShape;
    const n = try count(rows, width);
    // Weight and bias may be the same read-only buffer; each must be disjoint from both activations.
    try disjoint(&.{ .{ .at = x.ptr, .bytes = n * 2 }, .{ .at = out.ptr, .bytes = n * 2 } });
    for ([_]u64{ w, b }) |p| try disjoint(&.{
        .{ .at = p, .bytes = width * 2 }, .{ .at = x.ptr, .bytes = n * 2 }, .{ .at = out.ptr, .bytes = n * 2 },
    });
    try raw.layerNorm(o.l, x.ptr, w, b, out.ptr, rows, width, eps, o.stream);
}

pub fn gelu(o: Ops, x: t.Tensor, out: t.Tensor, rows: usize, width: usize) t.Error!void {
    if (x.kind != .bf16 or out.kind != .bf16) return error.BadShape;
    const n = try count(rows, width);
    if (x.ptr == out.ptr) {
        if (x.ptr == 0 or x.ptr % 2 != 0 or x.ptr > std.math.maxInt(u64) - n * 2) return error.BadShape;
    } else try disjoint(&.{ .{ .at = x.ptr, .bytes = n * 2 }, .{ .at = out.ptr, .bytes = n * 2 } });
    try raw.gelu(o.l, x.ptr, out.ptr, n, o.stream);
}

fn geometry(rows: usize, heads: usize, d: usize) t.Error!usize {
    if (heads == 0 or heads > 256 or d == 0 or d > 256 or d % 64 != 0) return error.BadShape;
    return count(rows, heads * d);
}

pub fn rope(o: Ops, qkv: t.Tensor, positions: u64, q: u64, k: u64, v: u64, rows: usize, heads: usize, d: usize, theta: f32) t.Error!void {
    if (qkv.kind != .bf16 or !std.math.isFinite(theta) or theta <= 0 or positions % 4 != 0 or q % 4 != 0) return error.BadShape;
    const n = try geometry(rows, heads, d);
    try disjoint(&.{
        .{ .at = qkv.ptr, .bytes = n * 6 }, .{ .at = positions, .bytes = rows * 8 },
        .{ .at = q, .bytes = n * 4 },       .{ .at = k, .bytes = n * 2 },
        .{ .at = v, .bytes = n * 2 },
    });
    try raw.rope(o.l, qkv.ptr, positions, q, k, v, rows, heads, d, theta, o.stream);
}

pub fn attentionRows(o: Ops, x: u64, out: t.Tensor, rows: usize, heads: usize, d: usize) t.Error!void {
    if (out.kind != .bf16 or x % 4 != 0) return error.BadShape;
    const n = try geometry(rows, heads, d);
    try disjoint(&.{ .{ .at = x, .bytes = n * 4 }, .{ .at = out.ptr, .bytes = n * 2 } });
    try raw.attentionRows(o.l, x, out.ptr, rows, heads, d, o.stream);
}

/// One image per call. No cross-image visibility or causal mask is used.
pub fn fullAttention(o: Ops, q: u64, k: u64, v: u64, out: u64, rows: usize, heads: usize, d: usize) t.Error!void {
    const n = try geometry(rows, heads, d);
    if (q % 4 != 0 or out % 4 != 0) return error.BadShape;
    try disjoint(&.{
        .{ .at = q, .bytes = n * 4 }, .{ .at = k, .bytes = n * 2 },
        .{ .at = v, .bytes = n * 2 }, .{ .at = out, .bytes = n * 4 },
    });
    try raw.full(o.l, q, k, v, out, rows, heads, d, 1 / @sqrt(@as(f32, @floatFromInt(d))), o.stream);
}

test "dots HIP vision: malformed geometry and buffer spans fail before a launch" {
    var o: Ops = undefined;
    o.l = undefined;
    try std.testing.expectError(error.BadShape, geometry(1, 0, 128));
    try std.testing.expectError(error.BadShape, geometry(1, 12, 127));
    try std.testing.expectError(error.BadShape, disjoint(&.{ .{ .at = 100, .bytes = 100 }, .{ .at = 102, .bytes = 2 } }));
    try std.testing.expectError(error.BadShape, layerNorm(o, .{ .ptr = 100, .kind = .bf16 }, 200, 300, .{ .ptr = 100, .kind = .bf16 }, 1, 1, 1e-6));
    try std.testing.expectError(error.BadShape, gelu(o, .{ .ptr = 100, .kind = .f32 }, .{ .ptr = 200, .kind = .bf16 }, 1, 1));
}
