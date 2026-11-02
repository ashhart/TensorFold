//! Admission for dense BF16 embeddings and Torch RoPE. The caller validates token IDs and device positions.
const std = @import("std");
const Ops = @import("ops.zig").Ops;
const t = @import("types.zig");
const raw = @import("../launches/dense_text.zig");
const count = @import("../launches/columns.zig").count;

fn span(at: u64, bytes: usize, alignment: usize) t.Error!u64 {
    if (at == 0 or at % alignment != 0) return error.BadShape;
    return std.math.add(u64, at, bytes) catch error.BadShape;
}

fn separate(at: u64, end: u64, out: u64, oend: u64) t.Error!void {
    if (at < oend and out < end) return error.BadShape;
}

pub fn embed(o: Ops, table: u64, ids: u64, out: t.Tensor, rows: usize, width: usize, vocab: usize) t.Error!void {
    if (out.kind != .bf16 or vocab == 0 or vocab > std.math.maxInt(c_int)) return error.BadShape;
    const n = try count(rows, width);
    const table_bytes = std.math.mul(usize, vocab, width * 2) catch return error.BadShape;
    const tend = try span(table, table_bytes, 2);
    const iend = try span(ids, rows * 4, 4);
    const oend = try span(out.ptr, n * 2, 2);
    try separate(table, tend, out.ptr, oend);
    try separate(ids, iend, out.ptr, oend);
    try raw.embed(o.l, table, ids, out.ptr, rows, width, vocab, o.stream);
}

pub fn rope(o: Ops, x: t.Tensor, positions: u64, out: t.Tensor, rows: usize, heads: usize, d: usize, theta: f32, cache: bool, total: usize) t.Error!void {
    if (x.kind != .bf16 or (out.kind != .bf16 and out.kind != .f32) or heads == 0 or heads > 256 or
        d == 0 or d > 256 or d % 2 != 0 or total == 0 or total > 131072 or
        !std.math.isFinite(theta) or theta <= 0) return error.BadShape;
    const n = try count(rows, heads * d);
    const on = try count(if (cache) total else rows, heads * d);
    const xend = try span(x.ptr, n * 2, 2);
    const pend = try span(positions, rows * 4, 4);
    const oend = try span(out.ptr, on * out.kind.size(), out.kind.size());
    try separate(x.ptr, xend, out.ptr, oend);
    try separate(positions, pend, out.ptr, oend);
    try raw.rope(o.l, x.ptr, positions, out.ptr, @backingInt(out.kind), rows, heads, d, theta, cache, total, o.stream);
}

test "dots HIP text: invalid spans and formats fail before device access" {
    var o: Ops = undefined;
    o.l = undefined;
    try std.testing.expectError(error.BadShape, embed(o, 100, 200, .{ .ptr = 300, .kind = .f32 }, 1, 1, 1));
    try std.testing.expectError(error.BadShape, embed(o, 100, 200, .{ .ptr = 100, .kind = .bf16 }, 1, 1, 1));
    try std.testing.expectError(error.BadShape, rope(o, .{ .ptr = 100, .kind = .bf16 }, 200, .{ .ptr = 400, .kind = .bf16 }, 1, 12, 127, 1000000, false, 1));
    try std.testing.expectError(error.BadShape, span(std.math.maxInt(u64) - 1, 4, 2));
}
