//! Bounds for the maintained HC sum, norm, mix and dense activation bodies.
const std = @import("std");
const wo = @import("core").weight_only;
pub const Op = enum { sum, norm, mix, act };
pub const Geometry = struct {
    hidden: usize,
    streams: usize,
    pub fn validate(g: Geometry) !void {
        if (g.hidden == 0 or g.hidden % 256 != 0 or g.streams < 2 or g.streams > 8 or g.hidden > std.math.maxInt(i32) / (128 * g.streams)) return error.Qwen4HcGeometry;
    }
};
pub const Call = struct { rows: usize, width: usize = 0, eps: f32 = 1e-6 };
pub fn sizes(g: Geometry, p: Call, op: Op) ![4]usize {
    try g.validate();
    if (p.rows == 0 or p.rows > 128 or !std.math.isFinite(p.eps) or p.eps <= 0) return error.Qwen4HcShape;
    const wide = try std.math.mul(usize, p.rows, try std.math.mul(usize, g.hidden, g.streams));
    const x = try std.math.mul(usize, wide, 2);
    const sums = try std.math.mul(usize, wide / 256, 4);
    return switch (op) {
        .sum => .{ x, sums, 0, 0 },
        .norm => .{ x, sums, try std.math.mul(usize, g.hidden * g.streams, 4), x },
        .mix => .{ x, x, try std.math.mul(usize, p.rows * g.hidden, 2), try std.math.mul(usize, p.rows * (g.hidden / 32), 4) },
        .act => blk: {
            if (p.width == 0 or p.width > 65535 * 256) return error.Qwen4HcShape;
            const count = try std.math.mul(usize, p.rows, p.width);
            if (count > std.math.maxInt(u32)) return error.Qwen4HcShape;
            const n = try std.math.mul(usize, count, 2);
            break :blk .{ n, n, n, 0 };
        },
    };
}
pub fn validate(g: Geometry, p: Call, op: Op, ranges: []const wo.Range) !void {
    return validateViews(g, p, op, ranges, true);
}
pub fn validateMetal(g: Geometry, p: Call, op: Op, ranges: []const wo.Range) !void {
    return validateViews(g, p, op, ranges, false);
}
fn validateViews(g: Geometry, p: Call, op: Op, ranges: []const wo.Range, auxiliary: bool) !void {
    const needs = try sizes(g, p, op);
    const count: usize = if (op == .sum) 2 else if (op == .act or (op == .mix and !auxiliary)) 3 else 4;
    if (ranges.len != count) return error.Qwen4HcBuffer;
    for (ranges, needs[0..count], 0..) |range, n, i| {
        if (range.address == 0 or range.len < n) return error.Qwen4HcBuffer;
        const align_to: usize = if ((op == .sum and i == 1) or (op == .norm and (i == 1 or i == 2)) or (op == .mix and i == 3)) 4 else 2;
        if (range.address % align_to != 0) return error.Qwen4HcAlignment;
        _ = try std.math.add(usize, range.address, range.len);
    }
    const output: usize = switch (op) {
        .sum => 1,
        .norm => 3,
        .mix, .act => 2,
    };
    for (output..count) |i| for (0..i) |j| {
        const x = ranges[i];
        const y = ranges[j];
        if (if (x.address <= y.address) y.address - x.address < x.len else x.address - y.address < y.len) return error.Qwen4HcAlias;
    };
}
