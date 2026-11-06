//! Host Gemma attention adjoints over captured windows and one cache generation; never a serving fallback.
const std = @import("std");

pub const Shape = struct {
    rows: usize,
    heads: usize,
    kv: usize,
    width: usize,
    capacity: usize,
    /// A ring snapshot owns logical tokens [first, first + capacity); linear storage requires first = 0.
    first: usize = 0,
    ring: bool = false,
};
/// The captured half-open logical mask, including a sliding image block's possibly noncausal upper bound.
pub const Row = struct { begin: usize = 0, end: usize };

fn times(a: usize, b: usize, c: usize) !usize {
    return std.math.mul(usize, try std.math.mul(usize, a, b), c);
}
fn keyAt(g: Shape, row: Row, j: usize) usize {
    const token = row.begin + j;
    return if (g.ring) token % g.capacity else token;
}
fn keyCount(row: Row) usize {
    return row.end - row.begin;
}
fn scalar(comptime T: type) void {
    if (T != f32 and T != f64) @compileError("Gemma host backward requires f32 or f64");
}
fn disjoint(output: []const u8, other: []const u8) !void {
    const a = @intFromPtr(output.ptr);
    const b = @intFromPtr(other.ptr);
    const n = output.len;
    const m = other.len;
    if (n != 0 and m != 0 and (if (a <= b) b - a < n else a - b < m)) return error.GemmaGradientAlias;
}
fn finite(comptime T: type, values: []const T) !void {
    for (values) |v| if (!std.math.isFinite(v)) return error.GemmaGradientNonfinite;
}

/// q/dout/dq [rows,heads,width], k/v/dk/dv [capacity,kv,width]; overwrites adjoints, accumulates shared keys.
pub fn backward(comptime T: type, a: std.mem.Allocator, g: Shape, rows: []const Row, q: []const T, k: []const T, v: []const T, dout: []const T, dq: []T, dk: []T, dv: []T) !void {
    scalar(T);
    if (g.rows == 0 or g.heads == 0 or g.kv == 0 or g.width == 0 or g.capacity == 0 or
        g.heads % g.kv != 0 or (!g.ring and g.first != 0) or rows.len != g.rows)
        return error.GemmaGradientShape;
    const scale: T = 1;
    const live_end = try std.math.add(usize, g.first, g.capacity);
    const nq = try times(g.rows, g.heads, g.width);
    const nk = try times(g.capacity, g.kv, g.width);
    _ = try std.math.mul(usize, nq, @sizeOf(T));
    _ = try std.math.mul(usize, nk, @sizeOf(T));
    _ = try std.math.mul(usize, rows.len, @sizeOf(Row));
    if (q.len != nq or dout.len != nq or dq.len != nq or k.len != nk or v.len != nk or dk.len != nk or dv.len != nk)
        return error.GemmaGradientShape;
    const outputs = [_][]T{ dq, dk, dv };
    for (outputs, 0..) |out, i| {
        for ([_][]const T{ q, k, v, dout }) |input| try disjoint(std.mem.sliceAsBytes(out), std.mem.sliceAsBytes(input));
        for (outputs[0..i]) |prior| try disjoint(std.mem.sliceAsBytes(out), std.mem.sliceAsBytes(prior));
        try disjoint(std.mem.sliceAsBytes(out), std.mem.sliceAsBytes(rows));
    }
    var largest: usize = 0;
    for (rows) |row| {
        if (row.begin < g.first or row.end > live_end or row.begin >= row.end)
            return error.GemmaGradientWindow;
        largest = @max(largest, keyCount(row));
    }
    try finite(T, q);
    try finite(T, dout);
    for (rows) |row| for (0..keyCount(row)) |j| {
        const base = keyAt(g, row, j) * g.kv * g.width;
        try finite(T, k[base..][0 .. g.kv * g.width]);
        try finite(T, v[base..][0 .. g.kv * g.width]);
    };
    const probabilities = try a.alloc(T, largest);
    defer a.free(probabilities);
    const dp = try a.alloc(T, largest);
    defer a.free(dp);
    for (outputs) |out| @memset(out, 0);
    errdefer for (outputs) |out| @memset(out, 0);
    for (rows, 0..) |row, r| for (0..g.heads) |h| {
        const head = h / (g.heads / g.kv);
        const at = (r * g.heads + h) * g.width;
        const count = keyCount(row);
        var top = -std.math.inf(T);
        for (probabilities[0..count], 0..) |*score, j| {
            const base = (keyAt(g, row, j) * g.kv + head) * g.width;
            score.* = 0;
            for (0..g.width) |d| score.* += q[at + d] * k[base + d];
            score.* *= scale;
            if (!std.math.isFinite(score.*)) return error.GemmaGradientNonfinite;
            top = @max(top, score.*);
        }
        var total: T = 0;
        for (probabilities[0..count]) |*p| {
            p.* = @exp(p.* - top);
            total += p.*;
        }
        var center: T = 0;
        for (probabilities[0..count], dp[0..count], 0..) |*p, *derivative, j| {
            p.* /= total;
            derivative.* = 0;
            const base = (keyAt(g, row, j) * g.kv + head) * g.width;
            for (0..g.width) |d| derivative.* += dout[at + d] * v[base + d];
            center += p.* * derivative.*;
        }
        for (probabilities[0..count], dp[0..count], 0..) |p, derivative, j| {
            const ds = p * (derivative - center) * scale;
            const base = (keyAt(g, row, j) * g.kv + head) * g.width;
            for (0..g.width) |d| {
                dq[at + d] += ds * k[base + d];
                dk[base + d] += ds * q[at + d];
                dv[base + d] += p * dout[at + d];
            }
        }
    };
    for (outputs) |out| try finite(T, out);
}

test {
    _ = @import("attention_backward_test.zig");
}
