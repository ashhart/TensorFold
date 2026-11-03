//! Host QSA adjoints with fixed selection and token-major GQA caches; no production inference fallback.
const std = @import("std");

pub const Shape = struct { rows: usize, heads: usize, kv: usize, width: usize, capacity: usize, scale: f64 = 0.0625 };
/// Dense rows read 0..end; sparse rows read exactly these unique causal token IDs, including the incomplete tail.
pub const Row = struct { end: usize, ids: ?[]const u32 = null };

fn times(a: usize, b: usize, c: usize) !usize {
    return std.math.mul(usize, try std.math.mul(usize, a, b), c);
}
fn keyAt(row: Row, j: usize) usize {
    return if (row.ids) |ids| ids[j] else j;
}
fn keyCount(row: Row) usize {
    return if (row.ids) |ids| ids.len else row.end;
}
fn scalar(comptime T: type) void {
    if (T != f32 and T != f64) @compileError("QSA host backward requires f32 or f64");
}
fn disjoint(comptime T: type, output: []T, other: []const T) !void {
    const a = @intFromPtr(output.ptr);
    const b = @intFromPtr(other.ptr);
    const n = try std.math.mul(usize, output.len, @sizeOf(T));
    const m = try std.math.mul(usize, other.len, @sizeOf(T));
    if (n != 0 and m != 0 and (if (a <= b) b - a < n else a - b < m)) return error.QsaGradientAlias;
}
fn finite(comptime T: type, values: []const T) !void {
    for (values) |v| if (!std.math.isFinite(v)) return error.QsaGradientNonfinite;
}

/// q/dout/dq [rows,heads,width], k/v/dk/dv [capacity,kv,width]; overwrites adjoints, accumulates shared keys.
pub fn backward(comptime T: type, a: std.mem.Allocator, g: Shape, rows: []const Row, q: []const T, k: []const T, v: []const T, dout: []const T, dq: []T, dk: []T, dv: []T) !void {
    scalar(T);
    if (g.rows == 0 or g.heads == 0 or g.kv == 0 or g.width == 0 or g.capacity == 0 or
        g.heads % g.kv != 0 or !std.math.isFinite(g.scale) or g.scale <= 0 or rows.len != g.rows)
        return error.QsaGradientShape;
    const scale: T = @floatCast(g.scale);
    if (!std.math.isFinite(scale) or scale <= 0) return error.QsaGradientShape;
    const nq = try times(g.rows, g.heads, g.width);
    const nk = try times(g.capacity, g.kv, g.width);
    if (q.len != nq or dout.len != nq or dq.len != nq or k.len != nk or v.len != nk or dk.len != nk or dv.len != nk)
        return error.QsaGradientShape;
    const outputs = [_][]T{ dq, dk, dv };
    for (outputs, 0..) |out, i| {
        for ([_][]const T{ q, k, v, dout }) |input| try disjoint(T, out, input);
        for (outputs[0..i]) |prior| try disjoint(T, out, prior);
    }
    var largest: usize = 0;
    for (rows) |row| {
        if (row.end == 0 or row.end > g.capacity or keyCount(row) == 0 or keyCount(row) > row.end) return error.QsaGradientSelection;
        if (row.ids) |ids| for (ids, 0..) |id, i| {
            if (id >= row.end or std.mem.indexOfScalar(u32, ids[0..i], id) != null) return error.QsaGradientSelection;
        };
        largest = @max(largest, keyCount(row));
    }
    try finite(T, q);
    try finite(T, dout);
    for (rows) |row| for (0..keyCount(row)) |j| {
        const base = keyAt(row, j) * g.kv * g.width;
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
            const base = (keyAt(row, j) * g.kv + head) * g.width;
            score.* = 0;
            for (0..g.width) |d| score.* += q[at + d] * k[base + d];
            score.* *= scale;
            if (!std.math.isFinite(score.*)) return error.QsaGradientNonfinite;
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
            const base = (keyAt(row, j) * g.kv + head) * g.width;
            for (0..g.width) |d| derivative.* += dout[at + d] * v[base + d];
            center += p.* * derivative.*;
        }
        for (probabilities[0..count], dp[0..count], 0..) |p, derivative, j| {
            const ds = p * (derivative - center) * scale;
            const base = (keyAt(row, j) * g.kv + head) * g.width;
            for (0..g.width) |d| {
                dq[at + d] += ds * k[base + d];
                dk[base + d] += ds * q[at + d];
                dv[base + d] += p * dout[at + d];
            }
        }
    };
    for (outputs) |out| try finite(T, out);
}

fn sigmoid(comptime T: type, x: T) T {
    const e = @exp(-@abs(x));
    return if (x >= 0) 1 / (1 + e) else e / (1 + e);
}

/// Stored context/sigmoid retain forward cuts; cast adjoints are straight-through, not rounding derivatives.
pub fn gateBackward(comptime T: type, context: []const T, logits: []const T, stored_sigmoid: []const T, dout: []const T, dx: []T, dg: []T) !void {
    scalar(T);
    const n = context.len;
    if (n == 0 or logits.len != n or stored_sigmoid.len != n or dout.len != n or dx.len != n or dg.len != n)
        return error.QsaGradientShape;
    for ([_][]T{ dx, dg }) |out| for ([_][]const T{ context, logits, stored_sigmoid, dout }) |input| try disjoint(T, out, input);
    try disjoint(T, dx, dg);
    for ([_][]const T{ context, logits, stored_sigmoid, dout }) |input| try finite(T, input);
    for (stored_sigmoid) |s| if (s < 0 or s > 1) return error.QsaGradientShape;
    for (context, logits, stored_sigmoid, dout, dx, dg) |x, gate, saved, dy, *to_x, *to_gate| {
        const s = sigmoid(T, gate);
        to_x.* = dy * saved;
        to_gate.* = dy * x * s * (1 - s);
    }
    if (finite(T, dx)) |_| {
        finite(T, dg) catch |err| {
            @memset(dx, 0);
            @memset(dg, 0);
            return err;
        };
    } else |err| {
        @memset(dx, 0);
        @memset(dg, 0);
        return err;
    }
}

test {
    _ = @import("qsa_backward_test.zig");
}
