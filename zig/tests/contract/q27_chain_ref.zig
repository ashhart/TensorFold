//! Independent FP64 causal attention and a cancellation witness for its FP32 value sums.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const D = 256;

pub fn stress(key: usize) f32 {
    return if (key < 32) 32768 else if (key < 96) 1 else if (key < 128) -32768 else 0;
}

pub fn reference(a: std.mem.Allocator, query: []const u16, keys: []const u16, values: []const u16, gate: ?[]const u16, rows: usize, heads: usize, kv: usize, start: usize, cap: usize, want: []f64, mag: []f64) !void {
    const scores = try a.alloc(f64, cap);
    defer a.free(scores);
    const output = try a.alloc(f64, D);
    defer a.free(output);
    const absolute = try a.alloc(f64, D);
    defer a.free(absolute);
    for (0..rows) |r| for (0..heads) |h| {
        const kh = h / (heads / kv);
        const count = start + r + 1;
        var top = -std.math.inf(f64);
        for (scores[0..count], 0..) |*s, key| {
            var sum: f64 = 0;
            for (0..D) |d| sum += @as(f64, c.fromBf16(query[(r * heads + h) * D + d])) * c.fromBf16(keys[(kh * cap + key) * D + d]);
            s.* = sum / 16;
            top = @max(top, s.*);
        }
        var total: f64 = 0;
        @memset(output, 0);
        @memset(absolute, 0);
        for (scores[0..count], 0..) |s, key| {
            const p = @exp(s - top);
            total += p;
            for (output, absolute, 0..) |*o, *abs, d| {
                const term = p * c.fromBf16(values[(kh * cap + key) * D + d]);
                o.* += term;
                abs.* += @abs(term);
            }
        }
        for (output, 0..) |o, d| {
            const at = (r * heads + h) * D + d;
            const factor: f64 = if (gate) |g| c.fromBf16(c.toBf16(@floatCast(1 / (1 + @exp(-@as(f64, c.fromBf16(g[(r * heads + h) * 2 * D + D + d]))))))) else 1;
            want[at] = o / total * factor;
            mag[at] = absolute[d] / total * @abs(factor);
        }
    };
}

pub fn stressReference(start: usize, rows: usize, width: usize, gated: bool, want: []f64, mag: []f64) void {
    for (0..rows) |r| {
        var sum: f64 = 0;
        for (0..start + r + 1) |key| sum += stress(key);
        const value = sum / @as(f64, @floatFromInt(start + r + 1)) * (if (gated) @as(f64, 0.5) else 1);
        @memset(want[r * width ..][0..width], value);
        @memset(mag[r * width ..][0..width], @abs(value));
    }
}

pub fn simulate(a: std.mem.Allocator, start: usize, rows: usize, width: usize, gated: bool, acc: c.Accum, order: contract.gates.Order, out: []f64) !void {
    const terms = try a.alloc(f64, start + rows);
    defer a.free(terms);
    for (terms, 0..) |*v, key| v.* = stress(key);
    for (0..rows) |r| {
        const value = contract.rows.sumIn(acc, order, terms[0 .. start + r + 1]) / @as(f64, @floatFromInt(start + r + 1)) * (if (gated) @as(f64, 0.5) else 1);
        @memset(out[r * width ..][0..width], value);
    }
}

test "long attention stress separates reduced sums in every contract order" {
    var want: [4]f64 = undefined;
    var mag: [4]f64 = undefined;
    var got: [4]f64 = undefined;
    stressReference(8193, 4, 1, false, &want, &mag);
    for ([_]contract.gates.Order{ .sequential, .pairwise, .strided }) |order| {
        try simulate(std.testing.allocator, 8193, 4, 1, false, .f32, order, &got);
        try std.testing.expectEqualSlices(f64, &want, &got);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try simulate(std.testing.allocator, 8193, 4, 1, false, acc, order, &got);
            try std.testing.expect(!std.math.isFinite(got[0]) or @abs(got[0] - want[0]) > 0.75 * @abs(want[0]));
        }
    }
}

test "FP64 chain formula is causal and applies the rounded gate at every output column" {
    const rows = 2;
    const heads = 6;
    const cap = 3;
    const query: [rows * heads * D]u16 = @splat(0);
    const keys: [cap * D]u16 = @splat(0);
    var values: [cap * D]u16 = undefined;
    @memset(values[0..D], c.toBf16(2));
    @memset(values[D .. 2 * D], c.toBf16(-2));
    @memset(values[2 * D ..], c.toBf16(std.math.nan(f32)));
    const gate: [rows * heads * 2 * D]u16 = @splat(0);
    var want: [rows * heads * D]f64 = undefined;
    var mag: [rows * heads * D]f64 = undefined;
    for ([_]bool{ false, true }) |gated| {
        try reference(std.testing.allocator, &query, &keys, &values, if (gated) &gate else null, rows, heads, 1, 0, cap, &want, &mag);
        for (want[0 .. heads * D]) |v| try std.testing.expectEqual(@as(f64, if (gated) 1 else 2), v);
        for (want[heads * D ..]) |v| try std.testing.expectEqual(@as(f64, 0), v);
        for (mag) |v| try std.testing.expectEqual(@as(f64, if (gated) 1 else 2), v);
    }
}
