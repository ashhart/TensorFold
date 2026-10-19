//! Signed pointwise BF16 endpoints from measured scalar extrema and the exact measured normalized pair.
const std = @import("std");
const accuracy = @import("rotary_accuracy.zig");
const fixture = @import("rotary_fixture.zig");
pub const Interval = struct { low: f64, high: f64 };
fn r32(x: f64) f32 {
    return @floatCast(x);
}
fn bf(x: f32) f64 {
    return fixture.from(fixture.bf(x));
}
fn radius(m: accuracy.Maximum, reference: f64) f64 {
    return std.math.nextAfter(f64, @min(m.absolute, m.ulps * accuracy.ulp(reference)), std.math.inf(f64));
}
fn ceil32(x: f64) f32 {
    const y = r32(x);
    return if (@as(f64, y) < x) std.math.nextAfter(f32, y, std.math.inf(f32)) else y;
}
fn floor32(x: f64) f32 {
    const y = r32(x);
    return if (@as(f64, y) > x) std.math.nextAfter(f32, y, -std.math.inf(f32)) else y;
}
fn coefficients(m: accuracy.Maximum, ref: f64, scale: f32) ![2]f32 {
    const error_ = radius(m, ref);
    const low = ceil32(std.math.nextAfter(f64, ref - error_, -std.math.inf(f64)));
    const high = floor32(std.math.nextAfter(f64, ref + error_, std.math.inf(f64)));
    if (low > high) return error.InvalidRotaryReceipt;
    return .{ low * scale, high * scale };
}
pub fn rotation(a: f32, b: f32, cosines: [2]f32, sines: [2]f32, upper: bool) Interval {
    @setFloatMode(.strict);
    var out: Interval = .{ .low = std.math.inf(f64), .high = -std.math.inf(f64) };
    const sign: f32 = if (upper) 1 else -1;
    for (cosines) |co| for (sines) |si| {
        const ac = a * co;
        const bs = sign * b * si;
        for ([_]f32{ ac + bs, @mulAdd(f32, a, co, bs), @mulAdd(f32, sign * b, si, ac) }) |value| {
            out.low = @min(out.low, bf(value));
            out.high = @max(out.high, bf(value));
        }
    };
    return out;
}
fn norm(row: *const accuracy.Row, input: []const u16, weights: []const f32, eps: f32) !*const accuracy.Norm {
    if (input.len != weights.len or input.len == 0) return error.UnrecordedRotaryFixture;
    for (weights) |w| if (@as(u32, @bitCast(w)) != @as(u32, @bitCast(weights[0]))) return error.UnrecordedRotaryFixture;
    for (row.normalization) |*n| {
        if (n.width == input.len and n.weight == @as(u32, @bitCast(weights[0])) and n.epsilon == @as(u32, @bitCast(eps)) and std.mem.eql(u16, n.input[0..n.width], input)) return n;
    }
    return error.UnrecordedRotaryFixture;
}
pub fn normalized(row: *const accuracy.Row, input: []const u16, pos: [3]i32, weights: []const f32, eps: f32, rotary: []const f32, intervals: []Interval) !void {
    @setFloatMode(.strict);
    if (input.len != intervals.len or try accuracy.domain.modeOf(rotary) != row.mode) return error.UnrecordedRotaryTable;
    const n = try norm(row, input, weights, eps);
    for (intervals, 0..) |*out, d| {
        const a = fixture.from(n.output[d]);
        if (d >= 64) {
            out.* = .{ .low = a, .high = a };
            continue;
        }
        const i = d % 32;
        const axis = if (row.role == .mrope) accuracy.domain.point(.mrope, i).axis else 0;
        const position = pos[axis];
        if (position < 0 or position >= accuracy.domain.maximum or @mod(position, @as(i32, @intCast(accuracy.domain.stride(row.role)))) != 0) return error.UnrecordedRotaryPosition;
        const frequency: f32 = @bitCast(row.frequency_bits[i]);
        const angle: f32 = @as(f32, @floatFromInt(position)) * frequency;
        const scale: f32 = if (row.mode == .yarn) rotary[2] else 1;
        const co = try coefficients(row.cosine, @cos(@as(f64, angle)), scale);
        const si = try coefficients(row.sine, @sin(@as(f64, angle)), scale);
        const b = fixture.from(n.output[if (d < 32) d + 32 else d - 32]);
        out.* = rotation(a, b, co, si, d >= 32);
    }
}

pub fn pooled(row: *const accuracy.Row, raw: []const u16, position: i32, weights: []const f32, eps: f32, rotary: []const f32, intervals: []Interval) !void {
    if (raw.len != 512) return error.UnrecordedRotaryFixture;
    var mean: [128]u16 = undefined;
    for (&mean, 0..) |*h, d| {
        var total: f32 = fixture.from(raw[d]);
        for (1..4) |k| total += fixture.from(raw[k * 128 + d]);
        h.* = fixture.bf(total * 0.25);
    }
    try normalized(row, &mean, @splat(position), weights, eps, rotary, intervals);
}
