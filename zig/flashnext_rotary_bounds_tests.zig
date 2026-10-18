//! Scalar phase and rounding controls for the conditional rotary probe allowance.
const std = @import("std");
const f = @import("src/families/flashnext/vision_contract.zig");
const c = @import("contract").spec;
fn fp32(input: []const u16, positions: [3]i32, weights: []const f32, rotary: []const f32, output: []u16) void {
    @setFloatMode(.strict);
    var total: f32 = 0;
    for (input) |h| {
        const x: f32 = c.fromBf16(h);
        total += x * x;
    }
    const denominator: f32 = total / @as(f32, @floatFromInt(input.len)) + @as(f32, 1e-5);
    const inv: f32 = @floatCast(1 / @sqrt(@as(f64, denominator)));
    var normed: [256]f32 = undefined;
    for (input, weights, 0..) |h, w, i| {
        const product: f32 = c.fromBf16(h) * inv;
        normed[i] = c.fromBf16(c.toBf16(product * w));
    }
    for (input, 0..) |_, d| {
        var value = normed[d];
        if (d < 64) {
            const i = d % 32;
            const exponent: f32 = -(@as(f32, @floatFromInt(i)) / 32) * rotary[0];
            const freq: f32 = if (rotary[1] != 0) rotary[3 + i] else @floatCast(@exp2(@as(f64, exponent)));
            const angle: f32 = @as(f32, @floatFromInt(positions[f.axis(i)])) * freq;
            const scale: f32 = if (rotary[1] != 0) rotary[2] else 1;
            const co: f32 = @as(f32, @floatCast(@cos(@as(f64, angle)))) * scale;
            const si: f32 = @as(f32, @floatCast(@sin(@as(f64, angle)))) * scale;
            const left: f32 = if (d < 32) normed[d] * co else normed[d - 32] * si;
            const right: f32 = if (d < 32) normed[d + 32] * si else normed[d] * co;
            value = if (d < 32) left - right else left + right;
        }
        output[d] = c.toBf16(value);
    }
}

fn witness(factor: f64, seed: usize, row: usize, head: usize, d: usize) !void {
    const width: usize = if (head < 26) 256 else 128;
    const source: usize = if (head < 24) head * 512 else if (head < 26) 24 * 512 + (head - 24) * 256 else 24 * 512 + 4 * 256 + (head - 26) * 128;
    const weight: f32 = if (head < 24) 1 else if (head < 26) 0.75 else 1.25;
    var input: [256]u16 = undefined;
    const weights: [256]f32 = @splat(weight);
    for (input[0..width], 0..) |*h, i| h.* = c.toBf16(@as(f32, @floatFromInt((row * f.packed_width + source + i + seed) % 17)) / 8 - 1);
    const pos: [3]i32 = @splat(@intCast((if (factor > 1) @as(usize, 262140) else 0) + row * 17));
    const rotary = f.table(23.253496170043945, factor);
    var want: [256]u16 = undefined;
    var got: [256]u16 = undefined;
    var mag: [256]f64 = undefined;
    var allow: [256]f64 = undefined;
    f.normalizedRotation(input[0..width], pos, weights[0..width], 1e-5, &rotary, .f64, .sequential, want[0..width], mag[0..width]);
    fp32(input[0..width], pos, weights[0..width], &rotary, got[0..width]);
    if (@hasDecl(f, "normalizedBounds")) {
        try f.normalizedBounds(input[0..width], pos, weights[0..width], 1e-5, &rotary, allow[0..width]);
    } else {
        for (want[0..width], mag[0..width], allow[0..width]) |h, m, *a| a.* = c.bound(f.spec(c, .target), .{ .rows = 128, .n = f.output_width, .k = 256 }, c.fromBf16(h), m);
    }
    const error_observed = @abs(@as(f64, c.fromBf16(got[d])) - c.fromBf16(want[d]));
    try std.testing.expect(error_observed <= allow[d]);
    try std.testing.expect(@abs(2 * @as(f64, c.fromBf16(want[d]))) > allow[d]);
    for (want[0..width], got[0..width], allow[0..width]) |w, g, a| try std.testing.expect(@abs(@as(f64, c.fromBf16(w)) - c.fromBf16(g)) <= a);
}
test "rotary bound: the original plain and scaled phase witnesses fit without accepting a sign change" {
    try witness(0, 2, 50, 0, 33);
    try witness(4, 11, 10, 26, 12);
}
test "rotary bound: every residue, all coordinate axes and context boundaries enclose the FP32 source stages" {
    const weights: [128]f32 = @splat(1.25);
    for ([_]f64{ 0, 4 }) |factor| for (0..17) |seed| for ([_]i32{ 0, 1, 262143, 262144, 1048575 }) |position| {
        var p: [128]u16 = undefined;
        for (&p, 0..) |*h, d| h.* = c.toBf16(@as(f32, @floatFromInt((d + seed) % 17)) / 8 - 1);
        const pos: [3]i32 = .{ position, @max(position - 3, 0), @max(position - 7, 0) };
        const rotary = f.table(23.253496170043945, factor);
        var want: [128]u16 = undefined;
        var got: [128]u16 = undefined;
        var allow: [128]f64 = undefined;
        f.normalizedRotation(&p, pos, &weights, 1e-5, &rotary, .f64, .sequential, &want, null);
        fp32(&p, pos, &weights, &rotary, &got);
        try f.normalizedBounds(&p, pos, &weights, 1e-5, &rotary, &allow);
        for (want, got, allow) |w, actual, e| {
            try std.testing.expect(std.math.isFinite(e) and e >= 0);
            try std.testing.expect(@abs(@as(f64, c.fromBf16(actual)) - c.fromBf16(w)) <= e);
        }
    };
}
test "rotary bound: normalization midpoint crossings survive endpoint rounding" {
    const p: [128]u16 = @splat(0x3f80);
    const w: [128]f32 = @splat(1.00390625);
    const rotary = f.table(23.253496170043945, 0);
    var allow: [128]f64 = undefined;
    try f.normalizedBounds(&p, .{ 0, 0, 0 }, &w, 1e-30, &rotary, &allow);
    try std.testing.expect(allow[64] >= 0.0078125);
    try std.testing.expect(allow[64] < 0.015625);
    try std.testing.expectError(error.InvalidVisionBoundInput, f.normalizedBounds(&p, .{ 1048576, 0, 0 }, &w, 1e-5, &rotary, &allow));
}
test "rotary bound: wrong axes and a wrong rotary scale still fail" {
    const p: [128]u16 = @splat(0x3f80);
    const w: [128]f32 = @splat(1.25);
    const rotary = f.table(23.253496170043945, 4);
    var wrong = rotary;
    wrong[2] *= 2;
    var want: [128]u16 = undefined;
    var got: [128]u16 = undefined;
    var allow: [128]f64 = undefined;
    f.normalizedRotation(&p, .{ 3, 5, 7 }, &w, 1e-5, &rotary, .f64, .sequential, &want, null);
    try f.normalizedBounds(&p, .{ 3, 5, 7 }, &w, 1e-5, &rotary, &allow);
    fp32(&p, .{ 3, 3, 3 }, &w, &rotary, &got);
    try std.testing.expect(@abs(@as(f64, c.fromBf16(got[1])) - c.fromBf16(want[1])) > allow[1]);
    fp32(&p, .{ 3, 5, 7 }, &w, &wrong, &got);
    try std.testing.expect(@abs(@as(f64, c.fromBf16(got[0])) - c.fromBf16(want[0])) > allow[0]);
}
test "rotary bound: an unsupported fixture cannot silently obtain a wider allowance" {
    var p: [128]u16 = @splat(0x3f80);
    const weights: [128]f32 = @splat(1.25);
    const rotary = f.table(23.253496170043945, 0);
    var allow: [128]f64 = undefined;
    p[0] = c.toBf16(0.1);
    try std.testing.expectError(error.UnsupportedVisionBoundFixture, f.normalizedBounds(&p, .{ 0, 0, 0 }, &weights, 1e-5, &rotary, &allow));
    p[0] = 0x3f80;
    try std.testing.expectError(error.InvalidVisionBoundInput, f.normalizedBounds(&p, .{ 0, 0, 0 }, &weights, 0, &rotary, &allow));
    var raw: [4 * 128]u16 = @splat(0);
    raw[0] = c.toBf16(1);
    raw[128] = c.toBf16(0x1p-30);
    try std.testing.expectError(error.InexactPoolBoundFixture, f.pooledBounds(&raw, 0, &weights, 1e-5, &rotary, &allow));
}
