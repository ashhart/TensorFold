//! A logprob row from one row of the XPU's logits on the host: the normalizer, the pick, the `count` best ids.
const std = @import("std");
const lanes = @import("lanes");

const top = lanes.logprob.max_top;
const lane_width = 8;

/// e^x for x <= 0 (under -80: 0), 1.5e-7 relative per unit of |x| at worst: 2^n by exponent bits, 2^f by Taylor.
fn expLow(x: @Vector(lane_width, f32)) @Vector(lane_width, f32) {
    const V = @Vector(lane_width, f32);
    const U = @Vector(lane_width, u32);
    const low: V = @splat(-80);
    const t = @max(x, low) * @as(V, @splat(1.4426950408889634)); // x in log2 units
    const n = @round(t);
    const f = (t - n) * @as(V, @splat(0.6931471805599453)); // the remainder in natural units, |f| <= 0.347
    var p: V = @splat(1.0 / 5040.0);
    inline for (.{ 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0 }) |c| p = p * f + @as(V, @splat(c));
    const biased = @as(@Vector(lane_width, i32), @intFromFloat(n)) + @as(@Vector(lane_width, i32), @splat(127));
    const bits: U = @bitCast(biased);
    const scale: V = @bitCast(bits << @as(@Vector(lane_width, u5), @splat(23)));
    return @select(f32, x < low, @as(V, @splat(0)), p * scale);
}

/// The best `top` logits so far, by logit descending, ties to the lower id (ids arrive in increasing order).
const Best = struct {
    raw: [top]f32 = undefined,
    ids: [top]u32 = undefined,
    filled: usize = 0,
    bar: f32 = -std.math.inf(f32), // a logit must pass it to be taken: the worst held once `top` are held

    fn add(b: *Best, id: usize, x: f32) void {
        if (!(x > b.bar)) return;
        var slot = @min(b.filled, top - 1);
        while (slot > 0 and x > b.raw[slot - 1]) : (slot -= 1) {
            b.raw[slot] = b.raw[slot - 1];
            b.ids[slot] = b.ids[slot - 1];
        }
        b.raw[slot] = x;
        b.ids[slot] = @intCast(id);
        b.filled = @min(b.filled + 1, top);
        if (b.filled == top) b.bar = b.raw[top - 1];
    }
};

/// The row of `logits` for `pick`: f64 normalizer, best ids by logit, ties to the lower id; NaN or +inf: NaN logprob.
pub fn row(logits: []const f32, pick: u32, count: u8) lanes.LogprobRow {
    const V = @Vector(lane_width, f32);
    var peak: V = @splat(-std.math.inf(f32));
    var finite = true;
    const inf: V = @splat(std.math.inf(f32));
    var from: usize = 0;
    while (from + lane_width <= logits.len) : (from += lane_width) {
        const v: V = logits[from..][0..lane_width].*;
        peak = @max(peak, v);
        finite = finite and @reduce(.And, v < inf);
    }
    var most: f32 = @reduce(.Max, peak);
    for (logits[from..]) |x| {
        most = @max(most, x);
        finite = finite and x < std.math.inf(f32);
    }
    const shift: V = @splat(most);
    var sum: f64 = 0;
    var block: V = @splat(0);
    var best: Best = .{};
    var at: usize = 0;
    while (at + lane_width <= logits.len) : (at += lane_width) {
        const v: V = logits[at..][0..lane_width].*;
        block += expLow(v - shift);
        if (at % 64 == 56) { // f32 sums of 64 terms, kept in f64
            sum += @reduce(.Add, block);
            block = @splat(0);
        }
        if (@reduce(.Or, v > @as(V, @splat(best.bar)))) for (logits[at..][0..lane_width], at..) |x, id| best.add(id, x);
    }
    sum += @reduce(.Add, block);
    for (logits[at..], at..) |x, id| {
        sum += @exp(x - most);
        best.add(id, x);
    }
    const lse: f32 = most + @as(f32, @floatCast(@log(sum)));
    const nan = std.math.nan(f32);
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = if (finite) logits[pick] - lse else nan, .count = @min(count, top) };
    r.ids = best.ids;
    for (best.raw[0..best.filled], 0..) |x, i| r.logprobs[i] = x - lse;
    return r;
}

test "draft sources: XPU logprob row: the vector exponential stays near the library's" {
    var x: [lane_width]f32 = undefined;
    var k: f32 = 0;
    while (k > -90) : (k -= 0.37) {
        for (&x, 0..) |*v, i| v.* = k - @as(f32, @floatFromInt(i)) * 0.0113;
        const got: [lane_width]f32 = expLow(x);
        for (x, got) |a, g| {
            const want: f64 = @exp(@as(f64, a));
            if (a < -80) {
                try std.testing.expectEqual(@as(f32, 0), g);
            } else {
                try std.testing.expectApproxEqRel(@as(f32, @floatCast(want)), g, 1.5e-7 * (1 - a));
            }
        }
    }
}

test "draft sources: XPU logprob row: normalizer, best ids in order, ties to the lower id, a pick outside them" {
    var l: [130]f32 = undefined;
    for (&l, 0..) |*x, i| x.* = @floatFromInt(i % 7);
    const r = row(&l, 5, 3);
    var exact: f64 = 0;
    for (l) |x| exact += @exp(@as(f64, x) - 6.0);
    const lse: f64 = 6.0 + @log(exact);
    try std.testing.expectEqual(@as(u8, 3), r.count);
    try std.testing.expectEqualSlices(u32, &.{ 6, 13, 20 }, r.ids[0..3]); // the lower ids of the equal best logits
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(6.0 - lse)), r.logprobs[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(5.0 - lse)), r.logprob, 1e-5); // 5 is no best id: still exact
    try std.testing.expectEqual(@as(u32, 5), r.token);
    const all = row(&l, 6, 200); // the count never passes max_top
    try std.testing.expectEqual(@as(u8, lanes.logprob.max_top), all.count);
}

test "draft sources: XPU logprob row: a NaN or +inf logit anywhere gives a NaN logprob, -inf is fine" {
    var l: [131]f32 = undefined;
    for (&l, 0..) |*x, i| x.* = @floatFromInt(i % 5);
    try std.testing.expect(!std.math.isNan(row(&l, 3, 2).logprob));
    for ([_]usize{ 0, 17, 128, 130 }) |at| { // in a vector block and in the scalar tail
        for ([_]f32{ std.math.nan(f32), std.math.inf(f32) }) |bad| {
            var m = l;
            m[at] = bad;
            try std.testing.expect(std.math.isNan(row(&m, 3, 2).logprob));
        }
    }
    l[40] = -std.math.inf(f32);
    try std.testing.expect(!std.math.isNan(row(&l, 3, 2).logprob));
}
