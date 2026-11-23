// The greedy and sampled picks over a logits row; kept apart from target.zig to keep it under the 600-line cap.
const std = @import("std");
const host = @import("dsv41");
const target_ = @import("target.zig");
const lanes = host.lanes;
const Target = target_.Target;

/// The keyed choice of a logits row over the whole vocabulary (greedy: the first maximum, ties to the lower id).
pub fn choose(t: *const Target, row: []const f32, draw: u64, smp: ?lanes.Sampling) !u32 {
    const s = smp orelse return argmax(row);
    if (s.temperature <= 0) return argmax(row);
    const vals = t.pick_vals[0..row.len];
    for (vals, row) |*v, x| v.* = x;
    return @intCast(try lanes.sampling.choose(t.a, vals, t.pick_ids[0..row.len], draw, s));
}

/// The first maximum, ties to the lower id, in 16-wide steps; a row holding a NaN takes the scalar scan.
pub fn argmax(row: []const f32) u32 {
    const L = 16;
    const V = @Vector(L, f32);
    var top: V = @splat(-std.math.inf(f32));
    var nan = false;
    var i: usize = 0;
    while (i + L <= row.len) : (i += L) {
        const v: V = row[i..][0..L].*;
        nan = nan or @reduce(.Or, v != v);
        top = @max(top, v);
    }
    var m = @reduce(.Max, top);
    for (row[i..]) |x| {
        nan = nan or x != x;
        m = @max(m, x);
    }
    if (nan) return scan(row);
    var j: usize = 0;
    while (j + L <= row.len) : (j += L) {
        const v: V = row[j..][0..L].*;
        if (@reduce(.Or, v == @as(V, @splat(m)))) break;
    }
    while (j < row.len) : (j += 1) if (row[j] == m) return @intCast(j);
    return 0;
}

pub fn scan(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = i;
    };
    return @intCast(best);
}

test "argmax: the first maximum, NaN rows as the scalar scan" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const row = try a.alloc(f32, 129280);
    defer a.free(row);
    for (0..200) |k| {
        for (row) |*x| x.* = rnd.floatNorm(f32);
        if (k % 5 == 1) row[rnd.uintLessThan(usize, row.len)] = 9.0;
        if (k % 5 == 2) row[rnd.uintLessThan(usize, row.len)] = std.math.nan(f32);
        if (k % 5 == 3) @memset(row, -std.math.inf(f32));
        if (k % 5 == 4) row[rnd.uintLessThan(usize, 7)] = std.math.nan(f32);
        try std.testing.expectEqual(scan(row), argmax(row));
    }
}
