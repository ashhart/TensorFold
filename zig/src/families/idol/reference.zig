//! Comparing a velocity with idol-sdk's: cosine, largest difference and worst row, and the gate against fp32.
const std = @import("std");

pub const Match = struct { max_abs: f64, cosine: f64, worst_row: f64 };

/// Largest absolute difference, cosine, and the largest relative L2 error of one row of `width`.
pub fn match(ours: []const f32, theirs: []const f32, width: usize) Match {
    std.debug.assert(ours.len == theirs.len and ours.len % width == 0);
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var most: f64 = 0;
    var worst: f64 = 0;
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |a, b| {
            const d = @as(f64, a) - b;
            dot += @as(f64, a) * b;
            aa += @as(f64, a) * a;
            bb += @as(f64, b) * b;
            rd += d * d;
            rb += @as(f64, b) * b;
            most = @max(most, @abs(d));
        }
        worst = @max(worst, @sqrt(rd / @max(rb, 1e-30)));
    }
    return .{ .max_abs = most, .cosine = dot / @sqrt(@max(aa * bb, 1e-300)), .worst_row = worst };
}

/// How far ours may sit from the fp32 ruler: this many times the torch fp16 reference's own distance.
pub const factor = 2.0;

/// Against the fp32 ruler: worst row and 1 - cosine within `factor` times the fp16 reference's own; non-finite fails.
pub fn within(ours: Match, reference: Match) bool {
    if (!std.math.isFinite(ours.cosine) or !std.math.isFinite(ours.worst_row)) return false;
    return ours.worst_row <= factor * reference.worst_row and (1 - ours.cosine) <= factor * (1 - reference.cosine);
}

/// The cosine half of `within`: for a velocity, whose worst row the gate takes over the whole run instead.
pub fn withinCosine(ours: Match, reference: Match) bool {
    return std.math.isFinite(ours.cosine) and (1 - ours.cosine) <= factor * (1 - reference.cosine);
}

test "a velocity matches itself exactly" {
    const v = [_]f32{ 1, -2, 3, 0.5, 0.25, -1 };
    const m = match(&v, &v, 3);
    try std.testing.expectEqual(@as(f64, 0), m.max_abs);
    try std.testing.expectEqual(@as(f64, 0), m.worst_row);
    try std.testing.expectApproxEqAbs(@as(f64, 1), m.cosine, 1e-15);
}

test "a planted 5% error in one row fails the gate though everything else repeats exactly" {
    const a = std.testing.allocator;
    const n = 64 * 16;
    const ruler = try a.alloc(f32, n);
    defer a.free(ruler);
    const fp16 = try a.alloc(f32, n);
    defer a.free(fp16);
    const ours = try a.alloc(f32, n);
    defer a.free(ours);
    var rng = std.Random.DefaultPrng.init(7);
    for (ruler, fp16, ours) |*r, *f, *o| {
        r.* = rng.random().floatNorm(f32);
        f.* = r.* * (1 + 1e-3 * rng.random().floatNorm(f32));
        o.* = r.* * (1 + 1e-3 * rng.random().floatNorm(f32));
    }
    const reference = match(fp16, ruler, 16);
    try std.testing.expect(within(match(ours, ruler, 16), reference));
    for (ours[5 * 16 ..][0..16]) |*o| o.* *= 1.05;
    try std.testing.expect(!within(match(ours, ruler, 16), reference));
}

test "non-finite values fail the gate" {
    const ruler = [_]f32{ 1, 2, 3, 4 };
    const bad = [_]f32{ 1, std.math.nan(f32), 3, 4 };
    try std.testing.expect(!within(match(&bad, &ruler, 2), match(&ruler, &ruler, 2)));
}
