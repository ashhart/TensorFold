//! A run's rows against float32 ones, gated by the reference's own bf16 distance from them (twice it, at most).
const std = @import("std");
const resize = @import("resize.zig");

pub const Match = struct { cosine: f64, worst_row: f64, worst_at: usize };

/// The cosine over all values and the largest relative L2 error of one row of `width`.
pub fn match(ours: []const f32, exact: []const f32, width: usize) Match {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var m: Match = .{ .cosine = 0, .worst_row = 0, .worst_at = 0 };
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], exact[row..][0..width]) |a32, b32| {
            const a: f64 = a32;
            const b: f64 = b32;
            dot += a * b;
            aa += a * a;
            bb += b * b;
            rd += (a - b) * (a - b);
            rb += b * b;
        }
        const rel = @sqrt(rd / @max(rb, 1e-30));
        if (rel > m.worst_row) {
            m.worst_row = rel;
            m.worst_at = row / width;
        }
    }
    m.cosine = dot / @sqrt(@max(aa * bb, 1e-300));
    return m;
}

/// The float32 rows rounded to bf16, against themselves: the closest any bf16 result can be.
pub fn rounding(a: std.mem.Allocator, exact: []const f32, width: usize) !Match {
    const r = try a.alloc(f32, exact.len);
    defer a.free(r);
    for (r, exact) |*o, x| o.* = resize.widen(resize.bf16(x));
    return match(r, exact, width);
}

pub const factor = 2.0;

pub const Verdict = struct { pass: bool, cosine_err: f64, cosine_limit: f64, row: f64, row_at: usize, row_limit: f64 };

/// Ours within `factor` of the larger of the reference's bf16 error and bf16 rounding's, in cosine and worst row.
pub fn verdict(ours: Match, reference: Match, rounded: Match) Verdict {
    const cosine_limit = factor * (1 - @min(reference.cosine, rounded.cosine));
    const row_limit = factor * @max(reference.worst_row, rounded.worst_row);
    const cosine_err = 1 - ours.cosine;
    return .{ .pass = cosine_err <= cosine_limit and ours.worst_row <= row_limit, .cosine_err = cosine_err, .cosine_limit = cosine_limit, .row = ours.worst_row, .row_at = ours.worst_at, .row_limit = row_limit };
}

test "a run equal to float32 passes, and one twice as far as the reference fails" {
    const exact = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const near = [_]f32{ 1.01, 2, 3, 4, 5, 6, 7, 8 };
    const far = [_]f32{ 1.5, 2, 3, 4, 5, 6, 7, 8 };
    const ref = match(&near, &exact, 4);
    const rnd = try rounding(std.testing.allocator, &exact, 4);
    try std.testing.expect(verdict(match(&exact, &exact, 4), ref, rnd).pass);
    try std.testing.expect(!verdict(match(&far, &exact, 4), ref, rnd).pass);
}
