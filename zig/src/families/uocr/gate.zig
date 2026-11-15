//! A run's rows against float32 ones, gated by the reference's bf16 distance (twice it, at most); a reply's ties.
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

test "uocr: a run equal to float32 passes, and one twice as far as the reference fails" {
    const exact = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const near = [_]f32{ 1.01, 2, 3, 4, 5, 6, 7, 8 };
    const far = [_]f32{ 1.5, 2, 3, 4, 5, 6, 7, 8 };
    const ref = match(&near, &exact, 4);
    const rnd = try rounding(std.testing.allocator, &exact, 4);
    try std.testing.expect(verdict(match(&exact, &exact, 4), ref, rnd).pass);
    try std.testing.expect(!verdict(match(&far, &exact, 4), ref, rnd).pass);
}

/// How many bf16 steps apart two logits are, a step being bf16's spacing at the larger one's size.
pub fn bf16Steps(x: f32, y: f32) f32 {
    const size = @max(@abs(x), @abs(y));
    if (size == 0) return 0;
    const ulp = std.math.ldexp(@as(f32, 1), std.math.frexp(size).exponent - 1 - 7);
    return @abs(x - y) / ulp;
}

test "uocr: bf16 steps between two logits follow bf16's spacing at their size" {
    try std.testing.expectEqual(@as(f32, 0), bf16Steps(24.5, 24.5));
    try std.testing.expectEqual(@as(f32, 1), bf16Steps(24.0, 24.125)); // [16, 32): a step is 0.125
    try std.testing.expectEqual(@as(f32, 2), bf16Steps(40.0, 40.5)); // [32, 64): a step is 0.25
    try std.testing.expectEqual(@as(f32, 4), bf16Steps(-30.0, -29.5));
    try std.testing.expectEqual(@as(f32, 0.5), bf16Steps(1.0, 1.00390625)); // [1, 2): a step is 1/128
}

/// A reference step's two highest logits: id, logit, id, logit.
pub const Top2 = [4]f64;

/// A departure from torch bf16's pick is a tie when both runs weigh the same two tokens, each within one bf16 step.
pub const Departure = struct {
    same_two: bool,
    ours: f32,
    theirs: f32,
    pub fn tie(d: Departure) bool {
        return d.same_two and d.ours <= 1 and d.theirs <= 1;
    }
};

/// Our bf16 logits `lg` picked `ours` where the reference picked `want`, with the reference's `top2` at that step.
pub fn departure(lg: []const u16, ours: u32, want: u32, top2: Top2) Departure {
    const t0: u32 = @intFromFloat(top2[0]);
    const t1: u32 = @intFromFloat(top2[2]);
    return .{
        .same_two = (t0 == ours and t1 == want) or (t0 == want and t1 == ours),
        .ours = bf16Steps(resize.widen(lg[ours]), resize.widen(lg[want])),
        .theirs = bf16Steps(@floatCast(top2[1]), @floatCast(top2[3])),
    };
}

test "uocr: a departure is a tie only with the same two contenders and both margins within one bf16 step" {
    var lg = std.mem.zeroes([8]u16);
    lg[3] = resize.bf16(24.125);
    lg[5] = resize.bf16(24.0);
    const t = departure(&lg, 3, 5, .{ 5, 24.0625, 3, 24.0 });
    try std.testing.expect(t.tie() and t.same_two and t.ours == 1 and t.theirs == 0.5);
    try std.testing.expect(!departure(&lg, 3, 5, .{ 5, 24.0625, 7, 24.0 }).tie()); // torch's second is another token
    try std.testing.expect(!departure(&lg, 3, 5, .{ 5, 24.5, 3, 24.0 }).tie()); // torch's margin is four steps
    lg[3] = resize.bf16(24.5);
    try std.testing.expect(!departure(&lg, 3, 5, .{ 5, 24.0625, 3, 24.0 }).tie()); // ours is four steps
}
