//! A committed token's log probability under the target's raw distribution, with its best tokens (OpenAI logprobs).
const std = @import("std");

/// The most alternatives a row carries (OpenAI's `top_logprobs` limit).
pub const max_top = 20;

/// The words a backend writes for one row: the log normalizer, the pick's logit, the best ids, their logits.
pub const words = 2 + 2 * max_top;

pub const Row = struct {
    token: u32, // the committed token
    logprob: f32, // its log probability; NaN when it was forced and is not among the best `count`
    count: u8 = 0,
    ids: [max_top]u32 = undefined, // logit descending, ties by the lower id
    logprobs: [max_top]f32 = undefined,

    /// One row from a backend's words: `pick` is the token whose logit the backend gathered, `count` best ids follow.
    pub fn fromWords(w: []const u32, pick: u32, count: u8) Row {
        std.debug.assert(count <= max_top and w.len >= words);
        const lse: f32 = @bitCast(w[0]);
        var r: Row = .{ .token = pick, .logprob = @as(f32, @bitCast(w[1])) - lse, .count = count };
        for (0..count) |i| {
            r.ids[i] = w[2 + i];
            r.logprobs[i] = @as(f32, @bitCast(w[2 + max_top + i])) - lse;
        }
        return r;
    }

    /// A forced token's row at this position: its log probability among the best, else NaN.
    pub fn forToken(r: Row, token: u32) Row {
        if (token == r.token) return r;
        var out = r;
        out.token = token;
        out.logprob = std.math.nan(f32);
        for (r.ids[0..r.count], r.logprobs[0..r.count]) |id, lp| {
            if (id == token) out.logprob = lp;
        }
        return out;
    }
};

test "a row reads the normalizer, the pick and the best ids from the backend's words" {
    var w: [words]u32 = @splat(0);
    w[0] = @bitCast(@as(f32, 2.5));
    w[1] = @bitCast(@as(f32, 2.0));
    w[2] = 7;
    w[3] = 9;
    w[2 + max_top] = @bitCast(@as(f32, 2.0));
    w[3 + max_top] = @bitCast(@as(f32, 1.0));
    const r = Row.fromWords(&w, 7, 2);
    try std.testing.expectEqual(@as(f32, -0.5), r.logprob);
    try std.testing.expectEqualSlices(u32, &.{ 7, 9 }, r.ids[0..2]);
    try std.testing.expectEqual(@as(f32, -1.5), r.logprobs[1]);
    try std.testing.expectEqual(@as(f32, -1.5), r.forToken(9).logprob);
    try std.testing.expect(std.math.isNan(r.forToken(3).logprob));
}
