//! A committed token's log probability under the target's raw distribution, with its best tokens (OpenAI logprobs).
const std = @import("std");

/// The most alternatives a row carries (OpenAI's `top_logprobs` limit).
pub const max_top = 20;

/// The words a backend writes for one row: the log normalizer, the pick's logit, the best ids, their logits, the rank.
pub const words = 3 + 2 * max_top;

pub const Row = struct {
    token: u32, // the committed token
    logprob: f32, // its log probability; NaN when it was forced and is not among the best `count`
    count: u8 = 0,
    ids: [max_top]u32 = undefined, // logit descending, ties by the lower id
    logprobs: [max_top]f32 = undefined,
    rank: u32 = 0, // vLLM's: the vocabulary's tokens at or above it, itself included, when the backend was asked for it

    /// One row from a backend's words: `pick` is the token whose logit the backend gathered, `count` best ids follow.
    pub fn fromWords(w: []const u32, pick: u32, count: u8) Row {
        std.debug.assert(count <= max_top and w.len >= words);
        const lse: f32 = @bitCast(w[0]);
        var r: Row = .{ .token = pick, .logprob = @as(f32, @bitCast(w[1])) - lse, .count = count, .rank = w[2 + 2 * max_top] };
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

/// A bf16 word as its f32 value.
fn bf16(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

/// The row of bf16 logits `l` for `pick`: an fp64 normalizer, `count` best ids, ties to the lower id.
pub fn bf16Row(l: []const u16, pick: u32, count: u8) Row {
    var most: f64 = -std.math.inf(f64);
    for (l) |w| most = @max(most, bf16(w));
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, bf16(w)) - most);
    const lse: f32 = @floatCast(most + @log(sum));
    var r: Row = .{ .token = pick, .logprob = bf16(l[pick]) - lse, .count = max_top };
    var filled: usize = 0;
    for (l, 0..) |w, id| {
        const x = bf16(w);
        if (filled == max_top and x <= r.logprobs[max_top - 1] + lse) continue;
        var at = @min(filled, max_top - 1);
        while (at > 0 and x > r.logprobs[at - 1] + lse) : (at -= 1) {
            r.ids[at] = r.ids[at - 1];
            r.logprobs[at] = r.logprobs[at - 1];
        }
        r.ids[at] = @intCast(id);
        r.logprobs[at] = x - lse;
        filled = @min(filled + 1, max_top);
    }
    r.count = @min(count, max_top);
    return r;
}

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

test "a bf16 row: the normalizer, the best ids in order, ties to the lower id" {
    var l: [40]u16 = undefined;
    for (&l, 0..) |*w, i| w.* = @intCast(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 7)))) >> 16);
    const r = bf16Row(&l, 6, 20);
    try std.testing.expectEqual(@as(u32, 6), r.ids[0]);
    try std.testing.expectEqual(@as(u32, 13), r.ids[1]);
    try std.testing.expect(r.logprobs[0] == r.logprob and r.logprobs[19] <= r.logprobs[18]);
}
