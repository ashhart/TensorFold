//! An MTP draft's chance under --mtp-confidence: the ranks' top-1 words with their softmax sums, read on rank 0.
const std = @import("std");

pub const Pick = struct { token: u32, chance: f64 = 1 };

/// Each rank's (value, id, sum, _) -> the largest value, lowest id on a tie; with sums, its whole-vocabulary softmax.
pub fn pick(words: []const [4]u32, sums: bool) Pick {
    var best = words[0];
    for (words[1..]) |p| {
        const v: f32 = @bitCast(p[0]);
        const b: f32 = @bitCast(best[0]);
        if (v > b or (v == b and p[1] < best[1])) best = p;
    }
    if (!sums) return .{ .token = best[1] };
    const top: f64 = @as(f32, @bitCast(best[0]));
    var z: f64 = 0;
    for (words) |p| z += @as(f64, @as(f32, @bitCast(p[2]))) * @exp(@as(f64, @as(f32, @bitCast(p[0]))) - top);
    return .{ .token = best[1], .chance = 1 / z };
}

fn word(v: f32, id: u32, sum: f32) [4]u32 {
    return .{ @bitCast(v), id, @bitCast(sum), 0 };
}

test "mtp chance: the ranks' pick keeps greedy's order, and its chance is the softmax over every rank's slice" {
    const t = std.testing;
    // three slices of logits {2, 0}, {3, 1}, {3, 0}: the max ties at 3, the lower id (rank 1's 10) wins
    const one = [_]f32{ 2, 0 };
    const two = [_]f32{ 3, 1 };
    const three = [_]f32{ 3, 0 };
    var words: [3][4]u32 = undefined;
    var whole: f64 = 0;
    for ([_][]const f32{ &one, &two, &three }, [_]u32{ 0, 10, 20 }, &words) |slice, lo, *w| {
        const m = @max(slice[0], slice[1]);
        var s: f32 = 0;
        for (slice) |x| s += @exp(x - m);
        w.* = word(m, lo + @as(u32, if (slice[0] >= slice[1]) 0 else 1), s);
        for (slice) |x| whole += @exp(@as(f64, x));
    }
    const p = pick(&words, true);
    try t.expectEqual(@as(u32, 10), p.token);
    try t.expectApproxEqRel(@exp(3.0) / whole, p.chance, 1e-6);
    try t.expectEqual(Pick{ .token = 10 }, pick(&words, false));
    // one rank: its slice's sum alone
    try t.expectApproxEqRel(1.0 / 1.5, pick(&.{word(4, 7, 1.5)}, true).chance, 1e-12);
}
