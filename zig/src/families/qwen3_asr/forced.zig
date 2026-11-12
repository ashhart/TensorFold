//! The teacher-forced tie rule (#2648's): on the reference's history, a differing top token passes only as a swap.
const std = @import("std");

/// The widest top-1/top-2 margin a swapped step may have, on either side, in logits.
pub const bar: f32 = 0.25;

/// A step's two best tokens and the first one's lead.
pub const Top = struct { first: u32, second: u32, margin: f32 };

pub const Step = enum { same, swap, fail };

/// Same top token, or the two tokens swapped with both margins at most `max`; anything else (NaN too) fails.
pub fn judge(ours: Top, ref: Top, max: f32) Step {
    if (ours.first == ref.first and !std.math.isNan(ours.margin)) return .same;
    const swapped = ours.first == ref.second and ours.second == ref.first;
    return if (swapped and ours.margin <= max and ref.margin <= max) .swap else .fail;
}

/// The reference's step from its topk pair and its greedy id: topk may list an exact tie (margin 0) either way round.
pub fn reference(first: u32, second: u32, margin: f32, greedy: u32) error{TopIsNotGreedy}!Top {
    var r: Top = .{ .first = first, .second = second, .margin = margin };
    if (r.first != greedy and r.second == greedy and r.margin == 0) std.mem.swap(u32, &r.first, &r.second);
    return if (r.first == greedy) r else error.TopIsNotGreedy;
}

fn widen(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// The two best of bf16 logits, each the first of equal values (torch.argmax's rule); a NaN makes the margin NaN.
pub fn top(logits: []const u16) Top {
    var a: u32 = 0;
    var b: ?u32 = null;
    var nan = false;
    for (logits, 0..) |bits, i| {
        const v = widen(bits);
        if (std.math.isNan(v)) nan = true;
        if (i == 0) continue;
        if (v > widen(logits[a])) {
            b = a;
            a = @intCast(i);
        } else if (b == null or v > widen(logits[b.?])) b = @intCast(i);
    }
    const second = b orelse a;
    const m = widen(logits[a]) - widen(logits[second]);
    return .{ .first = a, .second = second, .margin = if (nan) std.math.nan(f32) else m };
}

fn bf(x: f32) u16 {
    return @intCast(@as(u32, @bitCast(x)) >> 16);
}

test "top: the two best, the first of equals, the lead" {
    const t = top(&.{ bf(1), bf(3), bf(2.5), bf(3) });
    try std.testing.expectEqual(Top{ .first = 1, .second = 3, .margin = 0 }, t);
    try std.testing.expectEqual(Top{ .first = 0, .second = 2, .margin = 0.5 }, top(&.{ bf(4), bf(1), bf(3.5) }));
    try std.testing.expect(std.math.isNan(top(&.{ bf(1), 0x7fc0 }).margin));
}

test "judge: a swap within 0.25 on both sides passes; a wider margin, a third token or a NaN fails" {
    const ref: Top = .{ .first = 10, .second = 20, .margin = 0.125 };
    try std.testing.expectEqual(Step.same, judge(.{ .first = 10, .second = 30, .margin = 3 }, ref, bar));
    try std.testing.expectEqual(Step.swap, judge(.{ .first = 20, .second = 10, .margin = 0.25 }, ref, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 20, .second = 10, .margin = 0.375 }, ref, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 20, .second = 10, .margin = 0 }, .{ .first = 10, .second = 20, .margin = 0.5 }, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 20, .second = 30, .margin = 0 }, ref, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 30, .second = 10, .margin = 0 }, ref, bar));
    try std.testing.expectEqual(Step.fail, judge(.{ .first = 10, .second = 20, .margin = std.math.nan(f32) }, ref, bar));
}

test "reference: an exact tie listed the other way round is the greedy pick's; else the greedy id must lead" {
    try std.testing.expectEqual(Top{ .first = 20, .second = 10, .margin = 0 }, try reference(10, 20, 0, 20));
    try std.testing.expectEqual(Top{ .first = 10, .second = 20, .margin = 0 }, try reference(10, 20, 0, 10));
    try std.testing.expectError(error.TopIsNotGreedy, reference(10, 20, 0.125, 20));
    try std.testing.expectError(error.TopIsNotGreedy, reference(10, 20, 0, 30));
    // as on 1.7B 1089-134686-0000 step 21: the tie decides a swap, not a fail
    const ref = try reference(323, 11, 0, 11);
    try std.testing.expectEqual(Step.swap, judge(.{ .first = 323, .second = 11, .margin = 0.25 }, ref, bar));
}
