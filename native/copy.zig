//! Eight-token suffix matches propose verbatim continuations. Every token still
//! goes through the same target verification and sampling as a DFlash proposal.
const std = @import("std");
const Proposal = @import("drafter.zig").Proposal;
pub fn propose(context: []const i32, budget: usize) Proposal {
    var result = Proposal{};
    if (context.len < 17 or budget == 0) return result;
    const tail = context[context.len - 8 ..];
    var best: usize = 0;
    var end: ?usize = null;
    for (8..context.len - 7) |at| {
        if (!std.mem.eql(i32, context[at - 8 .. at], tail)) continue;
        var matched: usize = 8;
        while (matched < at and matched < context.len and context[at - matched - 1] == context[context.len - matched - 1]) matched += 1;
        if (matched >= best) {
            best = matched;
            end = at;
        }
    }
    if (end) |at| {
        result.len = @min(@min(budget, 31), context.len - at);
        for (0..result.len) |i| {
            result.tokens[i] = context[at + i];
            result.parents[i] = @as(i32, @intCast(i)) - 1;
        }
    }
    return result;
}
test "copy continuation requires eight matches and is a bounded chain" {
    const ctx = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 40, 41, 42, 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    const p = propose(&ctx, 3);
    try std.testing.expectEqual(@as(usize, 3), p.len);
    try std.testing.expectEqualSlices(i32, &.{ 40, 41, 42 }, p.tokens[0..3]);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, p.parents[0..3]);
    try std.testing.expectEqual(@as(usize, 0), propose(ctx[0..10], 31).len);
}
