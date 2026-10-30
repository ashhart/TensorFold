//! The batched MTP head's plan: its requests fit the admitted capacity, and its lanes' groups fit the piece.
const std = @import("std");
const plan = @import("cuda_mtp_plan.zig");

test "Flash Next head plan: a 64-row chunk with two one-row heads and four 16-row paths fits" {
    const admitted = 64;
    const rows = [_]usize{ 1, 1, 16, 16, 16, 16 };
    const total: usize = 66;
    const p = plan.plan(admitted, total);
    // the shared scratch and the scores are priced at startup: the head's requests stay within them
    try std.testing.expect(p.pass_rows <= admitted);
    try std.testing.expect(p.score_rows <= admitted);
    try std.testing.expectEqual(@as(usize, 32), p.piece);
    // the lanes' groups: each group's rows fit the piece, and every lane is in one
    var groups: [8]usize = undefined;
    const n = plan.groups(&rows, p.piece, groups[0..]);
    try std.testing.expectEqual(@as(usize, 3), n);
    var at: usize = 0;
    for (groups[0..n]) |g| {
        var sum: usize = 0;
        for (rows[at..][0..g]) |r| sum += r;
        try std.testing.expect(sum <= p.piece);
        at += g;
    }
    try std.testing.expectEqual(rows.len, at);
}

test "Flash Next head plan: one lane's rows never exceed the piece" {
    const rows = [_]usize{16};
    var groups: [4]usize = undefined;
    const p = plan.plan(64, 16);
    try std.testing.expectEqual(@as(usize, 1), plan.groups(&rows, p.piece, groups[0..]));
    try std.testing.expectEqual(@as(usize, 1), groups[0]);
}
