//! Regression: budget under one grid step, then a piece_end rounding up past it, ends the round (no overflow).
const std = @import("std");
const sched = @import("sched.zig");

fn seqOf(key: usize, n: u64, done: u64, t: f64) sched.Seq {
    return .{ .key = key, .n = n, .submitted = t, .done = done, .save_at = null };
}

fn planOnce(p: *sched.Planner) !struct { pieces: []sched.Piece, finals: []usize } {
    var pieces: std.ArrayList(sched.Piece) = .empty;
    var finals: std.ArrayList(usize) = .empty;
    try p.plan(null, &pieces, &finals);
    return .{ .pieces = try pieces.toOwnedSlice(p.gpa), .finals = try finals.toOwnedSlice(p.gpa) };
}

test "a prompt mid-prefill and two arrivals leave the round 5 rows: the next piece is one grid step, the rest skip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]u64{ 4096, 2048, 1024 }) |rows| {
        var p = sched.Planner.init(arena.allocator(), .{ .rows = rows, .short = 512, .sessions = false });
        // A: mid-prefill, 2040 left; B: a short arrival (299 left); C: a long arrival
        try p.seqs.append(p.gpa, seqOf(1, 4096 + 2041, 4096, 0));
        try p.seqs.append(p.gpa, seqOf(2, 300, 0, 2));
        try p.seqs.append(p.gpa, seqOf(3, 150_000, 0, 2.8));
        const r = try planOnce(&p); // B whole, A to its last grid point within the budget, then 5 (or fewer) rows left
        var taken: u64 = 0;
        for (r.pieces) |pc| taken += pc.end - pc.start;
        // the overshoot is at most one grid step less one row
        try std.testing.expect(taken <= rows + sched.grid - 1);
        try std.testing.expectEqual(@as(u64, 2), r.pieces[0].key);
    }
}

test "the smallest repro: 2040 of 2048 rows to one prompt, a second prompt's minimum grid step (16) exceeds the 8 left" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var p = sched.Planner.init(arena.allocator(), .{ .rows = 2048, .sessions = false });
    try p.seqs.append(p.gpa, seqOf(1, 2041, 0, 0));
    try p.seqs.append(p.gpa, seqOf(2, 9000, 0, 1));
    const r = try planOnce(&p);
    try std.testing.expectEqual(@as(usize, 2), r.pieces.len);
    try std.testing.expectEqual(@as(u64, 16), r.pieces[1].end); // budget 8 -> -8, then skips
}

test "edge cases: n 1 and cached n - 1 are finals at once; nothing is planned for them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var p = sched.Planner.init(arena.allocator(), .{ .rows = 2048, .sessions = false });
    try p.seqs.append(p.gpa, seqOf(1, 1, 0, 0));
    try p.seqs.append(p.gpa, seqOf(2, 5000, 4999, 1)); // the longest strict-prefix hit
    const r = try planOnce(&p);
    try std.testing.expectEqual(@as(usize, 0), r.pieces.len);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, r.finals);
}
