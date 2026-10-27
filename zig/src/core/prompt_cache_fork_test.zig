//! A re-sent document with a new question: a miss learns its fork, and the next question resumes it.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const fake = @import("prompt_cache_test.zig");
const Fake = fake.Fake;
const fresh = fake.fresh;
const Store = pc.Store;

const sys = [_]u32{ 1, 2, 3, 4 };
const doc = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
const q1 = [_]u32{ 91, 92 };
const q2 = [_]u32{ 93, 94 };
const q3 = [_]u32{ 95, 96 };
const tail = [_]u32{ 70, 71 }; // the template's generation prompt after the user message
const p1 = sys ++ doc ++ q1 ++ tail;
const p2 = sys ++ doc ++ q2 ++ tail;
const p3 = sys ++ doc ++ q3 ++ tail;
const history = sys.len + doc.len + q1.len; // the end of the user message: 22
const fork = sys.len + doc.len; // where the questions part: 20

test "a re-sent document learns its fork on a miss, and the next question resumes it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    // The question sits 3 tokens before the history's mark, nearer than min_gap: the fork keeps anyway.
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .fork = true }, 1 << 20);
    defer s.deinit();

    var p = try s.begin(a, &p1, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqualSlices(u32, &.{history}, p.marks);
    try std.testing.expectEqual(fresh(&p1), f.pass(&s, &p1, p));

    // Still a miss (nothing is kept at the fork yet), and the pass keeps the state the next question resumes.
    p = try s.begin(a, &p2, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqualSlices(u32, &.{ fork - 1, history }, p.marks);
    try std.testing.expectEqual(fresh(&p2), f.pass(&s, &p2, p));

    p = try s.begin(a, &p3, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, fork - 1), p.from);
    try std.testing.expectEqual(fresh(&p3), f.pass(&s, &p3, p));
    try std.testing.expectEqual(@as(u64, 1), s.counts.hits);
}

test "without the fork rule a re-sent document keeps no state to resume" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer s.deinit();

    var p = try s.begin(a, &p1, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(fresh(&p1), f.pass(&s, &p1, p));
    p = try s.begin(a, &p2, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{history}, p.marks);
    try std.testing.expectEqual(fresh(&p2), f.pass(&s, &p2, p));
    p = try s.begin(a, &p3, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(@as(u64, 0), s.counts.hits);
}

test "a follow-up that resumes the rewind state keeps its fork, near the history as it is" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .fork = true }, 1 << 20);
    defer s.deinit();
    const rewind = sys.len; // the start of the latest user message, as server/prompt.zig names it

    var p = try s.beginRewind(a, &p1, history, rewind, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ rewind - 1, history }, p.marks);
    try std.testing.expectEqual(fresh(&p1), f.pass(&s, &p1, p));

    // It resumes the rewind state and parts at the question: that fork keeps, though 3 tokens from the history.
    p = try s.beginRewind(a, &p2, history, rewind, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, rewind - 1), p.from);
    try std.testing.expectEqualSlices(u32, &.{ fork - 1, history }, p.marks);
    try std.testing.expectEqual(fresh(&p2), f.pass(&s, &p2, p));

    p = try s.beginRewind(a, &p3, history, rewind, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, fork - 1), p.from);
    try std.testing.expectEqual(fresh(&p3), f.pass(&s, &p3, p));
}

test "a planned family keeps the fork on its grid, however near the history's mark" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .planned = true, .grid = 4, .min_prompt = 0, .fork = true }, 1 << 20);
    defer s.deinit();

    var p = try s.begin(a, &p1, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{20}, p.marks); // the history's 22 on the grid, keyed with the question's token
    try std.testing.expectEqual(fresh(&p1), f.pass(&s, &p1, p));

    // The fork's 19 snaps to 16, 4 from the history's 20 and nearer than min_gap: compared snapped, it keeps.
    p = try s.begin(a, &p2, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqualSlices(u32, &.{ 16, 20 }, p.marks);
    try std.testing.expectEqual(fresh(&p2), f.pass(&s, &p2, p));

    p = try s.begin(a, &p3, history, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 16), p.from);
    try std.testing.expectEqual(fresh(&p3), f.pass(&s, &p3, p));
}

test "a miss keeps no fork where the prompts part before half the history, or share nothing" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .fork = true }, 1 << 20);
    defer s.deinit();
    _ = f.pass(&s, &p1, try s.begin(a, &p1, history, &.{}, &.{}, null, &.{}));

    const early = sys[0..2].* ++ [_]u32{ 8, 8 } ++ doc ++ q1 ++ tail; // parts at 2, under half the history
    const other: [p1.len]u32 = @splat(7); // shares no token
    for ([_][]const u32{ &early, &other }) |prompt| {
        const p = try s.begin(a, prompt, history, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(u32, 0), p.from);
        try std.testing.expectEqualSlices(u32, &.{history}, p.marks);
        try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
    }
}

test {
    _ = @import("prompt_cache_fork.zig");
}
