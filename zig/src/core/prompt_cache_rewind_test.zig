//! Rewind states: kept before the latest user message, so an edited or deleted last turn resumes there.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_test.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;
const Store = pc.Store;

test "editing the latest user resumes the private rewind and gives the fresh result" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 211);
    defer s.deinit();
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var p = try s.beginRewind(a, &first, 8, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 3, 8 }, p.marks);
    try std.testing.expectEqual(fresh(&first), f.pass(&s, &first, p));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    for (s.entries.items) |entry| try std.testing.expect(!entry.shared);
    try std.testing.expectEqual(@as(usize, 0), s.shared_keys.items.len);

    const edited = [_]u32{ 1, 2, 3, 4, 50, 6, 7, 8, 9, 10 };
    p = try s.beginRewind(a, &edited, 8, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 3), p.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, p));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expect(s.find(&edited, &.{}, &.{}).?.at == 8);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 51, 6, 7, 8, 9, 10 }, &.{}, &.{}).?.at == 3);
}

test "one state budget keeps the endpoint before the rewind" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 110);
    defer s.deinit();
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const p = try s.beginRewind(arena.allocator(), &first, 8, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{8}, p.marks);
    _ = f.pass(&s, &first, p);
    try std.testing.expectEqual(@as(u32, 8), s.entries.items[0].at);
}

test "planned rewind floors after lookahead and short user turns bypass min_gap" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .lookahead = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const starts = [_]u32{ 3, 6, 8 };
    const p = try s.beginRewind(arena.allocator(), &first, 9, 5, &.{}, &starts, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 3, 8 }, p.marks);
    try std.testing.expectEqual(fresh(&first), f.pass(&s, &first, p));
    try std.testing.expectEqual(@as(u32, 3), s.find(&.{ 1, 2, 3, 4, 50, 6, 7, 8, 9, 10, 11, 12 }, &starts, &.{}).?.at);
}

test "warm endpoint retains the rewind when two states fit" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .warm = true, .min_prompt = 0 }, 214);
    defer s.deinit();
    const warm = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const p = try s.beginRewind(a, &warm, warm.len, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{4}, p.marks);
    _ = f.pass(&s, &warm, p);
    try std.testing.expect(s.keep(&warm, warm.len, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    const edited = [_]u32{ 1, 2, 3, 4, 50, 6, 7, 8, 9, 10, 11 };
    const resumed = try s.beginRewind(a, &edited, 10, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), resumed.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, resumed));
}

test "a grid family keeps its rewind on the grid, and an edit resumes there equal to a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .planned = true, .grid = 4, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const first = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const p = try s.beginRewind(a, &first, 10, 7, &.{}, &.{}, null, &.{}); // no request starts: the grid's
    try std.testing.expectEqualSlices(u32, &.{ 4, 8 }, p.marks);
    try std.testing.expectEqual(fresh(&first), f.pass(&s, &first, p));
    const edited = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 50, 9, 10, 11, 12 };
    const q = try s.beginRewind(a, &edited, 10, 7, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), q.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, q));
    for (s.entries.items) |e| try std.testing.expectEqual(@as(u32, 0), e.at % 4);
}

test "a state kept past the pass's marks evicts the rewind last instead of failing the keep" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 215); // bytes 100 + at
    defer s.deinit();
    const turn = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const p = try s.beginRewind(arena.allocator(), &turn, 8, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 3, 8 }, p.marks);
    _ = f.pass(&s, &turn, p);
    const replied = turn ++ [_]u32{ 11, 12, 13, 14 }; // a reply's state, as the serial host keeps it after decode
    f.at = 13;
    try std.testing.expect(s.keep(&replied, 13, null, &.{}, &.{}));
    try std.testing.expect(s.held <= s.budget);
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u32, 13), s.entries.items[0].at);
}

test "a first turn's rewind on the system cut stays a shared cut" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .planned = true, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer s.deinit();
    const turn = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // a system block, the user's text from 4, the generation prompt at 7
    const starts = [_]u32{ 3, 7 }; // the server's cut before the conversation's own text, and the generation prompt
    const p = try s.beginRewind(arena.allocator(), &turn, 7, 4, &.{4}, &starts, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 3, 7 }, p.marks);
    _ = f.pass(&s, &turn, p);
    var cut: ?*pc.Entry = null;
    for (s.entries.items) |e| if (e.at == 3) {
        cut = e;
    };
    try std.testing.expect(cut.?.shared);
}

test "a history state past the budget is still counted refused when the rewind is kept" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 105); // the history at 8 needs 108
    defer s.deinit();
    const turn = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const p = try s.beginRewind(arena.allocator(), &turn, 8, 4, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{3}, p.marks);
    try std.testing.expectEqual(@as(u64, 1), s.counts.refused);
    try std.testing.expectEqual(fresh(&turn), f.pass(&s, &turn, p));
}

test "a rewind under min_gap past the resume point keeps no copy: the resumed state serves the edit and goes last" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .warm = true, .min_prompt = 0, .min_gap = 4 }, 222); // bytes 100 + at
    defer s.deinit();
    const before = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = f.pass(&s, &before, try s.begin(a, &before, 6, &.{}, &.{}, null, &.{})); // the state a turn resumes, as a reply's
    const turn = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // the user's text from 8, two tokens past it
    const p = try s.beginRewind(a, &turn, 11, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 6), p.from);
    try std.testing.expectEqualSlices(u32, &.{11}, p.marks);
    try std.testing.expectEqual(fresh(&turn), f.pass(&s, &turn, p));
    const replied = turn ++ [_]u32{ 13, 14 }; // a reply's state: the history's goes, the resumed state stays
    f.at = 13;
    try std.testing.expect(s.keep(&replied, 13, null, &.{}, &.{}));
    try std.testing.expect(s.held <= s.budget);
    const edited = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 50, 10, 11, 12 };
    const q = try s.beginRewind(a, &edited, 11, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 6), q.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, q));
}

test "the warm pass after a turn keeps that turn's rewind state, though it resumes later: the edit still resumes there" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .warm = true, .min_prompt = 0, .min_gap = 4 }, 230); // two states (bytes 100 + at)
    defer s.deinit();
    const before = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = f.pass(&s, &before, try s.begin(a, &before, 6, &.{}, &.{}, null, &.{}));
    const turn = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }; // resumes at 6, the user's text from 8
    _ = f.pass(&s, &turn, try s.beginRewind(a, &turn, 11, 8, &.{}, &.{}, null, &.{}));
    const replied = turn ++ [_]u32{ 13, 14 };
    f.at = 13;
    try std.testing.expect(s.keep(&replied, 13, null, &.{}, &.{})); // the reply's state
    const warm = replied ++ [_]u32{15}; // the next turn's opening, prefilled in the background from the reply's state
    const w = try s.beginRewind(a, &warm, warm.len, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 13), w.from);
    f.at = 15;
    try std.testing.expect(s.keep(&warm, 15, null, &.{}, &.{}));
    try std.testing.expect(s.held <= s.budget);
    const edited = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 50, 10, 11, 12 };
    const q = try s.beginRewind(a, &edited, 11, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 6), q.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, q));
}
