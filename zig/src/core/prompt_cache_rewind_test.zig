const std = @import("std");
const Store = @import("prompt_cache.zig").Store;
const fixture = @import("prompt_cache_test.zig");
const Fake = fixture.Fake;
const fresh = fixture.fresh;

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

test "a planned family's grid keeps a private rewind and resumed edits equal a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .planned = true, .grid = 4, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const p1 = try s.beginRewind(a, &t1, 9, 6, &.{6}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 4, 8 }, p1.marks);
    _ = f.pass(&s, &t1, p1);
    try std.testing.expectEqual(@as(usize, 0), s.shared_keys.items.len);
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 66, 7, 11, 12, 13, 14 };
    try std.testing.expect(s.find(&t2, &.{6}, &.{}) == null); // a request's own starts win over the grid
    const p2 = try s.beginRewind(a, &t2, 9, 6, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), p2.from);
    try std.testing.expectEqualSlices(u32, &.{8}, p2.marks);
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p2));
}
