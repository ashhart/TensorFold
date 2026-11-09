//! Planned endpoint room includes the current restore, without spending it on optional copies first.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const reserve = @import("prompt_cache_reserve.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const a = std.testing.allocator;
fn idle(_: *anyopaque, _: *const pc.Entry) bool {
    return false;
}

test "queued cache: a shared current restore funds its next endpoint after the borrow ends" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 116);
    defer s.deinit();
    const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    _ = f.pass(&s, words[0..12], .{});
    try std.testing.expect(s.keep(words[0..12], 12, null, &.{}, &.{}));
    s.entries.items[0].shared = true;
    s.reservation_guard = .{ .ctx = &f, .needed = idle };
    const l = try s.lookup(a, &words, 16, &.{}, &.{}, &.{});
    defer a.free(l.marks);
    try std.testing.expectEqualSlices(u32, &.{16}, l.marks);
    const from = l.entry.?;
    s.resumed(from, &words, true);
    _ = f.pass(&s, words[0..16], .{});
    try std.testing.expect(s.keep(&words, 16, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u32, 16), s.entries.items[0].at);
    try std.testing.expectEqual(@as(u64, 116), s.held);
}

test "queued cache: endpoint has priority over a cheaper optional rewind with exactly one slot" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .warm = .on, .lookahead = 1 }, 119);
    defer s.deinit();
    const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    _ = f.pass(&s, words[0..12], .{});
    try std.testing.expect(s.keep(&words, 12, null, &.{}, &.{}));
    s.most = 1;
    s.reservation_guard = .{ .ctx = &f, .needed = idle };
    const primary = reserve.endpointFor(&s, &words, 12, 20, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 19), primary);
    const marks = try a.dupe(u32, &.{ 4, 19 });
    const fitted = try reserve.fitting(&s, a, &words, marks, &.{}, s.entries.items[0], 20, primary);
    defer a.free(fitted);
    try std.testing.expectEqualSlices(u32, &.{19}, fitted);
}
