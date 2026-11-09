//! Planned endpoint room includes the current restore, without spending it on optional copies first.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const reserve = @import("prompt_cache_reserve.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const a = std.testing.allocator;
fn idle(_: *anyopaque, _: *const pc.Entry) bool {
    return false;
}

test "queued cache: a shared current restore cannot fund its next endpoint" {
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
    try std.testing.expectEqualSlices(u32, &.{}, l.marks);
    const from = l.entry.?;
    s.resumed(from, &words, true);
    _ = f.pass(&s, words[0..16], .{ .from = 12, .marks = l.marks });
    try std.testing.expectEqual(@as(u32, 12), s.entries.items[0].at);
    try std.testing.expectEqual(@as(u64, 112), s.held);
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
    const fitted = try reserve.fitting(&s, a, &words, marks, &.{}, &.{}, s.entries.items[0], 20, primary);
    defer a.free(fitted);
    try std.testing.expectEqualSlices(u32, &.{19}, fitted);
}

test "queued cache: branched rewind and multiply borrowed restores cannot promise their room" {
    for (0..3) |kind| {
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 116);
        defer s.deinit();
        const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
        _ = f.pass(&s, words[0..12], .{});
        try std.testing.expect(s.keep(words[0..12], 12, null, &.{}, &.{}));
        const e = s.entries.items[0];
        switch (kind) {
            0 => e.branched = true,
            1 => s.active_rewind_at = 12,
            else => e.borrowers = 1,
        }
        defer e.borrowers = 0;
        s.reservation_guard = .{ .ctx = &f, .needed = idle };
        const marks = try reserve.fitting(&s, a, &words, try a.dupe(u32, &.{16}), &.{}, &.{}, e, 20, 16);
        defer a.free(marks);
        try std.testing.expectEqual(@as(usize, 0), marks.len);
        try std.testing.expectEqual(@as(u64, 1), s.counts.trimmed);
    }
}

test "queued cache: debug divergence names both token ids and preserves prompt-end resume" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    const original = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    _ = f.pass(&s, original[0..8], .{});
    try std.testing.expect(s.keepBorn(&original, 8, 8, null, &.{}, &.{}));
    _ = f.pass(&s, original[0..14], .{});
    try std.testing.expect(s.keepBorn(&original, 8, 14, null, &.{}, &.{}));
    var changed = original;
    changed[12] = 99;
    var heard: @import("prompt_cache_note.zig").Heard = .{};
    @import("prompt_cache_note.zig").sink = &heard;
    defer @import("prompt_cache_note.zig").sink = null;
    const l = try s.lookup(a, &changed, 14, &.{}, &.{}, &.{});
    defer a.free(l.marks);
    defer if (l.entry) |e| s.release(e);
    try std.testing.expectEqual(@as(u32, 8), l.entry.?.at);
    try std.testing.expect(std.mem.indexOf(u8, heard.text(), "diverges at 12: kept token 13, request token 99; resuming at 8") != null);
    try std.testing.expect(@import("prompt_cache_divergence.zig").find(&s, &original, &.{}, l.entry) == null);
    changed[1] = 99;
    try std.testing.expect(@import("prompt_cache_divergence.zig").find(&s, &changed, &.{}, null) == null);
}

test "queued cache: begin reserves the warm lookahead endpoint after releasing its restore" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0, .min_gap = 0, .warm = .on, .lookahead = 1 }, 119);
    defer s.deinit();
    const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    _ = f.pass(&s, words[0..12], .{});
    try std.testing.expect(s.keep(&words, 12, null, &.{}, &.{}));
    s.reservation_guard = .{ .ctx = &f, .needed = idle };
    const plan = try s.begin(a, &words, 20, &.{}, &.{}, null, &.{});
    defer a.free(plan.marks);
    try std.testing.expectEqual(@as(u32, 12), plan.from);
    try std.testing.expectEqualSlices(u32, &.{19}, plan.marks);
    try std.testing.expectEqual(@as(u64, 0), s.held);
}
