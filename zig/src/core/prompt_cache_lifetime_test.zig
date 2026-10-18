//! Restoration borrowers survive invalidation; collection and budget eviction cannot free their state early.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;
const a = std.testing.allocator;
const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

test "an expired restore stays alive until its final borrower reports failure" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    _ = f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{}));
    const one = (try s.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{})).entry.?;
    const two = (try s.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{})).entry.?;
    const held = s.held;
    s.expired(one.saved);
    s.expired(one.saved);
    s.collectExpired();
    try std.testing.expect(s.find(&prompt, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 4), one.at);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    try std.testing.expectEqual(held, s.held);
    s.release(one); // cancellation of one borrower must leave the other's Entry and Saved intact
    try std.testing.expectEqual(@as(usize, 1), f.live);
    s.resumed(two, &prompt, false);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(u64, 0), s.held);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    try std.testing.expectEqual(fresh(&prompt), f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{})));
}

test "invalidating the longest prefix leaves a valid shorter prefix usable" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    _ = f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{}));
    _ = f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 8, &.{}, &.{}, null, &.{}));
    const saved = s.find(&prompt, &.{}, &.{}).?.saved;
    s.expired(saved);
    try std.testing.expectEqual(@as(u32, 4), s.find(&prompt, &.{}, &.{}).?.at);
    const plan = try s.begin(arena.allocator(), &prompt, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), plan.from);
    try std.testing.expectEqual(fresh(&prompt), f.pass(&s, &prompt, plan));
}

test "clear and forget invalidate borrowed entries without freeing them" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    _ = f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{}));
    const entry = (try s.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{})).entry.?;
    s.forget(&prompt, 4, &.{});
    s.clear();
    try std.testing.expect(s.find(&prompt, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    s.release(entry);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(u64, 0), s.held);
}

test "budget pressure refuses a keep rather than evicting an outstanding lookup" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 104);
    defer s.deinit();
    _ = f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{}));
    const entry = (try s.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{})).entry.?;
    const other = [_]u32{ 99, 98, 97, 96, 95 };
    try std.testing.expect(!s.keep(&other, 4, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u32, 4), entry.at);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    s.release(entry);
    f.at = 4;
    f.sum = fresh(other[0..4]);
    try std.testing.expect(s.keep(&other, 4, null, &.{}, &.{}));
}

test "GLM expiry: a failed restore reports whether the state was already expired on arrival" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    const Log = struct {
        calls: usize = 0,
        legacy: usize = 0,
        last: pc.RestoreDetails = undefined,
        fn old(ctx: *anyopaque, _: u32, _: anyerror) void {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            l.legacy += 1;
        }
        fn record(ctx: *anyopaque, report: pc.RestoreDetails) void {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            l.calls += 1;
            l.last = report;
        }
    };
    var log: Log = .{};
    s.restore_log = .{ .ctx = &log, .write = Log.old, .details = Log.record };
    for ([_]bool{ false, true }, 0..) |expired, i| {
        _ = f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{}));
        const entry = (try s.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{})).entry.?;
        const saved = entry.saved;
        if (expired) s.expired(saved);
        s.resumedWithReason(entry, &prompt, error.SnapshotStale);
        // The final release can destroy entry; inspect the captured report and store instead.
        try std.testing.expectEqual(i + 1, log.calls);
        try std.testing.expectEqual(@as(u64, i + 1), s.counts.failed);
        try std.testing.expectEqual(expired, log.last.expired);
        try std.testing.expectEqual(@as(u32, 4), log.last.kept);
        try std.testing.expectEqual(prompt.len, log.last.prompt);
        try std.testing.expectEqual(error.SnapshotStale, log.last.err);
        try std.testing.expect(!s.borrowed(saved));
    }
    try std.testing.expectEqual(@as(usize, 0), log.legacy);
}

test "GLM expiry: actual store borrow query lasts until the last release and excludes expired states" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    _ = f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{}));
    const first = (try s.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{})).entry.?;
    const second = (try s.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{})).entry.?;
    const saved = first.saved;
    try std.testing.expect(first == second and s.borrowed(saved));
    s.release(first);
    try std.testing.expect(s.borrowed(saved));
    s.expired(saved);
    try std.testing.expect(!s.borrowed(saved));
    s.collectExpired();
    try std.testing.expectEqual(@as(u32, 1), second.borrowers);
    s.release(second);
    try std.testing.expect(!s.borrowed(saved));
}
