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

test "a failed restore reports whether the state was already expired on arrival" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    _ = f.pass(&s, &prompt, try s.begin(arena.allocator(), &prompt, 4, &.{}, &.{}, null, &.{}));
    const live = (try s.lookup(arena.allocator(), &prompt, 4, &.{}, &.{}, &.{})).entry.?;
    s.resumed(live, &prompt, false); // refused while still valid: the backend, not a concurrent eviction
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    try std.testing.expect(live.expired);
    const other = [_]u32{ 11, 12, 13, 14, 15, 16 };
    _ = f.pass(&s, &other, try s.begin(arena.allocator(), &other, 4, &.{}, &.{}, null, &.{}));
    const dead = (try s.lookup(arena.allocator(), &other, 4, &.{}, &.{}, &.{})).entry.?;
    s.expired(dead.saved); // a concurrent admission invalidated it between lookup and restore
    s.resumed(dead, &other, false);
    try std.testing.expectEqual(@as(u64, 2), s.counts.failed);
}
