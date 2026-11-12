//! Promised room and protected states prevent futile eviction while fitting spare storage remains reusable.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const a = std.testing.allocator;

fn none(_: *anyopaque, _: *const pc.Entry) bool {
    return false;
}

fn put(f: *Fake, s: *pc.Store, words: []const u32, at: u32) !void {
    _ = f.pass(s, words[0..at], .{});
    try std.testing.expect(s.keepBorn(words, at, at, null, &.{}, &.{}));
}

fn fill(base: u32) [20]u32 {
    var w: [20]u32 = undefined;
    for (&w, 0..) |*x, i| x.* = base + @as(u32, @intCast(i));
    return w;
}

// R1: other passes' promises leave less than this state's bytes: refuse before evicting anything.
test "queued cache: a keep past the promised limit refuses without evicting" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 400);
    defer s.deinit();
    const b = fill(100);
    const d = fill(200);
    const c = fill(300);
    try put(&f, &s, &b, 8);
    try put(&f, &s, &d, 8);
    s.reservation_guard = .{ .ctx = &f, .needed = none, .promised = .{ .bytes = 350 } };
    defer s.reservation_guard = null;
    _ = f.pass(&s, c[0..12], .{});
    try std.testing.expect(!s.keepBorn(&c, 12, 12, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len); // B's and D's only states stay
}

// R1b: the same with promised slots.
test "queued cache: a keep with every slot promised refuses without evicting" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    s.most = 2;
    const b = fill(100);
    const d = fill(200);
    const c = fill(300);
    try put(&f, &s, &b, 8);
    try put(&f, &s, &d, 8);
    s.reservation_guard = .{ .ctx = &f, .needed = none, .promised = .{ .slots = 2 } };
    defer s.reservation_guard = null;
    _ = f.pass(&s, c[0..12], .{});
    try std.testing.expect(!s.keepBorn(&c, 12, 12, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
}

// R2: promises exceed the free bytes; an admission whose endpoint cannot fit evicts nothing for it.
test "queued cache: an admission evicts only when its endpoint then fits" {
    for ([_]u64{ 250, 150 }) |promised| {
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 108 + 108 + 100);
        defer s.deinit();
        const b = fill(100);
        const d = fill(200);
        const c = fill(300);
        try put(&f, &s, &b, 8);
        try put(&f, &s, &d, 8);
        s.reservation_guard = .{ .ctx = &f, .needed = none, .promised = .{ .bytes = promised } };
        defer s.reservation_guard = null;
        const l = try s.lookup(a, &c, 4, &.{}, &.{}, &.{});
        defer a.free(l.marks);
        // either the endpoint was reserved, or nothing was evicted for it
        try std.testing.expect(l.marks.len == 1 or s.entries.items.len == 2);
    }
}

// R3: a one-byte outside promise must not stop a keep from reusing spare storage that fits it.
test "queued cache: an outside promise still lets a keep reuse fitting spare storage" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.pooled(), .{ .min_prompt = 0 }, 300);
    defer s.deinit();
    const x = fill(100);
    const y = fill(200);
    try put(&f, &s, &x, 12);
    f.spare_bytes = 120; // a dropped state's storage, inside the budget beside x's 102
    s.reservation_guard = .{ .ctx = &f, .needed = none, .promised = .{ .bytes = 1 } };
    defer s.reservation_guard = null;
    _ = f.pass(&s, y[0..16], .{});
    try std.testing.expect(s.keepBorn(&y, 16, 16, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 208), s.held);
    try std.testing.expectEqual(@as(u64, 14), f.spare_bytes); // the spare took it: nothing freed, nothing new
}

// R3 control: with no outside promise the same keep reuses the spare.
test "queued cache: without a promise the keep reuses fitting spare storage" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.pooled(), .{ .min_prompt = 0 }, 300);
    defer s.deinit();
    const x = fill(100);
    const y = fill(200);
    try put(&f, &s, &x, 12);
    f.spare_bytes = 120;
    _ = f.pass(&s, y[0..16], .{});
    try std.testing.expect(s.keepBorn(&y, 16, 16, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(u64, 14), f.spare_bytes);
}

const Protect = struct {
    fn needed(_: *anyopaque, e: *const pc.Entry) bool {
        return e.tokens[0] >= 500;
    }
};

// R1c: queued states fill the store; a guarded keep that cannot fit evicts no idle conversation for nothing.
test "queued cache: a guarded keep that cannot fit beside protected states evicts nothing" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 330);
    defer s.deinit();
    const q1 = fill(500);
    const q2 = fill(600);
    const b = fill(100);
    const c = fill(300);
    try put(&f, &s, &b, 4);
    try put(&f, &s, &q1, 12);
    try put(&f, &s, &q2, 12);
    s.reservation_guard = .{ .ctx = &f, .needed = Protect.needed, .exact = true };
    defer s.reservation_guard = null;
    _ = f.pass(&s, c[0..16], .{});
    try std.testing.expect(!s.keepBorn(&c, 16, 16, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len); // B's only state stays
}

test "queued cache: admission refuses a promised slot deficit without evicting" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    s.most = 2;
    const b = fill(100);
    const d = fill(200);
    const c = fill(300);
    try put(&f, &s, &b, 8);
    try put(&f, &s, &d, 8);
    s.reservation_guard = .{ .ctx = &f, .needed = none, .promised = .{ .slots = 2 } };
    defer s.reservation_guard = null;
    const l = try s.lookup(a, &c, 4, &.{}, &.{}, &.{});
    defer a.free(l.marks);
    try std.testing.expectEqual(@as(usize, 0), l.marks.len);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
}
