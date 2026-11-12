//! Guarded admission preserves newest states, shared restores and the deepest queued match.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const a = std.testing.allocator;

fn none(_: *anyopaque, _: *const pc.Entry) bool {
    return false;
}

const Queued = struct {
    key: []const u32,
    fn needed(ptr: *anyopaque, e: *const pc.Entry) bool {
        const q: *const Queued = @ptrCast(@alignCast(ptr));
        return e.tokens.len > 0 and e.at < q.key.len and e.tokens.len <= q.key.len and std.mem.eql(u32, e.tokens, q.key[0..e.tokens.len]);
    }
};

fn put(f: *Fake, s: *pc.Store, words: []const u32, born: u32, at: u32) !void {
    _ = f.pass(s, words[0..at], .{});
    try std.testing.expect(s.keepBorn(words, born, at, null, &.{}, &.{}));
}

// A: the guarded reserve should free a superseded copy before an idle conversation's newest state.
test "queued cache: guarded reserve evicts superseded before another newest" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 108 + 108 + 112);
    defer s.deinit();
    var b: [20]u32 = undefined;
    var d: [20]u32 = undefined;
    var c: [20]u32 = undefined;
    for (0..20) |i| {
        b[i] = @intCast(100 + i);
        d[i] = @intCast(200 + i);
        c[i] = @intCast(300 + i);
    }
    try put(&f, &s, b[0..8], 8, 8); // B's newest, least recently used
    try put(&f, &s, d[0..8], 8, 8); // D's old turn
    try put(&f, &s, d[0..12], 12, 12); // D's newest supersedes it
    s.reservation_guard = .{ .ctx = &f, .needed = none };
    defer s.reservation_guard = null;
    const l = try s.lookup(a, &c, 4, &.{}, &.{}, &.{});
    defer a.free(l.marks);
    try std.testing.expectEqualSlices(u32, &.{4}, l.marks);
    try std.testing.expect(s.find(&b, &.{}, &.{}) != null); // B's newest stays
}

// B: a shared `from` is counted as room, but the keep evicts the oldest entry, a protected queued state.
test "queued cache: shared from room does not evict a protected queued state at keep" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 112 + 112 + 4);
    defer s.deinit();
    var q: [20]u32 = undefined;
    var c: [20]u32 = undefined;
    for (0..20) |i| {
        q[i] = @intCast(500 + i);
        c[i] = @intCast(1 + i);
    }
    try put(&f, &s, q[0..12], 12, 12); // the queued follow-up's state, older
    try put(&f, &s, c[0..12], 12, 12); // a shared system cut
    for (s.entries.items) |e| if (e.tokens[0] == 1) {
        e.shared = true;
    };
    var queued: Queued = .{ .key = &q };
    s.reservation_guard = .{ .ctx = &queued, .needed = Queued.needed };
    const l = try s.lookup(a, &c, 16, &.{}, &.{}, &.{});
    s.reservation_guard = null;
    defer a.free(l.marks);
    const from = l.entry.?;
    try std.testing.expectEqual(@as(u32, 12), from.at);
    s.resumed(from, &c, true);
    _ = f.pass(&s, c[0..16], .{ .from = 12, .marks = l.marks });
    try std.testing.expect(s.find(&q, &.{}, &.{}) != null); // the queued conversation's state survives
}

// C: protection covers a queued job's shorter, superseded copy, which it will not resume.
test "queued cache: a queued job's superseded copy can make room" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 108 + 112);
    defer s.deinit();
    var q: [20]u32 = undefined;
    var c: [20]u32 = undefined;
    for (0..20) |i| {
        q[i] = @intCast(500 + i);
        c[i] = @intCast(300 + i);
    }
    try put(&f, &s, q[0..8], 8, 8);
    try put(&f, &s, q[0..12], 12, 12); // the state the queued follow-up resumes
    var queued: Queued = .{ .key = &q };
    s.reservation_guard = .{ .ctx = &queued, .needed = Queued.needed };
    defer s.reservation_guard = null;
    const l = try s.lookup(a, &c, 4, &.{}, &.{}, &.{});
    defer a.free(l.marks);
    try std.testing.expectEqualSlices(u32, &.{4}, l.marks); // the new conversation keeps its endpoint
    try std.testing.expectEqual(@as(u32, 12), s.find(&q, &.{}, &.{}).?.at);
}

test "queued cache: unguarded reserve evicts superseded before another newest" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 108 + 108 + 112);
    defer s.deinit();
    var b: [20]u32 = undefined;
    var d: [20]u32 = undefined;
    var c: [20]u32 = undefined;
    for (0..20) |i| {
        b[i] = @intCast(100 + i);
        d[i] = @intCast(200 + i);
        c[i] = @intCast(300 + i);
    }
    try put(&f, &s, b[0..8], 8, 8); // B's newest, least recently used
    try put(&f, &s, d[0..8], 8, 8); // D's old turn
    try put(&f, &s, d[0..12], 12, 12); // D's newest supersedes it
    defer s.reservation_guard = null;
    const l = try s.lookup(a, &c, 4, &.{}, &.{}, &.{});
    defer a.free(l.marks);
    try std.testing.expectEqualSlices(u32, &.{4}, l.marks);
    try std.testing.expect(s.find(&b, &.{}, &.{}) != null); // B's newest stays
}

test "queued cache: unguarded a queued job's superseded copy can make room" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 108 + 112);
    defer s.deinit();
    var q: [20]u32 = undefined;
    var c: [20]u32 = undefined;
    for (0..20) |i| {
        q[i] = @intCast(500 + i);
        c[i] = @intCast(300 + i);
    }
    try put(&f, &s, q[0..8], 8, 8);
    try put(&f, &s, q[0..12], 12, 12); // the state the queued follow-up resumes
    defer s.reservation_guard = null;
    const l = try s.lookup(a, &c, 4, &.{}, &.{}, &.{});
    defer a.free(l.marks);
    try std.testing.expectEqualSlices(u32, &.{4}, l.marks); // the new conversation keeps its endpoint
    try std.testing.expectEqual(@as(u32, 12), s.find(&q, &.{}, &.{}).?.at);
}
