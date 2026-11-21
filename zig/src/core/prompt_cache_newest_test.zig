//! Returning conversations retain their newest state before redundant copies, including equal-born reply ends.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const a = std.testing.allocator;

fn put(f: *Fake, s: *pc.Store, words: []const u32, born: u32, at: u32) !void {
    _ = f.pass(s, words[0..at], .{});
    try std.testing.expect(s.keepBorn(words, born, at, null, &.{}, &.{}));
}

test "newest cache: eight cyclic conversations keep every reply end above the old three-copy fit" {
    const n = 8;
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .warm = .on, .min_prompt = 0 }, n * 176);
    defer s.deinit();
    var words: [n][80]u32 = undefined;
    for (&words, 1..) |*conversation, id| for (conversation, 0..) |*word, i| {
        word.* = @intCast(id * 1000 + i);
    };
    for (0..6) |turn| for (&words) |*conversation| {
        const len: u32 = @intCast(24 + turn * 8);
        if (turn > 0) {
            const e = s.find(conversation[0..len], &.{}, &.{}) orelse return error.NewestReplyEvicted;
            try std.testing.expectEqual(len - 4, e.at);
            const state: *Fake.State = @ptrCast(@alignCast(e.saved));
            try std.testing.expectEqual(e.at, state.at);
            try std.testing.expectEqual(@import("prompt_cache_test.zig").fresh(conversation[0..e.at]), state.sum);
            s.resumed(e, conversation[0..len], true);
        }
        try put(&f, &s, conversation[0..len], len, len - 8);
        try put(&f, &s, conversation[0..len], len, len);
        try put(&f, &s, conversation[0 .. len + 4], len, len + 4);
        try std.testing.expect(s.held <= s.budget);
        try std.testing.expectEqual(s.entries.items.len, f.live);
    };
    try std.testing.expect(s.counts.evicted > n);
}

test "newest cache: equal-born reply links evict its redundant endpoint before an older conversation" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .warm = .on, .min_prompt = 0 }, 10000);
    defer s.deinit();
    const other = [_]u32{ 90, 91, 92, 93, 94, 95, 96, 97 };
    const own = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    try put(&f, &s, &other, 8, 7);
    try put(&f, &s, own[0..8], 8, 7);
    try put(&f, &s, own[0..12], 8, 12);
    const victim = s.entries.items[s.victimBut(null, &.{}, 0).?];
    try std.testing.expectEqualSlices(u32, own[0..7], victim.tokens);
    s.links_lost = true;
    try std.testing.expectEqualSlices(u32, own[0..7], s.entries.items[s.victimBut(null, &.{}, 0).?].tokens);
}

test "newest cache: a reply replaces its own equal-born endpoint when only two newest states fit" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .warm = .on, .min_prompt = 0 }, 220);
    defer s.deinit();
    const other = [_]u32{ 90, 91, 92, 93, 94, 95, 96, 97 };
    const own = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    try put(&f, &s, &other, 8, 7);
    try put(&f, &s, own[0..8], 8, 7);
    try put(&f, &s, own[0..12], 8, 12);
    try std.testing.expect(s.find(&other, &.{}, &.{}) != null);
    try std.testing.expectEqual(@as(u32, 12), s.find(&own, &.{}, &.{}).?.at);
}

test "newest cache: an identical resend does not pin an endpoint ahead of another conversation's newest" {
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .warm = .on, .min_prompt = 0 }, 330);
    defer s.deinit();
    const other = [_]u32{ 90, 91, 92, 93, 94, 95, 96, 97 };
    const own = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    const new = [_]u32{ 50, 51, 52, 53, 54, 55, 56, 57 };
    try put(&f, &s, &other, 8, 7);
    try put(&f, &s, own[0..8], 8, 7);
    try put(&f, &s, own[0..12], 8, 12);
    const endpoint = s.find(own[0..8], &.{}, &.{}).?;
    s.resumed(endpoint, own[0..8], true);
    try std.testing.expect(!endpoint.branched);
    try put(&f, &s, &new, 8, 8);
    try std.testing.expect(s.find(&other, &.{}, &.{}) != null);
    try std.testing.expectEqual(@as(u32, 12), s.find(&own, &.{}, &.{}).?.at);
}

test "newest cache: a re-rendered reply does not pin its equal-born endpoint as a branch" {
    for ([_]bool{ false, true }) |scan| {
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 10000);
        defer s.deinit();
        const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        try put(&f, &s, words[0..10], 10, 7);
        try put(&f, &s, words[0..14], 10, 14);
        const endpoint = s.find(words[0..10], &.{}, &.{}).?;
        var next = words;
        next[10] = 99;
        s.links_lost = scan;
        s.resumed(endpoint, &next, true);
        try std.testing.expect(!endpoint.branched);
        try std.testing.expect(@import("prompt_cache_policy.zig").redundant(&s, endpoint, &.{}, 0));
    }
}
