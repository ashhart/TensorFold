//! A bounded served-prompt log plans a repeat mark; full token keys still decide every actual resume.
const std = @import("std");
pub const Log = struct {
    keys: std.ArrayList(u64) = .empty,
    pub const limit = 256;
    pub fn deinit(l: *Log, a: std.mem.Allocator) void {
        l.keys.deinit(a);
    }
    fn key(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
    }
    pub fn note(l: *Log, a: std.mem.Allocator, min: u32, tokens: []const u32) void {
        if (min == 0 or tokens.len < min) return;
        const k = key(tokens);
        if (std.mem.indexOfScalar(u64, l.keys.items, k) != null) return;
        if (l.keys.items.len == limit) _ = l.keys.orderedRemove(0);
        l.keys.append(a, k) catch {};
    }
    pub fn mark(l: *const Log, min: u32, tokens: []const u32, from: u32, lookahead: u32, planned: bool) u32 {
        if (min == 0 or planned or tokens.len < @max(min, 2) or lookahead >= tokens.len) return 0;
        const at = tokens.len - @max(lookahead, 1);
        if (at <= from or std.mem.indexOfScalar(u64, l.keys.items, key(tokens)) == null) return 0;
        return @intCast(at);
    }
};

test "repeat cache log: bounded history is only a mark hint and planned families never add cuts" {
    var log: Log = .{};
    defer log.deinit(std.testing.allocator);
    const prompt = [_]u32{ 1, 2, 3, 4 };
    try std.testing.expectEqual(@as(u32, 0), log.mark(4, &prompt, 0, 1, false));
    log.note(std.testing.allocator, 4, &prompt);
    try std.testing.expectEqual(@as(u32, 3), log.mark(4, &prompt, 0, 1, false));
    try std.testing.expectEqual(@as(u32, 2), log.mark(4, &prompt, 0, 2, false));
    try std.testing.expectEqual(@as(u32, 0), log.mark(4, &prompt, 0, 1, true));
    try std.testing.expectEqual(@as(u32, 0), log.mark(4, &prompt, 3, 1, false));
    try std.testing.expectEqual(@as(u32, 0), log.mark(0, &prompt, 0, 1, false));
    for (0..Log.limit + 1) |n| log.note(std.testing.allocator, 4, &.{ @intCast(n + 100), 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, Log.limit), log.keys.items.len);
    try std.testing.expectEqual(@as(u32, 0), log.mark(4, &prompt, 0, 1, false));
}
