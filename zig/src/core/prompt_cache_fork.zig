//! A miss's fork: where a prompt parts from the kept prompt sharing the longest prefix, so a later pass resumes there.
const std = @import("std");

/// The salt's kept entry sharing the longest prefix, its last prompt the fork mark; empty when none shares.
pub fn longestShared(comptime E: type, entries: []const *const E, prompt: []const u32, salt: ?u64) []const u32 {
    var best: ?*const E = null;
    var best_n: usize = 0;
    for (entries) |e| {
        if (e.expired or e.salt != salt or e.last.len == 0) continue;
        const n = std.mem.indexOfDiff(u32, e.last, prompt) orelse @min(e.last.len, prompt.len);
        if (n == 0 or n >= prompt.len or n <= best_n) continue;
        best = e;
        best_n = n;
    }
    return if (best) |b| b.last else &.{};
}

test "the longest shared kept prompt is the fork, and a whole-prompt match is not" {
    const E = struct { expired: bool = false, salt: ?u64 = null, last: []const u32 };
    const a = [_]u32{ 1, 2, 3, 4, 5 };
    const b = [_]u32{ 1, 2, 9, 4, 5 };
    const one: E = .{ .last = &a };
    const two: E = .{ .last = &b };
    const gone: E = .{ .expired = true, .last = &a };
    const kept = [_]*const E{ &one, &two, &gone };
    const prompt = [_]u32{ 1, 2, 3, 4, 50 };
    try std.testing.expectEqualSlices(u32, &a, longestShared(E, &kept, &prompt, null)); // four of five tokens shared
    const whole = [_]u32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(usize, 0), longestShared(E, kept[0..1], &whole, null).len);
    const none = [_]u32{ 9, 9 };
    try std.testing.expectEqual(@as(usize, 0), longestShared(E, &kept, &none, null).len);
}

test "the fork learns from the request salt's entries only" {
    const E = struct { expired: bool = false, salt: ?u64 = null, last: []const u32 };
    const a = [_]u32{ 1, 2, 3, 4, 5 };
    const b = [_]u32{ 1, 2, 9, 4, 5 };
    const one: E = .{ .salt = 7, .last = &a };
    const two: E = .{ .salt = 8, .last = &b };
    const open: E = .{ .last = &b };
    const kept = [_]*const E{ &one, &two };
    const prompt = [_]u32{ 1, 2, 3, 4, 50 };
    // Two salts sharing a prefix: each salt's fork comes only from its own entries.
    try std.testing.expectEqualSlices(u32, &a, longestShared(E, &kept, &prompt, 7));
    try std.testing.expectEqualSlices(u32, &b, longestShared(E, &kept, &prompt, 8));
    // One salt alone: the other salt's longer match never shapes its mark.
    try std.testing.expectEqual(@as(usize, 0), longestShared(E, kept[0..1], &prompt, 8).len);
    try std.testing.expectEqualSlices(u32, &a, longestShared(E, kept[0..1], &prompt, 7));
    // No salt on either side behaves as today; a salted entry is not a candidate for an unsalted request.
    try std.testing.expectEqual(@as(usize, 0), longestShared(E, &kept, &prompt, null).len);
    try std.testing.expectEqualSlices(u32, &b, longestShared(E, &[_]*const E{ &one, &open }, &prompt, null));
}
