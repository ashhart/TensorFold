//! A miss's fork: where a prompt parts from the kept prompt sharing the longest prefix, so a later pass resumes there.
const std = @import("std");

/// The kept entry sharing the longest prefix with `prompt`, its last prompt the fork mark; empty when none shares.
pub fn longestShared(comptime E: type, entries: []const *const E, prompt: []const u32) []const u32 {
    var best: ?*const E = null;
    var best_n: usize = 0;
    for (entries) |e| {
        if (e.expired or e.last.len == 0) continue;
        const n = std.mem.indexOfDiff(u32, e.last, prompt) orelse @min(e.last.len, prompt.len);
        if (n == 0 or n >= prompt.len or n <= best_n) continue;
        best = e;
        best_n = n;
    }
    return if (best) |b| b.last else &.{};
}

/// The mark a miss keeps at the fork: one before the part, as a state read at `at` is keyed with the token at `at`.
pub fn forkAt(comptime E: type, entries: []const *const E, prompt: []const u32, lookahead: u32, history_len: u32) u32 {
    const last = longestShared(E, entries, prompt);
    if (last.len == 0) return 0;
    const at: u32 = @intCast(std.mem.indexOfDiff(u32, last, prompt) orelse @min(last.len, prompt.len));
    if (at == 0 or at >= history_len or at < history_len / 2) return 0;
    return at -| lookahead;
}

test "the longest shared kept prompt is the fork, and a whole-prompt match is not" {
    const E = struct { expired: bool = false, last: []const u32 };
    const a = [_]u32{ 1, 2, 3, 4, 5 };
    const b = [_]u32{ 1, 2, 9, 4, 5 };
    const one: E = .{ .last = &a };
    const two: E = .{ .last = &b };
    const gone: E = .{ .expired = true, .last = &a };
    const kept = [_]*const E{ &one, &two, &gone };
    const prompt = [_]u32{ 1, 2, 3, 4, 50 };
    try std.testing.expectEqualSlices(u32, &a, longestShared(E, &kept, &prompt)); // five of six tokens shared
    const whole = [_]u32{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(usize, 0), longestShared(E, kept[0..1], &whole).len);
    const none = [_]u32{ 9, 9 };
    try std.testing.expectEqual(@as(usize, 0), longestShared(E, &kept, &none).len);
    // the fork sits one before the part under lookahead, past half the history, below it
    try std.testing.expectEqual(@as(u32, 3), forkAt(E, &kept, &prompt, 1, 6));
    try std.testing.expectEqual(@as(u32, 0), forkAt(E, &kept, &prompt, 1, 3)); // the history is at/before half
    try std.testing.expectEqual(@as(u32, 0), forkAt(E, &kept, &none, 1, 6)); // nothing shared
}
