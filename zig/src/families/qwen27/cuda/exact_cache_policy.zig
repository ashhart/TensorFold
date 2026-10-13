//! Private full-prompt cache eligibility and bounded ownership decisions.
const std = @import("std");
pub const max_tokens = 512;
pub const max_bytes = 256 * 1024 * 1024;

pub fn enabled(value: ?[]const u8) !bool {
    const text = value orelse return false;
    if (std.mem.eql(u8, text, "0")) return false;
    if (std.mem.eql(u8, text, "1")) return true;
    return error.InvalidExactPromptCacheFlag;
}

pub fn eligible(tokens: usize, chunks: usize, fresh: bool, cancelled: bool) bool {
    return tokens > 0 and tokens <= max_tokens and chunks <= max_tokens and fresh and !cancelled;
}

pub fn clean(finished: bool, normal_reason: bool, cancelled: bool, settled: bool) bool {
    return finished and normal_reason and !cancelled and settled;
}

pub const Decision = enum { refuse, busy, retain, evict };
pub fn admission(active_bytes: usize, pending_bytes: usize, requested: usize) Decision {
    if (requested == 0 or requested > max_bytes) return .refuse;
    if (pending_bytes != 0) return .busy;
    return if (active_bytes > max_bytes - requested) .evict else .retain;
}

test "flag is strict and disabled by default" {
    try std.testing.expect(!try enabled(null));
    try std.testing.expect(!try enabled("0"));
    try std.testing.expect(try enabled("1"));
    try std.testing.expectError(error.InvalidExactPromptCacheFlag, enabled("true"));
}

test "only bounded fresh uncancelled prompts can enter" {
    try std.testing.expect(eligible(1, 0, true, false));
    try std.testing.expect(eligible(512, 512, true, false));
    try std.testing.expect(!eligible(0, 0, true, false));
    try std.testing.expect(!eligible(513, 0, true, false));
    try std.testing.expect(!eligible(14, 0, false, false));
    try std.testing.expect(!eligible(14, 0, true, true));
}

test "clean completion never includes uncertain settle or cancellation" {
    try std.testing.expect(clean(true, true, false, true));
    try std.testing.expect(!clean(true, true, false, false));
    try std.testing.expect(!clean(true, true, true, true));
    try std.testing.expect(!clean(true, false, false, true));
    try std.testing.expect(!clean(false, true, false, true));
}

test "one pending owner and total byte bound include replacement with room" {
    try std.testing.expectEqual(.busy, admission(0, 1, 100));
    try std.testing.expectEqual(.refuse, admission(0, 0, max_bytes + 1));
    try std.testing.expectEqual(.retain, admission(100, 0, 100));
    try std.testing.expectEqual(.evict, admission(197995008, 0, 197995008));
    try std.testing.expectEqual(.retain, admission(0, 0, 197995008));
}
