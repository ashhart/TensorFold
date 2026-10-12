//! Bounded raw first-logit cache policy; decisions never cache sampling or a selected token.
const std = @import("std");
pub const row_bytes: usize = 248320 * 2; // shape.zig's fixed BF16 vocabulary row, already 256-byte aligned

pub fn enabled(text: ?[]const u8, cache_enabled: bool) !bool {
    const value = text orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (!std.mem.eql(u8, value, "1")) return error.InvalidExactCacheLogitsFlag;
    if (!cache_enabled) return error.FirstLogitsRequireExactCache;
    return true;
}

pub fn configured(cache_enabled: bool) !bool {
    const env = std.c.getenv("TF_PR608_EXACT_CACHE_LOGITS");
    return enabled(if (env) |v| std.mem.span(v) else null, cache_enabled);
}

pub fn canUse(healthy: bool, proven: bool, key_matches: bool, valid: bool, position_matches: bool, cancelled: bool) bool {
    return healthy and proven and key_matches and valid and position_matches and !cancelled;
}

pub fn promotable(on: bool, valid: bool) bool {
    return !on or valid;
}

/// Only successful submission grants validity; the existing promotion fence supplies completion.
pub fn capture(valid: *bool, copier: anytype, dst: u64, src: u64) !void {
    valid.* = false;
    try copier.copy(dst, src, row_bytes);
    valid.* = true;
}

test "raw logits are off by default and strictly require the full-prompt cache" {
    try std.testing.expect(!try enabled(null, false));
    try std.testing.expect(!try enabled("0", false));
    try std.testing.expect(try enabled("1", true));
    try std.testing.expectError(error.FirstLogitsRequireExactCache, enabled("1", false));
    try std.testing.expectError(error.InvalidExactCacheLogitsFlag, enabled("true", true));
    try std.testing.expectError(error.InvalidExactCacheLogitsFlag, enabled("", true));
}

test "current active key and actual stream restore proof are both required" {
    try std.testing.expect(canUse(true, true, true, true, true, false));
    try std.testing.expect(!canUse(false, true, true, true, true, false));
    try std.testing.expect(!canUse(true, false, true, true, true, false));
    try std.testing.expect(!canUse(true, true, false, true, true, false));
    try std.testing.expect(!canUse(true, true, true, false, true, false));
    try std.testing.expect(!canUse(true, true, true, true, false, false));
    try std.testing.expect(!canUse(true, true, true, true, true, true));
    try std.testing.expect(!promotable(true, false));
    try std.testing.expect(promotable(true, true));
    try std.testing.expect(promotable(false, false));
}

test "copy submission failure cannot create a valid raw row or promotion" {
    const Copier = struct {
        fail: bool,
        calls: usize = 0,
        fn copy(self: *@This(), dst: u64, src: u64, n: usize) !void {
            self.calls += 1;
            try std.testing.expectEqual(@as(u64, 256), dst);
            try std.testing.expectEqual(@as(u64, 512), src);
            try std.testing.expectEqual(@as(usize, 496640), n);
            if (self.fail) return error.InjectedCopyFailure;
        }
    };
    var valid = true;
    var copier: Copier = .{ .fail = true };
    try std.testing.expectError(error.InjectedCopyFailure, capture(&valid, &copier, 256, 512));
    try std.testing.expect(!valid and !promotable(true, valid));
    copier.fail = false;
    try capture(&valid, &copier, 256, 512);
    try std.testing.expect(valid and promotable(true, valid));
    try std.testing.expectEqual(@as(usize, 2), copier.calls);
}
