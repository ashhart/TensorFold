//! Host transaction checks prove deferred target state never reaches work after a miss or failed restore.
const std = @import("std");
const policy = @import("reset_skip.zig");
const work = .{ .calls = @as(u64, 96), .bytes = @as(u64, 153944064) };
const Mock = struct {
    hit: bool = true,
    cancel: bool = false,
    copy_error: bool = false,
    reset_error: bool = false,
    copies: usize = 0,
    resets: usize = 0,
    initialized: bool = false,
    pub fn cancelled(a: *Mock) bool {
        return a.cancel;
    }
    pub fn lookup(a: *Mock) !bool {
        a.copies += 1;
        if (a.copy_error) return error.PartialCopy;
        if (a.hit) a.initialized = true;
        return a.hit;
    }
    pub fn reset(a: *Mock) !void {
        a.resets += 1;
        if (a.reset_error) return error.ResetFailed;
        a.initialized = true;
    }
    fn target(a: *Mock) !void {
        if (!a.initialized) return error.UninitializedTarget;
    }
};

test "strict opt-in requires exact cache and rejects malformed flags" {
    try std.testing.expect(!try policy.compatible(null, true));
    try std.testing.expect(!try policy.compatible("0", false));
    try std.testing.expect(try policy.compatible("1", true));
    try std.testing.expectError(error.ResetSkipNeedsExactCache, policy.compatible("1", false));
    try std.testing.expectError(error.InvalidResetSkip, policy.compatible("true", true));
}

test "actual hit resolves fresh and retained state without reset" {
    for ([_]bool{ false, true }) |pooled| {
        var m: Mock = .{};
        var deferred = true;
        var stats: policy.Stats = .{ .enabled = true };
        try std.testing.expect(try policy.resolve(&m, &deferred, pooled, &stats, work));
        try m.target();
        try std.testing.expect(!deferred);
        try std.testing.expectEqual(@as(usize, 0), m.resets);
        try std.testing.expectEqual(@as(u64, 96), stats.avoided_fill_calls);
        try std.testing.expectEqual(work.bytes, stats.avoided_fill_bytes);
        try std.testing.expectEqual(@as(u64, 1), stats.restored_fresh + stats.restored_pool);
    }
}

test "positive hint then actual miss resets before target work" {
    var m: Mock = .{ .hit = false };
    var deferred = true;
    var stats: policy.Stats = .{};
    try std.testing.expectError(error.UninitializedTarget, m.target());
    try std.testing.expect(!try policy.resolve(&m, &deferred, false, &stats, work));
    try m.target();
    try std.testing.expectEqual(@as(usize, 1), m.copies);
    try std.testing.expectEqual(@as(usize, 1), m.resets);
    try std.testing.expectEqual(@as(u64, 1), stats.fallback_resets);
    try std.testing.expectEqual(@as(u64, 0), stats.avoided_fill_calls);
}

test "partial copy aborts without reset fallback or resolved admission" {
    var m: Mock = .{ .copy_error = true };
    var deferred = true;
    var stats: policy.Stats = .{};
    try std.testing.expectError(error.PartialCopy, policy.resolve(&m, &deferred, false, &stats, work));
    try std.testing.expect(deferred);
    try std.testing.expectEqual(@as(usize, 0), m.resets);
    try std.testing.expectEqual(@as(u64, 1), stats.failed_deferred_transactions);
}

test "cancelled deferred destination aborts before copying or resetting" {
    var m: Mock = .{ .cancel = true };
    var deferred = true;
    var stats: policy.Stats = .{};
    try std.testing.expectError(error.Cancelled, policy.resolve(&m, &deferred, true, &stats, work));
    try std.testing.expect(deferred);
    try std.testing.expectEqual(@as(usize, 0), m.copies + m.resets);
}

test "reset failure on actual miss remains unresolved and propagates" {
    var m: Mock = .{ .hit = false, .reset_error = true };
    var deferred = true;
    var stats: policy.Stats = .{};
    try std.testing.expectError(error.ResetFailed, policy.resolve(&m, &deferred, false, &stats, work));
    try std.testing.expect(deferred);
    try std.testing.expectEqual(@as(u64, 0), stats.fallback_resets + stats.avoided_fill_calls);
}

test "ordinary initialized path retains original lookup with no extra reset or cancel check" {
    var m: Mock = .{ .hit = false, .cancel = true, .initialized = true };
    var deferred = false;
    var stats: policy.Stats = .{};
    try std.testing.expect(!try policy.resolve(&m, &deferred, false, &stats, work));
    try m.target();
    try std.testing.expectEqual(@as(usize, 1), m.copies);
    try std.testing.expectEqual(@as(usize, 0), m.resets);
    try std.testing.expectEqual(@as(u64, 0), stats.failed_deferred_transactions);
}
