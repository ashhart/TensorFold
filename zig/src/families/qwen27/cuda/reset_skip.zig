//! Private prefill reset transaction: only actual full-state restoration resolves uninitialized target state.
const std = @import("std");
pub const max_capacity: usize = 512;
pub const Stats = struct {
    enabled: bool = false,
    deferred_fresh: u64 = 0,
    deferred_pool: u64 = 0,
    restored_fresh: u64 = 0,
    restored_pool: u64 = 0,
    avoided_fill_calls: u64 = 0,
    avoided_fill_bytes: u64 = 0,
    fallback_resets: u64 = 0,
    failed_deferred_transactions: u64 = 0,
};

pub fn compatible(raw: ?[]const u8, cache_enabled: bool) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (!std.mem.eql(u8, value, "1")) return error.InvalidResetSkip;
    if (!cache_enabled) return error.ResetSkipNeedsExactCache;
    return true;
}

pub fn configured(cache_enabled: bool) !bool {
    const raw = std.c.getenv("TF_PR608_EXACT_CACHE_SKIP_RESET");
    return compatible(if (raw) |value| std.mem.span(value) else null, cache_enabled);
}

/// Counts the actual reset domains, excluding attention KV, heads and alignment padding.
pub fn resetWork(g: anytype) struct { calls: u64, bytes: u64 } {
    var linear_layers: u64 = 0;
    for (0..g.layers) |i| if (i % 4 != 3) {
        linear_layers += 1;
    };
    return .{ .calls = linear_layers * 2, .bytes = linear_layers * @as(u64, @intCast(g.deltaBytes() + g.convBytes())) };
}

/// A hint never resolves deferred state: actual hit or original reset must happen before target work.
pub fn resolve(action: anytype, deferred: *bool, pooled: bool, stats: *Stats, work: anytype) !bool {
    if (!deferred.*) return action.lookup();
    errdefer stats.failed_deferred_transactions += 1;
    if (action.cancelled()) return error.Cancelled;
    const hit = try action.lookup();
    if (hit) {
        if (pooled) stats.restored_pool += 1 else stats.restored_fresh += 1;
        stats.avoided_fill_calls += work.calls;
        stats.avoided_fill_bytes += work.bytes;
    } else {
        try action.reset();
        stats.fallback_resets += 1;
    }
    deferred.* = false;
    return hit;
}

test {
    _ = @import("reset_skip_test.zig");
}
