//! What a MiniMax-M3 load may hold: weights, scratch and every stream's caches under a limit set below the panic line.
const std = @import("std");
const cfg = @import("config.zig");
const st = @import("state.zig");

/// The default share of RAM: past about 80% wired, macOS stalls until its watchdog restarts the Mac.
pub const default_share = 0.75;
/// Kept free inside the limit for the command buffers, the keepalive and the host.
pub const margin: u64 = 2 << 30;

/// Bytes for caches: `ram` times the share (or MINIMAX_LOAD_LIMIT_GB, at most the working set), less the rest.
pub fn caches(ram: u64, working_set: u64, text: ?[]const u8, resident: u64) error{BadLoadLimit}!u64 {
    var limit: u64 = @intFromFloat(@as(f64, @floatFromInt(ram)) * default_share);
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len > 0) {
        const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
        if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
        const most = @min(ram, working_set);
        const bytes = gb * 1e9; // compared as a float first: a huge limit would overflow the integer
        limit = if (bytes >= @as(f64, @floatFromInt(most))) most else @intFromFloat(bytes);
    }
    return if (limit > resident + margin) limit - resident - margin else 0;
}

/// The longest context one stream's caches fit in `bytes`, at most `most`.
pub fn longest(c: *const cfg.Config, bytes: u64, most: usize) usize {
    if (st.cacheBytes(c, 1) > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (st.cacheBytes(c, mid) <= bytes) lo = mid else hi = mid - 1;
    }
    return lo;
}

test "minimax_m3 budget: a share of RAM less the weights, a set limit capped at the working set, bad text refused" {
    const gib: u64 = 1 << 30;
    try std.testing.expectEqual(@as(u64, 192 * gib - 24 * gib - margin), try caches(256 * gib, 240 * gib, null, 24 * gib));
    try std.testing.expectEqual(@as(u64, 0), try caches(256 * gib, 240 * gib, null, 191 * gib));
    try std.testing.expectEqual(@as(u64, 64 * gib - 24 * gib - margin), try caches(256 * gib, 240 * gib, "68.719476736", 24 * gib));
    try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, "lots", 24 * gib));
    try std.testing.expectEqual(@as(u64, 240 * gib - 24 * gib - margin), try caches(256 * gib, 240 * gib, "1e30", 24 * gib));
}
