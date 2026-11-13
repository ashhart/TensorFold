//! What a Mac family's load may hold: weights, scratch and every stream's caches under a limit below the panic line.
const std = @import("std");

/// The default share of RAM: past about 80% wired, macOS stalls until its watchdog restarts the Mac.
pub const default_share = 0.75;
/// Kept free inside the limit for the command buffers, the keepalive and the host.
pub const margin: u64 = 2 << 30;

/// The limit: `ram` times the share, or `text`'s GB capped at the GPU's working set; more GB than `ram` is refused.
pub fn limit(ram: u64, working_set: u64, text: ?[]const u8) error{BadLoadLimit}!u64 {
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0) return @intFromFloat(@as(f64, @floatFromInt(ram)) * default_share);
    const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
    // compared as floats, so a value past u64's range never reaches @intFromFloat
    if (!std.math.isFinite(gb) or gb <= 0 or gb * 1e9 > @as(f64, @floatFromInt(ram))) return error.BadLoadLimit;
    return @min(@as(u64, @intFromFloat(gb * 1e9)), @min(ram, working_set));
}

/// Bytes for caches: the limit less what stays resident and the margin.
pub fn caches(ram: u64, working_set: u64, text: ?[]const u8, resident: u64) error{BadLoadLimit}!u64 {
    const most = try limit(ram, working_set, text);
    return if (most > resident + margin) most - resident - margin else 0;
}

/// The longest context one stream's caches (`cacheBytes(ctx, n)` for `n` tokens) fit in `bytes`, at most `most`.
pub fn longest(ctx: anytype, comptime cacheBytes: fn (@TypeOf(ctx), usize) usize, bytes: u64, most: usize) usize {
    if (cacheBytes(ctx, 1) > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (cacheBytes(ctx, mid) <= bytes) lo = mid else hi = mid - 1;
    }
    return lo;
}

test "load budget: a share of RAM less the weights, a set limit capped at the working set, bad or huge text refused" {
    const gib: u64 = 1 << 30;
    try std.testing.expectEqual(@as(u64, 192 * gib - 24 * gib - margin), try caches(256 * gib, 240 * gib, null, 24 * gib));
    try std.testing.expectEqual(@as(u64, 0), try caches(256 * gib, 240 * gib, null, 191 * gib));
    try std.testing.expectEqual(@as(u64, 64 * gib - 24 * gib - margin), try caches(256 * gib, 240 * gib, "68.719476736", 24 * gib));
    try std.testing.expectEqual(@as(u64, 240 * gib - 24 * gib - margin), try caches(256 * gib, 240 * gib, "270", 24 * gib));
    try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, "lots", 24 * gib));
    try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, "900", 24 * gib));
    try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, "1e30", 24 * gib));
    try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, "inf", 24 * gib));
    try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, "-1", 24 * gib));
}

test "load budget: the longest context is the last whose caches fit" {
    const per = struct {
        fn bytes(row: u64, n: usize) usize {
            return @intCast(row * n);
        }
    }.bytes;
    try std.testing.expectEqual(@as(usize, 4096), longest(@as(u64, 1024), per, 4096 * 1024, 1 << 20));
    try std.testing.expectEqual(@as(usize, 512), longest(@as(u64, 1024), per, 4096 * 1024, 512));
    try std.testing.expectEqual(@as(usize, 0), longest(@as(u64, 1024), per, 1000, 512));
}
