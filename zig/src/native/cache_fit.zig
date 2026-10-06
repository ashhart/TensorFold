//! The prompt cache's budget from memory: what 70% of RAM leaves past the server's footprint once loaded, less a margin.
const std = @import("std");
const builtin = @import("builtin");

pub const GiB: u64 = 1 << 30;
/// The share of RAM a served model may hold: the rest stays with macOS and other processes.
pub const SHARE_PERCENT = 70;
/// Kept back from the cache for buffers that grow with long prompts.
pub const MARGIN: u64 = 2 * GiB;

/// The cache's budget, what fits under the cap, and the cap itself (bytes).
pub const Fit = struct { budget: u64, room: u64, cap: u64 };

/// The budget for `total` bytes of RAM and a footprint of `ready` bytes once loaded: `gib` when given and it fits (or `over`),
/// else all that fits; a given budget past the room is refused.
pub fn fit(total: u64, ready: u64, gib: ?f64, over: bool) error{CacheOverCap}!Fit {
    const cap = total / 100 * SHARE_PERCENT;
    const room = cap -| ready -| MARGIN;
    const want = if (gib) |g| (if (g > 0) std.math.lossyCast(u64, g * GiB) else 0) else room;
    if (want > room and !over) return error.CacheOverCap;
    return .{ .budget = want, .room = room, .cap = cap };
}

/// This machine's RAM (null where the OS does not say).
pub fn ram() ?u64 {
    return std.process.totalSystemMemory() catch null;
}

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

/// This process's physical footprint as macOS counts it, Metal buffers included (rusage_info_v4; null elsewhere).
pub fn footprint() ?u64 {
    if (builtin.os.tag != .macos) return null;
    var info: [43]u64 = undefined;
    if (proc_pid_rusage(getpid(), 4, &info) != 0) return null;
    return info[9];
}

/// GiB with one decimal, for log lines and messages.
pub fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / GiB;
}

test "the default budget is what 70% of RAM leaves past the footprint at ready and the margin" {
    const ram256: u64 = 256 * GiB;
    const f = try fit(ram256, 172 * GiB, null, false); // this tip on a 256 GB M5 Ultra
    try std.testing.expectEqual(ram256 / 100 * 70, f.cap);
    try std.testing.expectEqual(f.cap - 174 * GiB, f.budget);
    try std.testing.expect(gibs(f.budget) > 5.0 and gibs(f.budget) < 5.3);
    try std.testing.expectEqual(@as(u64, 0), (try fit(ram256, 179 * GiB, null, false)).budget); // nothing left: no cache
    try std.testing.expectEqual(@as(u64, 0), (try fit(192 * GiB, 172 * GiB, null, false)).budget); // a 192 GB Mac: past the cap already
}

test "a given budget that fits is kept, a larger one is refused unless the override is set" {
    const ram256: u64 = 256 * GiB;
    try std.testing.expectEqual(@as(u64, 4 * GiB), (try fit(ram256, 172 * GiB, 4, false)).budget);
    try std.testing.expectEqual(@as(u64, 0), (try fit(ram256, 172 * GiB, 0, false)).budget);
    try std.testing.expectError(error.CacheOverCap, fit(ram256, 172 * GiB, 8, false));
    try std.testing.expectError(error.CacheOverCap, fit(ram256, 172 * GiB, 16, false));
    try std.testing.expectEqual(@as(u64, 16 * GiB), (try fit(ram256, 172 * GiB, 16, true)).budget);
    try std.testing.expectEqual(@as(u64, 64 * GiB), (try fit(512 * GiB, 172 * GiB, 64, false)).budget); // a 512 GB Mac has room
}

test "the OS readings: RAM and this process's footprint" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const r = ram() orelse return error.NoRam;
    const p = footprint() orelse return error.NoFootprint;
    try std.testing.expect(r >= 8 * GiB and p > 0 and p < r);
}
