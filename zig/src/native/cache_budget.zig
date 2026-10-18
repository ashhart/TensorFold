//! The shared single-host cache rule, independent of OS readings and GPU modules.
const std = @import("std");
pub const GiB: u64 = 1 << 30;
pub const Fit = struct { budget: u64, room: u64, cap: u64 };
pub const Policy = struct { margin: u64, ceiling: u64 };
pub fn fitUnder(cap: u64, ready: u64, gib: ?f64, over: bool, policy: Policy) error{CacheOverCap}!Fit {
    const room = cap -| ready -| policy.margin;
    const want = if (gib) |g| (if (g > 0) std.math.lossyCast(u64, g * GiB) else 0) else @min(room, policy.ceiling);
    if (want > room and !over) return error.CacheOverCap;
    return .{ .budget = want, .room = room, .cap = cap };
}
