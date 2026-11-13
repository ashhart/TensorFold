//! Shared native stall deadline parsing, in whole seconds; zero and absence leave the sentinel off.
const std = @import("std");
pub fn timeoutNs(value: ?[]const u8) !u64 {
    const text = value orelse return 0;
    const seconds = std.fmt.parseInt(u64, text, 10) catch return error.BadStallTimeout;
    return std.math.mul(u64, seconds, std.time.ns_per_s) catch error.BadStallTimeout;
}
