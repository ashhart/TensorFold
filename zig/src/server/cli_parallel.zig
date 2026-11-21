//! ``--parallel``: how many lanes a server asks for, and whether it named a number or "auto".
const std = @import("std");

/// ``--parallel``: "auto" is up to 8 requests at once; a number caps it.
pub fn parallel(text: []const u8) ?u32 {
    const t = std.mem.trim(u8, text, " ");
    if (std.ascii.eqlIgnoreCase(t, "auto")) return 8;
    const n = std.fmt.parseInt(i64, t, 10) catch return null;
    return @intCast(@max(1, @min(n, 4096)));
}

/// ``--parallel`` named a number (an engine that fits fewer refuses), not "auto" (it serves what fits).
pub fn parallelFixed(text: []const u8) bool {
    return !std.ascii.eqlIgnoreCase(std.mem.trim(u8, text, " "), "auto") and parallel(text) != null;
}
