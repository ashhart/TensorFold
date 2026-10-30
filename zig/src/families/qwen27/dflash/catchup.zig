//! The bounded incremental tap path; a larger backlog needs the existing full-context flush.
const std = @import("std");
pub const Range = struct { slot: u32, rows: u32, first: u64, begin: u64 };
pub fn fresh(ready: bool, begin: u64, end: u64, committed: u64, window: u32) ?Range {
    if (!ready or committed <= end or window == 0) return null;
    const behind = committed - end;
    const slot: u32 = @intCast(end % window);
    if (behind > 128 or slot + behind > window or end < committed -| (@as(u64, window) - 1)) return null;
    return .{ .slot = slot, .rows = @intCast(behind), .first = end, .begin = @max(begin, committed -| (@as(u64, window) - 1)) };
}
test "qwen27 window: a long zero-grant tap backlog takes the flush instead of a 128-row absorb" {
    try std.testing.expectEqual(@as(u32, 128), fresh(true, 0, 3, 131, 256).?.rows);
    try std.testing.expectEqual(@as(?Range, null), fresh(true, 0, 3, 132, 256));
    try std.testing.expectEqual(@as(?Range, null), fresh(true, 0, 240, 272, 256));
    try std.testing.expectEqual(@as(?Range, null), fresh(false, 0, 0, 3, 256));
}
