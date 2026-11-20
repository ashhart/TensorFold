//! Checkpoint prefetch in bounded Linux readahead windows.
const std = @import("std");

/// MADV_WILLNEED over `bytes` one 128 KiB readahead window a call: one call over a whole tensor reads its first window.
pub fn prefetch(bytes: []const u8) void {
    if (@import("builtin").os.tag != .linux or bytes.len == 0) return;
    const page = std.heap.pageSize();
    const step = 128 << 10;
    const lo = std.mem.alignBackward(usize, @intFromPtr(bytes.ptr), page);
    const hi = std.mem.alignForward(usize, @intFromPtr(bytes.ptr) + bytes.len, page);
    var at = lo;
    while (at < hi) : (at += step) _ = std.os.linux.madvise(@ptrFromInt(at), @min(step, hi - at), std.os.linux.MADV.WILLNEED);
}
