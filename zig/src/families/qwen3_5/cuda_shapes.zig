//! Captured launch geometry, separate from the live counts read through the staged device tables.
const std = @import("std");
const Round = @import("cuda_round.zig");

pub const Shape = struct {
    total: u16, // a prompt chunk's rows (up to Round.max_rows, never captured), or a window bucket
    nseg: u8,
    record: bool,
    prompt: bool,
    taps: bool,
    slots: u8,
    hidden: bool = false,
    paged: bool = false,
    mt2: bool = false,

    pub fn init(rows: usize, segments: usize, record: bool, taps: bool, slots: u8) Shape {
        std.debug.assert(rows > 0 and rows <= @as(usize, if (record) Round.most_verify else Round.max_rows) and segments > 0 and segments <= rows);
        const bucket: usize = if (rows == 1) 1 else std.mem.alignForward(usize, rows, 16);
        const windows = std.math.ceilPowerOfTwo(usize, segments) catch unreachable;
        return .{ .total = @intCast(if (record) bucket else rows), .nseg = @intCast(if (record) windows else segments), .record = record, .prompt = !record, .taps = taps, .slots = slots };
    }

    pub fn key(s: Shape) u32 {
        // captured shapes only: a window bucket's rows fit the low byte
        return @as(u32, @as(u8, @intCast(s.total))) | @as(u32, s.nseg) << 8 | @as(u32, @intFromBool(s.record)) << 16 | @as(u32, @intFromBool(s.prompt)) << 17 | @as(u32, @intFromBool(s.taps)) << 18 | @as(u32, s.slots) << 19 | @as(u32, @intFromBool(s.hidden)) << 27 | @as(u32, @intFromBool(s.paged)) << 28 | @as(u32, @intFromBool(s.mt2)) << 29;
    }
};

/// The measured FP8 row-tile envelope; context limits and prompt tiles are unchanged.
pub fn useMt2(qualified: bool, fp8: bool, prompt: bool, rows: usize, segments: usize, cached: usize) bool {
    return qualified and fp8 and !prompt and segments == 1 and rows > 0 and
        (rows == 1 or (rows <= 16 and cached >= 8192 and cached < 65536));
}

test "MT2 dispatch keeps context boundaries and graph identities separate" {
    const before = Shape.init(16, 1, true, false, 1);
    var after = before;
    after.mt2 = true;
    try std.testing.expectEqual(@as(u32, 1) << 29, before.key() ^ after.key());
    for ([_]usize{ 8192, 32768, 65535 }) |n| try std.testing.expect(useMt2(true, true, false, 16, 1, n));
    for ([_]usize{ 0, 8191, 65536, 131071 }) |n| try std.testing.expect(!useMt2(true, true, false, 16, 1, n));
    try std.testing.expect(useMt2(true, true, false, 1, 1, 131071));
    try std.testing.expect(!useMt2(false, true, false, 16, 1, 8192));
    try std.testing.expect(!useMt2(true, false, false, 16, 1, 8192));
    try std.testing.expect(!useMt2(true, true, true, 16, 1, 8192));
    try std.testing.expect(!useMt2(true, true, false, 16, 2, 8192));
    try std.testing.expect(!useMt2(true, true, false, 17, 1, 8192));
}
