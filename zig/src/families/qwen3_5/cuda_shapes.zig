//! Captured launch geometry, separate from the live counts read through the staged device tables.
const std = @import("std");
const Round = @import("cuda_round.zig");

pub const Shape = struct {
    total: u8,
    nseg: u8,
    record: bool,
    prompt: bool,
    taps: bool,
    slots: u8,
    hidden: bool = false,

    pub fn init(rows: usize, segments: usize, record: bool, taps: bool, slots: u8) Shape {
        std.debug.assert(rows > 0 and rows <= Round.most_verify and segments > 0 and segments <= rows);
        const bucket: usize = if (rows == 1) 1 else std.mem.alignForward(usize, rows, 16);
        const windows = std.math.ceilPowerOfTwo(usize, segments) catch unreachable;
        return .{ .total = @intCast(if (record) bucket else rows), .nseg = @intCast(if (record) windows else segments), .record = record, .prompt = !record, .taps = taps, .slots = slots };
    }

    pub fn key(s: Shape) u32 {
        return @as(u32, s.total) | @as(u32, s.nseg) << 8 | @as(u32, @intFromBool(s.record)) << 16 | @as(u32, @intFromBool(s.prompt)) << 17 | @as(u32, @intFromBool(s.taps)) << 18 | @as(u32, s.slots) << 19 | @as(u32, @intFromBool(s.hidden)) << 27;
    }
};
