//! XPU stream admission: how many streams a card's free memory holds, with no device needed (host-tested).
const std = @import("std");

pub const Memory = struct { free: u64, size: u64, allocated: u64 };

/// Streams of `slot_bytes` that fit beside `margin` and `reserve`, at most `want` and `max`; 0 when the first does not.
pub fn streamsThatFit(m: Memory, margin: u64, reserve: u64, slot_bytes: u64, want: u32, max: u32) u32 {
    const free = @min(m.free, m.size -| m.allocated);
    if (free < margin + reserve) return 0;
    const more: u32 = @intCast(@min((free - margin - reserve) / slot_bytes, max - 1));
    return @min(want, 1 + more);
}

const gib: u64 = 1 << 30;

test "a stream that does not fit is refused" {
    const m: Memory = .{ .free = 3 * gib, .size = 32 * gib, .allocated = 20 * gib };
    try std.testing.expectEqual(@as(u32, 0), streamsThatFit(m, 2 * gib, 2 * gib, gib, 8, 8));
}

test "streams follow the free memory, not the card's size" {
    const m: Memory = .{ .free = 10 * gib, .size = 32 * gib, .allocated = 18 * gib };
    try std.testing.expectEqual(@as(u32, 5), streamsThatFit(m, 2 * gib, 2 * gib, gib + gib / 2, 8, 8));
    try std.testing.expectEqual(@as(u32, 3), streamsThatFit(m, 2 * gib, 2 * gib, 3 * gib, 8, 8));
}

test "the size less what the engine allocated caps an over-reported free" {
    const m: Memory = .{ .free = 30 * gib, .size = 32 * gib, .allocated = 24 * gib };
    try std.testing.expectEqual(@as(u32, 3), streamsThatFit(m, 2 * gib, 2 * gib, 2 * gib, 8, 8));
}

test "at most the wanted streams and the engine's maximum" {
    const m: Memory = .{ .free = 30 * gib, .size = 32 * gib, .allocated = 0 };
    try std.testing.expectEqual(@as(u32, 2), streamsThatFit(m, 2 * gib, 0, gib, 2, 8));
    try std.testing.expectEqual(@as(u32, 8), streamsThatFit(m, 2 * gib, 0, gib, 16, 8));
}
