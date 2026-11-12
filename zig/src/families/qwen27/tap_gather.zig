//! Completed shared-memory tap planes gathered on the host without an extra GPU submission.
const std = @import("std");
/// `count` planes (the drafter's tap count, at most eight) of each kept row, packed row by row as the ring stores them.
pub fn gather(planes: []const u8, capacity: usize, written: usize, width: usize, count: usize, rows: []const u32, out: []u8) !void {
    if (count == 0 or count > 8) return error.BadTapGather;
    const bytes = try std.math.mul(usize, width, 2);
    const stride = try std.math.mul(usize, bytes, count);
    const source_size = try std.math.mul(usize, capacity, stride);
    const output_size = try std.math.mul(usize, rows.len, stride);
    if (planes.len < source_size or out.len < output_size) return error.ShortTapGather;
    if (written > capacity or rows.len > written) return error.BadTapGather;
    for (rows) |row| if (row >= written) return error.BadTapGather;
    for (rows, 0..) |row, i| {
        for (0..count) |plane| {
            const src = (plane * capacity + row) * bytes;
            @memcpy(out[i * stride + plane * bytes ..][0..bytes], planes[src..][0..bytes]);
        }
    }
}

test "qwen27 window: tap gather preserves every plane for a 32-row noncontiguous kept path" {
    var planes: [64 * 5 * 3]u16 = undefined;
    for (0..5) |p| for (0..64) |r| for (0..3) |c| {
        planes[(p * 64 + r) * 3 + c] = @intCast(((p + 1) << 12) | (r << 4) | c);
    };
    var rows: [32]u32 = undefined;
    for (&rows, 0..) |*r, i| r.* = @intCast(63 - i * 2);
    var out: [32 * 5 * 3 + 2]u16 = @splat(0xabab);
    try gather(std.mem.sliceAsBytes(&planes), 64, 64, 3, 5, &rows, std.mem.sliceAsBytes(out[0 .. 32 * 5 * 3]));
    for (rows, 0..) |r, i| for (0..5) |p| for (0..3) |c| {
        const expected: u16 = @intCast(((p + 1) << 12) | (r << 4) | c);
        try std.testing.expectEqual(expected, out[(i * 5 + p) * 3 + c]);
    };
    try std.testing.expectEqualSlices(u16, &.{ 0xabab, 0xabab }, out[32 * 5 * 3 ..]);
    try std.testing.expectError(error.BadTapGather, gather(std.mem.sliceAsBytes(&planes), 64, 64, 3, 5, &.{64}, std.mem.sliceAsBytes(&out)));
    try std.testing.expectError(error.ShortTapGather, gather(std.mem.sliceAsBytes(&planes), 64, 64, 3, 5, &rows, std.mem.sliceAsBytes(out[0..1])));
}

test "qwen27 window: tap gather refuses stale rows inside capacity and oversized kept paths" {
    const planes: [64 * 5]u16 = @splat(7);
    var out: [64 * 5]u16 = @splat(0xabab);
    const bytes = std.mem.sliceAsBytes(&planes);
    const dst = std.mem.sliceAsBytes(&out);
    try std.testing.expectError(error.BadTapGather, gather(bytes, 64, 2, 1, 5, &.{2}, dst));
    try std.testing.expectError(error.BadTapGather, gather(bytes, 64, 2, 1, 5, &.{ 0, 1, 0 }, dst));
    for (out) |v| try std.testing.expectEqual(@as(u16, 0xabab), v);
}

test "qwen27 window: tap gather copies all eight planes and places every later row at the eight-plane stride" {
    // Ornith's drafter taps eight layers; two windows' kept rows, each later row eight planes past the one before.
    var planes: [8 * 40 * 4]u16 = undefined;
    for (0..8) |p| for (0..40) |r| for (0..4) |c| {
        planes[(p * 40 + r) * 4 + c] = @intCast(((p + 1) << 12) | (r << 4) | c);
    };
    const rows = [_]u32{ 0, 2, 3, 17, 18, 39 };
    var out: [rows.len * 8 * 4 + 2]u16 = @splat(0xabab);
    try gather(std.mem.sliceAsBytes(&planes), 40, 40, 4, 8, &rows, std.mem.sliceAsBytes(out[0 .. rows.len * 8 * 4]));
    for (rows, 0..) |r, i| for (0..8) |p| for (0..4) |c| {
        const expected: u16 = @intCast(((p + 1) << 12) | (r << 4) | c);
        try std.testing.expectEqual(expected, out[(i * 8 + p) * 4 + c]);
    };
    try std.testing.expectEqualSlices(u16, &.{ 0xabab, 0xabab }, out[rows.len * 8 * 4 ..]);
    // a five-plane output is too short for eight planes, and a ninth plane or none is refused
    try std.testing.expectError(error.ShortTapGather, gather(std.mem.sliceAsBytes(&planes), 40, 40, 4, 8, &rows, std.mem.sliceAsBytes(out[0 .. rows.len * 5 * 4])));
    try std.testing.expectError(error.BadTapGather, gather(std.mem.sliceAsBytes(&planes), 40, 40, 4, 9, &rows, std.mem.sliceAsBytes(&out)));
    try std.testing.expectError(error.BadTapGather, gather(std.mem.sliceAsBytes(&planes), 40, 40, 4, 0, &rows, std.mem.sliceAsBytes(&out)));
}
