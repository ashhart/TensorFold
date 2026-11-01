//! Completed shared-memory tap planes gathered on the host without an extra GPU submission.
const std = @import("std");
pub fn gather(planes: []const u8, capacity: usize, written: usize, width: usize, rows: []const u32, out: []u8) !void {
    return gatherPlanes(5, planes, capacity, written, width, rows, out);
}
/// `count` planes of `capacity` rows each, row `rows[i]` of every plane side by side in out's row i.
pub fn gatherPlanes(count: usize, planes: []const u8, capacity: usize, written: usize, width: usize, rows: []const u32, out: []u8) !void {
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
    try gather(std.mem.sliceAsBytes(&planes), 64, 64, 3, &rows, std.mem.sliceAsBytes(out[0 .. 32 * 5 * 3]));
    for (rows, 0..) |r, i| for (0..5) |p| for (0..3) |c| {
        const expected: u16 = @intCast(((p + 1) << 12) | (r << 4) | c);
        try std.testing.expectEqual(expected, out[(i * 5 + p) * 3 + c]);
    };
    try std.testing.expectEqualSlices(u16, &.{ 0xabab, 0xabab }, out[32 * 5 * 3 ..]);
    try std.testing.expectError(error.BadTapGather, gather(std.mem.sliceAsBytes(&planes), 64, 64, 3, &.{64}, std.mem.sliceAsBytes(&out)));
    try std.testing.expectError(error.ShortTapGather, gather(std.mem.sliceAsBytes(&planes), 64, 64, 3, &rows, std.mem.sliceAsBytes(out[0..1])));
}

test "qwen27 window: tap gather refuses stale rows inside capacity and oversized kept paths" {
    const planes: [64 * 5]u16 = @splat(7);
    var out: [64 * 5]u16 = @splat(0xabab);
    const bytes = std.mem.sliceAsBytes(&planes);
    const dst = std.mem.sliceAsBytes(&out);
    try std.testing.expectError(error.BadTapGather, gather(bytes, 64, 2, 1, &.{2}, dst));
    try std.testing.expectError(error.BadTapGather, gather(bytes, 64, 2, 1, &.{ 0, 1, 0 }, dst));
    for (out) |v| try std.testing.expectEqual(@as(u16, 0xabab), v);
}
