//! Completed shared-memory tap planes gathered on the host without an extra GPU submission.
const std = @import("std");
pub fn gather(planes: []const u8, capacity: usize, width: usize, rows: []const u32, out: []u8) !void {
    const bytes = try std.math.mul(usize, width, 2);
    const stride = try std.math.mul(usize, bytes, 5);
    const source_size = try std.math.mul(usize, capacity, stride);
    const output_size = try std.math.mul(usize, rows.len, stride);
    if (planes.len < source_size or out.len < output_size) return error.ShortTapGather;
    for (rows, 0..) |row, i| {
        if (row >= capacity) return error.BadTapGather;
        for (0..5) |plane| {
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
    try gather(std.mem.sliceAsBytes(&planes), 64, 3, &rows, std.mem.sliceAsBytes(out[0 .. 32 * 5 * 3]));
    for (rows, 0..) |r, i| for (0..5) |p| for (0..3) |c| {
        const expected: u16 = @intCast(((p + 1) << 12) | (r << 4) | c);
        try std.testing.expectEqual(expected, out[(i * 5 + p) * 3 + c]);
    };
    try std.testing.expectEqualSlices(u16, &.{ 0xabab, 0xabab }, out[32 * 5 * 3 ..]);
    try std.testing.expectError(error.BadTapGather, gather(std.mem.sliceAsBytes(&planes), 64, 3, &.{64}, std.mem.sliceAsBytes(&out)));
    try std.testing.expectError(error.ShortTapGather, gather(std.mem.sliceAsBytes(&planes), 64, 3, &rows, std.mem.sliceAsBytes(out[0..1])));
}
