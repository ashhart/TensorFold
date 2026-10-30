//! One stored FP32 rotary table is rounded from binary64 frequencies before either backend uploads it.
const std = @import("std");
pub fn fill(theta: f64, rotary: usize, out: []f32) !void {
    if (!std.math.isFinite(theta) or theta <= 0 or rotary % 2 != 0 or out.len != 3 + rotary / 2) return error.Qwen4RotaryTable;
    out[0] = 0;
    out[1] = 1;
    out[2] = 1;
    for (out[3..], 0..) |*value, i| value.* = @floatCast(std.math.pow(f64, theta, -2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(rotary))));
}
test "Qwen4 rotary table: native devices consume the independently rounded FP32 frequency input" {
    var table: [19]f32 = undefined;
    try fill(10000, 32, &table);
    const expected = [_]u32{ 0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c3, 0x3d0186e2, 0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e37, 0x3ae91528, 0x3a83126f, 0x3a136a16, 0x39a5cb5f, 0x393a7753 };
    for (table[3..], expected) |value, bits| try std.testing.expectEqual(bits, @as(u32, @bitCast(value)));
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 1 }, table[0..3]);
    var disabled: [3]f32 = undefined;
    try fill(10000, 0, &disabled);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 1 }, &disabled);
    try std.testing.expectError(error.Qwen4RotaryTable, fill(10000, 32, table[0..18]));
    try std.testing.expectError(error.Qwen4RotaryTable, fill(std.math.nan(f64), 32, &table));
}
