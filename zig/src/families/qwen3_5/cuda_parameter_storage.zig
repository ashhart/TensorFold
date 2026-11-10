//! Storage conversion for a BF16 parameter also exported as exactly widened FP32 words.
const std = @import("std");

pub fn exactBf16(source: []const u8, destination: []u16) !void {
    if (source.len != try std.math.mul(usize, destination.len, 4)) return error.UnexpectedTensor;
    for (destination, 0..) |_, index| {
        const word = std.mem.readInt(u32, source[index * 4 ..][0..4], .little);
        if (word & 0xffff != 0) return error.InexactBf16Parameter;
    }
    for (destination, 0..) |*value, index| {
        const word = std.mem.readInt(u32, source[index * 4 ..][0..4], .little);
        value.* = @intCast(word >> 16);
    }
}

test "exact parameter storage retains all BF16 words, including signs and exceptional classes" {
    var source: [256 * 4]u8 = undefined;
    var actual: [256]u16 = undefined;
    for (0..256) |base| {
        for (0..256) |low| std.mem.writeInt(u32, source[low * 4 ..][0..4], @as(u32, @intCast(base * 256 + low)) << 16, .little);
        try exactBf16(&source, &actual);
        for (actual, 0..) |got, low| try std.testing.expectEqual(@as(u16, @intCast(base * 256 + low)), got);
    }
}

test "inexact parameter storage refuses rounding before changing the destination" {
    var source: [12]u8 = undefined;
    const words = [_]u32{ 0x3f800000, 0xbf800001, 0x40000000 };
    for (words, 0..) |word, i| std.mem.writeInt(u32, source[i * 4 ..][0..4], word, .little);
    var actual = [_]u16{ 0xa5a5, 0xa5a5, 0xa5a5 };
    try std.testing.expectError(error.InexactBf16Parameter, exactBf16(&source, &actual));
    try std.testing.expectEqualSlices(u16, &.{ 0xa5a5, 0xa5a5, 0xa5a5 }, &actual);
    try std.testing.expectError(error.UnexpectedTensor, exactBf16(source[0..11], &actual));
}
