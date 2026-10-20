const std = @import("std");
const npy = @import("npy.zig");

fn image(a: std.mem.Allocator, shape: []const u8, payload: usize) ![]u8 {
    const header = try std.fmt.allocPrint(a, "{{'descr': '<u2', 'fortran_order': False, 'shape': ({s}), }}", .{shape});
    defer a.free(header);
    const bytes = try a.alloc(u8, 10 + header.len + payload);
    @memset(bytes, 0);
    @memcpy(bytes[0..6], "\x93NUMPY");
    bytes[6] = 1;
    std.mem.writeInt(u16, bytes[8..10], @intCast(header.len), .little);
    @memcpy(bytes[10..][0..header.len], header);
    return bytes;
}

test "security: NumPy truncated version-two headers return an error without trapping" {
    const bytes = [_]u8{ 0x93, 'N', 'U', 'M', 'P', 'Y', 2, 0, 0, 0 };
    try std.testing.expectError(error.BadHeader, npy.parse(&bytes));
}

test "security: NumPy shape rank is bounded before writing dimensions" {
    const bytes = try image(std.testing.allocator, "1,1,1,1,1,1,1,1,1", 2);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(error.RankTooHigh, npy.parse(bytes));
}

test "security: NumPy shape products and truncated tensor payloads are refused" {
    const a = std.testing.allocator;
    const short = try image(a, "8,8", 2);
    defer a.free(short);
    try std.testing.expectError(error.BadPayload, npy.parse(short));
    const huge = try image(a, "18446744073709551615,2", 0);
    defer a.free(huge);
    try std.testing.expectError(error.BadHeader, npy.parse(huge));
}

test "NumPy streamed headers remain available without admitting an incomplete whole array" {
    const a = std.testing.allocator;
    const bytes = try image(a, "2,3", 12);
    defer a.free(bytes);
    const head = bytes[0 .. bytes.len - 12];
    const parsed = try npy.parseHeader(head);
    try std.testing.expectEqual(@as(usize, 6), parsed.count());
    try std.testing.expectEqual(@as(usize, 0), parsed.data.len);
    try std.testing.expectError(error.BadPayload, npy.parse(head));
    const modern = try a.alloc(u8, bytes.len + 2);
    defer a.free(modern);
    @memcpy(modern[0..8], bytes[0..8]);
    std.mem.writeInt(u32, modern[8..12], std.mem.readInt(u16, bytes[8..10], .little), .little);
    @memcpy(modern[12..], bytes[10..]);
    for ([_]u8{ 2, 3 }) |version| {
        modern[6] = version;
        const array = try npy.parse(modern);
        try std.testing.expectEqual(@as(usize, 6), array.count());
        try std.testing.expectEqual(@as(usize, 12), array.data.len);
    }
}
