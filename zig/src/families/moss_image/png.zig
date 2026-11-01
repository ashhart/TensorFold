//! An 8-bit RGB picture as a PNG: one IDAT of zlib-compressed rows, each with filter 0 (none).
const std = @import("std");
const flate = std.compress.flate;

/// The PNG of `rgb` [height, width, 3], allocated with `a`.
pub fn encode(a: std.mem.Allocator, rgb: []const u8, width: usize, height: usize) ![]u8 {
    if (width == 0 or height == 0 or rgb.len != width * height * 3) return error.PictureSize;
    if (width > std.math.maxInt(u31) or height > std.math.maxInt(u31)) return error.PictureSize;
    var z: std.Io.Writer.Allocating = try .initCapacity(a, rgb.len / 2 + 64);
    defer z.deinit();
    var window: [flate.max_window_len]u8 = undefined;
    var c = try flate.Compress.init(&z.writer, &window, .zlib, .default);
    const row = width * 3;
    for (0..height) |y| {
        try c.writer.writeByte(0);
        try c.writer.writeAll(rgb[y * row ..][0..row]);
    }
    try c.finish();
    var out: std.Io.Writer.Allocating = .init(a);
    errdefer out.deinit();
    try out.writer.writeAll("\x89PNG\r\n\x1a\n");
    var header: [13]u8 = undefined;
    std.mem.writeInt(u32, header[0..4], @intCast(width), .big);
    std.mem.writeInt(u32, header[4..8], @intCast(height), .big);
    header[8..13].* = .{ 8, 2, 0, 0, 0 };
    try chunk(&out.writer, "IHDR", &header);
    try chunk(&out.writer, "IDAT", z.written());
    try chunk(&out.writer, "IEND", "");
    return out.toOwnedSlice();
}

fn chunk(w: *std.Io.Writer, kind: *const [4]u8, data: []const u8) !void {
    try w.writeInt(u32, @intCast(data.len), .big);
    try w.writeAll(kind);
    try w.writeAll(data);
    var crc = std.hash.Crc32.init();
    crc.update(kind);
    crc.update(data);
    try w.writeInt(u32, crc.final(), .big);
}

test "a PNG of a small picture decodes back to its pixels" {
    const a = std.testing.allocator;
    var rgb: [5 * 3 * 3]u8 = undefined;
    for (&rgb, 0..) |*v, i| v.* = @truncate(i * 37);
    const png = try encode(a, &rgb, 5, 3);
    defer a.free(png);
    const picture = @import("../../core/picture/png.zig");
    const s = try picture.decode(a, png);
    defer a.free(s.data);
    try std.testing.expectEqual(@as(usize, 5), s.width);
    try std.testing.expectEqual(@as(usize, 3), s.height);
    for (0..15) |i| try std.testing.expectEqualSlices(u8, rgb[i * 3 ..][0..3], s.data[i * s.channels ..][0..3]);
    try std.testing.expectError(error.PictureSize, encode(a, rgb[0..10], 5, 3));
}
