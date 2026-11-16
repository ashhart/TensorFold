//! Page image sizes from headers alone: PNG, baseline JPEG, and WebP's three header forms.
const std = @import("std");
const size = @import("size.zig");
const harness = @import("harness_test.zig");
const t = std.testing;

/// A RIFF WEBP file of one chunk: its four-character code, then ``data`` (padded to 18 bytes).
fn webp(comptime code: *const [4]u8, data: []const u8) [30]u8 {
    var b: [30]u8 = @splat(0);
    @memcpy(b[0..4], "RIFF");
    @memcpy(b[8..12], "WEBP");
    @memcpy(b[12..16], code);
    @memcpy(b[20..][0..data.len], data);
    return b;
}

test "ocr size: each header form names its width and height; others and zero sides name none" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try t.expectEqual(@as(?[2]u32, .{ 300, 7 }), size.of(try harness.png(arena.allocator(), 300, 7)));
    // SOI, then a baseline frame header: length 17, precision 8, height 0x0102, width 0x0304
    const jpeg = "\xff\xd8\xff\xc0\x00\x11\x08\x01\x02\x03\x04\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01";
    try t.expectEqual(@as(?[2]u32, .{ 0x0304, 0x0102 }), size.of(jpeg));
    // VP8X: flags and reserved bytes, then width - 1 and height - 1 as 24-bit little-endian
    try t.expectEqual(@as(?[2]u32, .{ 640, 70000 }), size.of(&webp("VP8X", &.{ 0, 0, 0, 0, 0x7f, 0x02, 0x00, 0x6f, 0x11, 0x01 })));
    // VP8: frame tag, start code, then 14-bit sides (the top two bits are scale)
    try t.expectEqual(@as(?[2]u32, .{ 400, 300 }), size.of(&webp("VP8 ", &.{ 0, 0, 0, 0x9d, 0x01, 0x2a, 0x90, 0x41, 0x2c, 0x01 })));
    // VP8L: signature, then (width - 1) | (height - 1) << 14 in 28 bits
    const bits: u32 = (100 - 1) | ((50 - 1) << 14);
    var l: [5]u8 = undefined;
    l[0] = 0x2f;
    std.mem.writeInt(u32, l[1..5], bits, .little);
    try t.expectEqual(@as(?[2]u32, .{ 100, 50 }), size.of(&webp("VP8L", &l)));
    try t.expectEqual(@as(?[2]u32, null), size.of(&webp("VP8 ", &.{ 0, 0, 0, 1, 2, 3, 0x90, 0x01, 0x2c, 0x01 })));
    try t.expectEqual(@as(?[2]u32, null), size.of(&webp("ALPH", &.{})));
    try t.expectEqual(@as(?[2]u32, null), size.of("\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR\x00\x00\x00\x00\x00\x00\x00\x01"));
    try t.expectEqual(@as(?[2]u32, null), size.of("\x89PNG\r\n\x1a\n"));
    try t.expectEqual(@as(?[2]u32, null), size.of("GIF89a......"));
}
