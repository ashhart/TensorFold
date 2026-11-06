//! The resize and normalization against the 1B processor's own pixel bytes on 40 synthetic pages.
const std = @import("std");
const image = @import("image.zig");
const Config = @import("config.zig").Config;

/// Rows "h w CxHxW sha256" from `tools/zig/lighton_ocr_pixels.py`: the processor's CHW f32 pixels.
const expected = @embedFile("fixtures/processor_pixels.txt");

/// `page` in `tools/zig/lighton_ocr_pixels.py`: gradients plus hashed noise, in wrapping u32 arithmetic.
fn page(gpa: std.mem.Allocator, h: u32, w: u32) ![]u8 {
    const rgb = try gpa.alloc(u8, @as(usize, h) * w * 3);
    for (0..h) |y| for (0..w) |x| {
        const gy: i64 = @intCast(@as(u32, @intCast(y)) * 255 / @max(h - 1, 1));
        const gx: i64 = @intCast(@as(u32, @intCast(x)) * 255 / @max(w - 1, 1));
        const base = [3]i64{ @divFloor(gy + gx, 2), @intCast(@abs(gy - gx)), 255 - @divFloor(gy * 7 + gx * 3, 10) };
        for (0..3) |ch| {
            var k: u32 = (@as(u32, @intCast(y)) *% w +% @as(u32, @intCast(x))) *% 3 +% @as(u32, @intCast(ch));
            k *%= 0x9E3779B1;
            k ^= k >> 15;
            k *%= 0x85EBCA6B;
            k ^= k >> 13;
            const noise: i64 = @as(i64, k & 63) - 32;
            rgb[(y * w + x) * 3 + ch] = @intCast(std.math.clamp(base[ch] + noise, 0, 255));
        }
    };
    return rgb;
}

test "LightOnOCR prepared pixels equal the 1B processor's bytes on 40 pages" {
    const a = std.testing.allocator;
    const c = try Config.parse(a, @embedFile("fixtures/config.json"));
    var rows: usize = 0;
    var lines = std.mem.tokenizeScalar(u8, expected, '\n');
    while (lines.next()) |line| : (rows += 1) {
        var cols = std.mem.tokenizeScalar(u8, line, ' ');
        const h = try std.fmt.parseInt(u32, cols.next().?, 10);
        const w = try std.fmt.parseInt(u32, cols.next().?, 10);
        const shape = cols.next().?;
        const want = cols.next().?;
        const rgb = try page(a, h, w);
        defer a.free(rgb);
        const p = try image.prepareRgb(a, .{ .height = h, .width = w, .pixels = rgb }, 11, c, 1 << 20);
        defer a.free(p.pixels);
        const H = p.gh * image.patch;
        const W = p.gw * image.patch;
        var buf: [32]u8 = undefined;
        try std.testing.expectEqualStrings(shape, try std.fmt.bufPrint(&buf, "3x{d}x{d}", .{ H, W }));
        const chw = try a.alloc(f32, 3 * @as(usize, H) * W);
        defer a.free(chw);
        for (0..p.gh) |py| for (0..p.gw) |px| for (0..3) |ch| for (0..image.patch) |y| for (0..image.patch) |x| {
            const from = (py * p.gw + px) * image.patch_width + (ch * image.patch + y) * image.patch + x;
            chw[(ch * H + py * image.patch + y) * W + px * image.patch + x] = p.pixels[from];
        };
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(chw), &digest, .{});
        try std.testing.expectEqualStrings(want, &std.fmt.bytesToHex(digest, .lower));
    }
    try std.testing.expectEqual(@as(usize, 40), rows);
}
