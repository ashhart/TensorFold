const std = @import("std");
const image = @import("image.zig");
const Config = @import("config.zig").Config;
const a = std.testing.allocator;

fn config() !Config {
    return Config.parse(a, @embedFile("fixtures/config.json"));
}

test "LightOnOCR resize floors the longer side to 1540, then rounds each side up to 28" {
    const cases = [_][4]u32{
        .{ 1540, 1540, 1540, 1540 },
        .{ 1000, 2000, 784, 1540 },
        .{ 2000, 1000, 1540, 784 },
        .{ 900, 1280, 924, 1288 },
        .{ 29, 29, 56, 56 },
        .{ 1541, 1541, 1540, 1540 },
        .{ 28, 28, 28, 28 },
    };
    for (cases) |c| try std.testing.expectEqual(image.Size{ .h = c[2], .w = c[3] }, try image.targetSize(c[0], c[1], 1540));
    try std.testing.expectEqual(@as(u32, 3025), image.tokens(110, 110));
    try std.testing.expectEqual(@as(u32, 1540), image.tokens(56, 110));
    try std.testing.expectError(error.ImageTooSmall, image.targetSize(0, 100, 1540));
    try std.testing.expectError(error.ImageTooSmall, image.targetSize(1, 1000000, 1540));
}

test "LightOnOCR resize equals sizes and image tokens measured from the checkpoint processor" {
    // The full 1B processor (text and image) run as a black box: h, w, resized h, w, `<|image_pad|>` count.
    const cases = [_][5]u32{
        .{ 33, 3388, 28, 1540, 55 },      .{ 66, 6776, 28, 1540, 55 },       .{ 162, 16632, 28, 1540, 55 },
        .{ 2047, 12460, 252, 1540, 495 }, .{ 414, 2520, 252, 1540, 495 },    .{ 18172, 177, 1540, 28, 55 },
        .{ 1, 1, 28, 28, 1 },             .{ 10, 20, 28, 28, 1 },            .{ 1, 1540, 28, 1540, 55 },
        .{ 3, 4620, 28, 1540, 55 },       .{ 14000, 6500, 1540, 728, 1430 }, .{ 15, 2000, 28, 1540, 55 },
        .{ 1000, 2000, 784, 1540, 1540 }, .{ 900, 1280, 924, 1288, 1518 },   .{ 29, 29, 56, 56, 4 },
    };
    for (cases) |c| {
        const size = try image.targetSize(c[0], c[1], 1540);
        try std.testing.expectEqual(image.Size{ .h = c[2], .w = c[3] }, size);
        try std.testing.expectEqual(c[4], image.tokens(size.h / image.patch, size.w / image.patch));
    }
    // The processor refuses these: the shorter side floors to zero.
    for ([_][2]u32{ .{ 1, 1541 }, .{ 1541, 1 }, .{ 2, 3081 }, .{ 3, 4621 } }) |c|
        try std.testing.expectError(error.ImageTooSmall, image.targetSize(c[0], c[1], 1540));
}

test "LightOnOCR processor settings match the published checkpoint and reject incompatible arithmetic" {
    const c = try config();
    const sample = @embedFile("fixtures/processor_config.json");
    try image.validateProcessor(a, sample, c);
    const changes = [_][2][]const u8{
        .{ "\"patch_size\": 14", "\"patch_size\": 16" },
        .{ "\"longest_edge\": 1540", "\"longest_edge\": 1024" },
        .{ "\"resample\": 3", "\"resample\": 2" },
        .{ "\"do_normalize\": true", "\"do_normalize\": false" },
        .{ "\"do_center_crop\": null", "\"do_center_crop\": true" },
        .{ "\"do_pad\": null", "\"do_pad\": true" },
        .{ "\"channels_first\"", "\"channels_last\"" },
        .{ "\"PixtralImageProcessorFast\"", "\"Qwen2VLImageProcessor\"" },
        .{ "0.48145466", "0.5" },
        .{ "\"<|image_pad|>\"", "\"<image>\"" },
    };
    for (changes) |change| {
        const changed = try std.mem.replaceOwned(u8, a, sample, change[0], change[1]);
        defer a.free(changed);
        try std.testing.expectError(error.UnsupportedLightOnProcessor, image.validateProcessor(a, changed, c));
    }
    try std.testing.expectError(error.BadLightOnProcessor, image.validateProcessor(a, "[]", c));
}

/// Raster-ordered patch colors distinguish row order from Qwen's block/merge order without depending on resize.
fn pixels() [56 * 56 * 3]u8 {
    var rgb: [56 * 56 * 3]u8 = undefined;
    for (0..56) |y| for (0..56) |x| for (0..3) |ch| {
        rgb[(y * 56 + x) * 3 + ch] = @intCast((y / 14 * 4 + x / 14) * 13 + ch * 9);
    };
    return rgb;
}

test "LightOnOCR patches stay raster ordered and channel-first" {
    var rgb = pixels();
    const p = try image.prepareRgb(a, .{ .height = 56, .width = 56, .pixels = &rgb }, 11, try config(), 4);
    defer a.free(p.pixels);
    try std.testing.expectEqual(@as(u32, 4), p.gh);
    try std.testing.expectEqual(@as(u32, 4), p.gw);
    try std.testing.expectEqual(@as(u32, 4), p.tokens);
    try std.testing.expectEqual(@as(usize, 16 * 588), p.pixels.len);
    for (0..16) |patch_index| for (0..3) |ch| {
        const value: f32 = @floatFromInt(patch_index * 13 + ch * 9);
        const by: f32 = @floatCast(@as(f64, image.mean[ch]) / image.rescale);
        const scale: f32 = @floatCast(@as(f64, image.sd[ch]) / image.rescale);
        const expected = (value - by) / scale;
        try std.testing.expectEqual(expected, p.pixels[patch_index * 588 + ch * 196]);
        try std.testing.expectEqual(expected, p.pixels[patch_index * 588 + ch * 196 + 195]);
    };
}

test "LightOnOCR refuses too few image tokens and never changes resolution to fit" {
    var rgb = pixels();
    const c = try config();
    const source: image.Rgb = .{ .height = 56, .width = 56, .pixels = &rgb };
    try std.testing.expectError(error.ImageTokensExceeded, image.prepareRgb(a, source, 11, c, 3));
    try std.testing.expectError(error.ImageTokensExceeded, image.prepareRgb(a, source, 11, c, 0));
    var thin: [1541 * 3]u8 = @splat(0);
    try std.testing.expectError(error.ImageTooSmall, image.prepareRgb(a, .{ .height = 1, .width = 1541, .pixels = &thin }, 11, c, 4096));
    var tiny: [14 * 28 * 3]u8 = @splat(0);
    try std.testing.expectError(error.Invalid, image.prepareRgb(a, .{ .height = 56, .width = 56, .pixels = &tiny }, 11, c, 4096));
    try std.testing.expectError(error.Oversized, image.prepareRgb(a, .{ .height = std.math.maxInt(usize), .width = std.math.maxInt(usize), .pixels = &tiny }, 11, c, 4096));
}

test "LightOnOCR identity separates image bytes and grids while equal budgets keep the same image" {
    var rgb = pixels();
    const c = try config();
    const source: image.Rgb = .{ .height = 56, .width = 56, .pixels = &rgb };
    const first = try image.prepareRgb(a, source, 11, c, 4);
    defer a.free(first.pixels);
    const same = try image.prepareRgb(a, source, 11, c, 4096);
    defer a.free(same.pixels);
    const other = try image.prepareRgb(a, source, 12, c, 4096);
    defer a.free(other.pixels);
    try std.testing.expectEqual(first.hash, same.hash);
    try std.testing.expect(first.hash != other.hash);
    try std.testing.expectEqualSlices(f32, first.pixels, same.pixels);
    try std.testing.expect(first.hash < 1 << 62);
}

test "LightOnOCR image preparation releases temporary buffers on allocation failures" {
    try std.testing.checkAllAllocationFailures(a, prepareWithFailures, .{try config()});
}

test "LightOnOCR decodes encoded RGB images and rejects invalid encoded bytes" {
    const c = try config();
    const p = try image.prepare(std.testing.io, a, @embedFile("fixtures/solid.png"), c, 1);
    defer a.free(p.pixels);
    try std.testing.expectEqual(@as(u32, 2), p.gh);
    try std.testing.expectEqual(@as(u32, 2), p.gw);
    try std.testing.expectEqual(@as(u32, 1), p.tokens);
    try std.testing.expectEqual(@as(usize, 4 * 588), p.pixels.len);
    try std.testing.expectError(error.Invalid, image.prepare(std.testing.io, a, "not an image", c, 4096));
}

fn prepareWithFailures(gpa: std.mem.Allocator, c: Config) !void {
    var rgb = pixels();
    const p = try image.prepareRgb(gpa, .{ .height = 42, .width = 56, .pixels = rgb[0 .. 42 * 56 * 3] }, 11, c, 4096);
    gpa.free(p.pixels);
}
