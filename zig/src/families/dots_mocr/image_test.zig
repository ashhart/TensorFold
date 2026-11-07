//! Merge order, processor admission, content identity and budget failures for dots.mocr images.
const std = @import("std");
const image = @import("image.zig");
const cfg = @import("config.zig");
const cases = @import("config_test.zig");
const picture = @import("core").picture;
const sample = @embedFile("fixtures/preprocessor_config.json");

fn processor(a: std.mem.Allocator) !image.Processor {
    return image.Processor.parse(a, sample, try cfg.Config.parse(a, cases.sample, cases.generation));
}

fn pixelHash(pixels: []const f32) [64]u8 {
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (pixels) |v| {
        var le: [4]u8 = undefined;
        std.mem.writeInt(u32, &le, @bitCast(v), .little);
        hash.update(&le);
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

test "dots: native JPEG preparation equals the Hugging Face demo's pixels" {
    const path = @import("sys").getenv("TF_DOTS_DEMO_IMAGE") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const Manifest = struct { image_sha256: []const u8, pixel_sha256: []const u8, grid: [3]u32, shape: [2]usize };
    const manifest = try std.json.parseFromSlice(Manifest, a, @embedFile("fixtures/demo-reference.json"), .{ .ignore_unknown_fields = true });
    defer manifest.deinit();
    const want = manifest.value;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(32 << 20));
    defer a.free(bytes);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    try std.testing.expectEqualStrings(want.image_sha256, &std.fmt.bytesToHex(hash, .lower));
    const p = try processor(a);
    const prepared = try p.prepare(std.testing.io, a, bytes, 14400);
    defer a.free(prepared.pixels);
    try std.testing.expectEqual(want.grid, [3]u32{ 1, prepared.gh, prepared.gw });
    try std.testing.expectEqual(want.shape[0] * want.shape[1], prepared.pixels.len);
    try std.testing.expectEqualStrings(want.pixel_sha256, &pixelHash(prepared.pixels));
}

test "dots: processor pixels equal pinned Transformers 4.51.0 for every reference image" {
    const a = std.testing.allocator;
    const Case = struct { height: usize, width: usize, seed: usize, grid: [3]u32, sha256: []const u8 };
    const Manifest = struct { cases: []Case };
    const manifest = try std.json.parseFromSlice(Manifest, a, @embedFile("fixtures/processor-reference.json"), .{ .ignore_unknown_fields = true });
    defer manifest.deinit();
    const p = try processor(a);
    for (manifest.value.cases) |k| {
        const rgb = try a.alloc(u8, k.height * k.width * 3);
        defer a.free(rgb);
        for (0..k.height) |y| for (0..k.width) |x| for (0..3) |ch| {
            rgb[(y * k.width + x) * 3 + ch] = @intCast((7 * x + 13 * y + 50 * ch + (x * y) % 31 + k.seed) % 256);
        };
        const prepared = try p.prepareRgb(a, .{ .width = k.width, .height = k.height, .pixels = rgb }, k.seed, 14400);
        defer a.free(prepared.pixels);
        try std.testing.expectEqual(k.grid, [3]u32{ 1, prepared.gh, prepared.gw });
        try std.testing.expectEqualStrings(k.sha256, &pixelHash(prepared.pixels));
    }
}

test "dots: slow processor and axial positions keep the four-patch merge order" {
    const a = std.testing.allocator;
    const p = try processor(a);
    try std.testing.expectEqual(.pillow, p.patches.resize);
    try std.testing.expect(!p.patches.fused);
    try std.testing.expectEqual(@as(u32, 588), p.patches.width());
    const coords = try image.positions(a, 2, 4, 2);
    defer a.free(coords);
    const want = [_][2]u32{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 } };
    try std.testing.expectEqualSlices([2]u32, &want, coords);
    const rgb = try a.alloc(u8, 56 * 84 * 3);
    defer a.free(rgb);
    for (0..56) |y| for (0..84) |x| for (0..3) |ch| {
        rgb[(y * 84 + x) * 3 + ch] = @intCast((y / 14) * 4 + x / 14 + ch * 50);
    };
    const prepared = try p.prepareRgb(a, .{ .height = 56, .width = 84, .pixels = rgb }, 7, 14400);
    defer a.free(prepared.pixels);
    const pos = try image.positions(a, prepared.gh, prepared.gw, 2);
    defer a.free(pos);
    const table = picture.normalize.table(.unfused, p.patches.rescale, p.patches.mean, p.patches.std);
    for (pos, 0..) |yx, row| for (0..3) |ch| {
        const value = yx[0] * 4 + yx[1] + ch * 50;
        for (prepared.pixels[row * 588 + ch * 196 ..][0..196]) |v| try std.testing.expectEqual(table[ch][value], v);
    };
}

test "dots: reject incompatible preprocessing, malformed images and impossible grids" {
    const a = std.testing.allocator;
    const c = try cfg.Config.parse(a, cases.sample, cases.generation);
    const fast = try std.mem.replaceOwned(u8, a, sample, "Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast");
    defer a.free(fast);
    try std.testing.expectError(error.UnsupportedDotsConfig, image.Processor.parse(a, fast, c));
    const p = try processor(a);
    var px: [56 * 56 * 3]u8 = @splat(0);
    const rgb: picture.Rgb = .{ .height = 56, .width = 56, .pixels = &px };
    try std.testing.expectError(error.DotsImageBudget, p.prepareRgb(a, rgb, 0, 0));
    try std.testing.expectError(error.DotsImageSize, p.prepareRgb(a, .{ .height = 14, .width = 56, .pixels = &px }, 0, 100));
    try std.testing.expectError(error.Invalid, p.prepareRgb(a, .{ .height = 56, .width = 56, .pixels = px[0..3] }, 0, 100));
    try std.testing.expectError(error.DotsGrid, image.positions(a, 3, 4, 2));
    try std.testing.expectError(error.DotsGrid, image.positions(a, 2, 4, 0));
    const one = try p.prepareRgb(a, rgb, 10, 100);
    defer a.free(one.pixels);
    const two = try p.prepareRgb(a, rgb, 11, 100);
    defer a.free(two.pixels);
    try std.testing.expect(one.hash != two.hash);
}
