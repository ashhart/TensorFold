//! Mini's image preprocessing from PNG/JPEG against the HF formulas (pad, rescale, normalize, 40-pixel patch tokens).
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const image = @import("metal_vision_image.zig");

const t = std.testing;

const png_rgba = @embedFile("fixtures/media_rgba.png");
const raw_rgba = @embedFile("fixtures/media_rgba.raw"); // 45 x 37, four bytes a pixel
const png_rgb = @embedFile("fixtures/media_rgb.png");
const raw_rgb = @embedFile("fixtures/media_rgb.raw"); // 41 x 43
const jpeg_photo = @embedFile("fixtures/media_photo.jpg");
const raw_photo = @embedFile("fixtures/media_photo.raw"); // 50 x 30

/// convert_to_rgb of an RGBA uint8 image (integer arithmetic on int32, as the processor's).
fn composite(a: std.mem.Allocator, rgba: []const u8, w: usize, h: usize) ![]u8 {
    const out = try a.alloc(u8, w * h * 3);
    for (0..h) |y| for (0..w) |x| {
        const alpha: i32 = rgba[(y * w + x) * 4 + 3];
        const bg: i32 = if (((y / 8) + (x / 8)) % 2 == 0) 204 else 153;
        for (0..3) |c| {
            const v: i32 = rgba[(y * w + x) * 4 + c];
            out[(y * w + x) * 3 + c] = @intCast(@divFloor(v * alpha + bg * (255 - alpha) + 127, 255));
        }
    };
    return out;
}

/// pixel_values of one image as the model's patch embedding reads them: [tokens, merge row, merge column, 20, 20, 3].
fn oracle(a: std.mem.Allocator, rgb: []const u8, w: usize, h: usize) ![]f32 {
    const rows = (h + 39) / 40;
    const cols = (w + 39) / 40;
    // padded [3][rows * 40][cols * 40] on the 0..255 scale, filled with -1
    const hh = rows * 40;
    const ww = cols * 40;
    const padded = try a.alloc(f32, 3 * hh * ww);
    @memset(padded, -1.0);
    for (0..3) |c| for (0..h) |y| for (0..w) |x| {
        padded[(c * hh + y) * ww + x] = @floatFromInt(rgb[(y * w + x) * 3 + c]);
    };
    for (padded) |*v| v.* = (v.* * 0.00392156862745098 - 0.5) / 0.5;
    const out = try a.alloc(f32, rows * cols * 4 * 20 * 20 * 3);
    var at: usize = 0;
    // image.reshape(C, rows, 40, cols, 40).permute(1, 3, 2, 4, 0): token (row, col) holds [40, 40, 3] ...
    for (0..rows) |tr| for (0..cols) |tc| {
        // ... which reshape(-1, 2, 20, 2, 20, 3).transpose(2, 3) reorders to [merge row, merge col, 20, 20, 3]
        for (0..2) |mr| for (0..2) |mc| for (0..20) |py| for (0..20) |px| for (0..3) |c| {
            out[at] = padded[(c * hh + tr * 40 + mr * 20 + py) * ww + tc * 40 + mc * 20 + px];
            at += 1;
        };
    };
    return out;
}

fn expectPixels(want: []const f32, got: []const f32, tol: f32) !void {
    try t.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| if (!std.math.isFinite(g) or @abs(w - g) > tol) {
        std.debug.print("pixel {d}: want {d}, got {d}\n", .{ i, w, g });
        return error.TestExpectedEqual;
    };
}

test "an RGB PNG becomes the processor's patches: whole cells, -1 padding, merge-block order" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try image.prepare(t.io, a, png_rgb, 16);
    try t.expectEqual(@as(u32, 2), p.gh);
    try t.expectEqual(@as(u32, 2), p.gw);
    try t.expectEqual(@as(u32, 4), p.tokens);
    try t.expectEqual(@as(usize, image.tokensOf(41, 43)), p.tokens);
    try expectPixels(try oracle(a, raw_rgb, 41, 43), p.pixels, 1e-6);
}

test "an RGBA PNG is composited over the 8-pixel 204/153 checkerboard before padding" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try image.prepare(t.io, a, png_rgba, 16);
    try t.expectEqual(@as(u32, 1), p.gh);
    try t.expectEqual(@as(u32, 2), p.gw);
    const rgb = try composite(a, raw_rgba, 45, 37);
    try expectPixels(try oracle(a, rgb, 45, 37), p.pixels, 1e-6);
    // a fully transparent pixel shows the checkerboard itself, never the colour under the alpha
    try t.expectEqual(@as(u8, 0), raw_rgba[3]);
    try t.expectApproxEqAbs((@as(f32, 204) / 255.0 - 0.5) / 0.5, p.pixels[0], 1e-6);
}

test "a JPEG decodes through the native path to the same grid and pixels within the decoder's rounding" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = image.prepare(t.io, a, jpeg_photo, 16) catch |err| switch (err) {
        error.NoJpegLibrary, error.JpegLibraryVersion => return error.SkipZigTest,
        else => return err,
    };
    try t.expectEqual(@as(u32, 1), p.gh);
    try t.expectEqual(@as(u32, 2), p.gw);
    // Pillow's libjpeg and the engine's may differ by a few levels of IDCT rounding; padding and layout must not
    try expectPixels(try oracle(a, raw_photo, 50, 30), p.pixels, 4.0 / 127.5);
}

test "JPEG EXIF orientation changes the grid and turns the samples before patchification" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exif = "\xff\xe1\x00\x22Exif\x00\x00II*\x00\x08\x00\x00\x00" ++
        "\x01\x00\x12\x01\x03\x00\x01\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00";
    const turned_jpeg = try std.mem.concat(a, u8, &.{ jpeg_photo[0..2], exif, jpeg_photo[2..] });
    const p = image.prepare(t.io, a, turned_jpeg, 16) catch |err| switch (err) {
        error.NoJpegLibrary, error.JpegLibraryVersion => return error.SkipZigTest,
        else => return err,
    };
    try t.expectEqual(@as(u32, 2), p.gh);
    try t.expectEqual(@as(u32, 1), p.gw);
    const rotated = try a.alloc(u8, raw_photo.len);
    for (0..30) |y| for (0..50) |x| {
        @memcpy(rotated[(x * 30 + (29 - y)) * 3 ..][0..3], raw_photo[(y * 50 + x) * 3 ..][0..3]);
    };
    try expectPixels(try oracle(a, rotated, 30, 50), p.pixels, 4.0 / 127.5);
}

test "an image over the token budget is resized to fit and keeps the grid it reports; a budget of 0 refuses" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const one = try image.prepare(t.io, a, png_rgb, 1);
    try t.expectEqual(@as(u32, 1), one.tokens);
    try t.expectEqual(@as(usize, image.token_width), one.pixels.len);
    const pad: f32 = -1.0 - 2.0 / 255.0;
    var saw_pad = false;
    for (one.pixels) |v| {
        try t.expect(v >= pad - 1e-6 and v <= 1.0);
        saw_pad = saw_pad or @abs(v - pad) < 1e-6;
    }
    try t.expect(saw_pad);
    const whole = try image.prepare(t.io, a, png_rgb, 16);
    try t.expect(one.hash != whole.hash); // the hash covers the grid, so cached rows are never reused across budgets
    const again = try image.prepare(t.io, a, png_rgb, 16);
    try t.expectEqual(whole.hash, again.hash);
    try t.expectError(error.NoImageBudget, image.prepare(t.io, a, png_rgb, 0));
}

test "bytes that are not a supported picture are refused with words for the client" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectError(error.Invalid, image.prepare(t.io, a, "not a picture", 16));
    try t.expectError(error.Invalid, image.prepare(t.io, a, "", 16));
    // a PNG cut in the middle of its pixel data
    try t.expect(std.meta.isError(image.prepare(t.io, a, png_rgb[0 .. png_rgb.len / 2], 16)));
    try t.expect(image.problem(error.NoImageBudget) != null);
    try t.expect(image.problem(error.Invalid) == null); // the server words Invalid itself
}
