//! Mini's image preprocessing from PNG/JPEG against the HF formulas (pad, rescale, normalize, 40-pixel patch tokens).
const std = @import("std");
const core = @import("core");
const config = @import("../config.zig");
const image = @import("vision_image.zig");
const images = @import("images.zig");
const tables = @import("vision_tables.zig");

const picture = core.picture;
const t = std.testing;

const png_rgba = @embedFile("../fixtures/media_rgba.png");
const raw_rgba = @embedFile("../fixtures/media_rgba.raw"); // 45 x 37, four bytes a pixel
const png_rgb = @embedFile("../fixtures/media_rgb.png");
const raw_rgb = @embedFile("../fixtures/media_rgb.raw"); // 41 x 43
const jpeg_photo = @embedFile("../fixtures/media_photo.jpg");
const raw_photo = @embedFile("../fixtures/media_photo.raw"); // 50 x 30

pub fn rgbImage(a: std.mem.Allocator, w: usize, h: usize) !picture.Rgb {
    const px = try a.alloc(u8, w * h * 3);
    for (0..h) |y| for (0..w) |x| {
        px[(y * w + x) * 3] = @intCast(x * 255 / @max(w - 1, 1));
        px[(y * w + x) * 3 + 1] = @intCast(y * 255 / @max(h - 1, 1));
        px[(y * w + x) * 3 + 2] = @intCast((x + y) % 256);
    };
    return .{ .width = w, .height = h, .pixels = px };
}

test "the processor pads to 40-pixel cells with -1, normalizes to [-1, 1], and orders merge blocks then patches" {
    try t.expectEqual(@as(usize, 2), image.tokensOf(41, 40));
    try t.expectEqual(@as(usize, 1), image.tokensOf(40, 40));
    try t.expectEqual(@as(usize, 6), image.tokensOf(81, 80));
    var rgb = try rgbImage(t.allocator, 50, 30);
    defer rgb.deinit(t.allocator);
    const p = try image.fromRgb(t.allocator, rgb, 7, 16);
    defer t.allocator.free(p.pixels);
    try t.expectEqual(@as(u32, 1), p.gh);
    try t.expectEqual(@as(u32, 2), p.gw);
    try t.expectEqual(@as(usize, 2 * image.token_width), p.pixels.len);
    // token 0, patch 0 (rows 0..19, cols 0..19), pixel (0, 0) channel 0: byte 0 -> -1
    try t.expectEqual(@as(f32, -1.0), p.pixels[0]);
    // token 0, patch 1 = columns 20.., pixel (0, 0): x = 20 of 50 -> byte 20 * 255 / 49
    const byte: f32 = @floatFromInt(20 * 255 / 49);
    try t.expectApproxEqAbs(byte / 255.0 * 2 - 1, p.pixels[image.patch_width], 1e-6);
    // token 0, patch 2 starts at row 20 of a 30-high image: its rows 10.. are padding, (-1 / 255 - 0.5) / 0.5
    const pad: f32 = (-1.0 / 255.0 - 0.5) / 0.5;
    try t.expectApproxEqAbs(pad, p.pixels[2 * image.patch_width + (10 * image.patch + 0) * 3], 1e-6);
    // the same image under a one-token budget is resized to fit, aspect kept
    const small = try image.fromRgb(t.allocator, rgb, 7, 1);
    defer t.allocator.free(small.pixels);
    try t.expectEqual(@as(u32, 1), small.tokens);
    try t.expect(small.hash != p.hash);
    try t.expectError(error.NoImageBudget, image.fromRgb(t.allocator, rgb, 7, 0));
    try t.expect(image.problem(error.NoImageBudget) != null);
}

test "position taps are the half-pixel bilinear grid and the rotary table turns columns on the first frequencies" {
    const hd = 8;
    const n = comptime tables.patches(1, 2);
    var taps: [n * 4]u32 = undefined;
    var wts: [n * 4]f32 = undefined;
    var cs: [(n + 1) * hd]f32 = undefined;
    var sn: [(n + 1) * hd]f32 = undefined;
    tables.fill(1, 2, hd, &taps, &wts, &cs, &sn);
    // patch 0 sits at (0, 0) of a 2 x 4 patch grid: row source 7.5 on the 32-cell table, column source 3.5
    try t.expectEqual([4]u32{ 7 * 32 + 3, 7 * 32 + 4, 8 * 32 + 3, 8 * 32 + 4 }, taps[0..4].*);
    try t.expectEqual([4]f32{ 0.25, 0.25, 0.25, 0.25 }, wts[0..4].*);
    try t.expectEqual(@as(f32, 1.0), cs[0]);
    try t.expectEqual(@as(f32, 0.0), sn[0]);
    // patch 1 (row 0, column 1): positions (1, 2); pair 0 turns with a column frequency, pair 2 with a row one
    try t.expectApproxEqAbs(@cos(@as(f32, 2.0)), cs[2 * hd], 1e-6);
    try t.expectApproxEqAbs(@cos(@as(f32, 1.0)), cs[2 * hd + 4], 1e-6);
    try t.expectApproxEqAbs(@sin(@as(f32, 2.0)), sn[2 * hd + 1], 1e-6);
    var weight_sum: f32 = 0;
    for (wts[4..8]) |w| weight_sum += w;
    try t.expectApproxEqAbs(@as(f32, 1.0), weight_sum, 1e-5);
}

test "Mini's vision config is the tower this engine runs: 23 layers, 16 heads of 64, patch 20, merge 2, 1e-5 norms" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = (try config.parse(arena.allocator(), @embedFile("../fixtures/mini_config.json"), &why)).vision;
    try t.expectEqual(@as(u32, 1024), c.hidden);
    try t.expectEqual(@as(u32, 23), c.layers);
    try t.expectEqual(@as(u32, 16), c.heads);
    try t.expectEqual(@as(u32, image.patch), c.patch);
    try t.expectEqual(@as(u32, image.merge), c.merge);
    try t.expectEqual(@as(u32, image.grid), c.grid);
    try t.expectEqual(@as(u32, 4), c.mlp_ratio);
    try t.expectEqual(@as(u32, 4096), c.proj_hidden);
    try t.expectEqual(@as(u32, 2304), c.text_hidden);
    try t.expectEqual(@as(usize, 1200), image.patch_width); // the checkpoint's patch_embedding.weight [1024, 1200]
}

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

test "a picture's rows are logged as the lowercase SHA-256 of their BF16 bytes" {
    try t.expectEqualStrings("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", &images.digestHex("abc"));
    try t.expectEqualStrings("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", &images.digestHex(""));
}
