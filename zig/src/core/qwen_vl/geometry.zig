//! Qwen-VL image geometry: smart_resize to multiples of patch x merge inside the pixel bounds, patches in merge order.
const std = @import("std");
const picture = @import("../picture/picture.zig");
const Config = @import("config.zig").Config;

pub const Size = struct { h: u32, w: u32 };

/// The shared decoder's refusals, a budget too small for an image, and a side past 200 times the other.
pub const Error = picture.Error || error{ ImageTooSmall, AspectRatio };

/// One image ready for a tower: patches [gh * gw, patch width] in merge order, the patch grid and the image's tokens.
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

/// The words for a refusal only the image path knows: the decoder's, and a side past 200 times the other.
pub fn problem(err: anyerror) ?[]const u8 {
    if (err == error.AspectRatio) return "its longer side is more than 200 times its shorter side";
    return picture.decoderProblem(err);
}

/// transformers' smart_resize: sides multiples of `factor` (Python's round, halves to even), the area in the bounds.
pub fn smartResize(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64) error{AspectRatio}!Size {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    if (@max(h, w) / @min(h, w) > 200) return error.AspectRatio;
    const f: f64 = @floatFromInt(factor);
    var hb = roundEven(h / f) * f;
    var wb = roundEven(w / f) * f;
    if (hb * wb > @as(f64, @floatFromInt(max_pixels))) {
        const beta = @sqrt(h * w / @as(f64, @floatFromInt(max_pixels)));
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (hb * wb < @as(f64, @floatFromInt(min_pixels))) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ .h = @intFromFloat(hb), .w = @intFromFloat(wb) };
}

/// Python's round: a half goes to the even neighbour.
fn roundEven(x: f64) f64 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r;
}

/// An image's pixel bounds under a `max_tokens` budget: the processor's, its most cut to that many tokens.
pub fn bounds(c: Config, max_tokens: u32) [2]u64 {
    const per = @as(u64, c.factor()) * c.factor();
    const most = @min(c.max_pixels, @as(u64, max_tokens) * per);
    return .{ @min(c.min_pixels, most), most };
}

/// Bytes to patches: decoded as 0.6.6 decoded them (core/picture), resized with Pillow's bicubic, cut in merge order.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, c: Config, max_tokens: u32) Error!Prepared {
    var img = try picture.decode(io, gpa, bytes);
    defer img.deinit(gpa);
    return prepareRgb(gpa, img, std.hash.Wyhash.hash(0x51e7, bytes), c, max_tokens);
}

/// RGB to patches; the hash covers the bytes' hash and the grid, so one picture at two budgets is two images.
pub fn prepareRgb(gpa: std.mem.Allocator, img: picture.Rgb, bytes_hash: u64, c: Config, max_tokens: u32) Error!Prepared {
    if (max_tokens == 0) return error.ImageTooSmall;
    const b = bounds(c, max_tokens);
    const size = try smartResize(@intCast(img.height), @intCast(img.width), c.factor(), b[0], b[1]);
    const resized = try picture.resample.resize(gpa, .pillow, .of(img.pixels, img.width, img.height), size.w, size.h);
    defer gpa.free(resized);
    const grid = [2]u32{ size.h / c.patch, size.w / c.patch };
    const pixels = try patchify(gpa, .{ .width = size.w, .height = size.h, .pixels = resized }, c);
    return .{ .pixels = pixels, .gh = grid[0], .gw = grid[1], .tokens = grid[0] * grid[1] / (c.merge * c.merge), .hash = std.hash.Wyhash.hash(bytes_hash, std.mem.asBytes(&grid)) };
}

/// Patches in merge order (block row, block column, row, column), each [channel, frame, y, x], the frame repeated.
pub fn patchify(gpa: std.mem.Allocator, img: picture.Rgb, c: Config) error{OutOfMemory}![]f32 {
    const P = c.patch;
    const M = c.merge;
    const gh = img.height / P;
    const gw = img.width / P;
    const width = c.patchWidth();
    const out = try gpa.alloc(f32, gh * gw * width);
    // x * rescale in float64 then float32, then (x - mean) / std in float32: the processor's arithmetic
    var lut: [3][256]f32 = undefined;
    for (&lut, 0..) |*row, ch| for (row, 0..) |*v, i| {
        const scaled: f32 = @floatCast(@as(f64, @floatFromInt(i)) * c.rescale);
        v.* = (scaled - c.mean[ch]) / c.std[ch];
    };
    var n: usize = 0;
    for (0..gh / M) |by| for (0..gw / M) |bx| for (0..M) |my| for (0..M) |mx| {
        const y0 = (by * M + my) * P;
        const x0 = (bx * M + mx) * P;
        const dst = out[n * width ..][0..width];
        for (0..3) |ch| for (0..c.temporal) |t| for (0..P) |dy| for (0..P) |dx| {
            dst[((ch * c.temporal + t) * P + dy) * P + dx] = lut[ch][img.pixels[((y0 + dy) * img.width + x0 + dx) * 3 + ch]];
        };
        n += 1;
    };
    return out;
}

const sample = @import("config.zig");

fn sampleConfig(a: std.mem.Allocator) !Config {
    return (try Config.parse(a, sample.sample_config, sample.sample_preprocessor, sample.sample_processor)).?;
}

test "smart_resize as the processor rounds, grows and shrinks" {
    // the gate pictures' sizes (h, w) at a 4,096-token budget, as transformers' processor grids them
    const cases = [_]struct { h: u32, w: u32, want: Size }{
        .{ .h = 900, .w = 1280, .want = .{ .h = 896, .w = 1280 } },
        .{ .h = 80, .w = 1600, .want = .{ .h = 64, .w = 1600 } }, // 2.5 rounds to 2
        .{ .h = 700, .w = 1225, .want = .{ .h = 704, .w = 1216 } },
        .{ .h = 200, .w = 120, .want = .{ .h = 352, .w = 224 } }, // under 65,536 pixels: grown
        .{ .h = 80, .w = 360, .want = .{ .h = 128, .w = 544 } },
        .{ .h = 64, .w = 96, .want = .{ .h = 224, .w = 320 } },
    };
    for (cases) |k| try std.testing.expectEqual(k.want, try smartResize(k.h, k.w, 32, 65536, 4096 * 1024));
    // over the budget: the largest grid inside it
    const big = try smartResize(3000, 4000, 32, 65536, 256 * 1024);
    try std.testing.expect(@as(u64, big.h) * big.w <= 256 * 1024 and big.h % 32 == 0 and big.w % 32 == 0);
    try std.testing.expectError(error.AspectRatio, smartResize(10, 2010, 32, 65536, 4096 * 1024));
    try std.testing.expectEqual(@as(f64, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f64, 4), roundEven(3.5));
    try std.testing.expectEqual(@as(f64, 3), roundEven(2.75));
}

test "patches in merge order, each channel by frame by row by column, normalized" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try sampleConfig(a);
    // 64 x 32 pixels: a 4 x 2 patch grid, two merge blocks stacked; each pixel's red is its patch's raster index
    var px: [64 * 32 * 3]u8 = undefined;
    for (0..64) |y| for (0..32) |x| {
        px[(y * 32 + x) * 3] = @intCast((y / 16) * 2 + x / 16);
        px[(y * 32 + x) * 3 + 1] = 255;
        px[(y * 32 + x) * 3 + 2] = 0;
    };
    const out = try patchify(a, .{ .width = 32, .height = 64, .pixels = &px }, c);
    try std.testing.expectEqual(@as(usize, 8 * 1536), out.len);
    const red = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 }; // block (0,0): patches 0 1 2 3, then block (1,0): 4 5 6 7
    for (red, 0..) |r, n| {
        const p = out[n * 1536 ..][0..1536];
        const want: f32 = (@as(f32, @floatCast(@as(f64, @floatFromInt(r)) * c.rescale)) - 0.5) / 0.5;
        try std.testing.expectEqual(want, p[0]); // channel 0, frame 0
        try std.testing.expectEqual(want, p[256]); // channel 0, frame 1: the same image
        try std.testing.expectEqual(@as(f32, 1), p[512]); // green 255
        try std.testing.expectEqual(@as(f32, -1), p[1024]); // blue 0
    }
}

test "bounds follow the budget, and the hash names the grid" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try sampleConfig(a);
    try std.testing.expectEqual([2]u64{ 65536, 4096 * 1024 }, bounds(c, 4096));
    try std.testing.expectEqual([2]u64{ 32 * 1024, 32 * 1024 }, bounds(c, 32)); // under the processor's floor
    var px: [96 * 64 * 3]u8 = @splat(128);
    const img: picture.Rgb = .{ .width = 96, .height = 64, .pixels = &px };
    const grown = try prepareRgb(a, img, 7, c, 4096);
    try std.testing.expectEqual([3]u32{ 14, 20, 70 }, [3]u32{ grown.gh, grown.gw, grown.tokens });
    // under a tight budget the processor's floor rounds up past it, as transformers' does: the server refuses that
    try std.testing.expectEqual(@as(u32, 20), (try prepareRgb(a, img, 7, c, 16)).tokens);
    const photo: picture.Rgb = .{ .width = 640, .height = 480, .pixels = try a.alloc(u8, 640 * 480 * 3) };
    @memset(photo.pixels, 9);
    const whole = try prepareRgb(a, photo, 7, c, 4096);
    const cut = try prepareRgb(a, photo, 7, c, 64);
    try std.testing.expectEqual([2]u32{ 300, 54 }, [2]u32{ whole.tokens, cut.tokens });
    try std.testing.expect(whole.hash != cut.hash); // the same bytes on another grid are another image to the cache
    try std.testing.expectError(error.ImageTooSmall, prepareRgb(a, img, 7, c, 0));
    try std.testing.expectEqualStrings("its longer side is more than 200 times its shorter side", problem(error.AspectRatio).?);
}
