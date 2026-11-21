//! An image's bytes to the tower's input as image_processor.py's load_image makes it (RGB, padded, patched).
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const png = @import("../../core/picture/png.zig");
const jpeg = @import("../../core/picture/jpeg.zig");
const webp = @import("../../core/picture/webp.zig");
const resample = @import("../../core/picture/resample.zig");
const identity = @import("../../core/picture/identity.zig");
const vision = @import("vision.zig");

pub const Error = picture.Error || error{ImageTooSmall};

/// The words for a refusal the server cannot word: the JPEG library this machine lacks, or a CMYK JPEG.
pub const decoderProblem = picture.decoderProblem;

/// One image for the tower: its patches (f32 [n_vit_h * n_vit_w, 588]), its patch grid and the span's tokens.
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

pub const Plan = struct { best_h: u32, best_w: u32 };

/// Python's round(): half to even.
fn roundEven(x: f64) f64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) return r - std.math.sign(x);
    return r;
}

fn tokensOf(best_h: u64, best_w: u64) u64 {
    const p = vision.patch;
    const r = vision.downsample;
    const gh = ((best_h / p) + r - 1) / r;
    const gw = ((best_w / p) + r - 1) / r;
    return gh * (gw + 1) + 2;
}

/// plan_image_grid: small images grown to min_pixels, sides rounded to patches, shrunk past `max_tokens`.
pub fn plan(width: u32, height: u32, max_tokens: u32) Error!Plan {
    const p: f64 = vision.patch;
    var w: f64 = @floatFromInt(width);
    var h: f64 = @floatFromInt(height);
    if (w * h > 0 and w * h < vision.min_pixels) {
        const ratio = @sqrt(@as(f64, vision.min_pixels) / (w * h));
        w = @trunc(w * ratio);
        h = @trunc(h * ratio);
    }
    var best_w: u64 = @intFromFloat(@ceil(w / p) * p);
    var best_h: u64 = @intFromFloat(@ceil(h / p) * p);
    if (tokensOf(best_h, best_w) > max_tokens) {
        if (max_tokens < 5) return error.ImageTooSmall;
        const m: f64 = @floatFromInt(max_tokens);
        const r = h / w;
        const max_w = @sqrt((m - 2) / r + 0.25) - 0.5;
        const max_h = max_w * r;
        const cell: f64 = p * vision.downsample;
        if (max_w < 1.0) {
            best_h = @intFromFloat(@floor((m - 2) / 2) * cell);
            best_w = @intFromFloat(cell);
        } else if (max_h < 1.0) {
            best_h = @intFromFloat(cell);
            best_w = @intFromFloat((m - 3) * cell);
        } else {
            const beta = @min(@floor(max_w) * cell / w, @floor(max_h) * cell / h);
            best_h = @intFromFloat(@floor(h * beta / p) * p);
            best_w = @intFromFloat(@floor(w * beta / p) * p);
        }
        if (best_h == 0 or best_w == 0 or tokensOf(best_h, best_w) > max_tokens) return error.ImageTooSmall;
    }
    return .{ .best_h = @intCast(best_h), .best_w = @intCast(best_w) };
}

/// The bytes as convert("RGB") sees them: PNG, JPEG or WebP samples as stored, alpha dropped, gray spread to three.
fn rgb(io: std.Io, a: std.mem.Allocator, bytes: []const u8) Error!picture.Rgb {
    const s = switch (picture.sniff(bytes)) {
        .png => try png.decode(a, bytes),
        .jpeg => try jpeg.decode(io, a, bytes),
        .webp => try webp.decode(io, a, bytes),
        .unknown => return error.Invalid,
    };
    defer a.free(s.data);
    const out = try a.alloc(u8, s.width * s.height * 3);
    for (0..s.width * s.height) |i| {
        const px = s.data[i * s.channels ..][0..s.channels];
        for (0..3) |ch| out[i * 3 + ch] = if (s.channels >= 3) px[ch] else px[0];
    }
    return .{ .width = s.width, .height = s.height, .pixels = out };
}

/// ImageOps.pad(image, (bw, bh)): contain's aspect-kept resize (Pillow bicubic), centered on gray 127.
fn pad(a: std.mem.Allocator, img: picture.Rgb, bw: usize, bh: usize) ![]u8 {
    const W: f64 = @floatFromInt(img.width);
    const H: f64 = @floatFromInt(img.height);
    var rw = bw;
    var rh = bh;
    const im_ratio = W / H;
    const dest_ratio = @as(f64, @floatFromInt(bw)) / @as(f64, @floatFromInt(bh));
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            rh = @intFromFloat(roundEven(H / W * @as(f64, @floatFromInt(bw))));
        } else {
            rw = @intFromFloat(roundEven(W / H * @as(f64, @floatFromInt(bh))));
        }
    }
    const out = try a.alloc(u8, bw * bh * 3);
    @memset(out, 127);
    if (rw == 0 or rh == 0) return out;
    const resized = try resample.resize(a, .pillow, resample.View.of(img.pixels, img.width, img.height), rw, rh);
    defer a.free(resized);
    const x0: usize = if (rw != bw) @intFromFloat(roundEven(@as(f64, @floatFromInt(bw - rw)) * 0.5)) else 0;
    const y0: usize = if (rw == bw and rh != bh) @intFromFloat(roundEven(@as(f64, @floatFromInt(bh - rh)) * 0.5)) else 0;
    for (0..rh) |y| @memcpy(out[((y0 + y) * bw + x0) * 3 ..][0 .. rw * 3], resized[y * rw * 3 ..][0 .. rw * 3]);
    return out;
}

/// The image's bytes to tower input at most `max_tokens` span tokens (and at most the model's 1,024).
pub fn prepare(io: std.Io, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) Error!Prepared {
    var img = try rgb(io, a, bytes);
    defer img.deinit(a);
    try picture.fits(img.width, img.height);
    const pl = try plan(@intCast(img.width), @intCast(img.height), @min(max_tokens, vision.max_tokens));
    const bw: usize = pl.best_w;
    const bh: usize = pl.best_h;
    const canvas = try pad(a, img, bw, bh);
    defer a.free(canvas);
    const p = vision.patch;
    const gh = bh / p;
    const gw = bw / p;
    const pixels = try a.alloc(f32, gh * gw * vision.patch_values);
    for (0..gh) |py| for (0..gw) |px| {
        const patch = pixels[(py * gw + px) * vision.patch_values ..][0..vision.patch_values];
        for (0..3) |ch| for (0..p) |y| for (0..p) |x| {
            const v: f32 = @floatFromInt(canvas[((py * p + y) * bw + px * p + x) * 3 + ch]);
            patch[(ch * p + y) * p + x] = (v / 255.0 - 0.5) / 0.5;
        };
    };
    return .{
        .pixels = pixels,
        .gh = @intCast(gh),
        .gw = @intCast(gw),
        .tokens = vision.spanTokens(@intCast(gh), @intCast(gw)),
        .hash = identity.withGrid(identity.ofBytes(0xD5EE41, bytes), .{ @intCast(gh), @intCast(gw) }),
    };
}

/// An f32 value to bf16, nearest even (the tower reads its patches as bf16, as load_image casts them).
pub fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

test "plans: the example images' grids, a tiny image grown, a huge one held to 1,024 tokens" {
    const corn = try plan(450, 308, 1024);
    try std.testing.expectEqual(Plan{ .best_h = 462, .best_w = 658 }, corn);
    const carrots = try plan(1024, 701, 1024);
    try std.testing.expectEqual(@as(u32, 51 * 14), carrots.best_h);
    try std.testing.expectEqual(@as(u32, 74 * 14), carrots.best_w);
    const big = try plan(8000, 6000, 1024);
    try std.testing.expect(tokensOf(big.best_h, big.best_w) <= 1024);
    try std.testing.expectEqual(@as(f64, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f64, 4), roundEven(3.5));
}
