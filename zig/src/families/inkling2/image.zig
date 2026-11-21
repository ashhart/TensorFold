//! Inkling 2's image processor on the host: Inkling2ImageProcessor's 40-pixel patches and the tower's input tables.
const std = @import("std");
const core = @import("core");

const picture = core.picture;
const b16 = core.bf16_round;

pub const side = 40; // pixels a decoder token covers: the native 20-pixel patch, 2 x 2 merged
pub const patch = 20;
pub const merge = 2;
pub const grid = 32; // the learned position table's side
pub const patch_width = 3 * patch * patch; // a native patch's inputs: row, column, channel
pub const pairs = 32; // rotary pairs a 64-wide head: 16 by column, then 16 by row
const theta: f32 = 10000;
const pad: f32 = -1; // pixels outside the picture, on the 0-255 scale
const mean: f32 = 0.5 * 255.0; // the fused mean and std (0.5 each, rescale 1/255) as fp32
const std_: f32 = 0.5 * 255.0;

/// The release's placeholder: one per decoder image token, its rows replaced by the tower's.
pub const image_token: u32 = 200054;
/// The chat template's marker ahead of an image's placeholders.
pub const image_marker = "<|content_image|>";

/// One picture: its normalized patches ([tokens][40][40][3] fp32, the processor's pixel_values) and its grid.
pub const Prepared = struct {
    pixels: []f32,
    rows: u32,
    cols: u32,

    pub fn tokens(p: Prepared) u32 {
        return p.rows * p.cols;
    }

    pub fn deinit(p: *Prepared, a: std.mem.Allocator) void {
        a.free(p.pixels);
        p.* = undefined;
    }
};

/// Decoder tokens a picture of `w` x `h` pixels takes: no resize, one per 40-pixel cell of the padded grid.
pub fn tokensFor(w: usize, h: usize) usize {
    return ((h + side - 1) / side) * ((w + side - 1) / side);
}

/// PNG, JPEG or WebP bytes as the processor reads them: EXIF-oriented, alpha over its 8-pixel checkerboard.
pub fn decode(io: std.Io, a: std.mem.Allocator, bytes: []const u8, limits: picture.Limits) picture.Error!picture.Rgb {
    const s = switch (picture.sniff(bytes)) {
        .png => try picture.png.decodeWithLimits(a, bytes, limits),
        .jpeg => try picture.jpeg.decodeWithLimits(io, a, bytes, limits),
        .webp => try picture.webp.decodeWithLimits(io, a, bytes, limits),
        .unknown => return error.Invalid,
    };
    defer a.free(s.data);
    if (s.channels != 4) return picture.finish(a, s);
    // orient the alpha with the colours (finish on a 4-channel copy keeps it), then composite
    const rgba = try a.alloc(u8, s.width * s.height * 4);
    defer a.free(rgba);
    const turned = orient4(s, rgba);
    const out = try a.alloc(u8, turned.w * turned.h * 3);
    for (0..turned.h) |y| for (0..turned.w) |x| {
        const px = rgba[(y * turned.w + x) * 4 ..][0..4];
        const bg: u32 = if ((y / 8 + x / 8) % 2 == 0) 204 else 153;
        for (out[(y * turned.w + x) * 3 ..][0..3], px[0..3]) |*c, v| c.* = over(v, px[3], bg);
    };
    return .{ .width = turned.w, .height = turned.h, .pixels = out };
}

/// The processor's composite: (c * alpha + bg * (255 - alpha) + 127) // 255 in integers.
pub fn over(c: u8, alpha: u8, bg: u32) u8 {
    return @intCast((@as(u32, c) * alpha + bg * (255 - alpha) + 127) / 255);
}

/// picture.finish's transpose of 4-channel samples into `out`.
fn orient4(s: picture.Samples, out: []u8) struct { w: usize, h: usize } {
    const o = if (s.orientation >= 2 and s.orientation <= 8) s.orientation else 1;
    const w, const h = if (o >= 5) .{ s.height, s.width } else .{ s.width, s.height };
    for (0..h) |y| for (0..w) |x| {
        const sx, const sy = switch (o) {
            2 => .{ s.width - 1 - x, y },
            3 => .{ s.width - 1 - x, s.height - 1 - y },
            4 => .{ x, s.height - 1 - y },
            5 => .{ y, x },
            6 => .{ y, s.height - 1 - x },
            7 => .{ s.width - 1 - y, s.height - 1 - x },
            8 => .{ s.width - 1 - y, x },
            else => .{ x, y },
        };
        @memcpy(out[(y * w + x) * 4 ..][0..4], s.data[(sy * s.width + sx) * 4 ..][0..4]);
    };
    return .{ .w = w, .h = h };
}

/// RGB rows to the processor's pixel_values: padded with -1 to the 40-pixel grid, (v - 127.5) / 127.5 in fp32.
pub fn patches(a: std.mem.Allocator, img: picture.Rgb) !Prepared {
    const rows: u32 = @intCast((img.height + side - 1) / side);
    const cols: u32 = @intCast((img.width + side - 1) / side);
    const px = try a.alloc(f32, @as(usize, rows) * cols * side * side * 3);
    for (0..rows) |r| for (0..cols) |c| for (0..side) |y| for (0..side) |x| {
        const iy = r * side + y;
        const ix = c * side + x;
        const at = ((((r * cols + c) * side + y) * side) + x) * 3;
        for (0..3) |ch| {
            const v: f32 = if (iy < img.height and ix < img.width) @floatFromInt(img.pixels[(iy * img.width + ix) * 3 + ch]) else pad;
            px[at + ch] = (v - mean) / std_;
        }
    };
    return .{ .pixels = px, .rows = rows, .cols = cols };
}

/// Bytes to patches: decode, then patches; refuses pictures over `max_tokens`.
pub fn prepare(io: std.Io, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !Prepared {
    var img = try decode(io, a, bytes, .{});
    defer img.deinit(a);
    if (tokensFor(img.width, img.height) > max_tokens) return error.ImageTooLarge;
    return patches(a, img);
}

/// The words for a refusal of prepare's own.
pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.ImageTooLarge => "image too large for the tower's limit (one token per 40 x 40 pixels, no resize)",
        error.Invalid => picture.problem(error.Invalid),
        error.Oversized => picture.problem(error.Oversized),
        error.Animated => picture.problem(error.Animated),
        else => picture.decoderProblem(err),
    };
}

/// Native patches a picture holds: 4 a token, in merge order (token raster, then the 2 x 2 block raster).
pub fn patchCount(p: Prepared) usize {
    return @as(usize, p.tokens()) * merge * merge;
}

/// A merge-order patch's (row, column) on the native patch grid.
pub fn place(i: usize, cols: u32) [2]usize {
    const t = i / (merge * merge);
    const in = i % (merge * merge);
    return .{ t / cols * merge + in / merge, t % cols * merge + in % merge };
}

/// The patch embedding's input rows as bf16 bits: [patches][row][column][channel] (the tower's reshape).
pub fn patchRows(p: Prepared, out: []u16) void {
    for (0..patchCount(p)) |i| {
        const t = i / (merge * merge);
        const my = i % (merge * merge) / merge;
        const mx = i % merge;
        for (0..patch) |y| for (0..patch) |x| for (0..3) |ch| {
            const v = p.pixels[((t * side + my * patch + y) * side + mx * patch + x) * 3 + ch];
            out[i * patch_width + (y * patch + x) * 3 + ch] = b16.bits(v);
        };
    }
}

/// One axis's two bilinear taps and weights (torch's align_corners=False interpolation, edge taps clamped).
const Taps = struct { t: [2]u32, w: [2]f32 };

fn axis(index: usize, size: usize) Taps {
    const src = (@as(f32, @floatFromInt(index)) + 0.5) * @as(f32, grid) / @as(f32, @floatFromInt(size)) - 0.5;
    const fl = @floor(src);
    var out: Taps = undefined;
    for (0..2) |o| {
        const raw = @as(i64, @intFromFloat(fl)) + @as(i64, @intCast(o));
        out.t[o] = @intCast(std.math.clamp(raw, 0, grid - 1));
        out.w[o] = @max(1 - @abs(src - fl - @as(f32, @floatFromInt(o))), 0);
    }
    return out;
}

/// Each patch's four position-table rows and fp32 weights (h0 w0, h0 w1, h1 w0, h1 w1), as the tower resamples.
pub fn positionTaps(p: Prepared, taps: []u32, weights: []f32) void {
    const H = @as(usize, p.rows) * merge;
    const W = @as(usize, p.cols) * merge;
    for (0..patchCount(p)) |i| {
        const rc = place(i, p.cols);
        const h = axis(rc[0], H);
        const w = axis(rc[1], W);
        for (0..2) |a| for (0..2) |b| {
            taps[i * 4 + a * 2 + b] = h.t[a] * grid + w.t[b];
            weights[i * 4 + a * 2 + b] = h.w[a] * w.w[b];
        };
    }
}

/// Rotary cos and sin a sequence row ([rows][32] fp32): row 0 the class token at (0, 0), patch i at its place + 1.
pub fn rotary(p: Prepared, cs: []f32, sn: []f32, on: bool) void {
    var inv: [pairs / 2]f32 = undefined;
    for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, pairs));
    for (0..patchCount(p) + 1) |r| {
        const rc: [2]usize = if (r == 0) .{ 0, 0 } else blk: {
            const q = place(r - 1, p.cols);
            break :blk .{ q[0] + 1, q[1] + 1 };
        };
        for (0..pairs) |j| {
            const pos: f32 = @floatFromInt(if (j < pairs / 2) rc[1] else rc[0]);
            const ang = pos * inv[j % (pairs / 2)];
            cs[r * pairs + j] = if (on) @cos(ang) else 1;
            sn[r * pairs + j] = if (on) @sin(ang) else 0;
        }
    }
}

test "tokens: one per 40-pixel cell, padded up" {
    try std.testing.expectEqual(@as(usize, 15), tokensFor(200, 120));
    try std.testing.expectEqual(@as(usize, 117), tokensFor(333, 517));
    try std.testing.expectEqual(@as(usize, 2), tokensFor(37, 45));
}

test "checkerboard composite as the processor rounds it" {
    try std.testing.expectEqual(@as(u8, 10), over(10, 255, 204));
    try std.testing.expectEqual(@as(u8, 204), over(10, 0, 204));
    try std.testing.expectEqual(@as(u8, (100 * 128 + 153 * 127 + 127) / 255), over(100, 128, 153));
}

test "merge order and bilinear taps against torch's interpolation formula" {
    try std.testing.expectEqual([2]usize{ 0, 1 }, place(1, 5));
    try std.testing.expectEqual([2]usize{ 1, 0 }, place(2, 5));
    try std.testing.expectEqual([2]usize{ 0, 2 }, place(4, 5));
    try std.testing.expectEqual([2]usize{ 2, 0 }, place(20, 5));
    // a 6-patch axis on the 32 grid: src = (i + 0.5) * 32 / 6 - 0.5
    const a0 = axis(0, 6);
    try std.testing.expectEqual([2]u32{ 2, 3 }, a0.t);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8333333), a0.w[0], 1e-6);
    const e = axis(0, 64); // src -0.25: floor -1 clamped to 0 twice, weights 0.25 and 0.75
    try std.testing.expectEqual([2]u32{ 0, 0 }, e.t);
    try std.testing.expectEqual([2]f32{ 0.25, 0.75 }, e.w);
}

test "patches: padding reads -1 on the 0-255 scale, the patch rows reorder by 20-pixel blocks" {
    const a = std.testing.allocator;
    var px: [41 * 3]u8 = @splat(0);
    px[40 * 3] = 255;
    var p = try patches(a, .{ .width = 41, .height = 1, .pixels = &px });
    defer p.deinit(a);
    try std.testing.expectEqual(@as(u32, 2), p.cols);
    try std.testing.expectEqual(@as(f32, -1), p.pixels[0]);
    try std.testing.expectEqual(@as(f32, (-1 - 127.5) / 127.5), p.pixels[3 * side]);
    try std.testing.expectEqual(@as(f32, 1), p.pixels[side * side * 3]);
    const rows = try a.alloc(u16, patchCount(p) * patch_width);
    defer a.free(rows);
    patchRows(p, rows);
    try std.testing.expectEqual(b16.bits(1), rows[4 * patch_width]); // token 1's first patch, pixel (0, 0)
}
