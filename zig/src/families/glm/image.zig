//! An image's bytes to GLM5-Next vision patches as TensorFold 0.6.6's --vision made them (Pillow, then mlx-vlm).
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const vision = @import("vision.zig");

/// One image ready for the tower: its patches, its patch grid and its tokens (gh * gw / 4).
pub const Prepared = struct {
    pixels: []f32, // [gh * gw, 1176]
    gh: u32,
    gw: u32,
    tokens: u32,
    hash: u64, // of the image's bytes and grid: other images, or this one at another budget, are other prompts
};

/// The shared decoder's refusals, and a token budget too small for an image.
pub const Error = picture.Error || error{ImageTooSmall};

/// The words for a refusal the server cannot word: the JPEG library this machine lacks, or a CMYK JPEG.
pub const decoderProblem = picture.decoderProblem;

// ---- the processor's geometry ----

pub const Geometry = struct { target_h: u32, target_w: u32, content_h: u32, content_w: u32 };

fn alignUp(v: u64, f: u64) u64 {
    return (v + f - 1) / f * f;
}

/// smart_resize and _resize_geometry: the padded canvas under `max_tokens` (at least `min_tokens`), the content size.
pub fn geometry(h: u32, w: u32, min_tokens: u32, max_tokens: u32) Error!Geometry {
    const factor: u64 = vision.patch * vision.merge;
    const frames: u64 = vision.temporal;
    const per_token = frames * factor * factor;
    const min_px = @as(u64, min_tokens) * per_token;
    const max_px = @as(u64, max_tokens) * per_token;
    if (min_tokens == 0 or max_tokens < min_tokens) return error.ImageTooSmall;
    const H: u64 = h;
    const W: u64 = w;
    var ah = alignUp(H, factor);
    var aw = alignUp(W, factor);
    var pixels = frames * ah * aw;
    if (pixels < min_px) {
        const scale = @sqrt(@as(f64, @floatFromInt(min_px)) / @as(f64, @floatFromInt(frames * H * W)));
        ah = alignUp(@max(1, @as(u64, @intFromFloat(@ceil(@as(f64, @floatFromInt(H)) * scale)))), factor);
        aw = alignUp(@max(1, @as(u64, @intFromFloat(@ceil(@as(f64, @floatFromInt(W)) * scale)))), factor);
        pixels = frames * ah * aw;
    }
    if (pixels > max_px) { // the tallest content that fits, by bisection on its height
        if (max_px < frames * factor * factor) return error.ImageTooSmall;
        var best: [2]u64 = .{ factor, factor };
        var low: i64 = 1;
        var high: i64 = @intCast(H);
        while (low <= high) {
            const ch: u64 = @intCast(@divFloor(low + high, 2));
            const cw: u64 = @max(1, W * ch / H); // math.floor(width * content_h / height)
            const cand: [2]u64 = .{ alignUp(ch, factor), alignUp(cw, factor) };
            if (frames * cand[0] * cand[1] <= max_px) {
                best = cand;
                low = @as(i64, @intCast(ch)) + 1;
            } else high = @as(i64, @intCast(ch)) - 1;
        }
        ah = best[0];
        aw = best[1];
    }
    var scale = @min(@as(f64, @floatFromInt(ah)) / @as(f64, @floatFromInt(H)), @as(f64, @floatFromInt(aw)) / @as(f64, @floatFromInt(W)));
    if (frames * H * W >= min_px) scale = @min(1.0, scale);
    const chf: u64 = @intFromFloat(@floor(@as(f64, @floatFromInt(H)) * scale));
    const cwf: u64 = @intFromFloat(@floor(@as(f64, @floatFromInt(W)) * scale));
    return .{ .target_h = @intCast(ah), .target_w = @intCast(aw), .content_h = @intCast(@max(1, @min(ah, chf))), .content_w = @intCast(@max(1, @min(aw, cwf))) };
}

/// The resized content padded with black, scaled, normalized and cut into 2x2 windows of 14x14 patches, row-major.
pub fn patchify(gpa: std.mem.Allocator, content: picture.Rgb, g: Geometry, mean: [3]f32, sd: [3]f32) Error![]f32 {
    const P = vision.patch;
    const M = vision.merge;
    const gh = g.target_h / P;
    const gw = g.target_w / P;
    const out = try gpa.alloc(f32, @as(usize, gh) * gw * vision.patch_width);
    const rescale: f32 = @floatCast(@as(f64, 1.0) / 255.0);
    var row: usize = 0;
    for (0..gh / M) |wy| for (0..gw / M) |wx| for (0..M) |my| for (0..M) |mx| {
        const py0 = (wy * M + my) * P;
        const px0 = (wx * M + mx) * P;
        const dst = out[row * vision.patch_width ..][0..vision.patch_width];
        for (0..3) |ch| for (0..vision.temporal) |t| for (0..P) |dy| for (0..P) |dx| {
            const y = py0 + dy;
            const x = px0 + dx;
            const v: u8 = if (y < content.height and x < content.width) content.pixels[(y * content.width + x) * 3 + ch] else 0;
            const f = @as(f32, @floatFromInt(v)) * rescale;
            dst[((ch * vision.temporal + t) * P + dy) * P + dx] = (f - mean[ch]) / sd[ch];
        };
        row += 1;
    };
    return out;
}

/// Bytes to patches: decoded as 0.6.6 decoded them (core's picture), sized for `max_tokens`, resized, padded and cut.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, c: vision.Config, max_tokens: u32) Error!Prepared {
    var img = try picture.decode(io, gpa, bytes);
    defer img.deinit(gpa);
    return prepareRgb(gpa, img, std.hash.Wyhash.hash(0x1a6e, bytes), c, max_tokens);
}

/// RGB to patches (Pillow's bicubic resize, the padded canvas cut); the hash covers the bytes' hash and the grid.
pub fn prepareRgb(gpa: std.mem.Allocator, img: picture.Rgb, bytes_hash: u64, c: vision.Config, max_tokens: u32) Error!Prepared {
    const g = try geometry(@intCast(img.height), @intCast(img.width), @min(c.min_tokens, max_tokens), max_tokens);
    const resized = try picture.resample.resize(gpa, .pillow, .of(img.pixels, img.width, img.height), g.content_w, g.content_h);
    defer gpa.free(resized);
    const content: picture.Rgb = .{ .width = g.content_w, .height = g.content_h, .pixels = resized };
    const pixels = try patchify(gpa, content, g, c.mean, c.std);
    const grid = [2]u32{ g.target_h / vision.patch, g.target_w / vision.patch };
    return .{ .pixels = pixels, .gh = grid[0], .gw = grid[1], .tokens = grid[0] * grid[1] / vision.patches_per_token, .hash = std.hash.Wyhash.hash(bytes_hash, std.mem.asBytes(&grid)) };
}

test "geometry follows smart_resize" {
    // the reference dumps' grids: 517x389 pads to 532x392 unscaled; 96x96 grows to 112; 20x15 to the 16-token floor
    try std.testing.expectEqual(Geometry{ .target_h = 392, .target_w = 532, .content_h = 389, .content_w = 517 }, try geometry(389, 517, 16, 4096));
    try std.testing.expectEqual(Geometry{ .target_h = 112, .target_w = 112, .content_h = 112, .content_w = 112 }, try geometry(96, 96, 16, 4096));
    const tiny = try geometry(15, 20, 16, 4096);
    try std.testing.expectEqual(@as(u32, 8 * 14), tiny.target_h);
    try std.testing.expectEqual(@as(u32, 10 * 14), tiny.target_w);
    // a large image under a small budget: the canvas fits it
    const big = try geometry(1500, 2000, 16, 256);
    try std.testing.expect(@as(u64, big.target_h) * big.target_w * 2 <= 256 * 2 * 28 * 28);
    try std.testing.expect(big.content_h <= big.target_h and big.content_w <= big.target_w);
}

test "the hash names the grid: one picture at two budgets is two images to the cache, at one budget the same" {
    const gpa = std.testing.allocator;
    const c: vision.Config = .{ .image_token = 1, .image_start = 2, .image_end = 3, .mean = @splat(0.5), .std = @splat(0.5), .min_tokens = 16, .max_tokens = 4096 };
    const pixels = try gpa.alloc(u8, 640 * 480 * 3);
    defer gpa.free(pixels);
    @memset(pixels, 9);
    const photo: picture.Rgb = .{ .width = 640, .height = 480, .pixels = pixels };
    const whole = try prepareRgb(gpa, photo, 7, c, 4096);
    defer gpa.free(whole.pixels);
    const cut = try prepareRgb(gpa, photo, 7, c, 64);
    defer gpa.free(cut.pixels);
    const again = try prepareRgb(gpa, photo, 7, c, 4096);
    defer gpa.free(again.pixels);
    try std.testing.expect(whole.tokens > cut.tokens);
    try std.testing.expect(whole.hash != cut.hash); // another grid: neither resumes a state the other kept
    try std.testing.expectEqual(whole.hash, again.hash); // the same picture at the same budget resumes
}
