//! An image's bytes to GLM5-Next vision patches as TensorFold 0.6.6's --vision makes them: decoded as its
//! vision/images.py does with PIL (ImageIO here: the stored values, no colour management; EXIF orientation applied;
//! alpha composited over white with PIL's integer blend), then mlx-vlm's Glm5NextImageProcessor: sized under the
//! token budget, resized with PIL's 8-bit bicubic resampler, padded with black, normalized, and cut into 2x2 windows
//! of 14x14 patches, each patch's frame given twice (temporal_patch_size 2).
const std = @import("std");
const vision = @import("vision.zig");

pub const Rgb = struct {
    w: u32,
    h: u32,
    px: []u8, // row-major RGB, 3 bytes a pixel

    pub fn deinit(r: Rgb, gpa: std.mem.Allocator) void {
        gpa.free(r.px);
    }
};

/// One image ready for the tower: its patches, its patch grid and its tokens (gh * gw / 4).
pub const Prepared = struct {
    pixels: []f32, // [gh * gw, 1176]
    gh: u32,
    gw: u32,
    tokens: u32,
    hash: u64, // of the image's bytes: two prompts with the same placeholders but other images are other prompts
};

pub const Error = error{ BadImage, ImageTooLarge, Animated, ImageTooSmall, OutOfMemory };

/// images.py's ImageLimits: an image's encoded bytes, sides and pixels.
pub const max_bytes = 10 << 20;
pub const max_side = 8192;
pub const max_pixels = 16 << 20;

// ---- ImageIO and CoreGraphics (C APIs; the server binary links both frameworks) ----
const CF = *anyopaque;
const CGRect = extern struct { x: f64, y: f64, w: f64, h: f64 };
extern "c" fn CFDataCreate(alloc: ?CF, bytes: [*]const u8, len: isize) ?CF;
extern "c" fn CFDataGetBytePtr(data: CF) [*]const u8;
extern "c" fn CFDataGetLength(data: CF) isize;
extern "c" fn CFRelease(cf: CF) void;
extern "c" fn CGImageSourceCreateWithData(data: CF, options: ?CF) ?CF;
extern "c" fn CGImageSourceGetCount(src: CF) usize;
extern "c" fn CGImageSourceCreateImageAtIndex(src: CF, index: usize, options: ?CF) ?CF;
extern "c" fn CGImageGetWidth(img: CF) usize;
extern "c" fn CGImageGetHeight(img: CF) usize;
extern "c" fn CGImageGetBitsPerComponent(img: CF) usize;
extern "c" fn CGImageGetBitsPerPixel(img: CF) usize;
extern "c" fn CGImageGetBytesPerRow(img: CF) usize;
extern "c" fn CGImageGetBitmapInfo(img: CF) u32;
extern "c" fn CGImageGetColorSpace(img: CF) ?CF;
extern "c" fn CGColorSpaceGetModel(cs: CF) i32;
extern "c" fn CGImageGetDataProvider(img: CF) ?CF;
extern "c" fn CGDataProviderCopyData(p: CF) ?CF;
extern "c" fn CGColorSpaceCreateDeviceRGB() ?CF;
extern "c" fn CGColorSpaceRelease(cs: CF) void;
extern "c" fn CGBitmapContextCreate(data: ?*anyopaque, w: usize, h: usize, bpc: usize, bpr: usize, cs: CF, info: u32) ?CF;
extern "c" fn CGContextDrawImage(ctx: CF, rect: CGRect, img: CF) void;
extern "c" fn CGContextRelease(ctx: CF) void;
extern "c" fn CGContextSetRGBFillColor(ctx: CF, r: f64, g: f64, b: f64, a: f64) void;
extern "c" fn CGContextFillRect(ctx: CF, rect: CGRect) void;
extern "c" fn CGImageSourceCopyPropertiesAtIndex(src: CF, index: usize, options: ?CF) ?CF;
extern "c" fn CFDictionaryGetValue(dict: CF, key: ?*const anyopaque) ?CF;
extern "c" fn CFNumberGetValue(num: CF, kind: isize, out: *anyopaque) bool;
extern var kCGImagePropertyOrientation: ?*const anyopaque;

const alpha_mask: u32 = 0x1f;
const order_mask: u32 = 0x7000;
const order_32_little: u32 = 0x2000;
const order_16_little: u32 = 0x1000;
const float_components: u32 = 0x100;

/// A PNG, JPEG, GIF, WebP, HEIC or any other single-frame image ImageIO reads, as 8-bit RGB: upright by its EXIF
/// orientation (PIL's exif_transpose) and over white where it has alpha (images.py's paste onto a white canvas).
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8) Error!Rgb {
    if (bytes.len == 0) return error.BadImage;
    if (bytes.len > max_bytes) return error.ImageTooLarge;
    const data = CFDataCreate(null, bytes.ptr, @intCast(bytes.len)) orelse return error.OutOfMemory;
    defer CFRelease(data);
    const src = CGImageSourceCreateWithData(data, null) orelse return error.BadImage;
    defer CFRelease(src);
    const frames = CGImageSourceGetCount(src);
    if (frames == 0) return error.BadImage;
    if (frames > 1) return error.Animated; // images.py: animated and multipage images are refused
    const img = CGImageSourceCreateImageAtIndex(src, 0, null) orelse return error.BadImage;
    defer CFRelease(img);
    const w = CGImageGetWidth(img);
    const h = CGImageGetHeight(img);
    if (w == 0 or h == 0) return error.BadImage;
    if (w > max_side or h > max_side or w * h > max_pixels) return error.ImageTooLarge;
    const out = try gpa.alloc(u8, w * h * 3);
    errdefer gpa.free(out);
    if (!(try raw(img, w, h, out))) try drawn(img, w, h, out);
    const rgb: Rgb = .{ .w = @intCast(w), .h = @intCast(h), .px = out };
    const o = orientation(src);
    if (o <= 1 or o > 8) return rgb;
    defer rgb.deinit(gpa);
    return orient(gpa, rgb, o);
}

/// The first image's EXIF orientation (1-8; 1 or 0: as stored).
fn orientation(src: CF) u32 {
    const props = CGImageSourceCopyPropertiesAtIndex(src, 0, null) orelse return 1;
    defer CFRelease(props);
    const num = CFDictionaryGetValue(props, kCGImagePropertyOrientation) orelse return 1;
    var v: i32 = 1;
    if (!CFNumberGetValue(num, 9, &v)) return 1; // kCFNumberIntType
    return if (v < 1) 1 else @intCast(v);
}

/// PIL's exif_transpose: orientation 2-8 as FLIP_LEFT_RIGHT, ROTATE_180, FLIP_TOP_BOTTOM, TRANSPOSE, ROTATE_270,
/// TRANSVERSE, ROTATE_90; each output pixel read from its source pixel.
pub fn orient(gpa: std.mem.Allocator, src: Rgb, o: u32) Error!Rgb {
    const w: usize = src.w;
    const h: usize = src.h;
    const swap = o >= 5;
    const ow = if (swap) h else w;
    const oh = if (swap) w else h;
    const out = try gpa.alloc(u8, ow * oh * 3);
    for (0..oh) |y| for (0..ow) |x| {
        const s: [2]usize = switch (o) {
            2 => .{ w - 1 - x, y },
            3 => .{ w - 1 - x, h - 1 - y },
            4 => .{ x, h - 1 - y },
            5 => .{ y, x },
            6 => .{ y, h - 1 - x },
            7 => .{ w - 1 - y, h - 1 - x },
            else => .{ w - 1 - y, x },
        };
        @memcpy(out[(y * ow + x) * 3 ..][0..3], src.px[(s[1] * w + s[0]) * 3 ..][0..3]);
    };
    return .{ .w = @intCast(ow), .h = @intCast(oh), .px = out };
}

/// PIL's paste of a pixel with alpha `a` onto white: DIV255(255 * (255 - a) + c * a).
fn overWhite(c: u8, a: u8) u8 {
    const t: u32 = 255 * (255 - @as(u32, a)) + @as(u32, c) * a + 128;
    return @intCast(((t >> 8) + t) >> 8);
}

/// The decoder's own values for 8- and 16-bit gray and RGB layouts (what PIL reads); false for any other layout.
fn raw(img: CF, w: usize, h: usize, out: []u8) Error!bool {
    const info = CGImageGetBitmapInfo(img);
    if (info & float_components != 0) return false;
    const cs = CGImageGetColorSpace(img) orelse return false;
    const model = CGColorSpaceGetModel(cs); // 0 monochrome, 1 RGB
    if (model != 0 and model != 1) return false;
    const bpc = CGImageGetBitsPerComponent(img);
    const bpp = CGImageGetBitsPerPixel(img);
    if (bpc != 8 and bpc != 16) return false;
    const alpha = info & alpha_mask; // 0 none, 1/2 premultiplied last/first, 3/4 last/first, 5/6 skip last/first
    if (alpha == 1 or alpha == 2 or alpha > 6) return false;
    const colours: usize = if (model == 1) 3 else 1;
    const comps: usize = colours + @intFromBool(alpha != 0);
    if (bpp != comps * bpc) return false;
    const first = alpha == 2 or alpha == 4 or alpha == 6; // the alpha (or skipped) component leads
    const order = info & order_mask;
    const little = (bpc == 8 and order == order_32_little) or (bpc == 16 and order == order_16_little);
    if (bpc == 8 and order != 0 and order != order_32_little) return false;
    if (bpc == 8 and little and comps != 4) return false;
    const provider = CGImageGetDataProvider(img) orelse return false;
    const data = CGDataProviderCopyData(provider) orelse return false;
    defer CFRelease(data);
    const base = CFDataGetBytePtr(data);
    const len: usize = @intCast(CFDataGetLength(data));
    const bpr = CGImageGetBytesPerRow(img);
    if (bpr * (h - 1) + w * bpp / 8 > len) return false;
    const step = bpc / 8;
    for (0..h) |y| {
        const row = base + y * bpr;
        for (0..w) |x| {
            const px = row + x * comps * step;
            var c: [4]u8 = undefined;
            for (0..comps) |i| {
                // a 32-bit little-endian pixel stores its components reversed; a 16-bit one each component's bytes
                const slot = if (bpc == 8 and little) comps - 1 - i else i;
                const at = px + slot * step;
                c[i] = if (step == 1) at[0] else if (little) at[1] else at[0]; // 16 bits: the high byte, as PIL's 8-bit view
            }
            const lead: usize = @intFromBool(first and alpha != 0);
            const o = out[(y * w + x) * 3 ..][0..3];
            if (colours == 3) {
                o.* = .{ c[lead], c[lead + 1], c[lead + 2] };
            } else {
                o.* = .{ c[lead], c[lead], c[lead] };
            }
            if (alpha == 3 or alpha == 4) { // a real alpha band: over white, as images.py pastes it
                const a = if (first) c[0] else c[comps - 1];
                for (o) |*v| v.* = overWhite(v.*, a);
            }
        }
    }
    return true;
}

/// Any other layout (palette, CMYK, float, premultiplied): drawn into an RGB context over white.
fn drawn(img: CF, w: usize, h: usize, out: []u8) Error!void {
    const cs = CGColorSpaceCreateDeviceRGB() orelse return error.BadImage;
    defer CGColorSpaceRelease(cs);
    const tmp = std.heap.page_allocator.alloc(u8, w * h * 4) catch return error.OutOfMemory;
    defer std.heap.page_allocator.free(tmp);
    @memset(tmp, 0);
    const ctx = CGBitmapContextCreate(tmp.ptr, w, h, 8, w * 4, cs, 5) orelse return error.BadImage; // RGBX
    defer CGContextRelease(ctx);
    CGContextSetRGBFillColor(ctx, 1, 1, 1, 1);
    CGContextFillRect(ctx, .{ .x = 0, .y = 0, .w = @floatFromInt(w), .h = @floatFromInt(h) });
    CGContextDrawImage(ctx, .{ .x = 0, .y = 0, .w = @floatFromInt(w), .h = @floatFromInt(h) }, img);
    for (0..w * h) |i| @memcpy(out[i * 3 ..][0..3], tmp[i * 4 ..][0..3]);
}

// ---- the processor's geometry ----

pub const Geometry = struct { target_h: u32, target_w: u32, content_h: u32, content_w: u32 };

fn alignUp(v: u64, f: u64) u64 {
    return (v + f - 1) / f * f;
}

/// smart_resize and _resize_geometry (num_frames = temporal_patch_size, as _process_one calls it): the padded canvas
/// under `max_tokens` (at least `min_tokens`), and the content size the image is resized to inside it.
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

// ---- PIL's ImagingResample for 8-bit images (Resample.c), bicubic (a = -0.5) ----

const precision_bits = 32 - 8 - 2;

fn bicubic(x0: f64) f64 {
    const a = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}

const Coeffs = struct { bounds: []u32, kk: []i32, ksize: usize };

/// precompute_coeffs and normalize_coeffs_8bpc: each output pixel's first input pixel, count and fixed-point weights.
fn coeffs(gpa: std.mem.Allocator, in_size: u32, out_size: u32) Error!Coeffs {
    const scale = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const filterscale = @max(scale, 1.0);
    const support = 2.0 * filterscale;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    const bounds = try gpa.alloc(u32, @as(usize, out_size) * 2);
    errdefer gpa.free(bounds);
    const kk = try gpa.alloc(i32, @as(usize, out_size) * ksize);
    errdefer gpa.free(kk);
    var pre = try gpa.alloc(f64, ksize);
    defer gpa.free(pre);
    const ss = 1.0 / filterscale;
    for (0..out_size) |xx| {
        const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
        var xmin_i: i64 = @intFromFloat(center - support + 0.5); // (int) truncates toward zero, as C does
        if (xmin_i < 0) xmin_i = 0;
        var xmax_i: i64 = @intFromFloat(center + support + 0.5);
        if (xmax_i > in_size) xmax_i = in_size;
        xmax_i -= xmin_i;
        const xmin: usize = @intCast(xmin_i);
        const xmax: usize = @intCast(xmax_i);
        var ww: f64 = 0;
        for (0..xmax) |x| {
            const w = bicubic((@as(f64, @floatFromInt(x + xmin)) - center + 0.5) * ss);
            pre[x] = w;
            ww += w;
        }
        for (0..ksize) |x| {
            const k = if (x < xmax) (if (ww != 0) pre[x] / ww else pre[x]) else 0;
            const v = k * @as(f64, @floatFromInt(@as(i64, 1) << precision_bits));
            kk[xx * ksize + x] = @intFromFloat(if (v < 0) v - 0.5 else v + 0.5); // C's (int) of the rounded value
        }
        bounds[xx * 2] = @intCast(xmin);
        bounds[xx * 2 + 1] = @intCast(xmax);
    }
    return .{ .bounds = bounds, .kk = kk, .ksize = ksize };
}

fn clip8(v: i64) u8 {
    const s = v >> precision_bits;
    return if (s < 0) 0 else if (s > 255) 255 else @intCast(s);
}

/// Image.resize((w, h), BICUBIC) on an 8-bit RGB image: horizontal pass, then vertical, each rounded to 8 bits.
pub fn resize(gpa: std.mem.Allocator, src: Rgb, w: u32, h: u32) Error!Rgb {
    var cur = src;
    var owned = false;
    errdefer if (owned) cur.deinit(gpa);
    if (w != src.w) { // horizontal (PIL limits it to the rows the vertical pass reads: all of them here)
        const c = try coeffs(gpa, src.w, w);
        defer gpa.free(c.bounds);
        defer gpa.free(c.kk);
        const out = try gpa.alloc(u8, @as(usize, w) * cur.h * 3);
        for (0..cur.h) |y| for (0..w) |xx| {
            const xmin = c.bounds[xx * 2];
            const xmax = c.bounds[xx * 2 + 1];
            const k = c.kk[xx * c.ksize ..];
            for (0..3) |ch| {
                var ss: i64 = 1 << (precision_bits - 1);
                for (0..xmax) |x| ss += @as(i64, cur.px[(y * cur.w + x + xmin) * 3 + ch]) * k[x];
                out[(y * w + xx) * 3 + ch] = clip8(ss);
            }
        };
        if (owned) cur.deinit(gpa);
        cur = .{ .w = w, .h = cur.h, .px = out };
        owned = true;
    }
    if (h != src.h) {
        const c = try coeffs(gpa, src.h, h);
        defer gpa.free(c.bounds);
        defer gpa.free(c.kk);
        const out = try gpa.alloc(u8, @as(usize, cur.w) * h * 3);
        for (0..h) |yy| {
            const ymin = c.bounds[yy * 2];
            const ymax = c.bounds[yy * 2 + 1];
            const k = c.kk[yy * c.ksize ..];
            for (0..cur.w) |x| for (0..3) |ch| {
                var ss: i64 = 1 << (precision_bits - 1);
                for (0..ymax) |y| ss += @as(i64, cur.px[((y + ymin) * cur.w + x) * 3 + ch]) * k[y];
                out[(yy * cur.w + x) * 3 + ch] = clip8(ss);
            };
        }
        if (owned) cur.deinit(gpa);
        cur = .{ .w = cur.w, .h = h, .px = out };
        owned = true;
    }
    if (!owned) cur = .{ .w = src.w, .h = src.h, .px = try gpa.dupe(u8, src.px) };
    return cur;
}

/// The content (already resized) padded with black to the canvas, scaled by 1/255, normalized by channel, cut into
/// 2x2 windows of 14x14 patches in row-major window order, each patch [channel, frame (twice), row, column].
pub fn patchify(gpa: std.mem.Allocator, content: Rgb, g: Geometry, mean: [3]f32, sd: [3]f32) Error![]f32 {
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
            const v: u8 = if (y < content.h and x < content.w) content.px[(y * content.w + x) * 3 + ch] else 0;
            const f = @as(f32, @floatFromInt(v)) * rescale;
            dst[((ch * vision.temporal + t) * P + dy) * P + dx] = (f - mean[ch]) / sd[ch];
        };
        row += 1;
    };
    return out;
}

/// Bytes to patches: decoded, sized for `max_tokens` (at least min(16, max_tokens)), resized, padded and cut.
pub fn prepare(gpa: std.mem.Allocator, bytes: []const u8, c: vision.Config, max_tokens: u32) Error!Prepared {
    const img = try decode(gpa, bytes);
    defer img.deinit(gpa);
    return prepareRgb(gpa, img, std.hash.Wyhash.hash(0x1a6e, bytes), c, max_tokens);
}

pub fn prepareRgb(gpa: std.mem.Allocator, img: Rgb, hash: u64, c: vision.Config, max_tokens: u32) Error!Prepared {
    const g = try geometry(img.h, img.w, @min(c.min_tokens, max_tokens), max_tokens);
    const content = try resize(gpa, img, g.content_w, g.content_h);
    defer content.deinit(gpa);
    const pixels = try patchify(gpa, content, g, c.mean, c.std);
    const gh = g.target_h / vision.patch;
    const gw = g.target_w / vision.patch;
    return .{ .pixels = pixels, .gh = gh, .gw = gw, .tokens = gh * gw / vision.patches_per_token, .hash = hash };
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

test "orientations move pixels as PIL's transposes do" {
    const gpa = std.testing.allocator;
    // a 3x2 image of pixel indices: 0 1 2 / 3 4 5
    var px: [18]u8 = undefined;
    for (0..6) |i| @memset(px[i * 3 ..][0..3], @intCast(i));
    const src: Rgb = .{ .w = 3, .h = 2, .px = &px };
    const want = [_][]const u8{ &.{ 2, 1, 0, 5, 4, 3 }, &.{ 5, 4, 3, 2, 1, 0 }, &.{ 3, 4, 5, 0, 1, 2 }, &.{ 0, 3, 1, 4, 2, 5 }, &.{ 3, 0, 4, 1, 5, 2 }, &.{ 5, 2, 4, 1, 3, 0 }, &.{ 2, 5, 1, 4, 0, 3 } };
    for (want, 2..) |w, o| {
        const out = try orient(gpa, src, @intCast(o));
        defer out.deinit(gpa);
        for (w, 0..) |v, i| try std.testing.expectEqual(v, out.px[i * 3]);
    }
}

test "alpha over white follows PIL's DIV255" {
    try std.testing.expectEqual(@as(u8, 255), overWhite(0, 0));
    try std.testing.expectEqual(@as(u8, 10), overWhite(10, 255));
    try std.testing.expectEqual(@as(u8, 132), overWhite(10, 128)); // DIV255(255 * 127 + 10 * 128)
}

test "resize keeps a flat image flat and its size" {
    const gpa = std.testing.allocator;
    const px = try gpa.alloc(u8, 7 * 5 * 3);
    defer gpa.free(px);
    @memset(px, 200);
    const out = try resize(gpa, .{ .w = 7, .h = 5, .px = px }, 13, 3);
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 13), out.w);
    for (out.px) |v| try std.testing.expectEqual(@as(u8, 200), v);
}
