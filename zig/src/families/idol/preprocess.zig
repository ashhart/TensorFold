//! idol-sdk's preprocess_image on the host, byte for byte as Pillow 13's LANCZOS and torch's CPU compute it.
const std = @import("std");
const Allocator = std.mem.Allocator;
const picture = @import("../../core/picture/picture.zig");

pub const size = 1024;
/// CROP_MARGIN by mode: renders and generated images ('synthetic'), photos ('real').
pub const Mode = enum { synthetic, real };
pub fn margin(m: Mode) f64 {
    return switch (m) {
        .synthetic => 1.2,
        .real => 1.7,
    };
}

/// 8-bit pixels: 3 (Pillow's RGB, or L read as RGB) or 4 (RGBA) channels a pixel.
pub const Image = struct {
    width: usize,
    height: usize,
    channels: usize,
    pixels: []u8,

    pub fn deinit(im: *Image, a: Allocator) void {
        a.free(im.pixels);
        im.* = undefined;
    }
};

/// The words for a photo this path refuses, where Pillow's mode would take another path through the SDK.
pub const unsupported = "only 8-bit RGB, RGBA or grayscale PNGs, RGB or grayscale JPEGs and WebPs are supported";

/// PNG, JPEG or WebP bytes as Image.open gives them: no EXIF transpose, RGBA only where Pillow's mode is RGBA.
pub fn decode(io: std.Io, a: Allocator, bytes: []const u8) !Image {
    const format = picture.sniff(bytes);
    var rgba = false;
    switch (format) {
        .png => {
            // IHDR: depth at 24, colour type at 25; Pillow's P, LA, I;16 and 1-bit modes take other paths
            if (bytes.len < 33 or bytes[24] != 8) return error.UnsupportedPhoto;
            switch (bytes[25]) {
                0, 2 => {},
                6 => rgba = true,
                else => return error.UnsupportedPhoto,
            }
        },
        .webp => rgba = webpAlpha(bytes),
        .jpeg => {},
        .unknown => return error.UnsupportedPhoto,
    }
    const s = switch (format) {
        .png => try picture.png.decode(a, bytes),
        .jpeg => try picture.jpeg.decode(io, a, bytes),
        .webp => try picture.webp.decode(io, a, bytes),
        .unknown => unreachable,
    };
    const ch: usize = if (rgba) 4 else 3;
    if (s.channels == ch) return .{ .width = s.width, .height = s.height, .channels = ch, .pixels = s.data };
    defer a.free(s.data);
    const out = try a.alloc(u8, s.width * s.height * ch);
    for (0..s.width * s.height) |i| @memcpy(out[i * ch ..][0..ch], s.data[i * s.channels ..][0..ch]);
    return .{ .width = s.width, .height = s.height, .channels = ch, .pixels = out };
}

/// Whether Pillow opens a WebP as RGBA: VP8X's alpha flag, or a lossless image's alpha bit.
fn webpAlpha(b: []const u8) bool {
    if (b.len < 30) return false;
    if (std.mem.eql(u8, b[12..16], "VP8X")) return b[20] & 0x10 != 0;
    if (std.mem.eql(u8, b[12..16], "VP8L")) return (std.mem.readInt(u32, b[21..25], .little) >> 28) & 1 != 0;
    return false;
}

// --- Pillow's resample (Resample.c), LANCZOS, 8 bits: double coefficients, fixed point with 22 fraction bits -------

const precision = 22;

extern "c" fn sin(x: f64) f64;

fn sinc(x0: f64) f64 {
    if (x0 == 0.0) return 1.0;
    const x = x0 * std.math.pi;
    return sin(x) / x;
}

fn lanczos(x: f64) f64 {
    return if (-3.0 <= x and x < 3.0) sinc(x) * sinc(x / 3) else 0.0;
}

/// One axis's taps: each output's first input and fixed-point coefficients (precompute_coeffs, normalize_coeffs_8bpc).
const Taps = struct {
    first: []usize,
    count: []usize,
    k: []i32,
    ksize: usize,

    fn init(a: Allocator, in: usize, out: usize) !Taps {
        const scale = @as(f64, @floatFromInt(in)) / @as(f64, @floatFromInt(out));
        const fs = @max(scale, 1.0);
        const support = 3.0 * fs;
        const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const t: Taps = .{ .first = try a.alloc(usize, out), .count = try a.alloc(usize, out), .k = try a.alloc(i32, out * ksize), .ksize = ksize };
        const w = try a.alloc(f64, ksize);
        defer a.free(w);
        for (0..out) |xx| {
            const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
            const ss = 1.0 / fs;
            const lo: i64 = @max(@as(i64, @intFromFloat(center - support + 0.5)), 0);
            const hi: i64 = @min(@as(i64, @intFromFloat(center + support + 0.5)), @as(i64, @intCast(in)));
            const n: usize = @intCast(hi - lo);
            var ww: f64 = 0;
            for (0..n) |x| {
                w[x] = lanczos((@as(f64, @floatFromInt(@as(i64, @intCast(x)) + lo)) - center + 0.5) * ss);
                ww += w[x];
            }
            const k = t.k[xx * ksize ..][0..ksize];
            @memset(k, 0);
            for (0..n) |x| {
                const v = if (ww != 0.0) w[x] / ww else w[x];
                const f = v * @as(f64, 1 << precision);
                k[x] = @intFromFloat(if (v < 0) -0.5 + f else 0.5 + f);
            }
            t.first[xx] = @intCast(lo);
            t.count[xx] = n;
        }
        return t;
    }

    fn deinit(t: *Taps, a: Allocator) void {
        a.free(t.first);
        a.free(t.count);
        a.free(t.k);
    }
};

fn clip8(ss: i32) u8 {
    return @intCast(std.math.clamp(ss >> precision, 0, 255));
}

/// Image.resize(LANCZOS) of 8-bit pixels: width, then height; a pass keeping its size is skipped, as Pillow skips it.
pub fn resample(a: Allocator, src: []const u8, w: usize, h: usize, ch: usize, ow: usize, oh: usize) ![]u8 {
    var cur = try a.dupe(u8, src);
    var cw = w;
    if (ow != w) {
        var t = try Taps.init(a, w, ow);
        defer t.deinit(a);
        const out = try a.alloc(u8, ow * h * ch);
        for (0..h) |y| for (0..ow) |x| for (0..ch) |c| {
            var ss: i32 = 1 << (precision - 1);
            const k = t.k[x * t.ksize ..];
            for (0..t.count[x]) |i| ss += @as(i32, cur[(y * w + t.first[x] + i) * ch + c]) * k[i];
            out[(y * ow + x) * ch + c] = clip8(ss);
        };
        a.free(cur);
        cur = out;
        cw = ow;
    }
    if (oh != h) {
        var t = try Taps.init(a, h, oh);
        defer t.deinit(a);
        const out = try a.alloc(u8, cw * oh * ch);
        for (0..oh) |y| {
            const k = t.k[y * t.ksize ..];
            for (0..cw * ch) |xc| {
                var ss: i32 = 1 << (precision - 1);
                for (0..t.count[y]) |i| ss += @as(i32, cur[(t.first[y] + i) * cw * ch + xc]) * k[i];
                out[y * cw * ch + xc] = clip8(ss);
            }
        }
        a.free(cur);
        cur = out;
    }
    return cur;
}

fn muldiv255(v: u32, b: u32) u8 {
    const t = v * b + 128;
    return @intCast((t + (t >> 8)) >> 8);
}

/// Image.resize of `im` as Pillow does it: RGBA through RGBa (premultiplied by MULDIV255, back by 255 v / a truncated).
pub fn resize(a: Allocator, im: Image, ow: usize, oh: usize) !Image {
    if (im.width == ow and im.height == oh) return .{ .width = ow, .height = oh, .channels = im.channels, .pixels = try a.dupe(u8, im.pixels) };
    if (im.channels == 3) return .{ .width = ow, .height = oh, .channels = 3, .pixels = try resample(a, im.pixels, im.width, im.height, 3, ow, oh) };
    const pre = try a.dupe(u8, im.pixels);
    defer a.free(pre);
    for (0..im.width * im.height) |i| {
        const px = pre[i * 4 ..][0..4];
        for (px[0..3]) |*c| c.* = muldiv255(c.*, px[3]);
    }
    const out = try resample(a, pre, im.width, im.height, 4, ow, oh);
    for (0..ow * oh) |i| {
        const px = out[i * 4 ..][0..4];
        const al: u32 = px[3];
        if (al == 0 or al == 255) continue;
        for (px[0..3]) |*c| c.* = @intCast(@min(255, (255 * @as(u32, c.*)) / al));
    }
    return .{ .width = ow, .height = oh, .channels = 4, .pixels = out };
}

/// The first step: the short side to 1,024, each side int(round(side * 1024 / short)) (Python's round: half to even).
pub fn resizedSize(w: usize, h: usize) [2]usize {
    const scale = @as(f64, size) / @as(f64, @floatFromInt(@min(w, h)));
    return .{ @max(1, roundEven(@as(f64, @floatFromInt(w)) * scale)), @max(1, roundEven(@as(f64, @floatFromInt(h)) * scale)) };
}

fn roundEven(v: f64) usize {
    const f = @floor(v);
    const d = v - f;
    const r = if (d > 0.5) f + 1 else if (d < 0.5) f else if (@mod(f, 2) == 0) f else f + 1;
    return @intFromFloat(r);
}

/// has_real_alpha: Pillow's mode RGBA with some alpha under 255.
pub fn hasRealAlpha(im: Image) bool {
    if (im.channels != 4) return false;
    for (0..im.width * im.height) |i| if (im.pixels[i * 4 + 3] < 255) return true;
    return false;
}

/// BiRefNet's input [1024 * 1024, 3]: x / 255, torch's CPU bilinear (align_corners) to 1,024, (x - mean) / std.
pub fn input(a: Allocator, im: Image, out: []f32) !void {
    std.debug.assert(out.len == size * size * 3);
    const w = im.width;
    const h = im.height;
    const v = try a.alloc(f32, w * h * 3);
    defer a.free(v);
    for (0..w * h) |i| for (0..3) |c| {
        v[i * 3 + c] = @as(f32, @floatFromInt(im.pixels[i * im.channels + c])) / 255.0;
    };
    const mean = [3]f32{ 0.485, 0.456, 0.406 };
    const sd = [3]f32{ 0.229, 0.224, 0.225 };
    const ys = axis(h, size);
    const xs = axis(w, size);
    for (0..size) |oy| {
        const y = ys.at(oy);
        for (0..size) |ox| {
            const x = xs.at(ox);
            for (0..3) |c| {
                const t = bilinear(v, w, c, y, x);
                out[(oy * size + ox) * 3 + c] = (t - mean[c]) / sd[c];
            }
        }
    }
}

/// One axis of an align_corners resize: scale (in - 1) / (out - 1) in fp32 (0 for one output).
pub const Axis = struct {
    scale: f32,
    in: usize,
    pub const Tap = struct { lo: usize, hi: usize, l0: f32, l1: f32 };
    pub fn at(ax: Axis, o: usize) Tap {
        const real = ax.scale * @as(f32, @floatFromInt(o));
        const lo = @min(@as(usize, @intFromFloat(real)), ax.in - 1);
        const l1 = @min(@max(real - @as(f32, @floatFromInt(lo)), 0), 1);
        return .{ .lo = lo, .hi = @min(lo + 1, ax.in - 1), .l0 = 1 - l1, .l1 = l1 };
    }
};

pub fn axis(in: usize, out: usize) Axis {
    const scale: f32 = if (out > 1) @as(f32, @floatFromInt(in - 1)) / @as(f32, @floatFromInt(out - 1)) else 0;
    return .{ .scale = scale, .in = in };
}

/// torch's CPU bilinear of channel c, fma(t0, w0, t1 w1) along each axis: its bytes on a photo resized both ways.
fn bilinear(v: []const f32, w: usize, c: usize, y: Axis.Tap, x: Axis.Tap) f32 {
    const row0 = @mulAdd(f32, v[(y.lo * w + x.lo) * 3 + c], x.l0, v[(y.lo * w + x.hi) * 3 + c] * x.l1);
    const row1 = @mulAdd(f32, v[(y.hi * w + x.lo) * 3 + c], x.l0, v[(y.hi * w + x.hi) * 3 + c] * x.l1);
    return @mulAdd(f32, row0, y.l0, row1 * y.l1);
}

/// remove_background's alpha [h, w]: sigmoid, bilinear (align_corners), clamp * 255 truncated, in fp32 as on MPS.
pub fn alpha(a: Allocator, logits: []const f32, w: usize, h: usize, out: []u8) !void {
    std.debug.assert(logits.len == size * size and out.len == w * h);
    const s = try a.alloc(f32, logits.len);
    defer a.free(s);
    for (s, logits) |*o, l| o.* = 1.0 / (1.0 + @exp(-l));
    const ys = axis(size, h);
    const xs = axis(size, w);
    for (0..h) |oy| {
        const y = ys.at(oy);
        for (0..w) |ox| {
            const x = xs.at(ox);
            const top = (1 - x.l1) * s[y.lo * size + x.lo] + x.l1 * s[y.lo * size + x.hi];
            const bottom = (1 - x.l1) * s[y.hi * size + x.lo] + x.l1 * s[y.hi * size + x.hi];
            const v = (1 - y.l1) * top + y.l1 * bottom;
            out[oy * w + ox] = @intFromFloat(std.math.clamp(v, 0, 1) * 255);
        }
    }
}

/// The steps after the alpha: erode, crop, resize and paste on black, into the 1024 x 1024 RGB the encoders see.
pub fn finish(a: Allocator, im: Image, given: ?[]const u8, crop_margin: f64, out: []u8) !void {
    std.debug.assert(out.len == size * size * 3);
    const w = im.width;
    const h = im.height;
    const al = try a.alloc(u8, w * h);
    defer a.free(al);
    for (al, 0..) |*o, i| o.* = if (given) |g| g[i] else im.pixels[i * 4 + 3];
    // ImageFilter.MinFilter(3): the minimum over the 3 x 3 neighbours inside the picture
    const er = try a.alloc(u8, w * h);
    defer a.free(er);
    var left: usize = w;
    var top: usize = h;
    var right: usize = 0;
    var bottom: usize = 0;
    for (0..h) |y| for (0..w) |x| {
        var m: u8 = 255;
        for (@max(y, 1) - 1..@min(y + 2, h)) |yy| for (@max(x, 1) - 1..@min(x + 2, w)) |xx| {
            m = @min(m, al[yy * w + xx]);
        };
        er[y * w + x] = m;
        if (m != 0) {
            left = @min(left, x);
            right = @max(right, x);
            top = @min(top, y);
            bottom = @max(bottom, y);
        }
    };
    if (left > right) return error.NoObject;
    const cx = @as(f64, @floatFromInt(left + right)) / 2;
    const cy = @as(f64, @floatFromInt(top + bottom)) / 2;
    const half = @as(f64, @floatFromInt(@max(right - left, bottom - top))) / 2 * crop_margin;
    // Python's int() truncates toward zero; Pillow's crop fills outside the picture with zeros
    const box = [4]i64{ @intFromFloat(cx - half), @intFromFloat(cy - half), @intFromFloat(cx + half), @intFromFloat(cy + half) };
    if (box[2] <= box[0] or box[3] <= box[1]) return error.NoObject;
    const cw: usize = @intCast(box[2] - box[0]);
    const ch: usize = @intCast(box[3] - box[1]);
    const crop = try a.alloc(u8, cw * ch * 4);
    defer a.free(crop);
    @memset(crop, 0);
    for (0..ch) |y| {
        const sy = box[1] + @as(i64, @intCast(y));
        if (sy < 0 or sy >= h) continue;
        for (0..cw) |x| {
            const sx = box[0] + @as(i64, @intCast(x));
            if (sx < 0 or sx >= w) continue;
            const s: usize = @intCast(sy * @as(i64, @intCast(w)) + sx);
            const d = crop[(y * cw + x) * 4 ..][0..4];
            @memcpy(d[0..3], im.pixels[s * im.channels ..][0..3]);
            d[3] = er[s];
        }
    }
    var big = try resize(a, .{ .width = cw, .height = ch, .channels = 4, .pixels = crop }, size, size);
    defer big.deinit(a);
    // paste on black with the alpha as the mask: v * a / 255 by Pillow's BLEND
    for (0..size * size) |i| {
        const px = big.pixels[i * 4 ..][0..4];
        for (0..3) |c| out[i * 3 + c] = muldiv255(px[c], px[3]);
    }
}

/// preprocess's first step on a decoded photo: its short side to 1,024.
pub fn resizeFirst(a: Allocator, im: Image) !Image {
    const s = resizedSize(im.width, im.height);
    return resize(a, im, s[0], s[1]);
}

test "Python's round halves to even, as preprocess's int(round(...)) sizes do" {
    try std.testing.expectEqual(@as(usize, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(usize, 4), roundEven(3.5));
    try std.testing.expectEqual(@as(usize, 1365), resizedSize(1280, 960)[0]);
    try std.testing.expectEqual(@as(usize, 1024), resizedSize(1280, 960)[1]);
}

test "Pillow's LANCZOS fixed point keeps a flat picture flat and its premultiply round-trips opaque pixels" {
    const a = std.testing.allocator;
    const src = try a.alloc(u8, 37 * 53 * 3);
    defer a.free(src);
    @memset(src, 200);
    const out = try resample(a, src, 37, 53, 3, 100, 64);
    defer a.free(out);
    for (out) |v| try std.testing.expectEqual(@as(u8, 200), v);
    for (0..256) |v| try std.testing.expectEqual(@as(u8, @intCast(v)), muldiv255(@intCast(v), 255));
}
