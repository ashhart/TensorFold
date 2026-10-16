//! An image's bytes to Qwen3.5 vision patches as 0.6.6's --vision makes them: PIL's decode, smart_resize and bicubic.
const std = @import("std");
const vision = @import("vision.zig");

extern fn stbi_load_from_memory(buffer: [*]const u8, len: c_int, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[*]u8;
extern fn stbi_load_16_from_memory(buffer: [*]const u8, len: c_int, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[*]u16;
extern fn stbi_is_16_bit_from_memory(buffer: [*]const u8, len: c_int) c_int;
extern fn stbi_info_from_memory(buffer: [*]const u8, len: c_int, x: *c_int, y: *c_int, comp: *c_int) c_int;
extern fn stbi_image_free(p: ?*anyopaque) void;

pub const Rgb = struct {
    w: u32,
    h: u32,
    px: []u8, // row-major RGB, 3 bytes a pixel

    pub fn deinit(r: Rgb, gpa: std.mem.Allocator) void {
        gpa.free(r.px);
    }
};

/// One image ready for the tower: its patches [gh * gw, 1536] in merge order, its grid and its tokens (gh * gw / 4).
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

pub const Error = error{ BadImage, ImageTooLarge, Animated, ImageTooSmall, OutOfMemory };

/// images.py's ImageLimits: sides and decoded pixels (the server bounds the encoded bytes).
pub const max_side = 8192;
pub const max_pixels = 16 << 20;
const factor = vision.patch * vision.merge; // 32

/// Bytes to patches for at most `max_tokens` tokens (0.6.6's per-image cap: the request's budget shared, 4096 at most).
pub fn prepare(gpa: std.mem.Allocator, bytes: []const u8, max_tokens: u32) Error!Prepared {
    const img = try decode(gpa, bytes);
    defer img.deinit(gpa);
    return prepareRgb(gpa, img, std.hash.Wyhash.hash(0x51e7, bytes), max_tokens);
}

pub fn prepareRgb(gpa: std.mem.Allocator, img: Rgb, hash: u64, max_tokens: u32) Error!Prepared {
    const tokens = @min(max_tokens, vision.max_patches / (vision.merge * vision.merge));
    if (tokens == 0) return error.ImageTooSmall;
    const size = try smartResize(img.h, img.w, factor * factor, @as(u64, tokens) * factor * factor);
    const resized = try resize(gpa, img, size.w, size.h);
    defer resized.deinit(gpa);
    const gh = size.h / vision.patch;
    const gw = size.w / vision.patch;
    return .{ .pixels = try patchify(gpa, resized), .gh = gh, .gw = gw, .tokens = gh * gw / (vision.merge * vision.merge), .hash = hash };
}

pub const Size = struct { h: u32, w: u32 };

/// transformers' smart_resize: sides multiples of `factor` (half-to-even), pixels within [min_pixels, max_pixels].
pub fn smartResize(height: u32, width: u32, min_pixels: u64, max_pixels_: u64) Error!Size {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    if (@max(h, w) / @min(h, w) > 200) return error.BadImage;
    const f: f64 = factor;
    var hb: f64 = roundEven(h / f) * f;
    var wb: f64 = roundEven(w / f) * f;
    if (hb * wb > @as(f64, @floatFromInt(max_pixels_))) {
        const beta = @sqrt(h * w / @as(f64, @floatFromInt(max_pixels_)));
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (hb * wb < @as(f64, @floatFromInt(min_pixels))) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ .h = @intFromFloat(hb), .w = @intFromFloat(wb) };
}

fn roundEven(x: f64) f64 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r;
}

/// The resized image as x / 255 (float64 to float32), normalized by 0.5 and 0.5, in 16x16 patches of 2x2 merge order.
pub fn patchify(gpa: std.mem.Allocator, img: Rgb) Error![]f32 {
    const P = vision.patch;
    const M = vision.merge;
    const T = vision.temporal;
    const gh = img.h / P;
    const gw = img.w / P;
    const out = try gpa.alloc(f32, @as(usize, gh) * gw * vision.patch_width);
    var lut: [256]f32 = undefined;
    for (&lut, 0..) |*v, i| {
        const scaled: f32 = @floatCast(@as(f64, @floatFromInt(i)) * (1.0 / 255.0));
        v.* = (scaled - 0.5) / 0.5;
    }
    var row: usize = 0;
    for (0..gh / M) |by| for (0..gw / M) |bx| for (0..M) |my| for (0..M) |mx| {
        const y0 = (by * M + my) * P;
        const x0 = (bx * M + mx) * P;
        const dst = out[row * vision.patch_width ..][0..vision.patch_width];
        for (0..3) |ch| for (0..T) |t| for (0..P) |dy| for (0..P) |dx| {
            dst[((ch * T + t) * P + dy) * P + dx] = lut[img.px[((y0 + dy) * img.w + x0 + dx) * 3 + ch]];
        };
        row += 1;
    };
    return out;
}

// ---- decoding (stb_image), EXIF orientation and alpha, as PIL gives images.py its RGB ----

/// TurboJPEG's decompression API (libturbojpeg.so.0), opened on first use; null when the library is absent.
const Turbo = struct {
    init: *const fn () callconv(.c) ?*anyopaque,
    header: *const fn (?*anyopaque, [*]const u8, c_ulong, *c_int, *c_int, *c_int, *c_int) callconv(.c) c_int,
    decompress: *const fn (?*anyopaque, [*]const u8, c_ulong, [*]u8, c_int, c_int, c_int, c_int, c_int) callconv(.c) c_int,
    destroy: *const fn (?*anyopaque) callconv(.c) c_int,

    var state = std.atomic.Value(u8).init(0); // 0: not opened, 1: opening, 2: done (api set or not)
    var api: ?Turbo = null;
    var lib: ?std.DynLib = null;

    fn load() void {
        var l = std.DynLib.open("libturbojpeg.so.0") catch return;
        api = .{
            .init = l.lookup(@FieldType(Turbo, "init"), "tjInitDecompress") orelse return l.close(),
            .header = l.lookup(@FieldType(Turbo, "header"), "tjDecompressHeader3") orelse return l.close(),
            .decompress = l.lookup(@FieldType(Turbo, "decompress"), "tjDecompress2") orelse return l.close(),
            .destroy = l.lookup(@FieldType(Turbo, "destroy"), "tjDestroy") orelse return l.close(),
        };
        lib = l;
    }

    fn get() ?Turbo {
        if (state.cmpxchgStrong(0, 1, .acquire, .acquire) == null) {
            load();
            state.store(2, .release);
        }
        while (state.load(.acquire) != 2) std.atomic.spinLoopHint();
        return api;
    }
};

const tjpf_rgb = 0;
const tjcs_cmyk = 3;
const tjcs_ycck = 4;

/// A JPEG through libjpeg-turbo as PIL decodes it to RGB rows; null without TurboJPEG or for CMYK (PIL converts those).
fn turboJpeg(gpa: std.mem.Allocator, bytes: []const u8) Error!?Rgb {
    const t = Turbo.get() orelse return null;
    const h = t.init() orelse return null;
    defer _ = t.destroy(h);
    var w: c_int = 0;
    var ht: c_int = 0;
    var sub: c_int = 0;
    var cs: c_int = 0;
    if (t.header(h, bytes.ptr, bytes.len, &w, &ht, &sub, &cs) != 0) return error.BadImage;
    if (cs == tjcs_cmyk or cs == tjcs_ycck) return null;
    if (w <= 0 or ht <= 0) return error.BadImage;
    if (w > max_side or ht > max_side or @as(u64, @intCast(w)) * @as(u64, @intCast(ht)) > max_pixels) return error.ImageTooLarge;
    const W: usize = @intCast(w);
    const H: usize = @intCast(ht);
    const out = try gpa.alloc(u8, W * H * 3);
    errdefer gpa.free(out);
    if (t.decompress(h, bytes.ptr, bytes.len, out.ptr, w, w * 3, ht, tjpf_rgb, 0) != 0) return error.BadImage;
    return .{ .w = @intCast(W), .h = @intCast(H), .px = out };
}

/// A single-frame PNG or JPEG as 8-bit RGB: upright by EXIF, alpha over white, as PIL's convert("RGB") gives.
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8) Error!Rgb {
    if (bytes.len < 8 or bytes.len > std.math.maxInt(c_int)) return error.BadImage;
    const n: c_int = @intCast(bytes.len);
    if (std.mem.startsWith(u8, bytes, "\x89PNG\r\n\x1a\n") and pngAnimated(bytes)) return error.Animated;
    if (std.mem.startsWith(u8, bytes, "\xff\xd8\xff")) if (try turboJpeg(gpa, bytes)) |img| return upright(gpa, img, bytes);
    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    if (stbi_info_from_memory(bytes.ptr, n, &w, &h, &comp) == 0) return error.BadImage;
    if (w <= 0 or h <= 0) return error.BadImage;
    if (w > max_side or h > max_side or @as(u64, @intCast(w)) * @as(u64, @intCast(h)) > max_pixels) return error.ImageTooLarge;
    const W: usize = @intCast(w);
    const H: usize = @intCast(h);
    const out = try gpa.alloc(u8, W * H * 3);
    errdefer gpa.free(out);
    if (stbi_is_16_bit_from_memory(bytes.ptr, n) != 0 and comp == 1) {
        const p = stbi_load_16_from_memory(bytes.ptr, n, &w, &h, &comp, 0) orelse return error.BadImage;
        defer stbi_image_free(p);
        for (0..W * H) |i| @memset(out[i * 3 ..][0..3], @intCast(@min(p[i], 255)));
    } else {
        const p = stbi_load_from_memory(bytes.ptr, n, &w, &h, &comp, 0) orelse return error.BadImage;
        defer stbi_image_free(p);
        const c: usize = @intCast(comp);
        for (0..W * H) |i| {
            const s = p[i * c ..][0..c];
            const px = out[i * 3 ..][0..3];
            switch (c) {
                1 => @memset(px, s[0]),
                2 => @memset(px, overWhite(s[0], s[1])),
                3 => @memcpy(px, s[0..3]),
                4 => for (0..3) |k| {
                    px[k] = overWhite(s[k], s[3]);
                },
                else => return error.BadImage,
            }
        }
    }
    return upright(gpa, .{ .w = @intCast(W), .h = @intCast(H), .px = out }, bytes);
}

/// The decoded image turned upright by the EXIF orientation its bytes carry (it takes `img`).
fn upright(gpa: std.mem.Allocator, img: Rgb, bytes: []const u8) Error!Rgb {
    const o = orientation(bytes);
    if (o < 2 or o > 8) return img;
    defer img.deinit(gpa);
    return orient(gpa, img, o);
}

/// An APNG (an acTL chunk before the image data) has more than one frame: images.py refuses it.
fn pngAnimated(bytes: []const u8) bool {
    var at: usize = 8;
    while (at + 8 <= bytes.len) {
        const len = std.mem.readInt(u32, bytes[at..][0..4], .big);
        const kind = bytes[at + 4 ..][0..4];
        if (std.mem.eql(u8, kind, "acTL")) return true;
        if (std.mem.eql(u8, kind, "IDAT")) return false;
        at += 12 + @as(usize, len);
    }
    return false;
}

/// The EXIF orientation (tag 0x0112 of IFD0) a JPEG's APP1 or a PNG's eXIf chunk carries; 1 when none.
pub fn orientation(bytes: []const u8) u32 {
    if (std.mem.startsWith(u8, bytes, "\x89PNG\r\n\x1a\n")) {
        var at: usize = 8;
        while (at + 12 <= bytes.len) {
            const len: usize = std.mem.readInt(u32, bytes[at..][0..4], .big);
            if (at + 12 + len > bytes.len) break;
            if (std.mem.eql(u8, bytes[at + 4 ..][0..4], "eXIf")) return tiffOrientation(bytes[at + 8 ..][0..len]);
            at += 12 + len;
        }
        return 1;
    }
    var at: usize = 2;
    while (at + 4 <= bytes.len and bytes[at] == 0xff) {
        const marker = bytes[at + 1];
        if (marker == 0xd9 or marker == 0xda) break; // end of image, start of scan
        const len: usize = std.mem.readInt(u16, bytes[at + 2 ..][0..2], .big);
        if (len < 2 or at + 2 + len > bytes.len) break;
        const seg = bytes[at + 4 .. at + 2 + len];
        if (marker == 0xe1 and std.mem.startsWith(u8, seg, "Exif\x00\x00")) return tiffOrientation(seg[6..]);
        at += 2 + len;
    }
    return 1;
}

fn tiffOrientation(t: []const u8) u32 {
    if (t.len < 8) return 1;
    const endian: std.builtin.Endian = if (std.mem.eql(u8, t[0..2], "II")) .little else if (std.mem.eql(u8, t[0..2], "MM")) .big else return 1;
    if (std.mem.readInt(u16, t[2..4], endian) != 42) return 1;
    const ifd: usize = std.mem.readInt(u32, t[4..8], endian);
    if (ifd + 2 > t.len) return 1;
    const count: usize = std.mem.readInt(u16, t[ifd..][0..2], endian);
    for (0..count) |i| {
        const e = ifd + 2 + i * 12;
        if (e + 12 > t.len) return 1;
        if (std.mem.readInt(u16, t[e..][0..2], endian) != 0x0112) continue;
        if (std.mem.readInt(u16, t[e + 2 ..][0..2], endian) != 3) return 1; // SHORT
        return std.mem.readInt(u16, t[e + 8 ..][0..2], endian);
    }
    return 1;
}

// Below: GLM's image.zig (families/glm) PIL arithmetic, copied since the CUDA module cannot import that family's files.

/// PIL's exif_transpose for orientations 2-8; each output pixel read from its source pixel.
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
    if (w != src.w) {
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

test "smart_resize rounds half to even and respects the pixel bounds" {
    try std.testing.expectEqual(Size{ .h = 32, .w = 64 }, try smartResize(15, 20, 1024, 4096 * 1024));
    try std.testing.expectEqual(Size{ .h = 384, .w = 512 }, try smartResize(389, 517, 1024, 4096 * 1024));
    try std.testing.expectEqual(@as(f64, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f64, 2), roundEven(1.5));
}
