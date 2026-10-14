//! Image inputs for Qwen-style vision towers, as the Python engine prepares them (tensorfold/vision/images.py and
//! qwen_processing.py with transformers' Qwen2VL processor): decode (JPEG, PNG; EXIF orientation applied, alpha over
//! white), smart_resize to multiples of patch x merge within a pixel budget, an antialiased bicubic resize (PIL's
//! filter and bounds with torch's uint8 fixed point: int16 weights at the most precision the axis allows),
//! (x - 127.5) / 127.5, and the patches in merge-block order,
//! each [channel, frame (the image twice), 16, 16] -- 1,536 values.

const std = @import("std");
/// GLM-5.3-Flash's preparation (its canvas, padding, normalization and patch order) on this module's decode and resize.
pub const glm = @import("glm_image.zig");

extern fn stbi_info_from_memory(buffer: [*]const u8, len: c_int, x: *c_int, y: *c_int, comp: *c_int) c_int;
extern fn stbi_load_from_memory(buffer: [*]const u8, len: c_int, x: *c_int, y: *c_int, comp: *c_int, req: c_int) ?[*]u8;
extern fn stbi_image_free(p: ?*anyopaque) void;

pub const Limits = struct {
    max_dimension: u32 = 8192,
    max_pixels: u64 = 16 * 1024 * 1024,
};

/// An RGB image, rows top to bottom, three bytes a pixel.
pub const Image = struct {
    w: u32,
    h: u32,
    rgb: []u8,

    pub fn deinit(im: *Image, gpa: std.mem.Allocator) void {
        gpa.free(im.rgb);
    }
};

pub const Error = error{ UnsupportedImage, ImageTooLarge, OutOfMemory, BadAspectRatio };

/// JPEG or PNG bytes to RGB: EXIF orientation applied (JPEG), any alpha composited over white.
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, limits: Limits) Error!Image {
    if (bytes.len == 0 or bytes.len > std.math.maxInt(c_int)) return error.UnsupportedImage;
    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    if (stbi_info_from_memory(bytes.ptr, @intCast(bytes.len), &w, &h, &comp) == 0) return error.UnsupportedImage;
    if (w <= 0 or h <= 0 or w > limits.max_dimension or h > limits.max_dimension or
        @as(u64, @intCast(w)) * @as(u64, @intCast(h)) > limits.max_pixels) return error.ImageTooLarge;
    const alpha = comp == 2 or comp == 4;
    const want: c_int = if (alpha) 4 else 3;
    var got: c_int = 0;
    const px = stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &w, &h, &comp, want) orelse return error.UnsupportedImage;
    defer stbi_image_free(px);
    const n: usize = @intCast(w * h);
    _ = &got;
    var rgb = try gpa.alloc(u8, n * 3);
    errdefer gpa.free(rgb);
    if (alpha) {
        // PIL's paste of RGBA onto white with the alpha as mask: out = white + (src - white) * a / 255, rounded
        for (0..n) |i| {
            const a: i32 = px[i * 4 + 3];
            for (0..3) |c| {
                const s: i32 = px[i * 4 + c];
                const t: i32 = (s - 255) * a + 128;
                rgb[i * 3 + c] = @intCast(255 + ((t + (t >> 8)) >> 8));
            }
        }
    } else @memcpy(rgb, px[0 .. n * 3]);
    var im: Image = .{ .w = @intCast(w), .h = @intCast(h), .rgb = rgb };
    const o = orientation(bytes);
    if (o > 1) {
        const t = try transpose(gpa, im, o);
        im.deinit(gpa);
        im = t;
    }
    return im;
}

/// A JPEG's EXIF orientation (1-8; 1 when absent or not a JPEG).
fn orientation(b: []const u8) u16 {
    if (b.len < 4 or b[0] != 0xFF or b[1] != 0xD8) return 1;
    var i: usize = 2;
    while (i + 4 <= b.len and b[i] == 0xFF) {
        const marker = b[i + 1];
        const len = (@as(usize, b[i + 2]) << 8) | b[i + 3];
        if (marker == 0xDA or len < 2 or i + 2 + len > b.len) break;
        if (marker == 0xE1 and len >= 16 and std.mem.eql(u8, b[i + 4 .. i + 10], "Exif\x00\x00")) {
            const t = b[i + 10 .. i + 2 + len];
            const le = t[0] == 'I';
            const rd16 = struct {
                fn f(x: []const u8, at: usize, little: bool) u16 {
                    return if (little) std.mem.readInt(u16, x[at..][0..2], .little) else std.mem.readInt(u16, x[at..][0..2], .big);
                }
            }.f;
            const rd32 = struct {
                fn f(x: []const u8, at: usize, little: bool) u32 {
                    return if (little) std.mem.readInt(u32, x[at..][0..4], .little) else std.mem.readInt(u32, x[at..][0..4], .big);
                }
            }.f;
            if (t.len < 8) return 1;
            const ifd = rd32(t, 4, le);
            if (ifd + 2 > t.len) return 1;
            const count = rd16(t, ifd, le);
            for (0..count) |e| {
                const at = ifd + 2 + e * 12;
                if (at + 12 > t.len) return 1;
                if (rd16(t, at, le) == 0x0112) {
                    const v = rd16(t, at + 8, le);
                    return if (v >= 1 and v <= 8) v else 1;
                }
            }
            return 1;
        }
        i += 2 + len;
    }
    return 1;
}

/// PIL's exif_transpose for orientations 2-8.
fn transpose(gpa: std.mem.Allocator, im: Image, o: u16) Error!Image {
    const swap = o >= 5;
    const W: u32 = if (swap) im.h else im.w;
    const H: u32 = if (swap) im.w else im.h;
    const out = try gpa.alloc(u8, im.rgb.len);
    for (0..H) |y| for (0..W) |x| {
        // the source pixel of output (x, y)
        const sx: u32, const sy: u32 = switch (o) {
            2 => .{ im.w - 1 - @as(u32, @intCast(x)), @intCast(y) },
            3 => .{ im.w - 1 - @as(u32, @intCast(x)), im.h - 1 - @as(u32, @intCast(y)) },
            4 => .{ @intCast(x), im.h - 1 - @as(u32, @intCast(y)) },
            5 => .{ @intCast(y), @intCast(x) },
            6 => .{ @intCast(y), im.h - 1 - @as(u32, @intCast(x)) },
            7 => .{ im.w - 1 - @as(u32, @intCast(y)), im.h - 1 - @as(u32, @intCast(x)) },
            8 => .{ im.w - 1 - @as(u32, @intCast(y)), @intCast(x) },
            else => .{ @intCast(x), @intCast(y) },
        };
        @memcpy(out[(y * W + x) * 3 ..][0..3], im.rgb[(sy * im.w + sx) * 3 ..][0..3]);
    };
    return .{ .w = W, .h = H, .rgb = out };
}

/// Python's round(): halves to even.
fn pyRound(x: f64) f64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return 2.0 * @round(x / 2.0);
    return r;
}

/// transformers' smart_resize: both sides multiples of ``factor``, the area within [min_pixels, max_pixels].
pub fn smartResize(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64) Error![2]u32 {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const f: f64 = @floatFromInt(factor);
    if (@max(h, w) / @min(h, w) > 200) return error.BadAspectRatio;
    var hb = pyRound(h / f) * f;
    var wb = pyRound(w / f) * f;
    if (hb * wb > @as(f64, @floatFromInt(max_pixels))) {
        const beta = @sqrt(h * w / @as(f64, @floatFromInt(max_pixels)));
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (hb * wb < @as(f64, @floatFromInt(min_pixels))) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ @intFromFloat(hb), @intFromFloat(wb) };
}


fn bicubic(x0: f64) f64 {
    const a = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

const Coeffs = struct { ksize: usize, bounds: []usize, kk: []i32, precision: u5 };

/// PIL's precompute_coeffs for an axis of ``in_size`` resampled to ``out_size``, as fixed point the way torch's uint8
/// antialias kernel takes them: int16 weights at the largest precision (< 22 bits) that keeps the largest in range.
fn coeffs(gpa: std.mem.Allocator, in_size: usize, out_size: usize) Error!Coeffs {
    const scale = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const filterscale = @max(scale, 1.0);
    const support = 2.0 * filterscale;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    const bounds = try gpa.alloc(usize, out_size * 2);
    const kk = try gpa.alloc(i32, out_size * ksize);
    const pre_all = try gpa.alloc(f64, out_size * ksize);
    defer gpa.free(pre_all);
    for (0..out_size) |xx| {
        const pre = pre_all[xx * ksize ..][0..ksize];
        const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
        var ww: f64 = 0;
        const ss = 1.0 / filterscale;
        var xmin_i: i64 = @intFromFloat(center - support + 0.5);
        if (xmin_i < 0) xmin_i = 0;
        var xmax_i: i64 = @intFromFloat(center + support + 0.5);
        if (xmax_i > @as(i64, @intCast(in_size))) xmax_i = @intCast(in_size);
        const xmin: usize = @intCast(xmin_i);
        const xmax: usize = @intCast(xmax_i - xmin_i);
        @memset(pre, 0);
        for (0..xmax) |x| {
            const wgt = bicubic((@as(f64, @floatFromInt(x + xmin)) - center + 0.5) * ss);
            pre[x] = wgt;
            ww += wgt;
        }
        for (0..xmax) |x| {
            if (ww != 0.0) pre[x] /= ww;
        }
        bounds[xx * 2] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    var w_max: f64 = 0;
    for (pre_all) |v| w_max = @max(w_max, @abs(v));
    var precision: u5 = 0;
    while (precision < 22) : (precision += 1) {
        const next: i64 = @intFromFloat(0.5 + w_max * @as(f64, @floatFromInt(@as(i64, 1) << (precision + 1))));
        if (next >= (1 << 15)) break;
    }
    const one: f64 = @floatFromInt(@as(i64, 1) << precision);
    for (pre_all, 0..) |v, i| kk[i] = @intFromFloat(if (v < 0) -0.5 + v * one else 0.5 + v * one);
    return .{ .ksize = ksize, .bounds = bounds, .kk = kk, .precision = precision };
}

fn clip8(v: i64, precision: u5) u8 {
    const r = v >> precision;
    return @intCast(std.math.clamp(r, 0, 255));
}

/// The antialiased bicubic resize (PIL's ImagingResample, 8 bits: the horizontal pass, then the vertical, each
/// rounding to bytes).
pub fn resize(gpa: std.mem.Allocator, im: Image, ow: u32, oh: u32) Error!Image {
    var cur = im;
    var owned = false;
    defer if (owned) cur.deinit(gpa);
    if (ow != im.w) {
        const c = try coeffs(gpa, im.w, ow);
        defer gpa.free(c.bounds);
        defer gpa.free(c.kk);
        const out = try gpa.alloc(u8, @as(usize, ow) * im.h * 3);
        for (0..im.h) |y| for (0..ow) |x| {
            const xmin = c.bounds[x * 2];
            const xmax = c.bounds[x * 2 + 1];
            const k = c.kk[x * c.ksize ..];
            for (0..3) |ch| {
                var ss: i64 = @as(i64, 1) << (c.precision - 1);
                for (0..xmax) |j| ss += @as(i64, cur.rgb[(y * cur.w + xmin + j) * 3 + ch]) * k[j];
                out[(y * ow + x) * 3 + ch] = clip8(ss, c.precision);
            }
        };
        cur = .{ .w = ow, .h = im.h, .rgb = out };
        owned = true;
    }
    if (oh != cur.h) {
        const c = try coeffs(gpa, cur.h, oh);
        defer gpa.free(c.bounds);
        defer gpa.free(c.kk);
        const out = try gpa.alloc(u8, @as(usize, cur.w) * oh * 3);
        for (0..oh) |y| {
            const ymin = c.bounds[y * 2];
            const ymax = c.bounds[y * 2 + 1];
            const k = c.kk[y * c.ksize ..];
            for (0..cur.w) |x| for (0..3) |ch| {
                var ss: i64 = @as(i64, 1) << (c.precision - 1);
                for (0..ymax) |j| ss += @as(i64, cur.rgb[((ymin + j) * cur.w + x) * 3 + ch]) * k[j];
                out[(y * cur.w + x) * 3 + ch] = clip8(ss, c.precision);
            };
        }
        if (owned) cur.deinit(gpa);
        cur = .{ .w = cur.w, .h = oh, .rgb = out };
        owned = true;
    }
    if (!owned) return .{ .w = im.w, .h = im.h, .rgb = try gpa.dupe(u8, im.rgb) };
    owned = false;
    return cur;
}

/// One image prepared for the tower: its patches (fp32 [gh * gw, 1536], merge-block order) and grid [1, gh, gw].
pub const Prepared = struct {
    patches: []f32,
    grid: [3]i64,

    pub fn tokens(p: Prepared) usize {
        return @intCast(@divExact(p.grid[1] * p.grid[2], 4));
    }

    pub fn deinit(p: *Prepared, gpa: std.mem.Allocator) void {
        gpa.free(p.patches);
    }
};

pub const PATCH = 16;
pub const MERGE = 2;
pub const TEMPORAL = 2;
pub const FACTOR = PATCH * MERGE;

/// ``im`` resized within ``max_tokens`` visual tokens (min_pixels as the Python frontend passes it: one merged patch),
/// normalized and cut into patches.
pub fn prepare(gpa: std.mem.Allocator, im: Image, max_tokens: u64) Error!Prepared {
    const cap = max_tokens * FACTOR * FACTOR;
    const size = try smartResize(im.h, im.w, FACTOR, @min(FACTOR * FACTOR, cap), cap);
    var r = try resize(gpa, im, size[1], size[0]);
    defer r.deinit(gpa);
    const gh: usize = size[0] / PATCH;
    const gw: usize = size[1] / PATCH;
    const per = 3 * TEMPORAL * PATCH * PATCH;
    const out = try gpa.alloc(f32, gh * gw * per);
    var n: usize = 0;
    var bh: usize = 0;
    while (bh < gh / MERGE) : (bh += 1) {
        var bw: usize = 0;
        while (bw < gw / MERGE) : (bw += 1) {
            for (0..MERGE) |mh| for (0..MERGE) |mw| {
                const py = (bh * MERGE + mh) * PATCH;
                const px = (bw * MERGE + mw) * PATCH;
                const dst = out[n * per ..][0..per];
                for (0..3) |c| for (0..TEMPORAL) |t| for (0..PATCH) |y| for (0..PATCH) |x| {
                    const v: f32 = @floatFromInt(r.rgb[((py + y) * r.w + px + x) * 3 + c]);
                    dst[((c * TEMPORAL + t) * PATCH + y) * PATCH + x] = (v - 127.5) / 127.5;
                };
                n += 1;
            };
        }
    }
    return .{ .patches = out, .grid = .{ 1, @intCast(gh), @intCast(gw) } };
}

// -- videos ---------------------------------------------------------------------------------------------------------

/// A video's frame size for the tower (the Python frontend's video_size): Qwen3-VL's smart_resize with its per-frame
/// cap (at most 768 tokens a frame group, at least ~134), the whole video within ``budget`` tokens.
pub fn videoSize(frames: u32, height0: u32, width0: u32, budget: u64) Error![2]u32 {
    const factor: f64 = FACTOR;
    const shortest: f64 = 128 * FACTOR * FACTOR;
    const longest: f64 = @as(f64, @floatFromInt(budget)) * TEMPORAL * FACTOR * FACTOR;
    const per_frame = @max(@min(768 * FACTOR * FACTOR, @floor(longest / @as(f64, @floatFromInt(@max(1, frames))))), @floor(shortest * 1.05));
    const max_pixels = per_frame * @as(f64, @floatFromInt(frames));
    if (frames < TEMPORAL) return error.UnsupportedImage;
    var height: f64 = @floatFromInt(height0);
    var width: f64 = @floatFromInt(width0);
    if (height < factor or width < factor) {
        const scale = @max(factor / height, factor / width);
        height = @trunc(height * scale);
        width = @trunc(width * scale);
    }
    if (@max(height, width) / @min(height, width) > 200) return error.BadAspectRatio;
    var hb = pyRound(height / factor) * factor;
    var wb = pyRound(width / factor) * factor;
    const tb = pyRound(@as(f64, @floatFromInt(frames)) / TEMPORAL) * TEMPORAL;
    const n: f64 = @floatFromInt(frames);
    if (tb * hb * wb > max_pixels) {
        const beta = @sqrt(n * height * width / max_pixels);
        hb = @max(factor, @floor(height / beta / factor) * factor);
        wb = @max(factor, @floor(width / beta / factor) * factor);
    } else if (tb * hb * wb < shortest) {
        const beta = @sqrt(shortest / (n * height * width));
        hb = @ceil(height * beta / factor) * factor;
        wb = @ceil(width * beta / factor) * factor;
    }
    return .{ @intFromFloat(hb), @intFromFloat(wb) };
}

/// A decoded video's frame groups for the tower: each group's patches (fp32 [gh * gw, 1536], two consecutive frames
/// as the temporal pair; an odd last frame repeats), normalized as numpy does ((x * f32(1/255) - 0.5) / 0.5), and
/// each group's time (the mean of its two frames' seconds).
pub const Video = struct {
    patches: []f32, // groups x gh x gw x 1536
    groups: usize,
    gh: usize,
    gw: usize,
    times: []f64,

    pub fn groupPatches(v: Video, g: usize) []f32 {
        const per = v.gh * v.gw * 3 * TEMPORAL * PATCH * PATCH;
        return v.patches[g * per ..][0..per];
    }

    pub fn tokens(v: Video) usize {
        return v.gh * v.gw / (MERGE * MERGE);
    }

    pub fn deinit(v: *Video, gpa: std.mem.Allocator) void {
        gpa.free(v.patches);
        gpa.free(v.times);
    }
};

/// ``rgb``: ``frames`` RGB frames of ``h`` x ``w`` (already at the tower's size), their source ``indices`` and ``fps``.
pub fn prepareVideo(gpa: std.mem.Allocator, rgb: []const u8, frames: usize, h: usize, w: usize, indices: []const i32, fps: f64) Error!Video {
    if (h % FACTOR != 0 or w % FACTOR != 0 or frames < 2) return error.UnsupportedImage;
    const groups = (frames + TEMPORAL - 1) / TEMPORAL;
    const gh = h / PATCH;
    const gw = w / PATCH;
    const per = 3 * TEMPORAL * PATCH * PATCH;
    const out = try gpa.alloc(f32, groups * gh * gw * per);
    errdefer gpa.free(out);
    const scale: f32 = @floatCast(@as(f64, 1.0) / 255.0);
    var n: usize = 0;
    for (0..groups) |g| {
        for (0..gh / MERGE) |bh| for (0..gw / MERGE) |bw| for (0..MERGE) |mh| for (0..MERGE) |mw| {
            const py = (bh * MERGE + mh) * PATCH;
            const px = (bw * MERGE + mw) * PATCH;
            const dst = out[n * per ..][0..per];
            for (0..3) |c| for (0..TEMPORAL) |t| {
                const f = @min(g * TEMPORAL + t, frames - 1);
                const frame = rgb[f * h * w * 3 ..];
                for (0..PATCH) |y| for (0..PATCH) |x| {
                    const v: f32 = @as(f32, @floatFromInt(frame[((py + y) * w + px + x) * 3 + c])) * scale;
                    dst[((c * TEMPORAL + t) * PATCH + y) * PATCH + x] = (v - 0.5) / 0.5;
                };
            };
            n += 1;
        };
    }
    const times = try gpa.alloc(f64, groups);
    for (0..groups) |g| {
        const a: f64 = @as(f64, @floatFromInt(indices[g * TEMPORAL])) / fps;
        const b: f64 = @as(f64, @floatFromInt(indices[@min(g * TEMPORAL + TEMPORAL - 1, frames - 1)])) / fps;
        times[g] = (a + b) / 2;
    }
    return .{ .patches = out, .groups = groups, .gh = gh, .gw = gw, .times = times };
}

/// Python's f"{t:.1f}": one decimal, an exact tie rounded to even.
pub fn formatSeconds(buf: []u8, t: f64) ![]const u8 {
    const x: f128 = @as(f128, t) * 10; // exact: a double's 53 bits times ten fit f128's 113
    const fl = @floor(x);
    const frac = x - fl;
    var r = if (frac > 0.5) fl + 1 else fl;
    if (frac == 0.5 and @rem(fl, 2) != 0) r = fl + 1;
    const v: i64 = @intFromFloat(r);
    return std.fmt.bufPrint(buf, "{d}.{d}", .{ @divTrunc(v, 10), @as(u64, @intCast(@mod(v, 10))) });
}

test {
    _ = glm;
}
