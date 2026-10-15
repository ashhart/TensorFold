//! Nemotron 3.5 Bolt's image preparation (image_processing_nemotron_h_omni.py as vLLM runs it): one dynamic-resolution
//! tile, torch's fp32 bicubic antialiased resize, CLIP normalization and C-RADIO's 16x16 patches.
//!
//! The oracle is tools/zig/bolt_vision_reference.py (`preprocess`, `target_patches`); its arithmetic is copied here
//! step for step. Decoding is the shared decoder's (qwen_image.decode); this module starts from RGB bytes.

const std = @import("std");
const qi = @import("qwen_image.zig");

pub const Error = qi.Error;
pub const Image = qi.Image;

pub const PATCH = 16;
pub const MERGE = 2; // pixel shuffle v2: 2x2 patches a token (downsample_ratio 0.5)
/// One patch's values, (channel, row, column) order: patch_generator.embedder's input width.
pub const PER = 3 * PATCH * PATCH;
pub const MIN_PATCHES = 1024;
pub const MAX_PATCHES = 13312;
/// vLLM's budget, (max_model_len - 4) * 4 patches, before the max_num_patches cap (which always binds).
pub const MODEL_LEN_BUDGET = (16384 - 4) * MERGE * MERGE;
pub const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
pub const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

/// Python's round(): halves to even.
fn pyRound(x: f64) f64 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// The patch grid (rows, columns) of the single tile: target_patches in the reference, both sides even.
pub fn grid(width: u32, height: u32) Error![2]u32 {
    if (width == 0 or height == 0) return error.UnsupportedImage;
    const r: f64 = MERGE;
    const mn: f64 = MIN_PATCHES;
    const avail: f64 = @max(@min(@as(f64, MODEL_LEN_BUDGET), MAX_PATCHES), mn);
    const ph = pyRound(@as(f64, @floatFromInt(height)) / PATCH + 0.5);
    const pw = pyRound(@as(f64, @floatFromInt(width)) / PATCH + 0.5);
    const f = @min(@sqrt(avail / (ph * pw)), 1.0);
    var th = @floor(f * ph);
    var tw = @floor(f * pw);
    if (avail > mn and th * tw < mn) {
        if (th * tw == 0) return error.BadAspectRatio; // Python divides by zero here
        const up = @sqrt(mn / (th * tw));
        th = @ceil(up * th);
        tw = @ceil(up * tw);
    }
    if (@mod(th, r) != 0) {
        const up = th + (r - @mod(th, r));
        th = if (up * tw <= avail) up else @max(r, th - @mod(th, r));
    }
    if (@mod(tw, r) != 0) {
        const up = tw + (r - @mod(tw, r));
        tw = if (th * up <= avail) up else @max(r, tw - @mod(tw, r));
    }
    return .{ @intFromFloat(th), @intFromFloat(tw) };
}

/// torch's antialiased bicubic filter (Keys, a = -0.5), in fp32 as aten's aa_filter computes it.
fn aaCubic(x0: f32) f32 {
    const a: f32 = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

/// One axis' taps: output i reads input [first[i], first[i] + count[i]) with weights w[i * ksize ..].
const Taps = struct {
    ksize: usize,
    first: []usize,
    count: []usize,
    w: []f32,

    fn deinit(t: *Taps, gpa: std.mem.Allocator) void {
        gpa.free(t.first);
        gpa.free(t.count);
        gpa.free(t.w);
    }
};

/// aten's _compute_weights_aa for an axis (align_corners=False, no scale factor): fp32 throughout, then normalized.
fn taps(gpa: std.mem.Allocator, in_size: usize, out_size: usize) Error!Taps {
    const scale: f32 = @as(f32, @floatFromInt(in_size)) / @as(f32, @floatFromInt(out_size));
    const support: f32 = if (scale >= 1.0) 2.0 * scale else 2.0;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    const invscale: f32 = if (scale >= 1.0) 1.0 / scale else 1.0;
    var t: Taps = .{ .ksize = ksize, .first = try gpa.alloc(usize, out_size), .count = undefined, .w = undefined };
    errdefer gpa.free(t.first);
    t.count = try gpa.alloc(usize, out_size);
    errdefer gpa.free(t.count);
    t.w = try gpa.alloc(f32, out_size * ksize);
    @memset(t.w, 0);
    for (0..out_size) |i| {
        const center: f32 = scale * (@as(f32, @floatFromInt(i)) + 0.5);
        const lo: i64 = @max(@as(i64, @intFromFloat(center - support + 0.5)), 0);
        const hi: i64 = @min(@as(i64, @intFromFloat(center + support + 0.5)), @as(i64, @intCast(in_size)));
        const n: usize = @intCast(hi - lo);
        const w = t.w[i * ksize ..][0..n];
        var total: f32 = 0;
        for (w, 0..) |*v, j| {
            const x: f32 = (@as(f32, @floatFromInt(@as(i64, @intCast(j)) + lo)) - center + 0.5) * invscale;
            v.* = aaCubic(x);
            total += v.*;
        }
        if (total != 0) for (w) |*v| {
            v.* /= total;
        };
        t.first[i] = @intCast(lo);
        t.count[i] = n;
    }
    return t;
}

/// F.interpolate(mode="bicubic", align_corners=False, antialias=True) of fp32 planes [3, h, w] to [3, oh, ow]:
/// the width pass first, then the height pass (aten's separable order); a side already the size is not resampled.
pub fn resizePlanes(gpa: std.mem.Allocator, src: []const f32, w: u32, h: u32, ow: u32, oh: u32) Error![]f32 {
    var cur = try gpa.dupe(f32, src);
    var cw: usize = w;
    if (ow != w) {
        var t = try taps(gpa, w, ow);
        defer t.deinit(gpa);
        const out = try gpa.alloc(f32, 3 * @as(usize, h) * ow);
        for (0..3 * @as(usize, h)) |row| {
            const in = cur[row * cw ..][0..cw];
            for (0..ow) |x| {
                const k = t.w[x * t.ksize ..];
                const s = in[t.first[x]..];
                var acc: f32 = s[0] * k[0];
                for (1..t.count[x]) |j| acc += s[j] * k[j];
                out[row * ow + x] = acc;
            }
        }
        gpa.free(cur);
        cur = out;
        cw = ow;
    }
    if (oh != h) {
        var t = try taps(gpa, h, oh);
        defer t.deinit(gpa);
        const out = try gpa.alloc(f32, 3 * @as(usize, oh) * cw);
        for (0..3) |c| {
            const plane = cur[c * h * cw ..][0 .. h * cw];
            for (0..oh) |y| {
                const k = t.w[y * t.ksize ..];
                const o = out[(c * oh + y) * cw ..][0..cw];
                for (o, 0..) |*v, x| {
                    var acc: f32 = plane[t.first[y] * cw + x] * k[0];
                    for (1..t.count[y]) |j| acc += plane[(t.first[y] + j) * cw + x] * k[j];
                    v.* = acc;
                }
            }
        }
        gpa.free(cur);
        cur = out;
    }
    return cur;
}

/// RGB bytes resized to [3, oh, ow] fp32 and normalized as the reference does: (x / 255 - mean) / std in fp32.
pub fn pixelValues(gpa: std.mem.Allocator, im: Image, ow: u32, oh: u32) Error![]f32 {
    const n = @as(usize, im.w) * im.h;
    if (im.rgb.len != n * 3) return error.UnsupportedImage;
    const planes = try gpa.alloc(f32, 3 * n);
    defer gpa.free(planes);
    for (0..n) |i| for (0..3) |c| {
        planes[c * n + i] = @floatFromInt(im.rgb[i * 3 + c]);
    };
    const out = try resizePlanes(gpa, planes, im.w, im.h, ow, oh);
    for (0..3) |c| for (out[c * @as(usize, ow) * oh ..][0 .. @as(usize, ow) * oh]) |*v| {
        v.* = (v.* / 255.0 - MEAN[c]) / STD[c];
    };
    return out;
}

/// One image ready for the tower: its patches (fp32 [gh * gw, 768], row-major over the grid, each patch (c, dy, dx)).
pub const Prepared = struct {
    patches: []f32,
    gh: u32,
    gw: u32,

    pub fn count(p: Prepared) usize {
        return @as(usize, p.gh) * p.gw;
    }

    /// The `<image>` placeholders the prompt carries: one a 2x2 group of patches.
    pub fn tokens(p: Prepared) usize {
        return p.count() / (MERGE * MERGE);
    }

    pub fn deinit(p: *Prepared, gpa: std.mem.Allocator) void {
        gpa.free(p.patches);
    }
};

/// Planes [3, gh * 16, gw * 16] cut into patches, (c, dy, dx) within each (the reference's reshape and permute).
pub fn patchify(gpa: std.mem.Allocator, px: []const f32, gh: u32, gw: u32) Error![]f32 {
    const H = @as(usize, gh) * PATCH;
    const W = @as(usize, gw) * PATCH;
    if (px.len != 3 * H * W) return error.UnsupportedImage;
    const out = try gpa.alloc(f32, @as(usize, gh) * gw * PER);
    for (0..gh) |py| for (0..gw) |pxi| {
        const dst = out[(py * gw + pxi) * PER ..][0..PER];
        for (0..3) |c| for (0..PATCH) |dy| {
            @memcpy(dst[(c * PATCH + dy) * PATCH ..][0..PATCH], px[(c * H + py * PATCH + dy) * W + pxi * PATCH ..][0..PATCH]);
        };
    };
    return out;
}

/// An RGB image to the tower's patches: grid, resize, normalize, patchify.
pub fn prepare(gpa: std.mem.Allocator, im: Image) Error!Prepared {
    const g = try grid(im.w, im.h);
    const px = try pixelValues(gpa, im, g[1] * PATCH, g[0] * PATCH);
    defer gpa.free(px);
    return .{ .patches = try patchify(gpa, px, g[0], g[1]), .gh = g[0], .gw = g[1] };
}

const fixture = @embedFile("fixtures/nemotron_resize.bin");

fn testImage(gpa: std.mem.Allocator, w: u32, h: u32) !Image {
    const rgb = try gpa.alloc(u8, @as(usize, w) * h * 3);
    for (0..h) |y| for (0..w) |x| for (0..3) |c| {
        rgb[(y * w + x) * 3 + c] = @intCast((7 * x + 13 * y + 29 * c + (x * y) % 17) % 256);
    };
    return .{ .w = w, .h = h, .rgb = rgb };
}

test "the patch grid matches target_patches (tools/zig/bolt_image_fixture.py's table)" {
    const cases = [_][4]u32{
        .{ 2172, 724, 136, 46 }, .{ 640, 480, 40, 30 },     .{ 37, 23, 40, 28 },  .{ 4000, 3000, 132, 100 },
        .{ 16, 16, 32, 32 },     .{ 100, 5000, 8, 314 },    .{ 1024, 1024, 64, 64 }, .{ 512, 512, 32, 32 },
        .{ 3000, 100, 188, 8 },  .{ 1920, 1080, 120, 68 }, .{ 800, 600, 50, 38 },
    };
    for (cases) |c| {
        const g = try grid(c[0], c[1]);
        try std.testing.expectEqual(c[3], g[0]);
        try std.testing.expectEqual(c[2], g[1]);
        try std.testing.expect(@as(usize, g[0]) * g[1] <= MAX_PATCHES);
    }
}

test "resize and normalization match torch's bicubic antialias (up and down) to 5e-6" {
    const gpa = std.testing.allocator;
    var im = try testImage(gpa, 37, 23);
    defer im.deinit(gpa);
    const want: []align(1) const f32 = std.mem.bytesAsSlice(f32, fixture);
    var at: usize = 0;
    var worst: f32 = 0;
    for ([_][2]u32{ .{ 48, 32 }, .{ 16, 16 }, .{ 24, 8 } }) |s| {
        const got = try pixelValues(gpa, im, s[0], s[1]);
        defer gpa.free(got);
        for (got, want[at..][0..got.len]) |g, w| worst = @max(worst, @abs(g - w));
        at += got.len;
    }
    try std.testing.expectEqual(want.len, at);
    try std.testing.expect(worst < 5e-6);
}

test "patches are (c, dy, dx) over a row-major grid; prepare's count and tokens" {
    const gpa = std.testing.allocator;
    const px = try gpa.alloc(f32, 3 * 32 * 48);
    defer gpa.free(px);
    for (px, 0..) |*v, i| v.* = @floatFromInt(i);
    const p = try patchify(gpa, px, 2, 3);
    defer gpa.free(p);
    // patch (1, 2), channel 2, dy 5, dx 7 is pixel (2, 16 + 5, 32 + 7)
    try std.testing.expectEqual(@as(f32, @floatFromInt((2 * 32 + 21) * 48 + 39)), p[(1 * 3 + 2) * PER + (2 * PATCH + 5) * PATCH + 7]);
    var im = try testImage(gpa, 37, 23);
    defer im.deinit(gpa);
    var prep = try prepare(gpa, im);
    defer prep.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 28 * 40), prep.count());
    try std.testing.expectEqual(@as(usize, 280), prep.tokens());
    try std.testing.expectEqual(prep.count() * PER, prep.patches.len);
}
