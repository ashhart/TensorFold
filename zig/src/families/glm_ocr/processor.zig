//! GLM46V's image geometry, Torch uint8 bicubic, fused normalization and grouped temporal patches.
const std = @import("std");
const picture = @import("core").picture;
const c = @import("config.zig");
const Allocator = std.mem.Allocator;
pub const input_limits: picture.Limits = .{ .pixels = 32 << 20 };
const output_limits: picture.Limits = .{ .side = 16384, .pixels = c.max_pixels / c.temporal };
pub const Error = picture.Error || error{ BadPixels, AspectRatio, ImageBudget, ContextExceeded };

pub const Geometry = struct {
    h: usize,
    w: usize,

    pub fn patches(g: Geometry) usize {
        return (g.h / c.patch) * (g.w / c.patch);
    }
    pub fn tokens(g: Geometry) usize {
        return g.patches() / (c.merge * c.merge);
    }
    pub fn fits(g: Geometry, max_tokens: usize, text: usize, window: usize) Error!void {
        if (g.tokens() > max_tokens) return error.ImageBudget;
        if (window > c.context or g.tokens() > window or text > window - g.tokens()) return error.ContextExceeded;
    }
};

/// Pixel limits include both duplicated frames, as in the checkpoint's smart_resize.
pub fn geometry(height: usize, width: usize) Error!Geometry {
    if (height == 0 or width == 0) return error.Invalid;
    if (height > picture.max_side or width > picture.max_side) return error.Oversized;
    try input_limits.fits(width, height);
    var h: f64 = @floatFromInt(height);
    var w: f64 = @floatFromInt(width);
    const factor = c.patch * c.merge;
    if (h < factor or w < factor) {
        const scale = @max(factor / h, factor / w);
        h = @trunc(h * scale);
        w = @trunc(w * scale);
    }
    if (@max(h, w) / @min(h, w) > 200) return error.AspectRatio;
    var hb = roundEven(h / factor) * factor;
    var wb = roundEven(w / factor) * factor;
    if (c.temporal * hb * wb > c.max_pixels) {
        const beta = @sqrt(c.temporal * h * w / c.max_pixels);
        hb = @max(factor, @floor(h / beta / factor) * factor);
        wb = @max(factor, @floor(w / beta / factor) * factor);
    } else if (c.temporal * hb * wb < c.min_pixels) {
        const beta = @sqrt(c.min_pixels / (c.temporal * h * w));
        hb = @ceil(h * beta / factor) * factor;
        wb = @ceil(w * beta / factor) * factor;
    }
    const g: Geometry = .{ .h = @intFromFloat(hb), .w = @intFromFloat(wb) };
    try output_limits.fits(g.w, g.h);
    return g;
}

fn roundEven(v: f64) f64 {
    const base = @floor(v);
    const fraction = v - base;
    if (fraction == 0.5) return base + @mod(base, 2);
    return @round(v);
}

pub const Prepared = struct {
    geometry: Geometry,
    pixels: []f32, // [patches, channel, time, patch_y, patch_x], merge-cell order
    hash: u64,

    pub fn deinit(p: *Prepared, a: Allocator) void {
        a.free(p.pixels);
        p.* = undefined;
    }
};

pub fn prepare(io: std.Io, a: Allocator, bytes: []const u8, max_tokens: usize) Error!Prepared {
    var rgb = try picture.decodeWithLimits(io, a, bytes, input_limits);
    defer rgb.deinit(a);
    return prepareRgb(a, rgb, max_tokens);
}

pub fn prepareRgb(a: Allocator, rgb: picture.Rgb, max_tokens: usize) Error!Prepared {
    const g = try geometry(rgb.height, rgb.width);
    try g.fits(max_tokens, 0, c.context);
    if (rgb.pixels.len != rgb.height * rgb.width * 3) return error.BadPixels;
    const resized = try picture.resample.resize(a, .torch, .of(rgb.pixels, rgb.width, rgb.height), g.w, g.h);
    defer a.free(resized);
    const out = try a.alloc(f32, g.patches() * c.patch_values);
    const table = normalization();
    const grid_w = g.w / c.patch;
    for (0..g.h) |y| for (0..g.w) |x| {
        const py = y / c.patch;
        const px = x / c.patch;
        const row = ((py / c.merge) * (grid_w / c.merge) + px / c.merge) * 4 + (py % 2) * 2 + px % 2;
        const base = row * c.patch_values + (y % c.patch) * c.patch + x % c.patch;
        for (0..3) |ch| for (0..c.temporal) |t| {
            out[base + (ch * c.temporal + t) * c.patch * c.patch] = table[ch][resized[(y * g.w + x) * 3 + ch]];
        };
    };
    var hash: std.hash.Wyhash = .init(0x676c6d6f6372);
    hash.update(std.mem.sliceAsBytes(out));
    for ([_]usize{ g.h, g.w }) |v| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, @intCast(v), .little);
        hash.update(&bytes);
    }
    return .{ .geometry = g, .pixels = out, .hash = hash.final() };
}

fn normalization() [3][256]f32 {
    var out: [3][256]f32 = undefined;
    for (&out, 0..) |*channel, ch| for (channel, 0..) |*v, i| {
        const mean = c.mean[ch] * @as(f32, 255);
        const scale = c.deviation[ch] * @as(f32, 255);
        v.* = (@as(f32, @floatFromInt(i)) - mean) / scale;
    };
    return out;
}
