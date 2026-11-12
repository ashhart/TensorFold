//! Parse's aspect-preserving Pillow resize, NaViT white padding, and raw RGB patch order.
const std = @import("std");
const picture = @import("core").picture;
const c = @import("config.zig");
const Allocator = std.mem.Allocator;
pub const Error = picture.Error || error{ ImageTooSmall, BadPixels, ContextExceeded };

pub const Size = struct { h: usize, w: usize };
pub const Geometry = struct {
    content: Size,
    padded: Size,

    pub fn patches(g: Geometry) usize {
        return (g.padded.h / c.patch) * (g.padded.w / c.patch);
    }
    pub fn tokens(g: Geometry) usize {
        return g.patches() / 4 + 1; // four horizontal patches per row, plus the summary
    }
    pub fn fits(g: Geometry, text: usize, window: usize) Error!void {
        if (window > c.context or g.tokens() > window or text > window - g.tokens()) return error.ContextExceeded;
    }
};

/// Follow the processor's two sequential float64 truncations, including small-image preservation.
pub fn geometry(height: usize, width: usize) Error!Geometry {
    if (height == 0 or width == 0) return error.Invalid;
    if (height > picture.max_side or width > picture.max_side) return error.Oversized;
    try picture.fits(width, height);
    const ratio = @as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(height));
    var h = height;
    var w = width;
    if (h > c.max_height) {
        h = c.max_height;
        w = @intFromFloat(@as(f64, @floatFromInt(h)) * ratio);
    }
    if (w > c.max_width) {
        w = c.max_width;
        h = @intFromFloat(@as(f64, @floatFromInt(w)) / ratio);
    }
    if (h == 0 or w == 0) return error.ImageTooSmall;
    return .{ .content = .{ .h = h, .w = w }, .padded = .{
        .h = std.mem.alignForward(usize, h, 16),
        .w = std.mem.alignForward(usize, w, 64),
    } };
}

pub const Prepared = struct {
    geometry: Geometry,
    pixels: []f32, // [patches, 3, 16, 16], RGB after normalization
    hash: u64,

    pub fn deinit(p: *Prepared, a: Allocator) void {
        a.free(p.pixels);
        p.* = undefined;
    }
};

pub fn prepare(io: std.Io, a: Allocator, bytes: []const u8) Error!Prepared {
    var rgb = try picture.decode(io, a, bytes);
    defer rgb.deinit(a);
    return prepareRgb(a, rgb);
}

pub fn prepareRgb(a: Allocator, rgb: picture.Rgb) Error!Prepared {
    const g = try geometry(rgb.height, rgb.width);
    if (rgb.pixels.len != rgb.height * rgb.width * 3) return error.BadPixels;
    const resized = try picture.resample.resizeWith(a, .pillow, .bilinear, .of(rgb.pixels, rgb.width, rgb.height), g.content.w, g.content.h);
    defer a.free(resized);
    const out = try a.alloc(f32, g.patches() * 768);
    const table = normalization();
    const cols = g.padded.w / c.patch;
    for (0..g.padded.h) |y| for (0..g.padded.w) |x| {
        const base = ((y / 16) * cols + x / 16) * 768 + (y % 16) * 16 + x % 16;
        for (0..3) |ch| {
            const value = if (y < g.content.h and x < g.content.w)
                resized[(y * g.content.w + x) * 3 + ch]
            else
                255;
            out[base + ch * 256] = table[ch][value];
        }
    };
    // Content dimensions affect positional sampling even when the padded grid is equal.
    var h: std.hash.Wyhash = .init(0x7061727365);
    h.update(std.mem.sliceAsBytes(out));
    for ([_]usize{ g.content.h, g.content.w, g.padded.h, g.padded.w }) |v| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, @intCast(v), .little);
        h.update(&bytes);
    }
    return .{ .geometry = g, .pixels = out, .hash = h.final() };
}

fn normalization() [3][256]f32 {
    var out: [3][256]f32 = undefined;
    for (&out, 0..) |*channel, ch| for (channel, 0..) |*v, i| {
        const rescaled = @as(f32, @floatFromInt(i)) / @as(f32, 255);
        v.* = (rescaled - c.mean[ch]) / c.deviation[ch];
    };
    return out;
}
