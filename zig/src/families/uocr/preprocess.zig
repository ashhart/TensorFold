//! A page to the encoder's views, as the publisher's processor makes them: Pillow's letterbox to the 1024 global view
//! (ImageOps.pad, gray 127), and in gundam mode Pillow's resize to a grid of 640 tiles, each normalized to bf16.
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const resample = picture.resample;
const cfg = @import("config.zig");
const resize = @import("resize.zig");

/// Fewest and most tiles a gundam page splits into.
pub const min_tiles = 2;
pub const max_tiles = 16;
const pad_gray: u8 = 127;

/// A page's views: the global one [3, 1024, 1024] and `across` x `down` tiles [3, 640, 640] each (none when 1 x 1).
pub const Views = struct {
    global: []u16,
    tiles: []u16,
    across: usize,
    down: usize,

    pub fn tileCount(v: Views) usize {
        return if (v.across * v.down > 1) v.across * v.down else 0;
    }

    pub fn tile(v: Views, i: usize) []const u16 {
        const n = 3 * cfg.tile_size * cfg.tile_size;
        return v.tiles[i * n ..][0..n];
    }

    /// The page's image tokens: the global block and separator, and the tiles' rows with their newlines.
    pub fn tokens(v: Views) usize {
        const g = cfg.side(cfg.base_size);
        const t = cfg.side(cfg.tile_size);
        return g * (g + 1) + 1 + if (v.tileCount() > 0) (t * v.across + 1) * (t * v.down) else 0;
    }

    pub fn deinit(v: *Views, a: std.mem.Allocator) void {
        a.free(v.global);
        a.free(v.tiles);
        v.* = undefined;
    }
};

/// RGB rows from decoded samples as Pillow's convert('RGB') gives them: alpha dropped, never composited, and the
/// stored orientation kept (the publisher's `infer` applies no EXIF transpose).
pub fn rgbOf(a: std.mem.Allocator, s: picture.Samples) ![]u8 {
    const out = try a.alloc(u8, s.width * s.height * 3);
    for (0..s.width * s.height) |i| @memcpy(out[i * 3 ..][0..3], s.data[i * s.channels ..][0..3]);
    return out;
}

/// A decoded page: RGB rows the caller frees.
pub const Page = struct {
    rgb: []u8,
    width: usize,
    height: usize,

    pub fn deinit(p: *Page, a: std.mem.Allocator) void {
        a.free(p.rgb);
        p.* = undefined;
    }
};

/// A PNG, JPEG or WebP file's page, taken as the publisher's Image.open(...).convert('RGB') takes it.
pub fn decode(io: std.Io, a: std.mem.Allocator, bytes: []const u8) picture.Error!Page {
    const s = switch (picture.sniff(bytes)) {
        .png => try picture.png.decodeWithLimits(a, bytes, .{}),
        .jpeg => try picture.jpeg.decodeWithLimits(io, a, bytes, .{}),
        .webp => try picture.webp.decodeWithLimits(io, a, bytes, .{}),
        .unknown => return error.Invalid,
    };
    defer a.free(s.data);
    return .{ .rgb = try rgbOf(a, s), .width = s.width, .height = s.height };
}

/// Python's round() of a float: halves to even.
fn roundEven(x: f64) usize {
    const f = @floor(x);
    const d = x - f;
    const r = if (d > 0.5) f + 1 else if (d < 0.5) f else if (@mod(f, 2) == 0) f else f + 1;
    return @intFromFloat(r);
}

/// ToTensor and Normalize(0.5, 0.5) in float32, then bf16: [3, side, side] from packed RGB rows.
fn normalize(rgb: []const u8, side: usize, out: []u16) void {
    for (0..side * side) |p| for (0..3) |c| {
        const x: f32 = @as(f32, @floatFromInt(rgb[p * 3 + c])) / 255.0;
        out[c * side * side + p] = resize.bf16((x - 0.5) / 0.5);
    };
}

/// The tile grid (across, down) dynamic_preprocess picks: the candidates in the order Python's set and stable sort
/// leave them, the closest aspect ratio first met, and on an exact tie the later one when the page is big enough.
pub fn grid(width: usize, height: usize) [2]usize {
    const aspect = @as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(height));
    const area: f64 = @floatFromInt(width * height);
    var best: [2]usize = .{ 1, 1 };
    var best_diff = std.math.inf(f64);
    for (candidates) |r| {
        const diff = @abs(aspect - @as(f64, @floatFromInt(r[0])) / @as(f64, @floatFromInt(r[1])));
        if (diff < best_diff) {
            best_diff = diff;
            best = r;
        } else if (diff == best_diff and area > 0.5 * @as(f64, @floatFromInt(cfg.tile_size * cfg.tile_size * r[0] * r[1]))) best = r;
    }
    return best;
}

/// Every (i, j) with 2 <= i*j <= 16 in CPython's set iteration order, stably sorted by i*j (the publisher's order).
const candidates = [_][2]usize{
    .{ 1, 2 },  .{ 2, 1 },  .{ 3, 1 },  .{ 1, 3 }, .{ 2, 2 },  .{ 4, 1 },  .{ 1, 4 },  .{ 5, 1 },  .{ 1, 5 },  .{ 1, 6 },
    .{ 6, 1 },  .{ 3, 2 },  .{ 2, 3 },  .{ 7, 1 }, .{ 1, 7 },  .{ 4, 2 },  .{ 2, 4 },  .{ 1, 8 },  .{ 8, 1 },  .{ 1, 9 },
    .{ 3, 3 },  .{ 9, 1 },  .{ 2, 5 },  .{ 5, 2 }, .{ 10, 1 }, .{ 1, 10 }, .{ 11, 1 }, .{ 1, 11 }, .{ 12, 1 }, .{ 3, 4 },
    .{ 4, 3 },  .{ 1, 12 }, .{ 6, 2 },  .{ 2, 6 }, .{ 13, 1 }, .{ 1, 13 }, .{ 14, 1 }, .{ 2, 7 },  .{ 1, 14 }, .{ 7, 2 },
    .{ 1, 15 }, .{ 5, 3 },  .{ 15, 1 }, .{ 3, 5 }, .{ 1, 16 }, .{ 2, 8 },  .{ 16, 1 }, .{ 8, 2 },  .{ 4, 4 },
};

/// A page's views from its RGB rows; `tiles` is gundam mode (the publisher's default for a single page).
pub fn views(a: std.mem.Allocator, rgb: []const u8, width: usize, height: usize, tiles: bool) !Views {
    if (width == 0 or height == 0) return error.EmptyPage;
    const src = resample.View.of(rgb, width, height);
    const B = cfg.base_size;
    // ImageOps.contain then pad: the long side to 1024, the short one rounded, centred on gray
    var rw: usize = B;
    var rh: usize = B;
    if (width > height) rh = roundEven(@as(f64, @floatFromInt(height)) / @as(f64, @floatFromInt(width)) * B);
    if (width < height) rw = roundEven(@as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(height)) * B);
    const canvas = try a.alloc(u8, B * B * 3);
    defer a.free(canvas);
    @memset(canvas, pad_gray);
    if (rw > 0 and rh > 0) {
        const fitted = try resample.resize(a, .pillow, src, rw, rh);
        defer a.free(fitted);
        const x0 = roundEven(@as(f64, @floatFromInt(B - rw)) * 0.5);
        const y0 = roundEven(@as(f64, @floatFromInt(B - rh)) * 0.5);
        for (0..rh) |y| @memcpy(canvas[((y0 + y) * B + x0) * 3 ..][0 .. rw * 3], fitted[y * rw * 3 ..][0 .. rw * 3]);
    }
    var v: Views = .{ .global = try a.alloc(u16, 3 * B * B), .tiles = &.{}, .across = 1, .down = 1 };
    errdefer a.free(v.global);
    normalize(canvas, B, v.global);
    const T = cfg.tile_size;
    if (!tiles or (width <= T and height <= T)) return v;
    const g = grid(width, height);
    v.across = g[0];
    v.down = g[1];
    const big = try resample.resize(a, .pillow, src, T * g[0], T * g[1]);
    defer a.free(big);
    v.tiles = try a.alloc(u16, g[0] * g[1] * 3 * T * T);
    const crop = try a.alloc(u8, T * T * 3);
    defer a.free(crop);
    for (0..g[0] * g[1]) |t| {
        const x0 = (t % g[0]) * T;
        const y0 = (t / g[0]) * T;
        for (0..T) |y| @memcpy(crop[y * T * 3 ..][0 .. T * 3], big[((y0 + y) * T * g[0] + x0) * 3 ..][0 .. T * 3]);
        normalize(crop, T, v.tiles[t * 3 * T * T ..][0 .. 3 * T * T]);
    }
    return v;
}

test "uocr: Python's rounding, the tile grids the publisher's code picks, and the page's token counts" {
    try std.testing.expectEqual(@as(usize, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(usize, 4), roundEven(3.5));
    try std.testing.expectEqual(@as(usize, 724), roundEven(724.0775));
    // find_closest_aspect_ratio's answers in the reference: A4 at 150 dpi both ways, a square (a tie: the last
    // equal ratio), a wide strip, a page just over 640
    try std.testing.expectEqual([2]usize{ 2, 3 }, grid(1240, 1754));
    try std.testing.expectEqual([2]usize{ 4, 3 }, grid(1754, 1240));
    try std.testing.expectEqual([2]usize{ 4, 4 }, grid(2000, 2000));
    try std.testing.expectEqual([2]usize{ 16, 1 }, grid(10000, 300));
    try std.testing.expectEqual([2]usize{ 2, 2 }, grid(700, 650));
    const v: Views = .{ .global = &.{}, .tiles = &.{}, .across = 2, .down = 2 };
    try std.testing.expectEqual(@as(usize, 693), v.tokens());
    const b: Views = .{ .global = &.{}, .tiles = &.{}, .across = 1, .down = 1 };
    try std.testing.expectEqual(@as(usize, 273), b.tokens());
}
