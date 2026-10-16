//! Nemotron Omni's image processor: one tile within the patch bounds, torch's float bicubic, CLIP normalization.
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;
const Allocator = std.mem.Allocator;
const picture = core.picture;

/// An image for the tower: patches [gh * gw, 3 * patch * patch] (channel, row, column), grid, tokens, its identity.
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

pub const Error = picture.Error || error{ImageTooSmall};

/// The patch grid: rows then columns, both even.
pub const Grid = struct { h: u32, w: u32 };

/// A side's first estimate: Python's round(d / patch + 0.5), the next whole patch, and a whole odd count one more.
fn closest(d: u64, patch: u64) u64 {
    const q = d / patch;
    return if (d % patch != 0) q + 1 else q + (q & 1);
}

/// _compute_target_patches: the grid for a w x h picture under `room` patches, scaled up to the minimum, sides even.
pub fn grid(c: *const Config, w: u64, h: u64, room: u64) error{ImageTooSmall}!Grid {
    const ph = closest(h, c.patch);
    const pw = closest(w, c.patch);
    const factor = @min(@sqrt(@as(f64, @floatFromInt(room)) / @as(f64, @floatFromInt(ph * pw))), 1.0);
    var th: u64 = @intFromFloat(@floor(factor * @as(f64, @floatFromInt(ph))));
    var tw: u64 = @intFromFloat(@floor(factor * @as(f64, @floatFromInt(pw))));
    if (room > c.min_patches and th * tw < c.min_patches) {
        if (th * tw == 0) return error.ImageTooSmall; // the processor divides by zero here
        const up = @sqrt(@as(f64, @floatFromInt(c.min_patches)) / @as(f64, @floatFromInt(th * tw)));
        th = @intFromFloat(@ceil(up * @as(f64, @floatFromInt(th))));
        tw = @intFromFloat(@ceil(up * @as(f64, @floatFromInt(tw))));
    }
    const d = Config.shuffle;
    if (th % d != 0) th = if ((th + d - th % d) * tw <= room) th + d - th % d else @max(d, th - th % d);
    if (tw % d != 0) tw = if (th * (tw + d - tw % d) <= room) tw + d - tw % d else @max(d, tw - tw % d);
    if (th * tw > c.max_patches) return error.ImageTooSmall;
    return .{ .h = @intCast(th), .w = @intCast(tw) };
}

/// Python's round(): halves to the even neighbour.
fn pyRound(x: f64) u64 {
    const f = @floor(x);
    const d = x - f;
    const r: f64 = if (d > 0.5 or (d == 0.5 and @mod(f, 2) == 1)) f + 1 else f;
    return @intFromFloat(r);
}

/// _compute_target_patches_video: the aspect-preserving grid nearest `video_target_patches`, both sides even.
pub fn gridVideo(c: *const Config, w: u64, h: u64) Grid {
    const target: f64 = @floatFromInt(c.video_target_patches);
    const aspect = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(@max(h, 1)));
    var ph = @max(pyRound(@sqrt(target / aspect)), 1);
    var pw = @max(pyRound(@sqrt(target * aspect)), 1);
    const d = Config.shuffle;
    const rem_h = ph % d;
    const rem_w = pw % d;
    const ph_up = ph + (if (rem_h != 0) d - rem_h else 0);
    const pw_up = pw + (if (rem_w != 0) d - rem_w else 0);
    if (@as(f64, @floatFromInt(ph_up * pw_up)) <= target) {
        ph = ph_up;
        pw = pw_up;
    } else {
        ph = @max(d, ph - rem_h);
        pw = @max(d, pw - rem_w);
    }
    return .{ .h = @intCast(ph), .w = @intCast(pw) };
}

/// The patches one image may take when the request allows it `max_tokens` tokens: the processor's bound, or less.
pub fn budget(c: *const Config, max_tokens: u32) error{ImageTooSmall}!u64 {
    const room = @min(@as(u64, c.max_patches), @as(u64, max_tokens) * Config.shuffle * Config.shuffle);
    if (room < c.min_patches) return error.ImageTooSmall;
    return room;
}

/// Bytes to patches: decoded as the server decodes (core's picture), sized, resized and cut.
pub fn prepare(io: std.Io, a: Allocator, c: *const Config, bytes: []const u8, max_tokens: u32) Error!Prepared {
    var rgb = try picture.decode(io, a, bytes);
    defer rgb.deinit(a);
    return prepareRgb(a, c, rgb, picture.identity.ofBytes(0x7ad10, bytes), max_tokens);
}

/// RGB to patches: the grid, the resize (none at its own size), normalization, patches; the hash names bytes and grid.
pub fn prepareRgb(a: Allocator, c: *const Config, rgb: picture.Rgb, bytes_hash: u64, max_tokens: u32) Error!Prepared {
    const g = try grid(c, rgb.width, rgb.height, try budget(c, max_tokens));
    const th = g.h * c.patch;
    const tw = g.w * c.patch;
    const planes = try a.alloc(f32, 3 * th * tw);
    defer a.free(planes);
    try picture.resample.bicubicPlanes(a, .of(rgb.pixels, rgb.width, rgb.height), tw, th, planes);
    normalize(c, planes);
    const pixels = try a.alloc(f32, planes.len);
    cut(c, planes, g, pixels);
    const n: u32 = g.h * g.w;
    return .{ .pixels = pixels, .gh = g.h, .gw = g.w, .tokens = n / (Config.shuffle * Config.shuffle), .hash = picture.identity.withGrid(bytes_hash, .{ g.h, g.w }) };
}

/// Planes [3][h][w] of 0..255 samples to (x / 255 - mean) / std, in float32.
pub fn normalize(c: *const Config, planes: []f32) void {
    const per = planes.len / 3;
    for (0..3) |ch| for (planes[ch * per ..][0..per]) |*v| {
        v.* = (v.* / 255.0 - c.mean[ch]) / c.std[ch];
    };
}

/// Planes [3][gh * patch][gw * patch] cut into patches row by row, each channel by channel, then row by row.
pub fn cut(c: *const Config, planes: []const f32, g: Grid, out: []f32) void {
    const P = c.patch;
    const tw = g.w * P;
    const th = g.h * P;
    var at: usize = 0;
    for (0..g.h) |gy| for (0..g.w) |gx| for (0..3) |ch| for (0..P) |py| {
        @memcpy(out[at..][0..P], planes[(ch * th + gy * P + py) * tw + gx * P ..][0..P]);
        at += P;
    };
}

const test_config: Config = .{ .hidden = 1280, .layers = 32, .heads = 16, .ffn = 5120, .eps = 1e-6, .patch = 16, .prefix = 10, .side = 128, .projector_hidden = 20480, .text_hidden = 4096, .final_norm = true, .mean = .{ 0.48145466, 0.4578275, 0.40821073 }, .std = .{ 0.26862954, 0.26130258, 0.27577711 }, .min_patches = 1024, .max_patches = 13312, .max_model_len = 16384, .image_token = 18 };

test "grids as the processor picks them (its own outputs, budget 13,312)" {
    // (width, height) -> (target_w, target_h) from _compute_target_patches on the public checkpoint's processor
    const cases = [_][4]u64{
        .{ 640, 480, 40, 30 },    .{ 50, 40, 38, 28 },      .{ 1254, 1254, 80, 80 }, .{ 2172, 724, 136, 46 },
        .{ 2898, 1068, 182, 68 }, .{ 3000, 1340, 172, 76 }, .{ 720, 720, 46, 46 },   .{ 1, 1, 32, 32 },
        .{ 8192, 16, 512, 2 },    .{ 16, 8192, 2, 512 },
    };
    for (cases) |k| {
        const g = try grid(&test_config, k[0], k[1], 13312);
        try std.testing.expectEqual(Grid{ .h = @intCast(k[3]), .w = @intCast(k[2]) }, g);
    }
}

test "a request's token share bounds the budget; under the minimum it is refused" {
    try std.testing.expectEqual(@as(u64, 13312), try budget(&test_config, 4096));
    try std.testing.expectEqual(@as(u64, 4096), try budget(&test_config, 1024));
    try std.testing.expectError(error.ImageTooSmall, budget(&test_config, 255));
}

test "patches: channel, row, column inside each patch, patches row by row; normalized in float32" {
    var c = test_config;
    c.patch = 2;
    const g: Grid = .{ .h = 1, .w = 2 };
    var planes: [3 * 2 * 4]f32 = undefined;
    for (&planes, 0..) |*p, i| p.* = @floatFromInt(i);
    var out: [planes.len]f32 = undefined;
    cut(&c, &planes, g, &out);
    // patch 1 (columns 2..3), channel 1, row 1, column 0: plane value at (1 * 2 + 1) * 4 + 2
    try std.testing.expectEqual(@as(f32, 14.0), out[12 + 4 + 2]);
    normalize(&c, &planes);
    try std.testing.expectEqual((@as(f32, 14.0) / 255.0 - c.mean[1]) / c.std[1], planes[14]);
}

test "the hash names the grid: one picture at two budgets is two images to the cache, at one budget the same" {
    const a = std.testing.allocator;
    const pixels = try a.alloc(u8, 640 * 480 * 3);
    defer a.free(pixels);
    @memset(pixels, 9);
    const rgb: picture.Rgb = .{ .width = 640, .height = 480, .pixels = pixels };
    const whole = try prepareRgb(a, &test_config, rgb, 7, 4096);
    defer a.free(whole.pixels);
    const small = try prepareRgb(a, &test_config, rgb, 7, 256);
    defer a.free(small.pixels);
    const again = try prepareRgb(a, &test_config, rgb, 7, 4096);
    defer a.free(again.pixels);
    try std.testing.expect(whole.tokens > small.tokens);
    try std.testing.expect(whole.hash != small.hash); // another grid: neither resumes a state the other kept
    try std.testing.expectEqual(whole.hash, again.hash); // the same picture at the same budget resumes
}

test "video grids as the processor picks them (its own outputs, 1,024 patches)" {
    // (width, height) -> (target_w, target_h) from _compute_target_patches_video on the public checkpoint's processor
    const cases = [_][4]u64{
        .{ 640, 480, 36, 28 }, .{ 1920, 1080, 42, 24 }, .{ 1280, 720, 42, 24 }, .{ 3840, 2160, 42, 24 },
        .{ 100, 400, 16, 64 }, .{ 480, 360, 36, 28 },   .{ 1, 1, 32, 32 },      .{ 4000, 16, 506, 2 },
        .{ 16, 4000, 2, 506 }, .{ 1024, 1024, 32, 32 }, .{ 720, 1280, 24, 42 },
    };
    for (cases) |k| {
        const g = gridVideo(&test_config, k[0], k[1]);
        try std.testing.expectEqual(Grid{ .h = @intCast(k[3]), .w = @intCast(k[2]) }, g);
    }
}
