//! Pictures to a tower's patches: transformers' smart_resize, the processor's resize and normalization, merge order.
const std = @import("std");
const picture = @import("picture.zig");

/// The processor's patch geometry, pixel bounds and arithmetic.
pub const Patches = struct {
    patch: u32,
    temporal: u32,
    merge: u32,
    min_pixels: u64,
    max_pixels: u64,
    rescale: f64,
    mean: [3]f32,
    std: [3]f32,
    resize: picture.resample.Kind = .pillow, // Pillow's bicubic, or torchvision's antialiased uint8 bicubic
    fused: bool = false, // torchvision's backend: (x - mean / rescale) / (std / rescale) in float32

    /// A patch's values: channel by frame by row by column.
    pub fn width(p: Patches) u32 {
        return 3 * p.temporal * p.patch * p.patch;
    }

    /// Image sides are multiples of this many pixels.
    pub fn factor(p: Patches) u32 {
        return p.patch * p.merge;
    }
};

pub const Size = struct { h: u32, w: u32 };

/// The shared decoder's refusals, a budget too small for an image, and a side past 200 times the other.
pub const Error = picture.Error || error{ ImageTooSmall, AspectRatio };

/// One image ready for a tower: patches [gh * gw, patch width] in merge order, the patch grid and the image's tokens.
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

/// The words for a refusal only the image path knows: the decoder's, and a side past 200 times the other.
pub fn problem(err: anyerror) ?[]const u8 {
    if (err == error.AspectRatio) return "its longer side is more than 200 times its shorter side";
    return picture.decoderProblem(err);
}

/// transformers' smart_resize: sides multiples of `factor` (Python's round, halves to even), the area in the bounds.
pub fn smartResize(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64) error{AspectRatio}!Size {
    return smartResizeWith(height, width, factor, min_pixels, max_pixels, .{});
}

/// Where processors' smart_resize copies part: LFM2-VL's keeps a side at least one factor and has no aspect bound.
pub const Variant = struct { at_least_factor: bool = false, max_aspect: ?f64 = 200 };

pub fn smartResizeWith(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64, v: Variant) error{AspectRatio}!Size {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    if (v.max_aspect) |most| if (@max(h, w) / @min(h, w) > most) return error.AspectRatio;
    const f: f64 = @floatFromInt(factor);
    var hb = roundEven(h / f) * f;
    var wb = roundEven(w / f) * f;
    if (v.at_least_factor) {
        hb = @max(f, hb);
        wb = @max(f, wb);
    }
    if (hb * wb > @as(f64, @floatFromInt(max_pixels))) {
        const beta = @sqrt(h * w / @as(f64, @floatFromInt(max_pixels)));
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (hb * wb < @as(f64, @floatFromInt(min_pixels))) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ .h = @intFromFloat(hb), .w = @intFromFloat(wb) };
}

/// Python's round: a half goes to the even neighbour.
fn roundEven(x: f64) f64 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r;
}

/// An image's pixel bounds under a `max_tokens` budget: the processor's, its most cut to that many tokens.
pub fn bounds(p: Patches, max_tokens: u32) [2]u64 {
    const per = @as(u64, p.factor()) * p.factor();
    const most = @min(p.max_pixels, @as(u64, max_tokens) * per);
    return .{ @min(p.min_pixels, most), most };
}

/// Bytes to patches: decoded as 0.6.6 decoded them (core/picture), resized by `p.resize`, cut in merge order.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, p: Patches, max_tokens: u32) Error!Prepared {
    var img = try picture.decode(io, gpa, bytes);
    defer img.deinit(gpa);
    return prepareRgb(gpa, img, picture.identity.ofBytes(0x51e7, bytes), p, max_tokens);
}

/// RGB to patches; the hash covers the bytes' hash and the grid, so one picture at two budgets is two images.
pub fn prepareRgb(gpa: std.mem.Allocator, img: picture.Rgb, bytes_hash: u64, p: Patches, max_tokens: u32) Error!Prepared {
    if (max_tokens == 0) return error.ImageTooSmall;
    const b = bounds(p, max_tokens);
    const size = try smartResize(@intCast(img.height), @intCast(img.width), p.factor(), b[0], b[1]);
    const resized = try picture.resample.resize(gpa, p.resize, .of(img.pixels, img.width, img.height), size.w, size.h);
    defer gpa.free(resized);
    const grid = [2]u32{ size.h / p.patch, size.w / p.patch };
    const pixels = try patchify(gpa, .{ .width = size.w, .height = size.h, .pixels = resized }, p);
    return .{ .pixels = pixels, .gh = grid[0], .gw = grid[1], .tokens = grid[0] * grid[1] / (p.merge * p.merge), .hash = picture.identity.withGrid(bytes_hash, grid) };
}

/// Patches in merge order (block row, block column, row, column), each [channel, frame, y, x], the frame repeated.
pub fn patchify(gpa: std.mem.Allocator, img: picture.Rgb, p: Patches) error{OutOfMemory}![]f32 {
    const P = p.patch;
    const M = p.merge;
    const gh = img.height / P;
    const gw = img.width / P;
    const width = p.width();
    const out = try gpa.alloc(f32, gh * gw * width);
    const lut = picture.normalize.table(if (p.fused) .fused else .unfused, p.rescale, p.mean, p.std);
    var n: usize = 0;
    for (0..gh / M) |by| for (0..gw / M) |bx| for (0..M) |my| for (0..M) |mx| {
        const y0 = (by * M + my) * P;
        const x0 = (bx * M + mx) * P;
        const dst = out[n * width ..][0..width];
        for (0..3) |ch| for (0..p.temporal) |t| for (0..P) |dy| for (0..P) |dx| {
            dst[((ch * p.temporal + t) * P + dy) * P + dx] = lut[ch][img.pixels[((y0 + dy) * img.width + x0 + dx) * 3 + ch]];
        };
        n += 1;
    };
    return out;
}

test "smart_resize as the processor rounds, grows and shrinks" {
    // the gate pictures' sizes (h, w) at a 4,096-token budget, as transformers' processor grids them
    const cases = [_]struct { h: u32, w: u32, want: Size }{
        .{ .h = 900, .w = 1280, .want = .{ .h = 896, .w = 1280 } },
        .{ .h = 80, .w = 1600, .want = .{ .h = 64, .w = 1600 } }, // 2.5 rounds to 2
        .{ .h = 700, .w = 1225, .want = .{ .h = 704, .w = 1216 } },
        .{ .h = 200, .w = 120, .want = .{ .h = 352, .w = 224 } }, // under 65,536 pixels: grown
        .{ .h = 80, .w = 360, .want = .{ .h = 128, .w = 544 } },
        .{ .h = 64, .w = 96, .want = .{ .h = 224, .w = 320 } },
    };
    for (cases) |k| try std.testing.expectEqual(k.want, try smartResize(k.h, k.w, 32, 65536, 4096 * 1024));
    // over the budget: the largest grid inside it
    const big = try smartResize(3000, 4000, 32, 65536, 256 * 1024);
    try std.testing.expect(@as(u64, big.h) * big.w <= 256 * 1024 and big.h % 32 == 0 and big.w % 32 == 0);
    try std.testing.expectError(error.AspectRatio, smartResize(10, 2010, 32, 65536, 4096 * 1024));
    // LFM2-VL's variant: no aspect bound, a side at least one factor
    const lfm2: Variant = .{ .at_least_factor = true, .max_aspect = null };
    try std.testing.expectEqual(Size{ .h = 32, .w = 3648 }, try smartResizeWith(10, 2010, 32, 65536, 262144, lfm2));
    try std.testing.expectEqual(@as(f64, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f64, 4), roundEven(3.5));
    try std.testing.expectEqual(@as(f64, 3), roundEven(2.75));
}

test {
    _ = @import("patches_test.zig");
}
