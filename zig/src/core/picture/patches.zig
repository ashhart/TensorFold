//! Image patch preparation from published model geometry and the project's processor behavior tests.
const std = @import("std");
const picture = @import("picture.zig");

pub const Patches = struct {
    patch: u32,
    temporal: u32,
    merge: u32,
    min_pixels: u64,
    max_pixels: u64,
    rescale: f64,
    mean: [3]f32,
    std: [3]f32,
    resize: picture.resample.Kind = .pillow,
    fused: bool = false,

    pub fn width(p: Patches) u32 {
        return 3 * p.temporal * p.patch * p.patch;
    }

    pub fn factor(p: Patches) u32 {
        return p.patch * p.merge;
    }
};

pub const Size = struct { h: u32, w: u32 };
pub const Error = picture.Error || error{ ImageTooSmall, AspectRatio };
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };
pub const Variant = struct { at_least_factor: bool = false, max_aspect: ?f64 = 200 };

pub fn problem(err: anyerror) ?[]const u8 {
    if (err == error.AspectRatio) return "its longer side is more than 200 times its shorter side";
    return picture.decoderProblem(err);
}

fn roundEven(value: f64) f64 {
    const lower = @floor(value);
    const tail = value - lower;
    return lower + @as(f64, if (tail > 0.5 or (tail == 0.5 and @mod(lower, 2) != 0)) 1 else 0);
}

pub fn smartResize(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64) error{AspectRatio}!Size {
    return smartResizeWith(height, width, factor, min_pixels, max_pixels, .{});
}

pub fn smartResizeWith(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64, v: Variant) error{AspectRatio}!Size {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const unit: f64 = @floatFromInt(factor);
    const minimum: f64 = @floatFromInt(min_pixels);
    const maximum: f64 = @floatFromInt(max_pixels);
    if (v.max_aspect) |limit| if (@max(h, w) / @min(h, w) > limit) return error.AspectRatio;
    var rows = roundEven(h / unit) * unit;
    var cols = roundEven(w / unit) * unit;
    if (v.at_least_factor) {
        rows = @max(rows, unit);
        cols = @max(cols, unit);
    }
    if (rows * cols > maximum) {
        const divisor = @sqrt(h * w / maximum);
        rows = @floor(h / divisor / unit) * unit;
        cols = @floor(w / divisor / unit) * unit;
    } else if (rows * cols < minimum) {
        const multiplier = @sqrt(minimum / (h * w));
        rows = @ceil(h * multiplier / unit) * unit;
        cols = @ceil(w * multiplier / unit) * unit;
    }
    return .{ .h = @intFromFloat(@max(unit, rows)), .w = @intFromFloat(@max(unit, cols)) };
}

pub fn bounds(p: Patches, max_tokens: u32) [2]u64 {
    const unit: u64 = p.factor();
    const ceiling = @min(p.max_pixels, @as(u64, max_tokens) * unit * unit);
    return .{ @min(p.min_pixels, ceiling), ceiling };
}

pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, p: Patches, max_tokens: u32) Error!Prepared {
    var rgb = try picture.decode(io, gpa, bytes);
    defer rgb.deinit(gpa);
    return prepareRgb(gpa, rgb, picture.identity.ofBytes(0x51e7, bytes), p, max_tokens);
}

pub fn prepareRgb(gpa: std.mem.Allocator, img: picture.Rgb, bytes_hash: u64, p: Patches, max_tokens: u32) Error!Prepared {
    if (max_tokens == 0) return error.ImageTooSmall;
    const limits = bounds(p, max_tokens);
    const size = try smartResize(@intCast(img.height), @intCast(img.width), p.factor(), limits[0], limits[1]);
    const resized = try picture.resample.resize(gpa, p.resize, .of(img.pixels, img.width, img.height), size.w, size.h);
    defer gpa.free(resized);
    const pixels = try patchify(gpa, .{ .width = size.w, .height = size.h, .pixels = resized }, p);
    const grid = [2]u32{ size.h / p.patch, size.w / p.patch };
    return .{
        .pixels = pixels,
        .gh = grid[0],
        .gw = grid[1],
        .tokens = grid[0] * grid[1] / (p.merge * p.merge),
        .hash = picture.identity.withGrid(bytes_hash, grid),
    };
}

fn normalized(value: u8, channel: usize, p: Patches) f32 {
    if (p.fused) {
        const bias: f32 = @floatCast(@as(f64, p.mean[channel]) / p.rescale);
        const divisor: f32 = @floatCast(@as(f64, p.std[channel]) / p.rescale);
        const centered: f32 = @as(f32, @floatFromInt(value)) - bias;
        return centered / divisor;
    }
    const scaled: f32 = @floatCast(@as(f64, @floatFromInt(value)) * p.rescale);
    const centered: f32 = scaled - p.mean[channel];
    return centered / p.std[channel];
}

pub fn patchify(gpa: std.mem.Allocator, img: picture.Rgb, p: Patches) error{OutOfMemory}![]f32 {
    const gh = img.height / p.patch;
    const gw = img.width / p.patch;
    const out = try gpa.alloc(f32, gh * gw * p.width());
    var at: usize = 0;
    for (0..gh / p.merge) |block_y| for (0..gw / p.merge) |block_x| {
        for (0..p.merge) |merge_y| for (0..p.merge) |merge_x| {
            const top = (block_y * p.merge + merge_y) * p.patch;
            const left = (block_x * p.merge + merge_x) * p.patch;
            for (0..3) |channel| for (0..p.temporal) |_| {
                for (0..p.patch) |y| for (0..p.patch) |x| {
                    out[at] = normalized(img.pixels[((top + y) * img.width + left + x) * 3 + channel], channel, p);
                    at += 1;
                };
            };
        };
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
