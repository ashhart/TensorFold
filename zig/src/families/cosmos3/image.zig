//! Cosmos3-Edge's image processor: sides multiples of 32, antialiased bicubic, (x - 127.5) / 127.5, 2 x 2 blocks.
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const geometry = @import("../../core/qwen_vl/geometry.zig");
const c = @import("config.zig");

pub const factor = c.patch * c.merge;
pub const min_pixels: u64 = 65536;
pub const max_pixels: u64 = 16777216;

pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

pub fn decoderProblem(err: anyerror) ?[]const u8 {
    return geometry.problem(err);
}

/// Bytes (PNG, JPEG or WebP) to patches under a `max_tokens` budget.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !Prepared {
    var img = try picture.decode(io, gpa, bytes);
    defer img.deinit(gpa);
    return prepareRgb(gpa, img, picture.identity.ofBytes(0xc053, bytes), max_tokens);
}

pub fn prepareRgb(gpa: std.mem.Allocator, img: picture.Rgb, bytes_hash: u64, max_tokens: u32) !Prepared {
    if (max_tokens == 0) return error.ImageTooSmall;
    const most = @min(max_pixels, @as(u64, max_tokens) * factor * factor);
    const size = try geometry.smartResize(@intCast(img.height), @intCast(img.width), factor, @min(min_pixels, most), most);
    const resized = try picture.resample.resize(gpa, .torch, .of(img.pixels, img.width, img.height), size.w, size.h);
    defer gpa.free(resized);
    const gh = size.h / c.patch;
    const gw = size.w / c.patch;
    return .{ .pixels = try patchify(gpa, resized, size.w, gh, gw), .gh = gh, .gw = gw, .tokens = gh * gw / (c.merge * c.merge), .hash = picture.identity.withGrid(bytes_hash, .{ gh, gw }) };
}

/// Patches [gh * gw, 768]: block row, block column, row in block, column in block; each patch (y, x, channel).
pub fn patchify(gpa: std.mem.Allocator, rgb: []const u8, width: usize, gh: usize, gw: usize) ![]f32 {
    const P = c.patch;
    const M = c.merge;
    const out = try gpa.alloc(f32, gh * gw * c.patch_width);
    // transformers' torchvision backend folds the rescale into the mean and std: (x - 127.5) / 127.5 in float32
    var lut: [256]f32 = undefined;
    for (&lut, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) - 127.5) / 127.5;
    var n: usize = 0;
    for (0..gh / M) |by| for (0..gw / M) |bx| for (0..M) |my| for (0..M) |mx| {
        const y0 = (by * M + my) * P;
        const x0 = (bx * M + mx) * P;
        const dst = out[n * c.patch_width ..][0..c.patch_width];
        for (0..P) |dy| for (0..P) |dx| for (0..3) |ch| {
            dst[(dy * P + dx) * 3 + ch] = lut[rgb[((y0 + dy) * width + x0 + dx) * 3 + ch]];
        };
        n += 1;
    };
    return out;
}

test "a 640 x 480 frame is 40 x 30 patches, 300 rows, in block order with HWC patches" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const px = try a.alloc(u8, 640 * 480 * 3);
    for (0..480) |y| for (0..640) |x| {
        px[(y * 640 + x) * 3] = @intCast((y / 16) % 256);
        px[(y * 640 + x) * 3 + 1] = @intCast((x / 16) % 256);
        px[(y * 640 + x) * 3 + 2] = 255;
    };
    const p = try prepareRgb(a, .{ .width = 640, .height = 480, .pixels = px }, 1, 4096);
    try std.testing.expectEqual([3]u32{ 30, 40, 300 }, [3]u32{ p.gh, p.gw, p.tokens });
    const v = struct {
        fn of(i: u8) f32 {
            return (@as(f32, @floatFromInt(i)) - 127.5) / 127.5;
        }
    }.of;
    // patch 3 is block (0, 0)'s (1, 1): patch row 1, column 1
    try std.testing.expectEqual(v(1), p.pixels[3 * 768]);
    try std.testing.expectEqual(v(1), p.pixels[3 * 768 + 1]);
    try std.testing.expectEqual(@as(f32, 1), p.pixels[3 * 768 + 2]);
    // patch 4 is block (0, 1)'s first: patch row 0, column 2
    try std.testing.expectEqual(v(0), p.pixels[4 * 768]);
    try std.testing.expectEqual(v(2), p.pixels[4 * 768 + 1]);
}
