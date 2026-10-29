//! Qwen-VL image geometry: core/picture/patches over a checkpoint's processor settings.
const std = @import("std");
const picture = @import("../picture/picture.zig");
const patches = picture.patches;
const Config = @import("config.zig").Config;

pub const Size = patches.Size;
pub const Error = patches.Error;
pub const Prepared = patches.Prepared;
pub const problem = patches.problem;
pub const smartResize = patches.smartResize;

/// An image's pixel bounds under a `max_tokens` budget: the processor's, its most cut to that many tokens.
pub fn bounds(c: Config, max_tokens: u32) [2]u64 {
    return patches.bounds(c.patches(), max_tokens);
}

/// Bytes to patches: decoded as 0.6.6 decoded them (core/picture), resized with Pillow's bicubic, cut in merge order.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, c: Config, max_tokens: u32) Error!Prepared {
    return patches.prepare(io, gpa, bytes, c.patches(), max_tokens);
}

/// RGB to patches; the hash covers the bytes' hash and the grid, so one picture at two budgets is two images.
pub fn prepareRgb(gpa: std.mem.Allocator, img: picture.Rgb, bytes_hash: u64, c: Config, max_tokens: u32) Error!Prepared {
    return patches.prepareRgb(gpa, img, bytes_hash, c.patches(), max_tokens);
}

/// Patches in merge order (block row, block column, row, column), each [channel, frame, y, x], the frame repeated.
pub fn patchify(gpa: std.mem.Allocator, img: picture.Rgb, c: Config) error{OutOfMemory}![]f32 {
    return patches.patchify(gpa, img, c.patches());
}

const sample = @import("config.zig");

fn sampleConfig(a: std.mem.Allocator) !Config {
    return (try Config.parse(a, sample.sample_config, sample.sample_preprocessor, sample.sample_processor)).?;
}

test "patches in merge order, each channel by frame by row by column, normalized" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try sampleConfig(a);
    // 64 x 32 pixels: a 4 x 2 patch grid, two merge blocks stacked; each pixel's red is its patch's raster index
    var px: [64 * 32 * 3]u8 = undefined;
    for (0..64) |y| for (0..32) |x| {
        px[(y * 32 + x) * 3] = @intCast((y / 16) * 2 + x / 16);
        px[(y * 32 + x) * 3 + 1] = 255;
        px[(y * 32 + x) * 3 + 2] = 0;
    };
    const out = try patchify(a, .{ .width = 32, .height = 64, .pixels = &px }, c);
    try std.testing.expectEqual(@as(usize, 8 * 1536), out.len);
    const red = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 }; // block (0,0): patches 0 1 2 3, then block (1,0): 4 5 6 7
    for (red, 0..) |r, n| {
        const p = out[n * 1536 ..][0..1536];
        const want: f32 = (@as(f32, @floatCast(@as(f64, @floatFromInt(r)) * c.rescale)) - 0.5) / 0.5;
        try std.testing.expectEqual(want, p[0]); // channel 0, frame 0
        try std.testing.expectEqual(want, p[256]); // channel 0, frame 1: the same image
        try std.testing.expectEqual(@as(f32, 1), p[512]); // green 255
        try std.testing.expectEqual(@as(f32, -1), p[1024]); // blue 0
    }
}

test "bounds follow the budget, and the hash names the grid" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try sampleConfig(a);
    try std.testing.expectEqual([2]u64{ 65536, 4096 * 1024 }, bounds(c, 4096));
    try std.testing.expectEqual([2]u64{ 32 * 1024, 32 * 1024 }, bounds(c, 32)); // under the processor's floor
    var px: [96 * 64 * 3]u8 = @splat(128);
    const img: picture.Rgb = .{ .width = 96, .height = 64, .pixels = &px };
    const grown = try prepareRgb(a, img, 7, c, 4096);
    try std.testing.expectEqual([3]u32{ 14, 20, 70 }, [3]u32{ grown.gh, grown.gw, grown.tokens });
    // under a tight budget the processor's floor rounds up past it, as transformers' does: the server refuses that
    try std.testing.expectEqual(@as(u32, 20), (try prepareRgb(a, img, 7, c, 16)).tokens);
    const photo: picture.Rgb = .{ .width = 640, .height = 480, .pixels = try a.alloc(u8, 640 * 480 * 3) };
    @memset(photo.pixels, 9);
    const whole = try prepareRgb(a, photo, 7, c, 4096);
    const cut = try prepareRgb(a, photo, 7, c, 64);
    try std.testing.expectEqual([2]u32{ 300, 54 }, [2]u32{ whole.tokens, cut.tokens });
    try std.testing.expect(whole.hash != cut.hash); // the same bytes on another grid are another image to the cache
    try std.testing.expectError(error.ImageTooSmall, prepareRgb(a, img, 7, c, 0));
    try std.testing.expectEqualStrings("its longer side is more than 200 times its shorter side", problem(error.AspectRatio).?);
}
