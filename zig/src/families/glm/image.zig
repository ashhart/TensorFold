//! An image's bytes to GLM5-Next vision patches as TensorFold 0.6.6's --vision made them (Pillow, then mlx-vlm).
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const vision = @import("vision_config.zig");
const core = @import("image_core.zig");
const steps = core.For(picture);

pub const Prepared = core.Prepared;
pub const Geometry = core.Geometry;
pub const geometry = core.geometry;
pub const Error = steps.Error;
pub const decoderProblem = steps.decoderProblem;
pub const patchify = steps.patchify;
pub const prepare = steps.prepare;
pub const prepareRgb = steps.prepareRgb;

test "geometry follows smart_resize" {
    // the reference dumps' grids: 517x389 pads to 532x392 unscaled; 96x96 grows to 112; 20x15 to the 16-token floor
    try std.testing.expectEqual(Geometry{ .target_h = 392, .target_w = 532, .content_h = 389, .content_w = 517 }, try geometry(389, 517, 16, 4096));
    try std.testing.expectEqual(Geometry{ .target_h = 112, .target_w = 112, .content_h = 112, .content_w = 112 }, try geometry(96, 96, 16, 4096));
    const tiny = try geometry(15, 20, 16, 4096);
    try std.testing.expectEqual(@as(u32, 8 * 14), tiny.target_h);
    try std.testing.expectEqual(@as(u32, 10 * 14), tiny.target_w);
    // a large image under a small budget: the canvas fits it
    const big = try geometry(1500, 2000, 16, 256);
    try std.testing.expect(@as(u64, big.target_h) * big.target_w * 2 <= 256 * 2 * 28 * 28);
    try std.testing.expect(big.content_h <= big.target_h and big.content_w <= big.target_w);
}

test "the hash names the grid: one picture at two budgets is two images to the cache, at one budget the same" {
    const gpa = std.testing.allocator;
    const c: vision.Config = .{ .image_token = 1, .image_start = 2, .image_end = 3, .mean = @splat(0.5), .std = @splat(0.5), .min_tokens = 16, .max_tokens = 4096 };
    const pixels = try gpa.alloc(u8, 640 * 480 * 3);
    defer gpa.free(pixels);
    @memset(pixels, 9);
    const photo: picture.Rgb = .{ .width = 640, .height = 480, .pixels = pixels };
    const whole = try prepareRgb(gpa, photo, 7, c, 4096);
    defer gpa.free(whole.pixels);
    const cut = try prepareRgb(gpa, photo, 7, c, 64);
    defer gpa.free(cut.pixels);
    const again = try prepareRgb(gpa, photo, 7, c, 4096);
    defer gpa.free(again.pixels);
    try std.testing.expect(whole.tokens > cut.tokens);
    try std.testing.expect(whole.hash != cut.hash); // another grid: neither resumes a state the other kept
    try std.testing.expectEqual(whole.hash, again.hash); // the same picture at the same budget resumes
}
