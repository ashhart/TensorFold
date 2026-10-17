//! Actual Vision.begin reaches a no-CUDA tower shape sentinel after checking the production reservation.
const std = @import("std");
const vision = @import("cuda_vision.zig");
const radio = @import("radio");
const lanes = @import("lanes");
test "low independent image and video budgets reach the actual tower boundary" {
    for ([_][3]u32{ .{ 16, 512, 256 }, .{ 256, 16384, 2048 }, .{ 256, 512, 768 } }) |c| {
        var v: vision.Vision = undefined;
        v.gpa = std.testing.allocator;
        v.spans = &.{};
        defer v.end();
        v.rows = .{ .d = undefined, .buf = undefined, .hidden = 4096, .limit = try vision.reservation(.{ .image_tokens = c[0], .video_tokens = c[1], .max_images = 4 }, true) };
        v.tower.c.text_hidden = 1;
        var images: [8]lanes.stream.Image = undefined;
        const count = c[2] / 256;
        for (images[0..count], 0..) |*img, i| img.* = .{ .pixels = &.{}, .gh = 32, .gw = 32, .tokens = 256, .at = @intCast(1 + i * 300) };
        try std.testing.expectError(error.RowShape, v.begin(undefined, images[0..count], 0));
        try std.testing.expectEqual(c[2], radio.rows.total(v.spans));
    }
}

test "reservation rejects overflow and keeps image-only budget" {
    try std.testing.expectError(error.BadVisionLimits, vision.reservation(.{ .image_tokens = 1, .video_tokens = std.math.maxInt(u32), .max_images = 1 }, true));
    try std.testing.expectEqual(@as(u32, 256), try vision.reservation(.{ .image_tokens = 256, .video_tokens = 512, .max_images = 1 }, false));
}
