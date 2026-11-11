const std = @import("std");
const Size = @import("prompts.zig").Size;

pub const Mask = struct {
    size: Size,
    pixels: []u8,
    score: f32,
    box: [4]f32,
};

pub fn fromLogits(allocator: std.mem.Allocator, logits: []const f32, size: Size, score: f32) !Mask {
    if (logits.len != try size.pixels()) return error.MaskShapeMismatch;
    if (!std.math.isFinite(score)) return error.NonFiniteMaskScore;
    const pixels = try allocator.alloc(u8, logits.len);
    errdefer allocator.free(pixels);
    var x0 = size.width;
    var y0 = size.height;
    var x1: usize = 0;
    var y1: usize = 0;
    for (pixels, logits, 0..) |*pixel, value, i| {
        if (!std.math.isFinite(value)) return error.NonFiniteMaskLogits;
        pixel.* = if (value > 0) 255 else 0;
        if (pixel.* == 0) continue;
        x0 = @min(x0, i % size.width);
        y0 = @min(y0, i / size.width);
        x1 = @max(x1, i % size.width + 1);
        y1 = @max(y1, i / size.width + 1);
    }
    return .{ .size = size, .pixels = pixels, .score = score, .box = if (x1 == 0) .{ 0, 0, 0, 0 } else .{
        @floatFromInt(x0), @floatFromInt(y0), @floatFromInt(x1), @floatFromInt(y1),
    } };
}

test "SAM mask conversion keeps zero outside and reports the foreground bounds" {
    const a = std.testing.allocator;
    const mask = try fromLogits(a, &.{ -1, -0.0, 0, 1 }, .{ .width = 2, .height = 2 }, 0.25);
    defer a.free(mask.pixels);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 255 }, mask.pixels);
    try std.testing.expectEqual([4]f32{ 1, 1, 2, 2 }, mask.box);
    try std.testing.expectEqual(@as(f32, 0.25), mask.score);
    const empty = try fromLogits(a, &.{ -1, -0.0, 0, -1 }, .{ .width = 2, .height = 2 }, 0);
    defer a.free(empty.pixels);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, empty.box);
}

test "SAM mask conversion refuses nonfinite logits before producing apparently valid background" {
    const a = std.testing.allocator;
    var logits: [56]f32 = @splat(-1);
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| for ([_]usize{ 0, 15, 16, 55 }) |at| {
        logits[at] = bad;
        if (fromLogits(a, &logits, .{ .width = 7, .height = 8 }, 1)) |mask| {
            a.free(mask.pixels);
            return error.NonFiniteMaskWasAccepted;
        } else |err| try std.testing.expectEqual(error.NonFiniteMaskLogits, err);
        logits[at] = -1;
    };
}

test "SAM mask conversion refuses nonfinite confidence and inconsistent geometry" {
    const a = std.testing.allocator;
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |score| {
        if (fromLogits(a, &.{1}, .{ .width = 1, .height = 1 }, score)) |mask| {
            a.free(mask.pixels);
            return error.NonFiniteScoreWasAccepted;
        } else |err| try std.testing.expectEqual(error.NonFiniteMaskScore, err);
    }
    try std.testing.expectError(error.MaskShapeMismatch, fromLogits(a, &.{1}, .{ .width = 2, .height = 1 }, 1));
}
