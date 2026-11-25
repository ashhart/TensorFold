//! The region outline PaddleX's ocrImage draws: its thickness along the edges and its clipping to the image.
const std = @import("std");
const draw = @import("draw.zig");
const t = std.testing;

fn isRed(rgb: []const u8, w: u32, x: u32, y: u32) bool {
    const at = (y * w + x) * 3;
    return std.mem.eql(u8, rgb[at..][0..3], &draw.red);
}

test "ocr draw: an outline is 2 px thick along the polygon's edges" {
    const w = 12;
    const h = 10;
    var rgb: [w * h * 3]u8 = @splat(0);
    draw.outline(&rgb, w, h, .{ .{ 2, 2 }, .{ 8, 2 }, .{ 8, 7 }, .{ 2, 7 } }, draw.red, 2);
    for ([_][2]u32{ .{ 2, 2 }, .{ 5, 2 }, .{ 5, 1 }, .{ 8, 4 }, .{ 7, 4 }, .{ 2, 7 }, .{ 1, 5 }, .{ 8, 7 } }) |p| try t.expect(isRed(&rgb, w, p[0], p[1]));
    for ([_][2]u32{ .{ 5, 5 }, .{ 0, 0 }, .{ 11, 9 }, .{ 5, 3 }, .{ 3, 5 }, .{ 9, 4 }, .{ 5, 8 } }) |p| try t.expect(!isRed(&rgb, w, p[0], p[1]));
}

test "ocr draw: an outline is clipped to the image" {
    const w = 8;
    const h = 6;
    var rgb: [w * h * 3]u8 = @splat(0);
    draw.outline(&rgb, w, h, .{ .{ -5, -5 }, .{ 20, -5 }, .{ 20, 3 }, .{ -5, 3 } }, draw.red, 2);
    for (0..w) |x| try t.expect(isRed(&rgb, w, @intCast(x), 3));
    try t.expect(!isRed(&rgb, w, 3, 0));
    // corners far outside cost no work and write nothing out of bounds
    var far: [w * h * 3]u8 = @splat(0);
    const big = std.math.maxInt(i32);
    draw.outline(&far, w, h, .{ .{ -big, -big }, .{ big, -big }, .{ big, -big + 1 }, .{ -big, -big + 1 } }, draw.red, 2);
    for (far) |b| try t.expectEqual(@as(u8, 0), b);
    draw.outline(&far, w, h, .{ .{ 0, 0 }, .{ w, 0 }, .{ w, h }, .{ 0, h } }, draw.red, 2);
    try t.expect(isRed(&far, w, 0, 0) and isRed(&far, w, w - 1, h - 1) and !isRed(&far, w, 3, 3));
}
