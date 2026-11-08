//! The page budget, the image-encode budget and the region outline (design 7.2).
const std = @import("std");
const budget = @import("budget.zig");
const draw = @import("draw.zig");
const types = @import("types.zig");
const t = std.testing;

test "ocr budget: pages default to at most 16 and take only what is left" {
    const p = budget.Pages.init();
    try t.expect(p.left >= 1 and p.left <= 16);
    var q: budget.Pages = .{ .left = 3 };
    try t.expect(q.take(2));
    try t.expect(!q.take(2));
    try t.expectEqual(@as(u32, 1), q.left);
    try t.expect(q.take(1));
    try t.expect(!q.take(1));
}

test "ocr budget: the 257th encode faults before it runs" {
    var fault: types.Fault = undefined;
    var e = budget.Encodes.init(1 << 30, 1);
    for (0..256) |_| try e.before(&fault, 1, 1);
    try t.expectError(error.Faulted, e.before(&fault, 1, 1));
    try t.expectEqual(types.Class.unreadable, fault.class);
    try t.expectEqualStrings("too many image regions", fault.message);
    try t.expectEqual(@as(u32, 256), e.count);
}

test "ocr budget: encoded pixels stop at 4x the request's page pixels" {
    var fault: types.Fault = undefined;
    var e = budget.Encodes.init(100 * 100, 1);
    try t.expectEqual(@as(u64, 40_000), e.limit_pixels);
    try e.before(&fault, 100, 300);
    try e.before(&fault, 100, 100); // exactly at the limit
    try t.expectError(error.Faulted, e.before(&fault, 1, 1));
    try t.expectEqual(types.Class.unreadable, fault.class);
    try t.expectEqual(@as(u64, 40_000), e.pixels);
    try t.expectEqual(@as(u32, 2), e.count);
}

test "ocr budget: encoded bytes fault after the encode that passes 16 MiB and 2 MiB a page" {
    for ([_]struct { usize, u64 }{ .{ 1, 18 << 20 }, .{ 16, 48 << 20 } }) |c| {
        var fault: types.Fault = undefined;
        var e = budget.Encodes.init(1 << 30, c[0]);
        try t.expectEqual(c[1], e.max_bytes);
        try e.after(&fault, c[1] - 1);
        try e.after(&fault, 1); // exactly at the limit
        try t.expectError(error.Faulted, e.after(&fault, 1));
        try t.expectEqual(types.Class.unreadable, fault.class);
        try t.expectEqualStrings("too many image regions", fault.message);
    }
}

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
