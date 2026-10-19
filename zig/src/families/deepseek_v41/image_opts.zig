//! The prefill's block options for image prompts: pure functions of the forward's image settings and a segment's ids.
const std = @import("std");
const block = @import("block.zig");
const Forward = @import("forward.zig").Forward;

/// The first virtual id of an image position; text ids sit below it.
pub const VBASE: u32 = 1 << 24;

pub fn isVid(t: u32) bool {
    return t >= VBASE;
}

/// Whether a window's ids hold an image position.
pub fn hasImage(ids: []const u32) bool {
    for (ids) |t| if (isVid(t)) return true;
    return false;
}

/// The plan's options: the keep buffer when images are on, both MoE calls at full rows with the image bias.
pub fn planWith(o: block.Options, images: bool, bias: bool) block.Options {
    var p = o;
    p.image_keep = images;
    if (images and bias) p.image_rows = -1;
    return p;
}

/// The options the buffer plan sizes with.
pub fn planOptions(f: *const Forward) block.Options {
    return planWith(f.opts, f.images, f.image_bias);
}

/// A segment's options: the keep step when it holds an image position, its image rows under the image bias.
pub fn segmentWith(base: block.Options, images: bool, bias: bool, ids: []const u32) block.Options {
    var o = base;
    o.image_keep = images and hasImage(ids);
    if (o.image_keep and bias) {
        var n: i64 = 0;
        for (ids) |t| n += @intFromBool(isVid(t));
        o.image_rows = n;
    }
    return o;
}

/// A segment's options on the forward `f`.
pub fn segmentOptions(f: *const Forward, ids: []const u32) block.Options {
    return segmentWith(f.opts, f.images, f.image_bias, ids);
}

test "image options: text segments keep the forward's options, image ones the keep step and their rows" {
    const vid = VBASE + 7;
    try std.testing.expect(!hasImage(&.{ 1, 2, VBASE - 1 }));
    try std.testing.expect(hasImage(&.{ 1, vid }));
    const o: block.Options = .{};
    const off = planWith(o, false, true);
    try std.testing.expect(!off.image_keep);
    try std.testing.expectEqual(o.image_rows, off.image_rows);
    const on = planWith(o, true, false);
    try std.testing.expect(on.image_keep);
    try std.testing.expectEqual(o.image_rows, on.image_rows);
    try std.testing.expectEqual(@as(i64, -1), planWith(o, true, true).image_rows);
    const ids = [_]u32{ 5, vid, vid + 1, 9 };
    try std.testing.expectEqual(o, segmentWith(o, false, true, &ids)); // images off: the forward's options
    try std.testing.expectEqual(o, segmentWith(o, true, true, &.{ 5, 9 })); // a text segment
    const kept = segmentWith(o, true, false, &ids);
    try std.testing.expect(kept.image_keep);
    try std.testing.expectEqual(o.image_rows, kept.image_rows);
    try std.testing.expectEqual(@as(i64, 2), segmentWith(o, true, true, &ids).image_rows);
}
