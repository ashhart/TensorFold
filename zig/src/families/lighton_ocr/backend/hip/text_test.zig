//! HIP decoder frames are checked on the host before asynchronous metadata or cache changes.
const std = @import("std");
const text = @import("text.zig");
const config = @import("../../config.zig");
const a = std.testing.allocator;

fn cfg() !config.Config {
    return config.Config.parse(a, config.sample);
}

test "LightOn HIP text: asymmetric widths and physical verification slack are counted" {
    const c = try cfg();
    try std.testing.expectEqual(@as(usize, 28 * 2 * 16400 * 1024 * 2), try text.streamBytes(c, 16400));
    try std.testing.expectError(error.TextShape, text.streamBytes(c, 16401));
    try std.testing.expectError(error.TextShape, text.streamBytes(c, 0));
    const s = try text.Scratch.init(c, 16400);
    try std.testing.expectEqual(s.bytes, try text.scratchBytes(c, 16400));
    try std.testing.expect(s.qraw + text.max_rows * 2048 * 2 <= s.kraw);
    try std.testing.expect(s.kraw + text.max_rows * 1024 * 2 <= s.vraw);
    try std.testing.expect(s.qwide + text.max_rows * 2048 * 4 <= s.awide);
    try std.testing.expect(s.logits + text.max_rows * c.vocab * 2 <= s.bytes);
    inline for (@typeInfo(text.Scratch).@"struct".field_names) |field|
        try std.testing.expectEqual(@as(usize, 0), @field(s, field) % 256);
}

test "LightOn HIP text: several slots share one validated projection frame" {
    const c = try cfg();
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const spans = [_]text.Span{
        .{ .slot = 1, .first = 0, .rows = 3, .committed = 512 },
        .{ .slot = 0, .first = 3, .rows = 5, .committed = 1200 },
    };
    const images = [_]text.ImageRows{.{ .first = 1, .rows = 2, .features = 4096 }};
    try text.validate(c, 16400, 2, &ids, &spans, &images);
    const crossing = [_]text.ImageRows{.{ .first = 2, .rows = 2, .features = 4096 }};
    try std.testing.expectError(error.ImageRows, text.validate(c, 16400, 2, &ids, &spans, &crossing));
}

test "LightOn HIP text: malformed spans and image aliases fail before device access" {
    const c = try cfg();
    const ids = [_]u32{ 1, 2 };
    const valid = [_]text.Span{.{ .slot = 0, .first = 0, .rows = 2, .committed = 0 }};
    for ([_]text.Span{
        .{ .slot = 1, .first = 0, .rows = 2, .committed = 0 },
        .{ .slot = 0, .first = 1, .rows = 2, .committed = 0 },
        .{ .slot = 0, .first = 0, .rows = 0, .committed = 0 },
        .{ .slot = 0, .first = 0, .rows = 3, .committed = 0 },
        .{ .slot = 0, .first = 0, .rows = 2, .committed = 16399 },
        .{ .slot = 0, .first = 0, .rows = 2, .committed = std.math.maxInt(u32) },
    }) |bad| try std.testing.expectError(error.TextShape, text.validate(c, 16400, 1, &ids, &.{bad}, &.{}));
    const duplicate = [_]text.Span{
        .{ .slot = 0, .first = 0, .rows = 1, .committed = 0 },
        .{ .slot = 0, .first = 1, .rows = 1, .committed = 1 },
    };
    try std.testing.expectError(error.DuplicateTextSlot, text.validate(c, 16400, 2, &ids, &duplicate, &.{}));
    for ([_]text.ImageRows{
        .{ .first = 2, .rows = 1, .features = 4096 },
        .{ .first = 0, .rows = 0, .features = 4096 },
        .{ .first = 0, .rows = 1, .features = 0 },
        .{ .first = 0, .rows = 1, .features = 4098 },
    }) |bad| try std.testing.expectError(error.ImageRows, text.validate(c, 16400, 1, &ids, &valid, &.{bad}));
    const overlap = [_]text.ImageRows{
        .{ .first = 0, .rows = 2, .features = 4096 }, .{ .first = 1, .rows = 1, .features = 8192 },
    };
    try std.testing.expectError(error.ImageRows, text.validate(c, 16400, 1, &ids, &valid, &overlap));
    var t: text.Text = undefined;
    t.c = c;
    t.context = 16400;
    t.slots = 1;
    try std.testing.expectError(error.TextToken, t.forward(&.{ 1, c.vocab }, &valid, &.{}));
}

test "LightOn HIP text: cache accessors preserve independent slot and layer head strides" {
    const c = try cfg();
    var t: text.Text = undefined;
    t.c = c;
    t.context = 16400;
    t.slots = 3;
    t.kv = .{ .r = undefined, .ptr = @ptrFromInt(4096), .len = (try text.streamBytes(c, 16400)) * 3 };
    const one: u64 = 16400 * 1024 * 2;
    try std.testing.expectEqual(@as(u64, 4096), t.key(0, 0));
    try std.testing.expectEqual(@as(u64, 4096) + one, t.key(1, 0));
    try std.testing.expectEqual(@as(u64, 4096) + 3 * one, t.value(0, 0));
    try std.testing.expectEqual(@as(u64, 4096) + 6 * one, t.key(0, 1));
    try std.testing.expectEqual(@as(u64, 4096) + 167 * one, t.value(2, 27));
}
