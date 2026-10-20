//! Actual Vision.begin on the host: the reservation up to the tower's shape check, and a lane prefill's tower passes.
const std = @import("std");
const vision = @import("cuda_vision.zig");
const radio = @import("radio");
const lanes = @import("lanes");
const Engine = @import("cuda_engine.zig").Engine;
const Cuda = @import("cuda_lanes.zig").Cuda;
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

const Tower = struct {
    var encoded: [2]u32 = @splat(0);
    fn encode(spans: []const radio.Span) !void {
        for (spans) |sp| encoded[sp.image] += 1;
    }
};

test "a lane prefill runs each image of its prompt through the tower once" {
    const a = std.testing.allocator;
    vision.Vision.test_encode = Tower.encode;
    defer vision.Vision.test_encode = null;
    var v: vision.Vision = undefined;
    v.gpa = a;
    v.spans = &.{};
    v.rows.limit = 8;
    var e: Engine = undefined;
    e.max_len = 4096;
    e.serial = .{ .d = undefined, .handle = null };
    e.own = .{ .arena = null, .ptr = @splat(0), .cap = 1 }; // too few rows: the pass stops before its first chunk
    e.bound = &e.own;
    e.head = null;
    e.vision = &v;
    var b: Cuda = undefined;
    b.gpa = a;
    b.e = &e;
    b.lanes = .empty;
    defer b.lanes.deinit(a);
    b.own_free = true;
    b.paged_only = false;
    b.tapped = null;
    const images = [_]lanes.stream.Image{ .{ .pixels = &.{}, .gh = 2, .gw = 6, .at = 2, .tokens = 3 }, .{ .pixels = &.{}, .gh = 2, .gw = 4, .at = 6, .tokens = 2 } };
    var s = try lanes.Stream.init(a, .{ .id = "images", .prompt = &.{ 1, 2, 3, 3, 3, 4, 3, 3, 5, 6 }, .max_new = 4, .images = &images });
    defer s.deinit(a);
    try std.testing.expectError(error.ContextFull, b.backend().prefill(&s));
    try std.testing.expectEqual([2]u32{ 1, 1 }, Tower.encoded);
}
