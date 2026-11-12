//! Image setup lifetime through the real Vision methods and host-only encoders.
const std = @import("std");
const cuda = @import("cuda");
const vision = @import("cuda_vision.zig");
const radio = @import("radio");
const lanes = @import("lanes");
test "low independent image and video budgets reach the actual tower boundary" {
    for ([_][3]u32{ .{ 16, 512, 256 }, .{ 256, 16384, 2048 }, .{ 256, 512, 768 } }) |c| {
        var v: vision.Vision = undefined;
        v.gpa = std.testing.allocator;
        v.spans = &.{};
        v.encoder = null;
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

const request_images = [_]lanes.stream.Image{
    .{ .pixels = &.{}, .gh = 4, .gw = 4, .tokens = 4, .at = 1 },
    .{ .pixels = &.{}, .gh = 4, .gw = 4, .tokens = 4, .at = 7 },
};

const Encode = struct {
    calls: usize = 0,
    rows: u32 = 0,
    fail: bool = false,

    fn run(ctx: *anyopaque, _: cuda.Stream, _: []const lanes.stream.Image, spans: []const radio.Span, _: *const radio.Rows) anyerror!void {
        const e: *Encode = @ptrCast(@alignCast(ctx));
        e.calls += 1;
        e.rows = radio.rows.total(spans);
        if (e.fail) return error.EncodeFailed;
    }
};

fn fixture(gpa: std.mem.Allocator, encoder: *Encode, limit: u32) vision.Vision {
    return .{
        .gpa = gpa,
        .io = undefined,
        .cfg = undefined,
        .tower = undefined,
        .rows = .{ .d = undefined, .buf = undefined, .hidden = 1, .limit = limit },
        .marks = undefined,
        .limits = undefined,
        .mark_ids = undefined,
        .encoder = .{ .ctx = encoder, .encode = Encode.run },
    };
}

test "image rows refusal clears spans before a text pass" {
    var encoder: Encode = .{};
    var v = fixture(std.testing.allocator, &encoder, 7);
    defer v.end();
    try std.testing.expectError(error.ImageRows, v.begin(undefined, &request_images, 0));
    try std.testing.expectEqual(@as(usize, 0), encoder.calls);
    try std.testing.expectEqual(@as(usize, 0), v.spans.len);
    try v.splice(undefined, 0, 20, 0);
}

test "image encoder failure clears spans before a text pass and retry" {
    var encoder: Encode = .{ .fail = true };
    var v = fixture(std.testing.allocator, &encoder, 8);
    defer v.end();
    for (1..4) |attempt| {
        try std.testing.expectError(error.EncodeFailed, v.begin(undefined, &request_images, 0));
        try std.testing.expectEqual(attempt, encoder.calls);
        try std.testing.expectEqual(@as(u32, 8), encoder.rows);
        try std.testing.expectEqual(@as(usize, 0), v.spans.len);
        try v.splice(undefined, 0, 20, 0);
    }
    encoder.fail = false;
    try v.begin(undefined, &request_images, 0);
    try std.testing.expectEqual(@as(usize, 4), encoder.calls);
    try std.testing.expectEqual(@as(u32, 8), radio.rows.total(v.spans));
}

test "image success retains only the active prompt spans" {
    var encoder: Encode = .{};
    var v = fixture(std.testing.allocator, &encoder, 8);
    defer v.end();
    try v.begin(undefined, &request_images, 0);
    try std.testing.expectEqual(@as(u32, 8), encoder.rows);
    try std.testing.expectEqual(@as(u32, 8), radio.rows.total(v.spans));
    try v.begin(undefined, &request_images, 5);
    try std.testing.expectEqual(@as(usize, 2), encoder.calls);
    try std.testing.expectEqual(@as(u32, 4), encoder.rows);
    try std.testing.expectEqualSlices(radio.Span, &.{.{ .at = 7, .n = 4, .row = 0, .image = 1 }}, v.spans);
    try v.begin(undefined, &.{}, 0);
    try std.testing.expectEqual(@as(usize, 2), encoder.calls);
    try std.testing.expectEqual(@as(usize, 0), v.spans.len);
    try v.splice(undefined, 0, 20, 0);
}

fn allocationTrial(gpa: std.mem.Allocator) !void {
    var encoder: Encode = .{};
    var v = fixture(gpa, &encoder, 8);
    defer v.end();
    for (0..2) |_| {
        v.begin(undefined, &request_images, 0) catch |err| {
            try std.testing.expectEqual(@as(usize, 0), v.spans.len);
            try v.splice(undefined, 0, 20, 0);
            return err;
        };
        try std.testing.expectEqual(@as(u32, 8), radio.rows.total(v.spans));
    }
}

test "image planning allocation failure releases the prior prompt spans" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationTrial, .{});
}
