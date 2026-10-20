//! The actual lane prompt pass with a counting encoder: each request sends each of its images through the tower once.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const radio = @import("radio");
const vision = @import("cuda_vision.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Cuda = @import("cuda_lanes.zig").Cuda;
const a = std.testing.allocator;

const Count = struct {
    passes: usize = 0,
    images: usize = 0,

    fn encode(ctx: *anyopaque, _: cuda.Stream, _: []const lanes.stream.Image, spans: []const radio.Span, _: *const radio.Rows) anyerror!void {
        const c: *Count = @ptrCast(@alignCast(ctx));
        c.passes += 1;
        c.images += spans.len;
    }
};

test "a lane prompt pass encodes each image of a request once" {
    for ([_]usize{ 1, 3 }) |n| {
        var count: Count = .{};
        var v: vision.Vision = .{ .gpa = a, .io = undefined, .cfg = undefined, .tower = undefined, .rows = .{ .d = undefined, .buf = undefined, .hidden = 1, .limit = 64 }, .marks = undefined, .limits = undefined, .mark_ids = undefined, .encoder = .{ .ctx = &count, .encode = Count.encode } };
        var e: Engine = .{ .gpa = a, .io = undefined, .ctx = undefined, .stream = undefined, .k = undefined, .w = undefined, .b = undefined, .c = undefined, .max_len = 4096, .pinned = undefined, .history = undefined };
        // the own sequence (no device memory), too short for the prompt: the pass stops before its first chunk
        e.serial = .{ .d = undefined, .handle = null };
        e.own = .{ .arena = null, .ptr = @splat(0), .cap = 1 };
        e.bound = &e.own;
        e.vision = &v;
        var b: Cuda = .{ .gpa = a, .e = &e, .head = null, .pinned = undefined, .rows_out = undefined, .rows_dev = 0, .prompt_out = undefined, .prompt_dev = 0 };
        defer b.lanes.deinit(a);
        var images: [3]lanes.stream.Image = undefined;
        for (images[0..n], 0..) |*img, i| img.* = .{ .pixels = &.{}, .gh = 4, .gw = 4, .tokens = 4, .at = @intCast(1 + i * 6) };
        const prompt: [20]u32 = @splat(1);
        var s = try lanes.Stream.init(a, .{ .id = "images", .prompt = prompt[0 .. 6 * n + 2], .max_new = 4, .images = images[0..n] });
        defer s.deinit(a);
        for (1..3) |request| {
            try std.testing.expectError(error.ContextFull, b.backend().prefill(&s));
            try std.testing.expectEqual(request, count.passes);
            try std.testing.expectEqual(request * n, count.images);
        }
    }
}
