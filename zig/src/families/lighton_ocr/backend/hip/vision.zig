//! LightOnOCR image bounds and projected spans on HIP; each image remains an independent tower input.
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const cfg = @import("../../config.zig");
const image = @import("../../image.zig");
const CancelCheck = @import("lanes").stream.CancelCheck;
const cancels = @import("hip_serve").cancel;

pub const Limits = struct {
    image_tokens: u32,
    max_images: u32,

    pub fn check(l: Limits, context: usize) !void {
        if (l.image_tokens == 0 or l.image_tokens > context or l.max_images == 0 or l.max_images > 64)
            return error.BadImageLimits;
    }
};

/// One expanded image interval and the first BF16 feature row in its stream's reserved buffer.
pub const Span = struct { at: u32, tokens: u32, features: u64 };
pub const Rows = struct { first: u32, rows: u32, features: u64 };

/// Check every interval before encoding; the tower keeps odd patch edges while the projector drops unmatched edges.
pub fn validate(c: cfg.Config, limits: Limits, images: []const lanes.stream.Image, prompt_len: usize) !usize {
    if (images.len > limits.max_images) return error.TooManyImages;
    var count: usize = 0;
    var end: usize = 0;
    for (images) |im| {
        if (im.gh < 2 or im.gw < 2 or im.gh > 110 or im.gw > 110 or
            im.pixels.len != @as(usize, im.gh) * im.gw * c.patchWidth() or im.tokens != image.tokens(im.gh, im.gw))
            return error.BadImagePatches;
        if (im.at < end or im.at > prompt_len or im.tokens > prompt_len - im.at) return error.BadImageSpan;
        end = @as(usize, im.at) + im.tokens;
        count += im.tokens;
        if (count > limits.image_tokens) return error.ImageTokensExceeded;
    }
    return count;
}

/// Clip an image to one prompt chunk, naming rows in the shared flattened forward and advancing its feature pointer.
pub fn intersect(span: Span, first: u32, end: u32, flat_first: u32, hidden: u32) ?Rows {
    const lo = @max(span.at, first);
    const hi = @min(@as(u64, span.at) + span.tokens, end);
    if (lo >= hi) return null;
    return .{
        .first = flat_first + lo - first,
        .rows = @intCast(hi - lo),
        .features = span.features + @as(u64, lo - span.at) * hidden * 2,
    };
}

/// The graph is supplied by the HIP family; this module shares no runtime or operator with CUDA.
pub fn For(comptime Tower: type) type {
    return struct {
        io: std.Io,
        c: cfg.Config,
        limits: Limits,
        tower: *Tower,

        const Self = @This();
        pub const part_type = "image";
        pub const explain = image.explain;

        pub fn prepare(v: *const Self, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !image.Prepared {
            return image.prepare(v.io, a, bytes, v.c, @min(max_tokens, v.limits.image_tokens));
        }

        /// Write one request's images into its persistent feature reservation; no image can attend to another image.
        pub fn encode(v: *Self, images: []const lanes.stream.Image, prompt_len: usize, out: u64, spans: []Span) !void {
            return v.encodeCancelable(images, prompt_len, out, spans, null);
        }

        pub fn encodeCancelable(v: *Self, images: []const lanes.stream.Image, prompt_len: usize, out: u64, spans: []Span, cancel: ?CancelCheck) !void {
            try cancels.poll(cancel);
            _ = try validate(v.c, v.limits, images, prompt_len);
            if (spans.len != images.len) return error.BadImageSpan;
            var at: usize = 0;
            for (images, spans) |im, *span| {
                try cancels.poll(cancel);
                const dst = out + at * v.c.hidden * 2;
                try v.tower.encodeCancelable(im.pixels, im.gh, im.gw, dst, cancel);
                span.* = .{ .at = im.at, .tokens = im.tokens, .features = dst };
                at += im.tokens;
            }
            try cancels.poll(cancel);
        }
    };
}

test "LightOnOCR HIP image reservations reject invalid limits" {
    try (Limits{ .image_tokens = 4096, .max_images = 4 }).check(8192);
    try std.testing.expectError(error.BadImageLimits, (Limits{ .image_tokens = 0, .max_images = 4 }).check(8192));
    try std.testing.expectError(error.BadImageLimits, (Limits{ .image_tokens = 8193, .max_images = 4 }).check(8192));
    try std.testing.expectError(error.BadImageLimits, (Limits{ .image_tokens = 4096, .max_images = 65 }).check(8192));
}

test "LightOnOCR HIP image intersections preserve offsets at chunk seams" {
    const span: Span = .{ .at = 100, .tokens = 300, .features = 0x10000 };
    try std.testing.expectEqual(Rows{ .first = 50, .rows = 28, .features = 0x10000 }, intersect(span, 64, 128, 14, 1024).?);
    try std.testing.expectEqual(Rows{ .first = 7, .rows = 128, .features = 0x10000 + 28 * 2048 }, intersect(span, 128, 256, 7, 1024).?);
    try std.testing.expectEqual(Rows{ .first = 0, .rows = 16, .features = 0x10000 + 284 * 2048 }, intersect(span, 384, 512, 0, 1024).?);
    try std.testing.expect(intersect(span, 0, 100, 0, 1024) == null);
    try std.testing.expect(intersect(span, 400, 500, 0, 1024) == null);
}

test "LightOnOCR HIP image cancellation stops mid-layer sequence and the next request can reuse the reservation" {
    const MockTower = struct {
        queued: bool = false,
        completed_layers: usize = 0,
        started_images: usize = 0,
        finished_images: usize = 0,
        stop_after: ?usize = 3,
        polled_before_drain: bool = false,

        pub fn synchronize(m: *@This()) !void {
            if (m.queued) m.completed_layers += 1;
            m.queued = false;
        }

        fn check(ptr: *anyopaque) bool {
            const m: *@This() = @ptrCast(@alignCast(ptr));
            m.polled_before_drain = m.polled_before_drain or m.queued;
            return if (m.stop_after) |at| m.completed_layers >= at else false;
        }

        pub fn encodeCancelable(m: *@This(), _: []const f32, _: u32, _: u32, _: u64, cancel: ?CancelCheck) !void {
            try cancels.poll(cancel);
            errdefer m.synchronize() catch unreachable;
            m.started_images += 1;
            for (0..24) |_| {
                try cancels.drain(m, cancel);
                m.queued = true;
            }
            try cancels.drain(m, cancel);
            m.finished_images += 1;
        }
    };
    var mock: MockTower = .{};
    const c = try cfg.Config.parse(std.testing.allocator, cfg.sample);
    const pixels: [4 * 588]f32 = @splat(0);
    const images = [_]lanes.stream.Image{
        .{ .pixels = &pixels, .gh = 2, .gw = 2, .at = 0, .tokens = 1 },
        .{ .pixels = &pixels, .gh = 2, .gw = 2, .at = 1, .tokens = 1 },
    };
    var spans: [2]Span = @splat(.{ .at = 0, .tokens = 0, .features = 0 });
    var v: For(MockTower) = .{ .io = undefined, .c = c, .limits = .{ .image_tokens = 2, .max_images = 2 }, .tower = &mock };
    const cancel: CancelCheck = .{ .ptr = &mock, .check = MockTower.check };
    try std.testing.expectError(error.Cancelled, v.encodeCancelable(&images, 2, 0x10000, &spans, cancel));
    try std.testing.expectEqual(@as(usize, 3), mock.completed_layers);
    try std.testing.expectEqual(@as(usize, 1), mock.started_images);
    try std.testing.expectEqual(@as(usize, 0), mock.finished_images);
    try std.testing.expectEqual(@as(u32, 0), spans[0].tokens);
    try std.testing.expect(!mock.queued and !mock.polled_before_drain);
    mock.stop_after = null;
    try v.encodeCancelable(&images, 2, 0x10000, &spans, cancel);
    try std.testing.expectEqual(@as(usize, 51), mock.completed_layers);
    try std.testing.expectEqual(@as(usize, 2), mock.finished_images);
    try std.testing.expect(!mock.queued and !mock.polled_before_drain);
    try std.testing.expectEqual(Span{ .at = 0, .tokens = 1, .features = 0x10000 }, spans[0]);
    try std.testing.expectEqual(Span{ .at = 1, .tokens = 1, .features = 0x10000 + 2048 }, spans[1]);
    mock.stop_after = 0;
    try std.testing.expectError(error.Cancelled, v.encodeCancelable(&images, 2, 0x10000, &spans, cancel));
    try std.testing.expectEqual(@as(usize, 3), mock.started_images);
}
