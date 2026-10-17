//! Family video boundaries through the actual preparer, with no decoder library or GPU.
const std = @import("std");
const video = @import("video.zig");
const Config = @import("config.zig").Config;
pub const cfg: Config = .{ .hidden = 1280, .layers = 32, .heads = 16, .ffn = 5120, .eps = 1e-6, .patch = 16, .prefix = 10, .side = 128, .projector_hidden = 20480, .text_hidden = 4096, .final_norm = false, .mean = .{ 0.5, 0.5, 0.5 }, .std = .{ 0.25, 0.25, 0.25 }, .min_patches = 1024, .max_patches = 13312, .max_model_len = 16384, .image_token = 18, .video_temporal = 2 };
pub const Fake = struct {
    calls: usize = 0,
    short: bool = false,
    width: u32 = 64,
    height: u32 = 48,
    pub fn decode(ctx: *anyopaque, indices: []const u32, sink: video.Sink) anyerror!void {
        const f: *@This() = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        var pixels: [64 * 721 * 3]u8 = @splat(1);
        for (indices, 0..) |_, i| try sink.take(sink.ctx, i, if (f.short) pixels[0..0] else pixels[0 .. f.width * f.height * 3], f.width, f.height, f.width * 3);
    }
    pub fn source(f: *@This()) video.Source {
        return .{ .frames = 30, .rate_num = 30, .rate_den = 1, .width = f.width, .height = f.height, .ctx = f, .decode = decode };
    }
};
test "zero frame rate refuses before sampling or decoding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fake = .{};
    var src = f.source();
    src.rate_num = 0;
    try std.testing.expectError(error.BadVideo, video.prepare(arena.allocator(), &cfg, src, 512, 256));
    try std.testing.expectEqual(@as(usize, 0), f.calls);
}
test "frame indices beyond signed decoder range refuse before decoding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fake = .{};
    var src = f.source();
    src.frames = @as(u64, std.math.maxInt(i32)) + 1;
    src.rate_num = std.math.maxInt(i32);
    try std.testing.expectError(error.VideoTooLarge, video.prepare(arena.allocator(), &cfg, src, 512, 256));
    try std.testing.expectEqual(@as(usize, 0), f.calls);
}
test "oversized source dimensions refuse before decoding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fake = .{};
    var src = f.source();
    src.width = 8193;
    src.height = 8193;
    try std.testing.expectError(error.VideoTooLarge, video.prepare(arena.allocator(), &cfg, src, 512, 256));
    try std.testing.expectEqual(@as(usize, 0), f.calls);
}
test "short decoded RGB payload refuses before pixel reads" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fake = .{ .short = true };
    try std.testing.expectError(error.BadVideo, video.prepare(arena.allocator(), &cfg, f.source(), 512, 256));
}
test "zero configured video grid refuses before decoding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f: Fake = .{};
    var c = cfg;
    c.video_target_patches = 0;
    try std.testing.expectError(error.BadVideo, video.prepare(arena.allocator(), &c, f.source(), 512, 256));
    try std.testing.expectEqual(@as(usize, 0), f.calls);
}
test "tower geometry rejects huge products before CUDA access" {
    var tower: @import("tower.zig").Tower = undefined;
    tower.c = cfg;
    try std.testing.expectError(error.ImageShape, tower.features(undefined, .{ .pixels = &.{}, .gh = 0xffffffff, .gw = 0xffffffff, .tokens = 1, .at = 0 }));
}
test "prompt spans refuse overflow before any device use" {
    const rows = @import("rows.zig");
    const Image = @import("lanes").stream.Image;
    const bad = [_]Image{.{ .pixels = &.{}, .gh = 2, .gw = 2, .tokens = 2, .at = 0xffffffff }};
    try std.testing.expectError(error.ImageSpans, rows.plan(std.testing.allocator, &bad, 0));
}
test "row allocation products refuse before touching an undefined driver" {
    try std.testing.expectError(error.ImageRows, @import("rows.zig").Rows.init(undefined, 2, std.math.maxInt(usize)));
}

fn drop(a: std.mem.Allocator, p: video.Prepared) void {
    for (p.groups, p.labels) |g, label| {
        a.free(g.pixels);
        a.free(label);
    }
    a.free(p.groups);
    a.free(p.times);
    a.free(p.labels);
}
fn allocationSweep(a: std.mem.Allocator) !void {
    var f: Fake = .{};
    var c = cfg;
    c.patch = 1;
    c.video_target_patches = 4;
    const p = try video.prepare(a, &c, f.source(), 512, 256);
    defer drop(a, p);
    try std.testing.expectEqual(@as(usize, 1), p.groups.len);
}
test "temporary and returned video owners unwind every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationSweep, .{});
}
test "source frame dimension duration and sampling cap endpoint controls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = cfg;
    c.patch = 1;
    c.video_target_patches = 4;
    var f: Fake = .{};
    var src = f.source();
    try std.testing.expectError(error.BadVideo, video.prepare(a, &c, src, 512, 0));
    try std.testing.expectError(error.VideoTooLarge, video.prepare(a, &c, src, 512, 257));
    src.rate_den = 0;
    try std.testing.expectError(error.BadVideo, video.prepare(a, &c, src, 512, 256));
    src = f.source();
    src.frames = 0;
    try std.testing.expectError(error.BadVideo, video.prepare(a, &c, src, 512, 256));
    src = f.source();
    src.rate_num = 1;
    src.frames = 3601;
    try std.testing.expectError(error.VideoTooLarge, video.prepare(a, &c, src, 512, 256));
    src.frames = 3600;
    const small = try video.prepare(a, &c, src, 512, 256);
    try std.testing.expectEqual(@as(usize, 128), small.groups.len);
    f.height = 721;
    src = f.source();
    src.rate_num = 1;
    src.frames = 3600;
    const large = try video.prepare(a, &c, src, 512, 256);
    try std.testing.expectEqual(@as(usize, 64), large.groups.len);
    var huge = c;
    huge.patch = std.math.maxInt(usize);
    try std.testing.expectError(error.VideoTooLarge, video.prepare(a, &huge, src, 512, 256));
    try std.testing.expectError(error.VideoTooLarge, @import("image.zig").gridVideo(&c, std.math.maxInt(u64), std.math.maxInt(u64)));
}
