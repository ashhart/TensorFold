//! A video's frames to GLM5-Next vision patches as transformers 5.19's Glm5NextVideoProcessor makes them from PyAV's.
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const steps = @import("video_core.zig").For(picture);

pub const Error = steps.Error;
pub const sample_fps = steps.sample_fps;
pub const Source = steps.Source;
pub const Sink = steps.Sink;
pub const Plan = steps.Plan;
pub const sample = steps.sample;
pub const canvas = steps.canvas;
pub const plan = steps.plan;
pub const Pair = steps.Pair;
pub const Prepared = steps.Prepared;
pub const prepare = steps.prepare;
pub const patchify = steps.patchify;
pub const seconds = steps.seconds;

test "sampling as Glm5NextVideoProcessor.sample_frames, then the loader's distinct frames" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const src = struct {
        fn of(frames: u64, num: u64, den: u64, ctx: *anyopaque) Source {
            return .{
                .frames = frames,
                .rate_num = num,
                .rate_den = den,
                .width = 64,
                .height = 48,
                .ctx = ctx,
                .decode = undefined,
            };
        }
    }.of;
    // 80 frames at 25: 3.2 s, 6 frames at 0, 0.5, 1, 1.5, 2, 2.5 s (the reference run's indices)
    try std.testing.expectEqualSlices(u32, &.{ 0, 13, 25, 38, 50, 63 }, try sample(a, src(80, 25, 1, &dummy), 256));
    // 10 frames at 25 (0.4 s): nothing to sample
    try std.testing.expectError(error.VideoTooShort, sample(a, src(10, 25, 1, &dummy), 256));
    // a long video: 256 frames over the whole length
    const long = try sample(a, src(1500, 10, 1, &dummy), 256);
    try std.testing.expectEqual(@as(usize, 256), long.len);
    try std.testing.expectEqual(@as(u32, 1499), long[255]);
    try std.testing.expectError(error.BadVideo, sample(a, src(0, 25, 1, &dummy), 256));
}

test "seconds as Python's f\"{t:.1f}\"" {
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("0.2 seconds", try seconds(&buf, 0.25));
    try std.testing.expectEqualStrings("0.8 seconds", try seconds(&buf, 0.75));
    try std.testing.expectEqualStrings("0.5 seconds", try seconds(&buf, 0.52));
    try std.testing.expectEqualStrings("12.0 seconds", try seconds(&buf, 11.96));
}
