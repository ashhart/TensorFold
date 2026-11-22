//! Production readback decoding and the shared draw predicate, no GPU execution.
const std = @import("std");
const t = std.testing;
const hip = @import("hip");
const head = @import("head.zig");
const draw = @import("hip_serve").draw;

test "HIP tail body: mandatory draw wait precedes byte decoding and draft id mapping" {
    const Mock = struct {
        var words: [4]u32 = undefined;
        var waits: usize = 0;
        fn sync(_: hip.abi.Stream) callconv(.c) hip.abi.Result {
            waits += 1;
            words = .{ 2, 0, @bitCast(@as(f32, 0.875)), @bitCast(@as(f32, 0.5)) };
            return 0;
        }
    };
    var r: hip.Runtime = undefined;
    r.api.hipStreamSynchronize = Mock.sync;
    var out: [2]u32 = undefined;
    var probs: [2]f32 = undefined;
    try head.finishDraws(.{ .r = &r, .handle = null }, .{ .bytes = std.mem.sliceAsBytes(&Mock.words), .rows = 2 }, &.{ 23, 31, 47 }, &out, &probs);
    try t.expectEqual([2]u32{ 47, 23 }, out);
    try t.expectEqual([2]f32{ 0.875, 0.5 }, probs);
    try t.expectEqual(@as(usize, 1), Mock.waits);
}

test "HIP tail body: greedy predicate preserves masks processors and mixed sampled rows" {
    try t.expect(draw.needsArgmax(&.{.{ .sampling = null, .position = 3 }}));
    try t.expect(!draw.needsArgmax(&.{}));
    try t.expect(!draw.needsArgmax(&.{.{ .sampling = .{ .seed = 1234, .temperature = 0.7 }, .position = 3 }}));
    try t.expect(draw.needsArgmax(&.{ .{ .sampling = .{ .seed = 1234, .temperature = 0.7 }, .position = 3 }, .{ .sampling = null, .position = 4 } }));
    try t.expect(!draw.needsArgmax(&.{.{ .sampling = null, .position = 3, .mask = &.{1} }}));
    try t.expect(!draw.needsArgmax(&.{.{ .sampling = null, .position = 3, .forced = 2 }}));
    try t.expect(!draw.needsArgmax(&.{.{ .sampling = null, .position = 3, .repetition = 1.2 }}));
}
