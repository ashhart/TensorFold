//! Schedule and modulation geometry controls share the production host helpers.
const std = @import("std");
const s = @import("schedule.zig");
const m = @import("modulation_shape.zig");
const st = @import("safetensors");
const t = std.testing;

fn entry(shape: []const usize) st.Entry {
    var e = st.Entry{ .dtype = .bf16, .rank = @intCast(shape.len), .shape = @splat(1), .begin = 0, .end = 0 };
    @memcpy(e.shape[0..shape.len], shape);
    return e;
}

test "H3 schedule geometry rejects empty nodes counts and invalid points" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectError(error.H3BadSchedule, s.spread(a, &.{}, 1));
    try t.expectError(error.H3BadDimension, s.spread(a, &.{0.5}, 0));
    for ([_]usize{ 0, 1 }) |n| try t.expectError(error.H3BadSchedule, s.Schedule.fromPoints(a, n, 3));
    try t.expectError(error.H3RowCountOverflow, s.pointsForSteps(std.math.maxInt(i32)));
    try t.expectEqual(@as(usize, 2), try s.pointsForSteps(1));
    _ = try s.Schedule.fromPoints(a, 2, 3);
    try t.expectError(error.H3BadSchedule, s.Schedule.fromNodes(a, &.{}, 3));
    for ([_]f64{ -1, 0, 1.1, std.math.nan(f64), std.math.inf(f64) }) |n|
        try t.expectError(error.H3BadSchedule, s.Schedule.fromNodes(a, &.{n}, 3));
    for ([_]f64{ 0, -1, std.math.inf(f64), std.math.nan(f64) }) |shift|
        try t.expectError(error.H3BadSchedule, s.Schedule.fromNodes(a, &.{1}, shift));
    try t.expectError(error.BadContract, s.contract(a, "[]", null, null));
    try t.expectError(error.BadContract, s.contract(a, "{\"dmd_denoising_steps\":4}", null, null));
}

test "H3 schedule plan refuses mismatched schedules held rows sums and invalid timesteps" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = s.Rows{ .text = 1, .audio = 1, .video = 1 };
    try t.expectError(error.H3BadSchedule, s.plan(a, r, &.{}, &.{}));
    try t.expectError(error.H3BadSchedule, s.plan(a, r, &.{0.5}, &.{}));
    var held = r;
    held.held_audio = 2;
    try t.expectError(error.H3HeldAudioExceedsTotal, s.plan(a, held, &.{0.5}, &.{0.5}));
    var huge = r;
    huge.text = std.math.maxInt(usize);
    try t.expectError(error.H3RowCountOverflow, s.plan(a, huge, &.{0.5}, &.{0.5}));
    for ([_]f32{ -1, 1.1, std.math.nan(f32), std.math.inf(f32) }) |v|
        try t.expectError(error.H3BadSchedule, s.plan(a, r, &.{v}, &.{0.5}));
    _ = try s.plan(a, r, &.{ 0, 1 }, &.{ 1, 0 });
}

test "H3 external schedule slices checked before Euler indexing" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var v = try s.Schedule.fromPoints(a, 2, 3);
    try s.validate(v);
    v.sigmas = v.sigmas[0..1];
    try t.expectError(error.H3BadSchedule, s.validate(v));
    v = try s.Schedule.fromPoints(a, 2, 3);
    v.sigmas[0] = 0;
    try t.expectError(error.H3BadSchedule, s.validate(v));
    v.sigmas[0] = 1;
    v.timesteps[0] = std.math.nan(f32);
    try t.expectError(error.H3BadSchedule, s.validate(v));
}

test "H3 embedding shape refuses zero odd rank and mismatched widths" {
    _ = try m.embedding(entry(&.{ 4, 2 }), entry(&.{4}), entry(&.{ 8, 4 }), entry(&.{8}), 1);
    try t.expectError(error.H3BadEmbeddingShape, m.embedding(entry(&.{4}), entry(&.{4}), entry(&.{ 8, 4 }), entry(&.{8}), 1));
    try t.expectError(error.H3BadDimension, m.embedding(entry(&.{ 4, 0 }), entry(&.{4}), entry(&.{ 8, 4 }), entry(&.{8}), 1));
    try t.expectError(error.H3BadEmbeddingShape, m.embedding(entry(&.{ 4, 3 }), entry(&.{4}), entry(&.{ 8, 4 }), entry(&.{8}), 1));
    try t.expectError(error.H3BadEmbeddingShape, m.embedding(entry(&.{ 4, 2 }), entry(&.{4}), entry(&.{ 8, 3 }), entry(&.{8}), 1));
    try t.expectEqual(@as(usize, 8), try m.linear(4, 8, 4, 2, 2));
    try t.expectError(error.H3BadEmbeddingShape, m.linear(3, 8, 4, 2, 2));
    try t.expectError(error.H3BadEmbeddingShape, m.linear(4, 7, 4, 2, 2));
}

test "H3 reschedule tables and projection products checked before buffers" {
    try t.expectEqual(@as(usize, 6), try m.tables(2, 256));
    try t.expectEqual(@as(usize, 4608), try m.projection(2, 256, 3, 6, 256));
    try t.expectError(error.H3BadDimension, m.tables(std.math.maxInt(i32), 256));
    try t.expectError(error.H3BufferSizeOverflow, m.tables(1, std.math.maxInt(usize)));
    try t.expectError(error.H3BadDimension, m.projection(1, 1, 3, 6, std.math.maxInt(i32)));
    try t.expectError(error.H3BadDimension, m.projection(0, 256, 3, 6, 256));
}

test "H3 held frame at timestep one has a keyframe table entry" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const p = try s.plan(arena.allocator(), .{ .text = 1, .audio = 1, .video = 1, .hold = true }, &.{1}, &.{0});
    try t.expectEqualSlices(f32, &.{ 0, s.keyframe_noise }, p.table);
    try t.expectEqualSlices(i32, &.{ 1, 0, 1 }, p.index);
}
