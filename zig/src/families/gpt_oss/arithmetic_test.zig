//! Analytic host cases for sinks, window edges and four-expert selection; no full-model reference claim.
const std = @import("std");
const attention = @import("attention.zig");
const route = @import("route.zig");
const t = std.testing;

test "gpt-oss sink has probability mass but contributes no value" {
    var out: [1]f64 = undefined;
    try attention.combine(&.{0}, &.{8}, 1, 0, &out);
    try t.expectEqual(@as(f64, 4), out[0]);
    try attention.combine(&.{ 0, 0 }, &.{ 3, 9 }, 1, 0, &out);
    try t.expectEqual(@as(f64, 4), out[0]);
    try attention.combine(&.{1000}, &.{8}, 1, 1000, &out);
    try t.expectEqual(@as(f64, 4), out[0]);
    try attention.combine(&.{}, &.{}, 1, 0, &out);
    try t.expectEqual(@as(f64, 0), out[0]);
    try t.expectError(error.NonfiniteAttention, attention.combine(&.{std.math.nan(f64)}, &.{1}, 1, 0, &out));
}

test "gpt-oss FP64 attention reference scales QK and includes the sink without reading future keys" {
    const c = try @import("config_test.zig").load();
    var q: [64]f64 = @splat(0);
    q[0] = 8;
    var keys: [128]f64 = @splat(std.math.nan(f64));
    @memset(keys[0..64], 0);
    keys[0] = 1;
    var values: [128]f64 = @splat(std.math.nan(f64));
    @memset(values[0..64], 4);
    var out: [64]f64 = undefined;
    try attention.reference(t.allocator, &c, 0, 0, &q, &keys, &values, 0, &out);
    for (out) |v| try t.expectApproxEqAbs(@as(f64, 2.9242343145200196), v, 1e-14);
}

test "gpt-oss full and sliding attention use layer_types with explicit inclusive window bounds" {
    const c = try @import("config_test.zig").load();
    const slide = try attention.bounds(&c, 0, 128);
    try t.expectEqual(@as(usize, 1), slide.begin);
    try t.expectEqual(@as(usize, 129), slide.end);
    const full = try attention.bounds(&c, 1, 128);
    try t.expectEqual(@as(usize, 0), full.begin);
    try t.expectEqual(@as(usize, 129), full.end);
    try t.expectEqual(@as(usize, 0), (try attention.bounds(&c, 0, 127)).begin);
    try t.expectError(error.BadPosition, attention.bounds(&c, 0, c.context));
}

test "gpt-oss masked keys are omitted before logits and values are read" {
    const c = try @import("config_test.zig").load();
    var logits: [130]f64 = @splat(0);
    var values: [130]f64 = @splat(4);
    logits[0] = std.math.nan(f64);
    values[0] = std.math.nan(f64);
    logits[129] = std.math.nan(f64);
    values[129] = std.math.inf(f64);
    var out: [1]f64 = undefined;
    try attention.row(&c, 0, 128, &logits, &values, 1, -std.math.inf(f64), &out);
    try t.expectEqual(@as(f64, 4), out[0]);
    try t.expectError(error.NonfiniteAttention, attention.row(&c, 1, 128, &logits, &values, 1, 0, &out));
}

test "gpt-oss router selects experts_per_token raw logits with deterministic low-ID ties" {
    const c = try @import("config_test.zig").load();
    var logits: [32]f64 = @splat(-100);
    logits[7] = 9;
    logits[2] = 9;
    logits[20] = 8;
    logits[31] = 7;
    const selected = try route.select(&c, &logits);
    try t.expectEqual(@as(usize, 4), selected.len);
    try t.expectEqualSlices(u32, &.{ 2, 7, 20, 31 }, selected.ids[0..selected.len]);
    try t.expectEqualSlices(f64, &.{ 9, 9, 8, 7 }, selected.logits[0..selected.len]);
    var weights: [4]f64 = undefined;
    try route.softmaxSelected(&.{ 0, 0, 0, 0 }, &weights);
    try t.expectEqualSlices(f64, &.{ 0.25, 0.25, 0.25, 0.25 }, &weights);
    logits[0] = std.math.nan(f64);
    try t.expectError(error.NonfiniteRouter, route.select(&c, &logits));
    try t.expectError(error.BadRouterShape, route.select(&c, logits[0..31]));
}
