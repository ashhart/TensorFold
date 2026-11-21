//! The 27B's sampled rows drawn on threads equal the same rows drawn in order, with the 2B's real draw.
const std = @import("std");
const lanes = @import("lanes");
const draws = @import("native/qwen27_draws.zig");
const draw = @import("families/qwen3_5/draw.zig");
const vocab = @import("families/qwen3_5/config.zig").vocab;
const Rows = draws.Rows(draw.draw, lanes.Sampling);
const rows = 16;

test {
    _ = draws;
}

/// Logits that differ by row, with ties and repeated values, so each row's candidates and token are its own.
fn fill(words: []u16) void {
    for (words, 0..) |*w, i| {
        const r = i / vocab;
        const x: f32 = @as(f32, @floatFromInt((i * (37 + 2 * r) + r * 11) % 509)) / 16.0 - 12.0;
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
    }
}

fn expectSame(gpa: std.mem.Allocator, words: []const u16, sampling: ?lanes.Sampling, masks: []const u32) !void {
    var positions: [rows]u64 = undefined;
    for (&positions, 0..) |*p, i| p.* = 4096 + 7 * i;
    var walked: [rows]u32 = undefined;
    var threaded: [rows]u32 = undefined;
    const width = (vocab + 31) / 32;
    const walk = Rows{ .gpa = gpa, .words = words.ptr, .vocab = vocab, .width = width, .sampling = sampling, .positions = &positions, .out = &walked, .masks = masks };
    for (0..rows) |i| try walk.row(i);
    for (0..rows) |i| try std.testing.expectEqual(try draw.draw(gpa, words[i * vocab ..][0..vocab], sampling, positions[i], if (masks.len == 0) &.{} else masks[i * width ..][0..width]), walked[i]);
    for (0..8) |_| {
        @memset(&threaded, std.math.maxInt(u32));
        var on = walk;
        on.out = &threaded;
        try draws.each(on, rows);
        try std.testing.expectEqualSlices(u32, &walked, &threaded);
    }
}

test "qwen27 draws: a window's real keyed draws on threads equal the same rows drawn in order" {
    const gpa = std.testing.allocator;
    const words = try gpa.alloc(u16, rows * vocab);
    defer gpa.free(words);
    fill(words);
    // the served bounded top-k, the general path (top_k 0 allocates), and greedy rows
    try expectSame(gpa, words, .{ .seed = 1234, .top_k = 20, .temperature = 0.6, .top_p = 0.95 }, &.{});
    try expectSame(gpa, words, .{ .seed = 99, .top_k = 0, .temperature = 1.0, .top_p = 0.9, .min_p = 0.05 }, &.{});
    try expectSame(gpa, words, null, &.{});
    // each row its own grammar mask
    const width = (vocab + 31) / 32;
    const masks = try gpa.alloc(u32, rows * width);
    defer gpa.free(masks);
    for (masks, 0..) |*m, i| m.* = @truncate(0x9E3779B97F4A7C15 *% (i + 1));
    try expectSame(gpa, words, .{ .seed = 5, .top_k = 20, .temperature = 0.8 }, masks);
}

test "qwen27 draws: a nonfinite row fails the threaded window as it fails the walk" {
    const gpa = std.testing.allocator;
    const words = try gpa.alloc(u16, rows * vocab);
    defer gpa.free(words);
    fill(words);
    words[5 * vocab + 1000] = 0x7fc0;
    var positions: [rows]u64 = @splat(0);
    var out: [rows]u32 = undefined;
    const r = Rows{ .gpa = gpa, .words = words.ptr, .vocab = vocab, .width = (vocab + 31) / 32, .sampling = .{ .seed = 3, .top_k = 20, .temperature = 0.6 }, .positions = &positions, .out = &out, .masks = &.{} };
    try std.testing.expectError(error.NonfiniteQwenLogits, r.row(5));
    try std.testing.expectError(error.NonfiniteQwenLogits, draws.each(r, rows));
}
