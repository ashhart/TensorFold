//! The host draw over a row of bf16 logits, shared by the Metal and CUDA backends: greedy, keyed sampling or a mask.
const std = @import("std");
const lanes = @import("lanes");
const c = @import("config.zig");

/// A row's allowed-token bits (structured output): one bit a logit.
pub const mask_words = (c.vocab + 31) / 32;

pub fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// The unchanged top-k candidates for the shared fp64 keyed sampler; a `mask` keeps its allowed ones (-inf's draw).
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
    var best: ?usize = null;
    var allowed: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteQwenLogits;
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        allowed += 1;
        if (best == null or x > value(logits[best.?])) best = id;
    }
    if (best == null) return error.NoAllowedQwenToken;
    const sampling = settings orelse return @intCast(best.?);
    if (sampling.temperature <= 0) return @intCast(best.?);
    const k = if (sampling.top_k == 0) allowed else @min(allowed, sampling.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        const x = value(word);
        if (k == allowed) {
            values[filled] = x;
            ids[filled] = id;
            filled += 1;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, sampling));
}

test "candidate selection preserves full-vocabulary keyed draws and tie order" {
    const gpa = std.testing.allocator;
    const words = try gpa.alloc(u16, c.vocab);
    defer gpa.free(words);
    const values = try gpa.alloc(f64, c.vocab);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, c.vocab);
    defer gpa.free(ids);
    for (words, values, ids, 0..) |*w, *v, *id, i| {
        const x: f32 = @as(f32, @floatFromInt((i * 37) % 257)) / 8.0 - 16.0;
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
        v.* = value(w.*);
        id.* = i;
    }
    for ([_]u32{ 0, 1, 20, 257, c.vocab }) |k| {
        for ([_]u64{ 0, 128, 65536 }) |position| {
            const s: lanes.Sampling = .{ .seed = 1234, .top_k = k, .temperature = 0.7, .top_p = 0.95, .min_p = 0.1 };
            const want = try lanes.sampling.choose(gpa, values, ids, position, s);
            try std.testing.expectEqual(want, try draw(gpa, words, s, position, &.{}));
        }
    }
    try std.testing.expectEqual(try draw(gpa, words, null, 0, &.{}), try draw(gpa, words, .{ .seed = 7, .temperature = 0 }, 99, &.{}));
    words[0] = 0x7fc0;
    try std.testing.expectError(error.NonfiniteQwenLogits, draw(gpa, words, null, 0, &.{}));
}

test "a masked draw is the full-vocabulary draw with the other tokens at -inf" {
    const gpa = std.testing.allocator;
    const words = try gpa.alloc(u16, c.vocab);
    defer gpa.free(words);
    const values = try gpa.alloc(f64, c.vocab);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, c.vocab);
    defer gpa.free(ids);
    var mask: [mask_words]u32 = @splat(0);
    for (words, values, ids, 0..) |*w, *v, *id, i| {
        const x: f32 = @as(f32, @floatFromInt((i * 37) % 257)) / 8.0 - 16.0;
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
        const on = i % 7 == 3 or i % 1000 == 1;
        if (on) mask[i / 32] |= @as(u32, 1) << @intCast(i % 32);
        v.* = if (on) value(w.*) else -std.math.inf(f64);
        id.* = i;
    }
    for ([_]u32{ 0, 1, 20, 257, c.vocab }) |k| {
        for ([_]u64{ 0, 128, 65536 }) |position| {
            const s: lanes.Sampling = .{ .seed = 99, .top_k = k, .temperature = 0.8, .top_p = 0.9, .min_p = 0.05 };
            const want = try lanes.sampling.choose(gpa, values, ids, position, s);
            try std.testing.expectEqual(want, try draw(gpa, words, s, position, &mask));
        }
    }
    try std.testing.expectEqual(try lanes.sampling.choose(gpa, values, ids, 0, .{ .seed = 1, .temperature = 0 }), try draw(gpa, words, null, 0, &mask));
    @memset(&mask, 0);
    try std.testing.expectError(error.NoAllowedQwenToken, draw(gpa, words, null, 0, &mask));
}
