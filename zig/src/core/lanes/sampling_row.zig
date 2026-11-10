//! The host draw over a row of bf16 logits: greedy, the keyed fp64 sampler over the top_k, or a grammar mask's tokens.
const std = @import("std");
const sampling = @import("sampling.zig");
const grammar = @import("grammar.zig");
const topk = @import("sampling_row_topk.zig");
const Sampling = sampling.Sampling;

/// A bf16 word as the f64 the keyed sampler reads.
pub fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// The keyed fp64 draw over the top_k (heap-free for 1..128: sampling_row_topk); a `mask` keeps its allowed ones.
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?Sampling, position: u64, mask: []const u32) !u32 {
    if (settings) |s| if (try topk.draw(logits, s, position, mask)) |token| return token;
    var best: ?usize = null;
    var allowed: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteLogits;
        if (mask.len > 0 and !grammar.allows(mask, id)) continue;
        allowed += 1;
        if (best == null or x > value(logits[best.?])) best = id;
    }
    if (best == null) return error.NoAllowedToken;
    const s = settings orelse return @intCast(best.?);
    if (s.temperature <= 0) return @intCast(best.?);
    const k = if (s.top_k == 0) allowed else @min(allowed, s.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (mask.len > 0 and !grammar.allows(mask, id)) continue;
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
    return @intCast(try sampling.choose(gpa, values, ids, position, s));
}

const test_vocab = 4099;

fn testRow(words: []u16, values: []f64, ids: []u64) void {
    for (words, values, ids, 0..) |*w, *v, *id, i| {
        const x: f32 = @as(f32, @floatFromInt((i * 37) % 257)) / 8.0 - 16.0;
        w.* = @intCast(@as(u32, @bitCast(x)) >> 16);
        v.* = value(w.*);
        id.* = i;
    }
}

test "candidate selection preserves full-vocabulary keyed draws and tie order" {
    const gpa = std.testing.allocator;
    var words: [test_vocab]u16 = undefined;
    var values: [test_vocab]f64 = undefined;
    var ids: [test_vocab]u64 = undefined;
    testRow(&words, &values, &ids);
    for ([_]u32{ 0, 1, 20, 257, test_vocab }) |k| {
        for ([_]u64{ 0, 128, 65536 }) |position| {
            const s: Sampling = .{ .seed = 1234, .top_k = k, .temperature = 0.7, .top_p = 0.95, .min_p = 0.1 };
            const want = try sampling.choose(gpa, &values, &ids, position, s);
            try std.testing.expectEqual(want, try draw(gpa, &words, s, position, &.{}));
        }
    }
    const greedy = try draw(gpa, &words, .{ .seed = 7, .temperature = 0 }, 99, &.{});
    try std.testing.expectEqual(try draw(gpa, &words, null, 0, &.{}), greedy);
    words[0] = 0x7fc0;
    try std.testing.expectError(error.NonfiniteLogits, draw(gpa, &words, null, 0, &.{}));
}

test "a masked draw is the full-vocabulary draw with the other tokens at -inf" {
    const gpa = std.testing.allocator;
    var words: [test_vocab]u16 = undefined;
    var values: [test_vocab]f64 = undefined;
    var ids: [test_vocab]u64 = undefined;
    testRow(&words, &values, &ids);
    var mask: [(test_vocab + 31) / 32]u32 = @splat(0);
    for (&values, 0..) |*v, i| {
        const on = i % 7 == 3 or i % 1000 == 1;
        if (on) mask[i / 32] |= @as(u32, 1) << @intCast(i % 32) else v.* = -std.math.inf(f64);
    }
    for ([_]u32{ 0, 1, 20, 257, test_vocab }) |k| {
        for ([_]u64{ 0, 128, 65536 }) |position| {
            const s: Sampling = .{ .seed = 99, .top_k = k, .temperature = 0.8, .top_p = 0.9, .min_p = 0.05 };
            const want = try sampling.choose(gpa, &values, &ids, position, s);
            try std.testing.expectEqual(want, try draw(gpa, &words, s, position, &mask));
        }
    }
    const greedy = try sampling.choose(gpa, &values, &ids, 0, .{ .seed = 1, .temperature = 0 });
    try std.testing.expectEqual(greedy, try draw(gpa, &words, null, 0, &mask));
    @memset(&mask, 0);
    try std.testing.expectError(error.NoAllowedToken, draw(gpa, &words, null, 0, &mask));
}

test "a sampled top_k up to 128 draws without the heap and picks the full-row token" {
    const a = std.testing.allocator;
    var words: [test_vocab]u16 = undefined;
    var values: [test_vocab]f64 = undefined;
    var ids: [test_vocab]u64 = undefined;
    testRow(&words, &values, &ids);
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    for ([_]u32{ 1, 20, 128 }) |k| for ([_]u64{ 0, 191 }) |position| {
        const s: Sampling = .{ .seed = 1234, .temperature = 1, .top_p = 0.95, .top_k = k };
        const want = try sampling.choose(a, &values, &ids, position, s);
        try std.testing.expectEqual(want, try draw(failing.allocator(), &words, s, position, &.{}));
    };
    try std.testing.expectEqual(@as(usize, 0), failing.alloc_index);
}

test {
    _ = topk;
}
