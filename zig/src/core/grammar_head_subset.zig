//! A bounded list of legal vocabulary IDs from a staged grammar mask.
const std = @import("std");

pub const default_limit = 512;

pub fn capacity(enabled: bool, vocab: usize, threshold: usize) usize {
    return if (enabled) @min(threshold, vocab) else 0;
}

test "head subset switch and threshold" {
    try std.testing.expectEqual(0, capacity(false, 131072, default_limit));
    try std.testing.expectEqual(default_limit, capacity(true, 131072, default_limit));
    try std.testing.expectEqual(0, capacity(true, 131072, 0));
    try std.testing.expectEqual(31, capacity(true, 31, default_limit));
}

/// Returns zero for a dead or over-limit row. The destination capacity is the threshold.
pub fn collect(bits: []const u32, vocab: usize, ids: []u32) usize {
    var count: usize = 0;
    for (bits, 0..) |word, i| {
        const base = i * 32;
        if (base >= vocab) break;
        const tail = @min(vocab - base, 32);
        const mask = @as(u32, std.math.maxInt(u32)) >> @as(u5, @intCast(32 - tail));
        var legal = word & mask;
        while (legal != 0) : (legal &= legal - 1) {
            if (count == ids.len) return 0;
            ids[count] = @intCast(base + @ctz(legal));
            count += 1;
        }
    }
    return if (count == vocab) 0 else count;
}

test "legal head rows produce the same logits and mask bytes" {
    const vocab = 1031;
    const neg_inf = 0xff80;
    const x = [_]f32{ 0.25, -0.5, 0.75, 1 };
    const Ref = struct {
        fn dot(weights: [4]f32) u16 {
            var sum: f32 = 0;
            for (weights, x) |w, v| sum = @mulAdd(f32, w, v, sum);
            return @truncate(@as(u32, @bitCast(sum)) >> 16);
        }
    };
    var weights: [vocab][4]f32 = undefined;
    var full: [vocab]u16 = undefined;
    for (&weights, &full, 0..) |*w, *out, id| {
        w.* = .{ @floatFromInt(id % 7), @floatFromInt(id), -0.75, @floatFromInt(id % 23) };
        out.* = Ref.dot(w.*);
    }
    for ([_]usize{ 1, 7, 512 }) |legal| {
        var bits: [(vocab + 31) / 32]u32 = @splat(0);
        bits[bits.len - 1] = 0xffff_ff80; // outside this vocabulary
        for (0..legal) |i| {
            const id = vocab - 1 - i * 2;
            bits[id / 32] |= @as(u32, 1) << @as(u5, @truncate(id));
        }
        var ids: [default_limit]u32 = undefined;
        const count = collect(&bits, vocab, &ids);
        try std.testing.expectEqual(legal, count);
        var result: [vocab]u16 = @splat(neg_inf);
        for (ids[0..count]) |id| result[id] = Ref.dot(weights[id]);
        for (full, result, 0..) |expected, actual, id| {
            const allowed = bits[id / 32] & (@as(u32, 1) << @as(u5, @truncate(id))) != 0;
            try std.testing.expectEqual(if (allowed) expected else neg_inf, actual);
        }
        try std.testing.expectEqual(0, collect(&bits, vocab, ids[0 .. legal - 1]));
        try std.testing.expectEqual(legal, collect(&bits, vocab, ids[0..legal]));
    }
    var ids: [default_limit]u32 = undefined;
    try std.testing.expectEqual(0, collect(&.{0}, vocab, &ids));
    try std.testing.expectEqual(0, collect(&(@as([33]u32, @splat(0xffffffff))), vocab, &ids));
}
