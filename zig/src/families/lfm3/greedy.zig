//! Exact unmasked BF16 greedy selection: SIMD finite checks and maxima retain the lowest token on ties.
const std = @import("std");

pub fn select(words: []const u16) !u32 {
    var best: ?usize = null;
    var maximum: f32 = -std.math.inf(f32);
    var i: usize = 0;
    while (i + 16 <= words.len) : (i += 16) {
        const codes: @Vector(16, u16) = words[i..][0..16].*;
        if (@reduce(.Or, (codes & @as(@Vector(16, u16), @splat(0x7f80))) == @as(@Vector(16, u16), @splat(0x7f80))))
            return error.NonfiniteLfmLogits;
        const wide: @Vector(16, u32) = @intCast(codes);
        const values: @Vector(16, f32) = @bitCast(wide << @as(@Vector(16, u5), @splat(16)));
        const block = @reduce(.Max, values);
        if (block > maximum) {
            const array: [16]f32 = values;
            for (array, 0..) |value, j| if (value == block) {
                maximum = block;
                best = i + j;
                break;
            };
        }
    }
    for (words[i..], i..) |word, j| {
        const value: f32 = @bitCast(@as(u32, word) << 16);
        if (!std.math.isFinite(value)) return error.NonfiniteLfmLogits;
        if (value > maximum) {
            maximum = value;
            best = j;
        }
    }
    return @intCast(best orelse return error.NoAllowedLfmToken);
}

fn scalar(words: []const u16) !u32 {
    var best: ?usize = null;
    for (words, 0..) |word, i| {
        const value: f32 = @bitCast(@as(u32, word) << 16);
        if (!std.math.isFinite(value)) return error.NonfiniteLfmLogits;
        if (best == null or value > @as(f32, @bitCast(@as(u32, words[best.?]) << 16))) best = i;
    }
    return @intCast(best orelse return error.NoAllowedLfmToken);
}

test "SIMD greedy keeps the first finite maximum across vector and remainder boundaries" {
    var words: [97]u16 = undefined;
    for (1..words.len + 1) |len| {
        @memset(&words, 0xbf80);
        for (0..len) |at| {
            words[at] = 0x3f80;
            try std.testing.expectEqual(try scalar(words[0..len]), try select(words[0..len]));
            words[at] = 0xbf80;
        }
    }
    for ([_]u16{ 0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f }) |word| {
        @memset(&words, word);
        try std.testing.expectEqual(@as(u32, 0), try select(&words));
    }
    @memset(&words, 0x8000);
    words[7] = 0;
    try std.testing.expectEqual(@as(u32, 0), try select(&words));
    try std.testing.expectError(error.NoAllowedLfmToken, select(&.{}));
}

test "SIMD greedy rejects every nonfinite BF16 encoding even after an earlier maximum" {
    var words: [49]u16 = @splat(0x7f7f);
    for (0..65536) |bits| {
        const word: u16 = @intCast(bits);
        if (word & 0x7f80 != 0x7f80) continue;
        for ([_]usize{ 0, 15, 16, 31, 32, 48 }) |at| {
            words[at] = word;
            try std.testing.expectError(error.NonfiniteLfmLogits, select(&words));
            words[at] = 0x7f7f;
        }
    }
}

test "SIMD greedy matches scalar ordering over every finite BF16 code and randomized vocabularies" {
    const a = std.testing.allocator;
    const words = try a.alloc(u16, 128000);
    defer a.free(words);
    var at: usize = 0;
    for (0..65536) |bits| {
        const word: u16 = @intCast(bits);
        if (word & 0x7f80 == 0x7f80) continue;
        words[at] = word;
        at += 1;
    }
    try std.testing.expectEqual(try scalar(words[0..at]), try select(words[0..at]));
    var pair: [33]u16 = @splat(0xff7f);
    for (0..65536) |bits| {
        const word: u16 = @intCast(bits);
        if (word & 0x7f80 == 0x7f80) continue;
        pair[3] = word;
        pair[17] = word ^ 1;
        try std.testing.expectEqual(try scalar(&pair), try select(&pair));
        std.mem.swap(u16, &pair[3], &pair[17]);
        try std.testing.expectEqual(try scalar(&pair), try select(&pair));
    }
    std.mem.reverse(u16, words[0..at]);
    try std.testing.expectEqual(try scalar(words[0..at]), try select(words[0..at]));
    var prng = std.Random.DefaultPrng.init(8719);
    for (0..32) |_| {
        for (words) |*word| word.* = @intCast(@as(u32, @bitCast(prng.random().float(f32) * 32 - 16)) >> 16);
        try std.testing.expectEqual(try scalar(words), try select(words));
    }
}
