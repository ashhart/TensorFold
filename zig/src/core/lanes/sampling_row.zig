//! The host draw over a row of bf16 logits: greedy, the keyed fp64 sampler over the top_k, or a grammar mask's tokens.
const std = @import("std");
const sampling = @import("sampling.zig");
const grammar = @import("grammar.zig");
const Sampling = sampling.Sampling;

/// A bf16 word as the f64 the keyed sampler reads.
pub fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// A finite bf16 word's order key: numeric order as unsigned order, both zeros 0x8000, every finite key above zero.
pub fn key(word: u16) u16 {
    return if (word & 0x8000 != 0) 0 -% word else word ^ 0x8000;
}

/// `key` over a vector of finite words.
pub fn keys(comptime n: usize, words: @Vector(n, u16)) @Vector(n, u16) {
    const V = @Vector(n, u16);
    const negative = words & @as(V, @splat(0x8000)) != @as(V, @splat(0));
    return @select(u16, negative, @as(V, @splat(0)) -% words, words ^ @as(V, @splat(0x8000)));
}

/// The allowed bits of mask word `w`: all without a mask, none past its end.
pub fn maskWord(mask: []const u32, w: usize) u32 {
    if (mask.len == 0) return std.math.maxInt(u32);
    return if (w < mask.len) mask[w] else 0;
}

/// Lane i is bit i of `bits`: each 16-bit part spread over its 16 lanes and tested against the lane's own bit.
pub fn spread(comptime n: usize, bits: @Int(.unsigned, n)) @Vector(n, bool) {
    const V = @Vector(n, u16);
    const index, const lane = comptime table: {
        var index: [n]i32 = undefined;
        var lane: [n]u16 = undefined;
        for (0..n) |i| {
            index[i] = i / 16;
            lane[i] = 1 << (i % 16);
        }
        break :table .{ index, lane };
    };
    const parts: @Vector(n / 16, u16) = @bitCast(bits);
    return @shuffle(u16, parts, undefined, index) & @as(V, lane) != @as(V, @splat(0));
}

/// Logits a scan block: its two mask words are tested together, and a block neither allows is not ranked.
const block = 64;
const Block = @Vector(block, u16);
/// Each lane's place in its block.
const order: Block = std.simd.iota(u16, block);

/// A row's first largest allowed id (null when none is allowed) and how many ids are allowed.
pub const Scan = struct { best: ?usize, allowed: usize };

/// Every word is checked finite, allowed or not; only a larger allowed key replaces the best: ties keep the lowest id.
pub fn scan(logits: []const u16, mask: []const u32) error{NonfiniteLogits}!Scan {
    return if (mask.len == 0) scanRow(false, logits, mask) else scanRow(true, logits, mask);
}

/// `scan` with the mask's presence fixed at compile time, so an unmasked row reads no mask words.
fn scanRow(comptime masked: bool, logits: []const u16, mask: []const u32) error{NonfiniteLogits}!Scan {
    const exponent: Block = @splat(0x7f80);
    var widest: Block = @splat(0);
    var top: u16 = 0;
    var best: ?usize = null;
    var allowed: usize = 0;
    var at: usize = 0;
    while (at + block <= logits.len) : (at += block) {
        const row: Block = logits[at..][0..block].*;
        widest = @max(widest, row & exponent);
        const bits = if (masked) @as(u64, maskWord(mask, at / 32)) | @as(u64, maskWord(mask, at / 32 + 1)) << 32 else ~@as(u64, 0);
        if (bits == 0) continue;
        allowed += @popCount(bits);
        var ranks = keys(block, row);
        if (~bits != 0) ranks = @select(u16, spread(block, bits), ranks, @as(Block, @splat(0)));
        const most = @reduce(.Max, ranks);
        if (most <= top) continue;
        top = most;
        best = at + @reduce(.Min, @select(u16, ranks == @as(Block, @splat(most)), order, @as(Block, @splat(block))));
    }
    if (@reduce(.Max, widest) == 0x7f80) return error.NonfiniteLogits;
    for (logits[at..], at..) |word, id| {
        if (word & 0x7f80 == 0x7f80) return error.NonfiniteLogits;
        if (masked and !grammar.allows(mask, id)) continue;
        allowed += 1;
        if (key(word) > top) {
            top = key(word);
            best = id;
        }
    }
    return .{ .best = best, .allowed = allowed };
}

/// Inserts an allowed candidate among the best `values.len` after equal values, or appends it when `all` are kept.
inline fn keep(values: []f64, ids: []u64, filled: *usize, all: bool, x: f64, id: usize) void {
    const k = values.len;
    if (all) {
        values[filled.*] = x;
        ids[filled.*] = id;
        filled.* += 1;
        return;
    }
    if (filled.* == k and x <= values[k - 1]) return;
    var at = @min(filled.*, k - 1);
    while (at > 0 and x > values[at - 1]) : (at -= 1) {
        values[at] = values[at - 1];
        ids[at] = ids[at - 1];
    }
    values[at] = x;
    ids[at] = id;
    filled.* = @min(filled.* + 1, k);
}

/// The unchanged top-k candidates for the shared fp64 keyed sampler; a `mask` keeps its allowed ones (-inf's draw).
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?Sampling, position: u64, mask: []const u32) !u32 {
    const row = try scan(logits, mask);
    const best = row.best orelse return error.NoAllowedToken;
    const s = settings orelse return @intCast(best);
    if (s.temperature <= 0) return @intCast(best);
    const k = if (s.top_k == 0) row.allowed else @min(row.allowed, s.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    if (mask.len == 0) {
        for (logits, 0..) |word, id| keep(values, ids, &filled, k == row.allowed, value(word), id);
    } else for (mask[0..@min(mask.len, (logits.len + 31) / 32)], 0..) |bits, w| {
        var left = bits;
        while (left != 0) : (left &= left - 1) {
            const id = w * 32 + @ctz(left);
            if (id >= logits.len) break;
            keep(values, ids, &filled, k == row.allowed, value(logits[id]), id);
        }
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

test "bf16 keys preserve numeric order over every finite word and both zero encodings" {
    for (0..65536) |bits| {
        const a: u16 = @intCast(bits);
        if (a & 0x7f80 == 0x7f80) continue;
        try std.testing.expect(key(a) > 0);
        try std.testing.expectEqual(key(a), keys(1, .{a})[0]);
        for ([_]u16{ a ^ 0x8000, a +% 1, a -% 1, 0, 0x8000, 0x7f7f, 0xff7f }) |b| {
            if (b & 0x7f80 == 0x7f80) continue;
            try std.testing.expectEqual(value(a) < value(b), key(a) < key(b));
            try std.testing.expectEqual(value(a) == value(b), key(a) == key(b));
        }
    }
}

test "spread puts bit i of a mask word in lane i" {
    var prng = std.Random.DefaultPrng.init(64);
    for (0..512) |case| {
        const wide: u64 = if (case < 64) @as(u64, 1) << @intCast(case) else prng.random().int(u64);
        const lanes64: [64]bool = spread(64, wide);
        const lanes32: [32]bool = spread(32, @truncate(wide));
        for (lanes64, 0..) |on, i| try std.testing.expectEqual(wide >> @intCast(i) & 1 == 1, on);
        for (lanes32, 0..) |on, i| try std.testing.expectEqual(wide >> @intCast(i) & 1 == 1, on);
    }
}
