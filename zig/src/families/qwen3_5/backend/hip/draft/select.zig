//! A hold's chain from the frame's downloads: lanes' shared walker (core/lanes/draft_tree.zig) on the rows apart.
const std = @import("std");
const walker = @import("lanes").draft_tree;
const config = @import("config.zig");

pub const Candidates = walker.Candidates;
pub const chain = walker.chain;

comptime {
    // a row's candidates fit the walker's arrays
    std.debug.assert(config.max_top <= walker.lp_top);
}

fn testBf16(x: f32) u16 {
    return @intCast(@as(u32, @bitCast(x)) >> 16);
}

test "a chain takes each row's best logit when the selector is silent, ties to the earlier candidate" {
    const ids = [_]u32{ 3, 1, 2, 0, 1, 2 };
    const logits = [_]f32{ 1, 5, 5, 2, 0, -1 };
    const zero: [4 * 2]u16 = @splat(0);
    const c: Candidates = .{
        .apart = .{ .ids = &ids, .logits = &logits, .projected = &.{ 0, 0, 0, 0 } },
        .topk = 3,
        .rank = 2,
        .predecessor = &zero,
        .successor = &zero,
    };
    var out: [2]u32 = undefined;
    chain(c, 0, &out);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0 }, &out);
}

test "the selector's edge after the previous draft can overturn a row's unary order" {
    // vocab 4, rank 1: predecessor 3 pulls candidate 2 up by 1 * 2 * 4 = 8 over candidate 1's lead of 1
    const pred = [_]u16{ testBf16(0), testBf16(0), testBf16(0), testBf16(1) };
    const succ = [_]u16{ testBf16(0), testBf16(0), testBf16(4), testBf16(0) };
    const ids = [_]u32{ 1, 2 };
    const logits = [_]f32{ 3, 2 };
    const c: Candidates = .{
        .apart = .{ .ids = &ids, .logits = &logits, .projected = &.{2} },
        .topk = 2,
        .rank = 1,
        .predecessor = &pred,
        .successor = &succ,
    };
    var out: [1]u32 = undefined;
    chain(c, 3, &out);
    try std.testing.expectEqual(@as(u32, 2), out[0]);
    chain(c, 0, &out);
    try std.testing.expectEqual(@as(u32, 1), out[0]);
}

test "temperature 0 keeps the logits' order over the noise; a sampled rule is a fixed draw of its seed" {
    const ids = [_]u32{ 7, 8, 9 };
    const logits = [_]f32{ 0.3, 0.2, 0.1 };
    const zero: [10]u16 = @splat(0);
    var c: Candidates = .{
        .apart = .{ .ids = &ids, .logits = &logits, .projected = &.{0} },
        .topk = 3,
        .rank = 1,
        .predecessor = &zero,
        .successor = &zero,
        .sampling = .{ .seed = 5, .temperature = 0 },
    };
    var out: [1]u32 = undefined;
    chain(c, 0, &out);
    try std.testing.expectEqual(@as(u32, 7), out[0]);
    c.sampling = .{ .seed = 5, .temperature = 1.0 };
    c.position = 40;
    chain(c, 0, &out);
    const first = out[0];
    chain(c, 0, &out);
    try std.testing.expectEqual(first, out[0]);
    var seen = false;
    for (0..64) |p| {
        c.position = p;
        chain(c, 0, &out);
        seen = seen or out[0] != 7;
    }
    try std.testing.expect(seen);
}

/// A fixed fill: an LCG's next value in [-scale, scale) at 16 bits, most of them between bf16 steps.
fn testFill(s: *u32, scale: f32) f32 {
    s.* = s.* *% 1664525 +% 1013904223;
    return @as(f32, @floatFromInt(@as(i32, @intCast(s.* >> 16)) - 32768)) / 32768 * scale;
}

test "fixed rows pick a fixed chain, greedy and sampled" {
    const rows = 7;
    const topk = 20;
    const rank = 8;
    const vocab = 64;
    var s: u32 = 1;
    var ids: [rows * topk]u32 = undefined;
    var logits: [rows * topk]f32 = undefined;
    var projected: [rows * rank]f32 = undefined;
    var pred: [vocab * rank]u16 = undefined;
    var succ: [vocab * rank]u16 = undefined;
    for (&ids, &logits) |*id, *l| {
        id.* = @intFromFloat(@abs(testFill(&s, vocab)));
        l.* = testFill(&s, 2);
    }
    for (&projected) |*p| p.* = testFill(&s, 1);
    for (&pred, &succ) |*p, *q| {
        p.* = testBf16(testFill(&s, 0.5));
        q.* = testBf16(testFill(&s, 0.5));
    }
    var c: Candidates = .{
        .apart = .{ .ids = &ids, .logits = &logits, .projected = &projected },
        .topk = topk,
        .rank = rank,
        .predecessor = &pred,
        .successor = &succ,
    };
    var out: [rows]u32 = undefined;
    chain(c, 5, &out);
    try std.testing.expectEqualSlices(u32, &.{ 4, 58, 9, 58, 24, 46, 15 }, &out);
    c.sampling = .{ .seed = 11, .temperature = 0.7 };
    c.position = 100;
    chain(c, 5, &out);
    try std.testing.expectEqualSlices(u32, &.{ 4, 37, 52, 58, 46, 54, 61 }, &out);
}

test "a selector projection between bf16 steps keeps its f32 value (an f16 target's)" {
    // vocab 2, rank 1: hp 1 + 2^-10 lifts candidate 0 over candidate 1's unary 1 + 2^-11; hp at bf16 (1.0) would not
    const pred = [_]u16{ testBf16(1), testBf16(1) };
    const succ = [_]u16{ testBf16(1), testBf16(0) };
    const ids = [_]u32{ 0, 1 };
    const logits = [_]f32{ 0, 1 + 0x1p-11 };
    const c: Candidates = .{
        .apart = .{ .ids = &ids, .logits = &logits, .projected = &.{1 + 0x1p-10} },
        .topk = 2,
        .rank = 1,
        .predecessor = &pred,
        .successor = &succ,
    };
    var out: [1]u32 = undefined;
    chain(c, 0, &out);
    try std.testing.expectEqual(@as(u32, 0), out[0]);
}
