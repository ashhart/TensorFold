//! A hold's chain from its rows' candidates (cuda_draft_tree.zig's `chain`): each row's best after the one before.
const std = @import("std");
const lanes = @import("lanes");

/// A sampled stream's candidates carry the draw's own Gumbel noise at this weight (the 27B engine's draft_tree).
const noise_weight = 0.7;
/// Candidates a row keeps at most (config.max_top).
pub const most_top = 20;

/// A hold's rows: each row's top ids and logits (`stride` a row), its selector projection, and the codebooks.
pub const Candidates = struct {
    ids: []const u32,
    logits: []const f32,
    stride: usize,
    topk: usize,
    projected: []const f32, // (rows, rank)
    rank: usize,
    predecessor: []const u16, // (vocab, rank) bf16
    successor: []const u16,
    // a sampled stream's rule and its first draft's position: row r's candidates scored as the target's draw at r
    sampling: ?lanes.Sampling = null,
    position: u64 = 0,
};

fn bf32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// A candidate's score as the target's draw sees it: its logit at the stream's temperature plus the draw's noise.
fn drawn(c: Candidates, r: usize, token: u32, logit: f64) f64 {
    const s = c.sampling orelse return logit;
    const top: f32 = @bitCast(@as(u32, 0x3F7FFFFF));
    const u = @min(lanes.gpu_rule.uniform24(s.seed, @intCast(c.position + r), token), top);
    return logit / @max(s.temperature, 1e-6) + noise_weight * -@log(-@log(@as(f64, u)));
}

/// Row `r`'s best candidate after `pred`: unary logit + pred . projected . successor, ties to the earlier one.
pub fn choice(c: Candidates, r: usize, pred: u32) u32 {
    const ids = c.ids[r * c.stride ..][0..c.topk];
    const logits = c.logits[r * c.stride ..][0..c.topk];
    const hp = c.projected[r * c.rank ..][0..c.rank];
    var best: f32 = -std.math.inf(f32);
    var pick: usize = 0;
    for (ids, logits, 0..) |id, logit, j| {
        var edge: f32 = 0;
        for (0..c.rank) |k| edge += bf32(c.predecessor[pred * c.rank + k]) * hp[k] * bf32(c.successor[id * c.rank + k]);
        const score: f32 = @floatCast(drawn(c, r, id, logit + edge));
        if (score > best) {
            best = score;
            pick = j;
        }
    }
    return ids[pick];
}

/// The path through the rows' candidates: each row's best after the one before, the first after `pending`.
pub fn chain(c: Candidates, pending: u32, out: []u32) void {
    var pred = pending;
    for (out, 0..) |*o, r| {
        o.* = choice(c, r, pred);
        pred = o.*;
    }
}

fn testBf16(x: f32) u16 {
    return @intCast(@as(u32, @bitCast(x)) >> 16);
}

test "a chain takes each row's best logit when the selector is silent, ties to the earlier candidate" {
    const ids = [_]u32{ 3, 1, 2, 0, 1, 2 };
    const logits = [_]f32{ 1, 5, 5, 2, 0, -1 };
    const zero: [4 * 2]u16 = @splat(0);
    const c: Candidates = .{
        .ids = &ids,
        .logits = &logits,
        .stride = 3,
        .topk = 3,
        .projected = &.{ 0, 0, 0, 0 },
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
        .ids = &ids,
        .logits = &logits,
        .stride = 2,
        .topk = 2,
        .projected = &.{2},
        .rank = 1,
        .predecessor = &pred,
        .successor = &succ,
    };
    try std.testing.expectEqual(@as(u32, 2), choice(c, 0, 3));
    try std.testing.expectEqual(@as(u32, 1), choice(c, 0, 0));
}

test "temperature 0 keeps the logits' order over the noise; a sampled rule is a fixed draw of its seed" {
    const ids = [_]u32{ 7, 8, 9 };
    const logits = [_]f32{ 0.3, 0.2, 0.1 };
    const zero: [10]u16 = @splat(0);
    var c: Candidates = .{
        .ids = &ids,
        .logits = &logits,
        .stride = 3,
        .topk = 3,
        .projected = &.{0},
        .rank = 1,
        .predecessor = &zero,
        .successor = &zero,
        .sampling = .{ .seed = 5, .temperature = 0 },
    };
    try std.testing.expectEqual(@as(u32, 7), choice(c, 0, 0));
    c.sampling = .{ .seed = 5, .temperature = 1.0 };
    c.position = 40;
    const first = choice(c, 0, 0);
    try std.testing.expectEqual(first, choice(c, 0, 0));
    var seen = false;
    for (0..64) |p| {
        c.position = p;
        seen = seen or choice(c, 0, 0) != 7;
    }
    try std.testing.expect(seen);
}
