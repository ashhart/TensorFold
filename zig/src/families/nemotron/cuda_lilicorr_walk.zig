//! The candidate walk: how a LiLiCorr score lattice becomes the drafts. The reference
//! (dflash2 speculator's _selector_walk_kernel) walks a CHAIN — `previous` starts at 0 (so
//! step 0 reads the start scores), each step argmaxes the row (step, previous) and takes
//! that candidate's token. `start` is [topk] (slot 0's candidate entry scores), `pair` is
//! [(slots-1) * topk * topk]. Greedy is a plain argmax; the sampled walk perturbs the row
//! with keyed gumbel noise before the argmax (the draws machinery keys it here, the walk
//! itself unchanged). Realized scores return per step for the round loop's pricing.
const std = @import("std");
const head = @import("cuda_lilicorr_head.zig");

const slots = head.slots;
const topk = head.topk;

/// The walk score matrix [slots, topk, topk] the walk reads: step 0's rows are the start
/// scores (every predecessor row agrees), step s's rows are pair[s - 1] scaled.
pub fn walkMatrix(start: []const f32, pair: []const f32, scale: f32, dst: []f32) !void { // start [topk]
    if (dst.len != slots * topk * topk) return error.ShapeMismatch;
    for (0..topk * topk) |i| dst[i] = scale * start[i % topk];
    for (0..(slots - 1) * topk * topk) |i| dst[topk * topk + i] = scale * pair[i];
}

pub const Chain = struct {
    tokens: [slots]u32, // the walked candidates' token ids
    index: [slots]usize, // each step's picked candidate column
    scores: [slots]f32, // each step's picked walk score (the caller's scale applied)
};

/// The greedy (temperature 0) chain over `start` [slots] and `pair` [(slots-1) * topk * topk].
/// `cand` is [slots * topk] token ids (the drafter's per-slot top-k). Ties take the lowest
/// column (the reference kernel's argmax order). Negative or NaN rows refuse loudly.
pub fn chain(start: []const f32, pair: []const f32, cand: []const u32) !Chain { // start [topk], pair [(slots-1) * topk * topk]
    var out: Chain = .{ .tokens = undefined, .index = undefined, .scores = undefined };
    var previous: usize = 0;
    for (0..slots) |s| {
        const row = if (s == 0) start[0..topk] else pair[((s - 1) * topk + previous) * topk ..][0..topk];
        var best: usize = 0;
        var value = row[0];
        for (row[1..], 1..) |v, i| if (v > value) {
            value = v;
            best = i;
        };
        if (!(value > -std.math.floatMax(f32)) or std.math.isNan(value)) return error.BadWalkScore;
        out.index[s] = best;
        out.tokens[s] = cand[s * topk + best];
        out.scores[s] = value;
        previous = best;
    }
    return out;
}

/// The row scores of one realized chain (the realized_scores the reference caches): step s's
/// row of the walk matrix. Same lookup the sampler's cache makes.
pub fn realizedScores(start: []const f32, pair: []const f32, picked: []const usize, out: []f32) !void { // start [topk]
    if (picked.len != slots or out.len != slots) return error.ShapeMismatch;
    for (picked, 0..) |p, s| {
        if (p >= topk) return error.BadWalkScore;
        out[s] = if (s == 0) start[p] else pair[((s - 1) * topk + picked[s - 1]) * topk + p];
    }
}

test "the greedy chain follows the transition argmaxes and the start row" {
    const a = std.testing.allocator;
    const start = try a.alloc(f32, topk);
    defer a.free(start);
    const pair = try a.alloc(f32, (slots - 1) * topk * topk);
    defer a.free(pair);
    const cand = try a.alloc(u32, slots * topk);
    defer a.free(cand);
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    for (start) |*v| v.* = rand.float(f32) - 0.5;
    for (pair) |*v| v.* = rand.float(f32) - 0.5;
    for (cand, 0..) |*v, i| v.* = @intCast(1000 + i);
    // force the path: step 0 picks column 5, then column 3, then column 0
    start[5] = 9.0;
    for (0..topk) |j| pair[(0 * topk + 5) * topk + j] = if (j == 3) 7.0 else -7.0; // from (0, 5) into slot 1
    for (0..topk) |j| pair[(1 * topk + 3) * topk + j] = if (j == 0) 6.0 else -6.0; // from (1, 3)
    const got = try chain(start, pair, cand);
    try std.testing.expectEqual(@as(usize, 5), got.index[0]);
    try std.testing.expectEqual(@as(usize, 3), got.index[1]);
    try std.testing.expectEqual(@as(usize, 0), got.index[2]);
    try std.testing.expectEqual(@as(u32, 1005), got.tokens[0]); // slot 0, column 5
    try std.testing.expectEqual(@as(u32, 1000 + topk + 3), got.tokens[1]);
    try std.testing.expectEqual(@as(u32, 1000 + 2 * topk), got.tokens[2]);
    try std.testing.expectEqual(start[5], got.scores[0]);
    try std.testing.expectEqual(@as(f32, 7.0), got.scores[1]);
    try std.testing.expectEqual(@as(f32, 6.0), got.scores[2]);
}

test "ties take the lowest column and a NaN row refuses" {
    const a = std.testing.allocator;
    const start = try a.alloc(f32, topk);
    defer a.free(start);
    const pair = try a.alloc(f32, (slots - 1) * topk * topk);
    defer a.free(pair);
    const cand = try a.alloc(u32, slots * topk);
    defer a.free(cand);
    @memset(start, 1.0); // every column ties: the walk takes column 0
    @memset(pair, 0.5);
    @memset(cand, 42);
    const got = try chain(start, pair, cand);
    for (got.index) |i| try std.testing.expectEqual(@as(usize, 0), i);
    try std.testing.expectEqual(@as(f32, 1.0), got.scores[0]);
    start[0] = std.math.nan(f32);
    try std.testing.expectError(error.BadWalkScore, chain(start, pair, cand));
}

test "walkMatrix is the start row expanded then the scaled pairs, and realizedScores re-reads it" {
    const a = std.testing.allocator;
    const start = try a.alloc(f32, topk);
    defer a.free(start);
    const pair = try a.alloc(f32, (slots - 1) * topk * topk);
    defer a.free(pair);
    for (start, 0..) |*v, i| v.* = @floatFromInt(i);
    for (pair, 0..) |*v, i| v.* = @floatFromInt(100 + i);
    const walk = try a.alloc(f32, slots * topk * topk);
    defer a.free(walk);
    try walkMatrix(start, pair, 8.0, walk);
    try std.testing.expectEqual(@as(f32, 8.0 * 2.0), walk[2]); // step 0: the start row, column 2
    try std.testing.expectEqual(@as(f32, 8.0 * 3.0), walk[topk * 2 + 3]); // predecessor row 2 reads the same start row
    try std.testing.expectEqual(@as(f32, 8.0 * (100 + 65)), walk[topk * topk + 65]);
    const picked = [_]usize{ 3, 5, 1, 0, 7, 2, 4 };
    const got = try a.alloc(f32, slots);
    defer a.free(got);
    try realizedScores(start, pair, &picked, got);
    try std.testing.expectEqual(start[3], got[0]);
    try std.testing.expectEqual(pair[(0 * topk + 3) * topk + 5], got[1]);
    try std.testing.expectEqual(pair[(1 * topk + 5) * topk + 1], got[2]);
    try std.testing.expectError(error.BadWalkScore, realizedScores(start, pair, &[_]usize{ 3, 5, 1, 0, 7, 2, topk }, got));
}
