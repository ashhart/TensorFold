//! DFlash2's host path selector (rc_df_walk): each draft the best of a row's top 32 by logit plus codebook score.
const std = @import("std");
const core = @import("core");

const bf = core.bf16_round;

/// Group of products one simd_sum adds; the kernel's partial sums are one per 32 ranks.
const lane_group = 32;

/// Drafts the walk writes at most: a block's rows after the pending token.
pub const max_slots = 15;

/// One walk's inputs: the top-32 candidates of each row and the selector's weights, all as the kernels leave them.
pub const Inputs = struct {
    top: u32,
    rank: u32,
    vocab: u32,
    values: []const f32, // [slots, top] logits, value descending then id ascending
    ids: []const u32, // [slots, top] vocabulary ids; at or past `vocab` is no candidate
    hidden: []const u16, // [slots, rank] bf16 bits: the selector's hidden projection of each row
    pred: []const u16, // [vocab, rank] bf16 bits: the predecessor codebook
    succ: []const u16, // [vocab, rank] bf16 bits: the successor codebook
};

/// Drafts after `anchor` into `out` (rows to walk) as rc_df_walk; the count made before a row with no candidate.
pub fn walk(in: Inputs, anchor: u32, out: []u32) usize {
    const rank: usize = in.rank;
    const top: usize = in.top;
    std.debug.assert(rank % lane_group == 0 and rank <= 1024 and in.hidden.len >= out.len * rank);
    std.debug.assert(in.values.len >= out.len * top and in.ids.len >= out.len * top);
    std.debug.assert(anchor < in.vocab);
    var ah: [1024]f32 = undefined;
    var prev: usize = anchor;
    for (out, 0..) |*o, e| {
        const a = in.pred[prev * rank ..][0..rank];
        const h = in.hidden[e * rank ..][0..rank];
        // candidate c scores bf16(logit + bf16(sum_r bf16(A[prev][r] H[e][r]) B[c][r])), a tie to the lower c
        for (ah[0..rank], a, h) |*x, p, q| x.* = bf.round32(bf.value(p) * bf.value(q));
        var best: f32 = -std.math.inf(f32);
        var pick: usize = 0;
        for (0..top) |c| {
            const id = in.ids[e * top + c];
            if (id >= in.vocab) continue;
            const b = in.succ[@as(usize, id) * rank ..][0..rank];
            var pair: f32 = 0;
            var g: usize = 0;
            // the 32-rank partials in lane order (simd_sum's own order is the hardware's), then the partials in order
            while (g < rank) : (g += lane_group) {
                var part: f32 = 0;
                for (ah[g..][0..lane_group], b[g..][0..lane_group]) |x, y| part += x * bf.value(y);
                pair += part;
            }
            const s = bf.round32(in.values[e * top + c] + bf.round32(pair));
            if (s > best) {
                best = s;
                pick = c;
            }
        }
        const id = in.ids[e * top + pick];
        // the chain stops (the Metal walk goes on, and the caller refuses the id)
        if (id >= in.vocab) return e;
        o.* = id;
        prev = id;
    }
    return out.len;
}

const testing = std.testing;

/// rc_df_walk, thread by thread: 256 products, one partial a 32-lane group, thread 0 choosing; the whole chain runs.
fn metalWalk(in: Inputs, anchor: u32, seq: []u32) void {
    const rank: usize = in.rank;
    var part: [32][32]f32 = undefined; // [group][candidate]
    var prev: u32 = anchor;
    for (0..seq.len) |e| {
        for (0..in.top) |c| {
            const cand = in.ids[e * in.top + c];
            const row = @min(cand, in.vocab - 1);
            var g: usize = 0;
            while (g < rank / 32) : (g += 1) {
                var lane_sum: f32 = 0;
                for (0..32) |lane| {
                    const t = g * 32 + lane;
                    const ahv = bf.round32(bf.value(in.pred[@as(usize, @min(prev, in.vocab - 1)) * rank + t]) * bf.value(in.hidden[e * rank + t]));
                    lane_sum += ahv * bf.value(in.succ[@as(usize, row) * rank + t]);
                }
                part[g][c] = lane_sum;
            }
        }
        var best: f32 = -std.math.inf(f32);
        var bi: usize = 0;
        for (0..in.top) |c| {
            var pair: f32 = 0;
            for (0..rank / 32) |k| pair += part[k][c];
            const s = bf.round32(in.values[e * in.top + c] + bf.round32(pair));
            if (in.ids[e * in.top + c] < in.vocab and s > best) {
                best = s;
                bi = c;
            }
        }
        prev = in.ids[e * in.top + bi];
        seq[e] = prev;
    }
}

const Rows = struct {
    values: []f32,
    ids: []u32,
    hidden: []u16,
    pred: []u16,
    succ: []u16,

    fn free(r: Rows, a: std.mem.Allocator) void {
        a.free(r.values);
        a.free(r.ids);
        a.free(r.hidden);
        a.free(r.pred);
        a.free(r.succ);
    }
};

const T_RANK = 256;
const T_TOP = 32;
const T_VOCAB = 300;
const T_SLOTS = 15;

/// Fixed pseudo-random rows with small bf16 values (so sums are in range); logits coarse so ties occur.
fn rows(a: std.mem.Allocator) !Rows {
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const rnd = prng.random();
    const r: Rows = .{
        .values = try a.alloc(f32, T_SLOTS * T_TOP),
        .ids = try a.alloc(u32, T_SLOTS * T_TOP),
        .hidden = try a.alloc(u16, T_SLOTS * T_RANK),
        .pred = try a.alloc(u16, T_VOCAB * T_RANK),
        .succ = try a.alloc(u16, T_VOCAB * T_RANK),
    };
    for (r.hidden) |*x| x.* = bf.bits(rnd.float(f32) * 2 - 1);
    for (r.pred) |*x| x.* = bf.bits(rnd.float(f32) * 2 - 1);
    for (r.succ) |*x| x.* = bf.bits(rnd.float(f32) * 2 - 1);
    for (0..T_SLOTS) |e| {
        for (0..T_TOP) |c| {
            r.ids[e * T_TOP + c] = @intCast((e * 7 + c * 9) % T_VOCAB); // distinct within a row
            r.values[e * T_TOP + c] = @floatFromInt(@as(i32, 8) - @as(i32, @intCast(c / 4))); // runs of 4 equal logits
        }
    }
    return r;
}

fn inputs(r: Rows) Inputs {
    return .{ .top = T_TOP, .rank = T_RANK, .vocab = T_VOCAB, .values = r.values, .ids = r.ids, .hidden = r.hidden, .pred = r.pred, .succ = r.succ };
}

test "ricursive dflash2 walk: fixed rows equal rc_df_walk's transcription, ties included" {
    const a = testing.allocator;
    const r = try rows(a);
    defer r.free(a);
    // row 0's candidates 0 and 1 (ids 0 and 9) have one logit and one codebook row: the lower candidate wins the tie
    for (0..T_RANK) |k| r.succ[9 * T_RANK + k] = r.succ[0 * T_RANK + k];
    var want: [T_SLOTS]u32 = undefined;
    var got: [T_SLOTS]u32 = undefined;
    for ([_]u32{ 0, 1, 299 }) |anchor| {
        metalWalk(inputs(r), anchor, &want);
        try testing.expectEqual(@as(usize, T_SLOTS), walk(inputs(r), anchor, &got));
        try testing.expectEqualSlices(u32, &want, &got);
    }
    // a walk of fewer rows is the prefix of the longer one
    var few: [4]u32 = undefined;
    try testing.expectEqual(@as(usize, 4), walk(inputs(r), 0, &few));
    metalWalk(inputs(r), 0, &want);
    try testing.expectEqualSlices(u32, want[0..4], &few);
}

test "ricursive dflash2 walk: equal scores pick the lower candidate" {
    const a = testing.allocator;
    const r = try rows(a);
    defer r.free(a);
    // every candidate of row 0 has one id's codebook row and one logit: all scores tie, so candidate 0 is chosen
    for (0..T_TOP) |c| {
        r.ids[c] = @intCast(10 + c);
        r.values[c] = 1;
        for (0..T_RANK) |k| r.succ[(10 + c) * T_RANK + k] = r.succ[10 * T_RANK + k];
    }
    var out: [1]u32 = undefined;
    try testing.expectEqual(@as(usize, 1), walk(inputs(r), 0, &out));
    try testing.expectEqual(@as(u32, 10), out[0]);
}

test "ricursive dflash2 walk: ids past the vocabulary are skipped, a row with none stops the chain" {
    const a = testing.allocator;
    const r = try rows(a);
    defer r.free(a);
    // row 0: the first 31 candidates are no candidate
    for (0..T_TOP - 1) |c| r.ids[c] = 0xffffffff;
    // row 3: none at all
    for (0..T_TOP) |c| r.ids[3 * T_TOP + c] = @intCast(T_VOCAB + c);
    var out: [T_SLOTS]u32 = undefined;
    try testing.expectEqual(@as(usize, 3), walk(inputs(r), 2, &out));
    try testing.expectEqual(r.ids[T_TOP - 1], out[0]);
    // the Metal transcription agrees up to the stop; its row 3 pick is an id the caller refuses
    var seq: [T_SLOTS]u32 = undefined;
    metalWalk(inputs(r), 2, &seq);
    try testing.expectEqualSlices(u32, out[0..3], seq[0..3]);
    try testing.expect(seq[3] >= T_VOCAB);
}
