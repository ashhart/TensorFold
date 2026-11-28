//! DFlash2's path selector on rank 0: each draft among a row's 16 likeliest by the previous draft's codebook row.
const df = @import("dflash.zig");
const CW = df.CW;
const RANK = df.RANK;
const TOPK = df.TOPK;

const Cand = struct { v: f32, id: u32 };

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// Stream `i`'s 7 drafts: each row's best of every rank's 16 by logit plus codebook score; greedy, chance kept.
pub fn select(dr: *const df.Drafter, i: usize, M: usize, q: df.Req) void {
    const out = q.out.?;
    // Clamp an out-of-vocab anchor so codebook lookup and empty-row fallback stay in bounds.
    var prev: usize = if (q.anchor < df.VOCAB) q.anchor else 0;
    for (1..df.BLOCK) |t| {
        const r = i * df.BLOCK + t;
        var c: [TOPK]Cand = undefined;
        var n: usize = 0;
        for (0..dr.world) |rk| {
            const blk = dr.cand[rk * M * CW + r * CW ..][0..CW];
            for (0..TOPK) |j| {
                // df_top16 emits 0x7fffffff when a row has no winner; skip any id outside the codebook.
                if (blk[j] < 0 or blk[j] >= df.VOCAB) continue;
                const cand: Cand = .{ .v = @bitCast(blk[TOPK + j]), .id = @intCast(blk[j]) };
                if (n < TOPK) {
                    c[n] = cand;
                    n += 1;
                    continue;
                }
                var lo: usize = 0;
                for (1..TOPK) |m| if (c[m].v < c[lo].v) {
                    lo = m;
                };
                if (cand.v > c[lo].v) c[lo] = cand;
            }
        }
        if (n == 0) { // no candidate (cannot happen with a whole vocabulary): the anchor repeated, chance 0
            for (t - 1..df.BLOCK - 1) |k| {
                out[k] = @intCast(prev);
                if (q.probs) |p| p[k] = 0;
            }
            return;
        }
        const g = dr.gh[r * RANK ..][0..RANK];
        const P = dr.pred[prev * RANK ..][0..RANK];
        var pg: [RANK]f32 = undefined;
        for (&pg, P, g) |*o, p, x| o.* = bf(p) * x;
        var best: usize = 0;
        var sc: [TOPK]f32 = undefined;
        for (0..n) |m| {
            const Sc = dr.succ[@as(usize, c[m].id) * RANK ..][0..RANK];
            var dot: f32 = 0;
            for (pg, Sc) |a, b| dot += a * bf(b);
            sc[m] = c[m].v + dot;
            if (sc[m] > sc[best]) best = m;
        }
        var z: f64 = 0;
        for (sc[0..n]) |x| z += @exp(@as(f64, x - sc[best]) / dr.temp);
        out[t - 1] = c[best].id;
        if (q.probs) |p| p[t - 1] = 1.0 / z;
        prev = c[best].id;
    }
}

test "GLM embedding: DFlash selector skips sentinel and out-of-vocab candidate IDs" {
    const std = @import("std");
    const a = std.testing.allocator;
    const world: usize = 2;
    const M: usize = df.BLOCK;
    const book = try a.alloc(u16, df.VOCAB * RANK);
    defer a.free(book);
    @memset(book, 0);
    const gh = try a.alloc(f32, M * RANK);
    defer a.free(gh);
    @memset(gh, 0);
    const cand = try a.alloc(i32, world * M * CW);
    defer a.free(cand);
    const neg_inf: i32 = @bitCast(-std.math.inf(f32));
    for (0..world) |rk| for (0..M) |r| {
        const blk = cand[rk * M * CW + r * CW ..][0..CW];
        @memset(blk[0..TOPK], 0x7fffffff);
        @memset(blk[TOPK..CW], neg_inf);
    };
    // Row 1: rank 0 holds only out-of-range IDs (0x7fffffff, VOCAB, -1); rank 1 holds the max valid ID.
    const row1_r0 = cand[1 * CW ..][0..CW];
    row1_r0[0] = 0x7fffffff;
    row1_r0[TOPK + 0] = @bitCast(@as(f32, 10.0));
    row1_r0[1] = @intCast(df.VOCAB);
    row1_r0[TOPK + 1] = @bitCast(@as(f32, 9.0));
    row1_r0[2] = -1;
    row1_r0[TOPK + 2] = @bitCast(@as(f32, 8.0));
    const row1_r1 = cand[M * CW + 1 * CW ..][0..CW];
    row1_r1[0] = @intCast(df.VOCAB - 1);
    row1_r1[TOPK + 0] = @bitCast(@as(f32, 1.5));
    var dr: df.Drafter = undefined;
    dr.world = world;
    dr.temp = 1.0;
    dr.cand = cand;
    dr.gh = gh;
    dr.pred = book;
    dr.succ = book;
    var out: [df.BLOCK - 1]u32 = undefined;
    var probs: [df.BLOCK - 1]f64 = undefined;
    select(&dr, 0, M, .{ .slot = 0, .pos = 0, .n = 0, .anchor = 5, .out = &out, .probs = &probs });
    try std.testing.expectEqual(@as(u32, df.VOCAB - 1), out[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), probs[0], 1e-12);
    for (1..df.BLOCK - 1) |k| {
        try std.testing.expectEqual(@as(u32, df.VOCAB - 1), out[k]);
        try std.testing.expectEqual(@as(f64, 0.0), probs[k]);
    }
}

test "GLM embedding: DFlash selector clamps out-of-vocab anchor before codebook lookup" {
    const std = @import("std");
    const a = std.testing.allocator;
    const M: usize = df.BLOCK;
    const book = try a.alloc(u16, df.VOCAB * RANK);
    defer a.free(book);
    @memset(book, 0);
    // pred[0][0] = 1.0, pred[VOCAB-1][0] = -1.0, succ[2][0] = 4.0, gh[1][0] = 1.0 to separate clamped vs valid anchors.
    book[0] = 0x3f80;
    book[(df.VOCAB - 1) * RANK] = 0xbf80;
    book[2 * RANK] = 0x4080;
    const gh = try a.alloc(f32, M * RANK);
    defer a.free(gh);
    @memset(gh, 0);
    gh[1 * RANK] = 1.0;
    const cand = try a.alloc(i32, M * CW);
    defer a.free(cand);
    const neg_inf: i32 = @bitCast(-std.math.inf(f32));
    for (0..M) |r| {
        const blk = cand[r * CW ..][0..CW];
        @memset(blk[0..TOPK], 0x7fffffff);
        @memset(blk[TOPK..CW], neg_inf);
    }
    var dr: df.Drafter = undefined;
    dr.world = 1;
    dr.temp = 1.0;
    dr.cand = cand;
    dr.gh = gh;
    dr.pred = book;
    dr.succ = book;
    var out: [df.BLOCK - 1]u32 = undefined;
    var probs: [df.BLOCK - 1]f64 = undefined;
    // Empty rows with an out-of-vocab anchor fall back to in-vocab token 0 at chance 0.
    for ([_]u32{ @intCast(df.VOCAB), 0x7fffffff }) |bad_anchor| {
        select(&dr, 0, M, .{ .slot = 0, .pos = 0, .n = 0, .anchor = bad_anchor, .out = &out, .probs = &probs });
        for (out, probs) |tok, pr| {
            try std.testing.expectEqual(@as(u32, 0), tok);
            try std.testing.expectEqual(@as(f64, 0.0), pr);
        }
    }
    // Valid row-1 candidates with an out-of-vocab anchor use pred[0]; the greatest valid anchor keeps pred[VOCAB - 1].
    cand[1 * CW + 0] = 1;
    cand[1 * CW + TOPK + 0] = @bitCast(@as(f32, 2.0));
    cand[1 * CW + 1] = 2;
    cand[1 * CW + TOPK + 1] = @bitCast(@as(f32, 1.0));
    select(&dr, 0, M, .{ .slot = 0, .pos = 0, .n = 0, .anchor = @intCast(df.VOCAB - 1), .out = &out, .probs = &probs });
    try std.testing.expectEqual(@as(u32, 1), out[0]);
    for ([_]u32{ @intCast(df.VOCAB), 0x7fffffff }) |bad_anchor| {
        select(&dr, 0, M, .{ .slot = 0, .pos = 0, .n = 0, .anchor = bad_anchor, .out = &out, .probs = &probs });
        try std.testing.expectEqual(@as(u32, 2), out[0]);
        try std.testing.expect(probs[0] > 0.5);
        for (1..df.BLOCK - 1) |k| {
            try std.testing.expectEqual(@as(u32, 2), out[k]);
            try std.testing.expectEqual(@as(f64, 0.0), probs[k]);
        }
    }
}
