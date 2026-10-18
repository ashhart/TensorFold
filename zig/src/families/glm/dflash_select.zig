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
    var prev: usize = q.anchor;
    for (1..df.BLOCK) |t| {
        const r = i * df.BLOCK + t;
        var c: [TOPK]Cand = undefined;
        var n: usize = 0;
        for (0..dr.world) |rk| {
            const blk = dr.cand[rk * M * CW + r * CW ..][0..CW];
            for (0..TOPK) |j| {
                if (blk[j] < 0) continue;
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
