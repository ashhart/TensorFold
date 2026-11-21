//! Benchmark-only confidence on the collapsed DSpark residual and previous-token embedding.
const std = @import("std");
const c = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");

fn fromBF16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn roundBF16(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    return fromBF16(@truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16));
}

pub fn emit(s: *const st.Scratch, projection: wts.Tensor, embedding: wts.Tensor, pos: usize) void {
    const ids = s.draft_ids.slice(u32, c.draft_block + 1);
    const pre = s.pre[0].slice(f32, c.draft_block * c.hc);
    const residual = s.x.slice(u16, c.draft_block * c.hc * c.hidden);
    const w = projection.host(u16);
    const emb = embedding.host(u16);
    var scores: [c.draft_block]f64 = undefined;
    for (&scores, ids[0..c.draft_block], 0..) |*score, token, row| {
        const r = residual[row * c.hc * c.hidden ..][0 .. c.hc * c.hidden];
        var sum: f64 = 0;
        for (w[0..c.hidden], 0..) |weight, d| {
            var y: f32 = 0;
            for (0..c.hc) |stream| {
                y = @mulAdd(f32, pre[row * c.hc + stream], fromBF16(r[stream * c.hidden + d]), y);
            }
            sum += @as(f64, roundBF16(y)) * @as(f64, fromBF16(weight));
        }
        for (w[c.hidden..], emb[@as(usize, token) * c.markov_rank ..][0..c.markov_rank]) |weight, e| {
            sum += @as(f64, fromBF16(e)) * @as(f64, fromBF16(weight));
        }
        score.* = sum;
    }
    std.debug.print("confidence-probe pos={d} anchor={d} ids={any} scores={any}\n", .{ pos, ids[0], ids[1..], &scores });
}
