//! The head's multi-head attention on the device, in 64-row query blocks: FP32 scores, softmax and P V, one rounding.
const std = @import("std");
const js = @import("core").joint_schema;
const Stream = @import("../stream.zig").Stream;
const Head = @import("head.zig").Head;
const plan = @import("core").joint_schema.plan;

/// out = MHA(q rows, kv rows): BF16 Q, K, V; FP32 scores at 1/sqrt(head_dim), softmax and P V; one context rounding.
pub fn mha(h: *Head, s: Stream, m: js.tensors.AttentionOf(u64), q: u64, nq: usize, kv: u64, nk: usize, out: u64) !void {
    const W = h.cfg.width;
    const L = h.layout;
    const qp = h.at(L.qp);
    const kp = h.at(L.kp);
    const vp = h.at(L.vp);
    const vf = h.at(L.vf);
    const scores = h.at(L.scores);
    const ctx32 = h.at(L.ctx32);
    const ctx = h.at(L.ctx);
    // in_proj packs q, k and v by rows: each a W x W block of the weight and W of the bias
    for ([_]u64{ qp, kp, vp }, [_]u64{ q, kv, kv }, [_]usize{ nq, nk, nk }, 0..) |dst, src, rows, part|
        try h.linear(s, src, rows, m.in_weight + part * W * W * 2, m.in_bias + part * W * 2, dst, W, W);
    try h.fns.convert(s, true, vp, vf, nk * W);
    const heads: usize = h.cfg.heads;
    const dh = W / heads;
    const alpha: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(dh))));
    var b0: usize = 0;
    while (b0 < nq) : (b0 += plan.block) {
        const nb = @min(plan.block, nq - b0);
        const shape: @import("../blas.zig").Heads = .{ .keys = nk, .queries = nb, .heads = heads, .head_dim = dh };
        try h.blas.scores(qp + b0 * W * 2, kp, scores, shape, alpha);
        try h.fns.softmaxRows(s, scores, heads * nb, nk);
        try h.blas.context(scores, vf, ctx32, shape);
        try h.fns.convert(s, false, ctx32, ctx + b0 * W * 2, nb * W);
    }
    try h.linear(s, ctx, nq, m.out.weight, m.out.bias, out, W, W);
}

/// Linear, GELU, Linear over `rows` rows of x into out, each output rounded; the hidden rows in the scratch.
pub fn feedForward(h: *Head, s: Stream, up: js.tensors.LinearOf(u64), down: js.tensors.LinearOf(u64), x: u64, rows: usize, out: u64) !void {
    const W = h.cfg.width;
    const F = h.cfg.feedforward;
    const mid = h.at(h.layout.ffh);
    try h.linear(s, x, rows, up.weight, up.bias, mid, F, W);
    try h.fns.gelu(s, mid, rows * F);
    try h.linear(s, mid, rows, down.weight, down.bias, out, W, F);
}

/// EvidenceRoutingLayer on the options (`x`, n rows) over the memory rows: attention to LN_m(M), then the FFN.
pub fn evidence(h: *Head, s: Stream, e: js.tensors.EvidenceOf(u64), x: u64, n: usize, memory: u64, nm: usize) !void {
    const W = h.cfg.width;
    const normed = h.at(h.layout.normed);
    const mn = h.at(h.layout.mn);
    const att = h.at(h.layout.att);
    try h.norm(s, x, e.query_norm, normed, n, W);
    try h.norm(s, memory, e.memory_norm, mn, nm, W);
    try mha(h, s, e.attention, normed, n, mn, nm, att);
    try h.add(s, x, att, x, n * W);
    try h.norm(s, x, e.feedforward_norm, normed, n, W);
    try feedForward(h, s, e.up, e.down, normed, n, att);
    try h.add(s, x, att, x, n * W);
}

/// TransformerDecoderLayer, norm first, over the fields (`x`, n rows): self-attention, the unnormed memory, the FFN.
pub fn decoder(h: *Head, s: Stream, d: js.tensors.DecoderOf(u64), x: u64, n: usize, memory: u64, nm: usize) !void {
    const W = h.cfg.width;
    const normed = h.at(h.layout.normed);
    const att = h.at(h.layout.att);
    try h.norm(s, x, d.norm1, normed, n, W);
    try mha(h, s, d.self_attn, normed, n, normed, n, att);
    try h.add(s, x, att, x, n * W);
    try h.norm(s, x, d.norm2, normed, n, W);
    try mha(h, s, d.cross_attn, normed, n, memory, nm, att);
    try h.add(s, x, att, x, n * W);
    try h.norm(s, x, d.norm3, normed, n, W);
    try feedForward(h, s, d.up, d.down, normed, n, att);
    try h.add(s, x, att, x, n * W);
}

test "joint schema attention: query blocks of 64 cover every row once" {
    var covered: [130]u8 = @splat(0);
    var b0: usize = 0;
    while (b0 < covered.len) : (b0 += plan.block) for (b0..b0 + @min(plan.block, covered.len - b0)) |i| {
        covered[i] += 1;
    };
    for (covered) |c| try std.testing.expectEqual(@as(u8, 1), c);
}
