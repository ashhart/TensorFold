//! The head's attention layers on the host: torch MultiheadAttention, the evidence layers and the decoder layers.
const std = @import("std");
const tensors = @import("tensors.zig");
const ref = @import("reference.zig");
const Ctx = ref.Ctx;

/// MultiheadAttention, no mask; bf16 mode rounds Q, K, V and the context once, scores, softmax and P·V stay exact.
pub fn mha(c: Ctx, m: tensors.Attention, q: []const f64, nq: usize, kv: []const f64, nk: usize, heads: usize) ![]f64 {
    const W = m.in_weight.dim(1);
    const dh = W / heads;
    const qp = try ref.linearRows(c, q, nq, m.in_weight, m.in_bias, 0, W);
    const kp = try ref.linearRows(c, kv, nk, m.in_weight, m.in_bias, W, W);
    const vp = try ref.linearRows(c, kv, nk, m.in_weight, m.in_bias, 2 * W, W);
    const ctx = try c.a.alloc(f64, nq * W);
    const s = try c.a.alloc(f64, nk);
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(dh)));
    for (0..nq) |i| for (0..heads) |hh| {
        const qi = qp[i * W + hh * dh ..][0..dh];
        for (s, 0..) |*x, j| {
            var d: f64 = 0;
            for (qi, kp[j * W + hh * dh ..][0..dh]) |a, b| d += a * b;
            x.* = d * scale;
        }
        ref.softmax(s);
        const o = ctx[i * W + hh * dh ..][0..dh];
        @memset(o, 0);
        for (s, 0..) |p, j| for (o, vp[j * W + hh * dh ..][0..dh]) |*x, v| {
            x.* += p * v;
        };
        for (o) |*x| x.* = c.r(x.*);
    };
    return ref.linear(c, ctx, nq, m.out);
}

/// Linear, GELU, Linear, each output rounded.
fn feedForward(c: Ctx, up: tensors.Linear, down: tensors.Linear, x: []const f64, n: usize) ![]f64 {
    const mid = try ref.linear(c, x, n, up);
    ref.gelu(c, mid);
    return ref.linear(c, mid, n, down);
}

/// EvidenceRoutingLayer: the options attend to the normed memory, then a feed-forward step, both residual.
pub fn evidence(c: Ctx, e: *const tensors.Evidence, x: []f64, n: usize, memory: []const f64, nm: usize, heads: usize) !void {
    const qn = try ref.layerNorm(c, x, n, e.query_norm);
    const mn = try ref.layerNorm(c, memory, nm, e.memory_norm);
    ref.addInto(c, x, try mha(c, e.attention, qn, n, mn, nm, heads));
    ref.addInto(c, x, try feedForward(c, e.up, e.down, try ref.layerNorm(c, x, n, e.feedforward_norm), n));
}

/// TransformerDecoderLayer with norm_first: self-attention, attention to the unnormed memory, feed-forward.
pub fn decoder(c: Ctx, d: *const tensors.Decoder, x: []f64, n: usize, memory: []const f64, nm: usize, heads: usize) !void {
    const n1 = try ref.layerNorm(c, x, n, d.norm1);
    ref.addInto(c, x, try mha(c, d.self_attn, n1, n, n1, n, heads));
    ref.addInto(c, x, try mha(c, d.cross_attn, try ref.layerNorm(c, x, n, d.norm2), n, memory, nm, heads));
    ref.addInto(c, x, try feedForward(c, d.up, d.down, try ref.layerNorm(c, x, n, d.norm3), n));
}
