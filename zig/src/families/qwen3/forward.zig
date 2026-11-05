//! A batch of independent prompts through every layer; each input's final-normed last row comes out.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const Kind = hip.ops.Kind;

/// Rows of one block of row-wise work, and of one attention query block.
pub const SPAN = 2048;

pub const Error = hip.ops.Error;

fn take(o: Ops, kind: Kind, n: usize) Error!Tensor {
    return .{ .ptr = try o.arena.take(n * kind.size()), .kind = kind };
}

fn at(t: Tensor, n: usize) Tensor {
    return .{ .ptr = t.ptr + n * t.kind.size(), .kind = t.kind };
}

/// head_dim ** -0.5 in double, as Python has it, passed as fp32.
pub fn scaleOf(head_dim: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(head_dim), -0.5));
}

/// Each input's final-normed last row into `out` (ends.len, hidden); inputs end at `ends`, start at position 0.
pub fn lastRows(ops: Ops, m: *const model.Model, ids: u64, ends: []const u32, out: Tensor) Error!void {
    var o = ops;
    o.prefill = true; // prefill's kernels at any row count: an input's rows have the same bits alone or batched
    const s = m.spec;
    const total: usize = if (ends.len == 0) return else ends[ends.len - 1];
    var longest: usize = 0;
    var begin: usize = 0;
    for (ends) |e| {
        if (e <= begin) return error.BadShape;
        longest = @max(longest, e - begin);
        begin = e;
    }
    const x = try take(o, m.act, total * s.hidden);
    var start: usize = 0;
    while (start < total) : (start += SPAN) {
        const rows = @min(SPAN, total - start);
        try o.embedRows(m.embed, ids + start * 4, rows, at(x, start * s.hidden));
    }
    for (m.layers) |*layer| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        try layerRows(o, m, layer, x, ends, total, longest, null);
    }
    for (ends, 0..) |e, i| try o.rms(at(x, (e - 1) * s.hidden), m.final_norm, at(out, i * s.hidden), 1, s.hidden, @floatCast(s.eps));
}

/// Device copies of a layer's projection inputs, row-major: input-normed, attention, post-normed and silu(gate) * up.
pub const Taps = struct { normed: u64, attn: u64, post: u64, act: u64 };

/// One input's rows through the embedding and layer 0, its projections' inputs copied into `taps` (the GEMM checks).
pub fn layerInputs(ops: Ops, m: *const model.Model, ids: u64, total: u32, taps: Taps) Error!void {
    var o = ops;
    o.prefill = true;
    if (total == 0 or total > SPAN) return error.BadShape;
    const x = try take(o, m.act, total * m.spec.hidden);
    try o.embedRows(m.embed, ids, total, x);
    try layerRows(o, m, &m.layers[0], x, &.{total}, total, total, taps);
}

/// One decoder layer over every row: norms, projections and MLP in SPAN blocks, attention input by input.
fn layerRows(o: Ops, m: *const model.Model, l: *const model.Layer, x: Tensor, ends: []const u32, total: usize, longest: usize, taps: ?Taps) Error!void {
    const s = m.spec;
    const hd = s.head_dim;
    const qw = s.heads * hd;
    const kvw = s.kv_heads * hd;
    const q = try take(o, m.act, total * qw);
    const k = try take(o, m.act, total * kvw);
    const v = try take(o, m.act, total * kvw);
    var start: usize = 0;
    while (start < total) : (start += SPAN) {
        const rows = @min(SPAN, total - start);
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const normed = try take(o, m.act, rows * s.hidden);
        try o.rms(at(x, start * s.hidden), l.input_norm, normed, rows, s.hidden, @floatCast(s.eps));
        if (taps) |tp| try copyRows(o, normed, at(.{ .ptr = tp.normed, .kind = m.act }, start * s.hidden), rows * s.hidden);
        try copyRows(o, try o.project(normed, l.q, rows, false), at(q, start * qw), rows * qw);
        try copyRows(o, try o.project(normed, l.k, rows, false), at(k, start * kvw), rows * kvw);
        try copyRows(o, try o.project(normed, l.v, rows, false), at(v, start * kvw), rows * kvw);
    }
    const attn = try take(o, m.act, total * qw);
    // one input's flat cache, reused by each input in turn: (kv_heads, longest, d)
    const cache: Ops.Cache = .{ .k = try o.arena.take(kvw * longest * m.act.size()), .v = try o.arena.take(kvw * longest * m.act.size()), .kind = m.act, .kv_heads = s.kv_heads, .total = longest, .d = hd };
    var a: usize = 0;
    for (ends) |b| {
        defer a = b;
        try attention(o, m, l, q, k, v, attn, cache, a, b);
    }
    if (taps) |tp| try copyRows(o, attn, .{ .ptr = tp.attn, .kind = m.act }, total * qw);
    start = 0;
    while (start < total) : (start += SPAN) {
        const rows = @min(SPAN, total - start);
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const xs = at(x, start * s.hidden);
        const y = try o.project(at(attn, start * qw), l.o, rows, false);
        const normed = try take(o, m.act, rows * s.hidden);
        // the residual add and the post-attention norm in one launch, with add-then-rms's bits
        try o.residualRms(xs, y, l.post_norm, normed, rows, s.hidden, @floatCast(s.eps));
        if (taps) |tp| try copyRows(o, normed, at(.{ .ptr = tp.post, .kind = m.act }, start * s.hidden), rows * s.hidden);
        const gate = try o.project(normed, l.gate, rows, false);
        const up = try o.project(normed, l.up, rows, false);
        const act = try take(o, m.act, rows * s.inter);
        try o.siluMul(gate, up, act, rows * s.inter);
        if (taps) |tp| try copyRows(o, act, at(.{ .ptr = tp.act, .kind = m.act }, start * s.inter), rows * s.inter);
        const z = try o.project(act, l.down, rows, false);
        try o.add(xs, z, xs, rows * s.hidden);
    }
}

/// Input rows [a, b): head norms, RoPE from position 0, its cache, causal attention, head-major rows back.
fn attention(o: Ops, m: *const model.Model, l: *const model.Layer, q: Tensor, k: Tensor, v: Tensor, attn: Tensor, c: Ops.Cache, a: usize, b: usize) Error!void {
    const s = m.spec;
    const hd = s.head_dim;
    const qw = s.heads * hd;
    const kvw = s.kv_heads * hd;
    const theta: f32 = @floatCast(s.theta);
    const len = b - a;
    var p: usize = 0;
    while (p < len) : (p += SPAN) {
        const rows = @min(SPAN, len - p);
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const qn = try take(o, m.act, rows * qw);
        try o.rms(at(q, (a + p) * qw), l.q_norm, qn, rows * s.heads, hd, @floatCast(s.eps));
        const kn = try take(o, m.act, rows * kvw);
        try o.rms(at(k, (a + p) * kvw), l.k_norm, kn, rows * s.kv_heads, hd, @floatCast(s.eps));
        // q rotated, rounded to the activation dtype, then widened: (heads, rows, d) fp32 for the kernel
        const q32 = try o.arena.of(f32, rows * qw);
        try o.ropePrefill(qn, .{ .ptr = q32, .kind = .f32 }, rows * hd, hd, rows, s.heads, hd, hd, p, theta);
        // k rotated into the cache (kv_heads, total, d) from slot p (the kernel writes row r at r * s_row); v as it is
        try o.ropePrefill(kn, .{ .ptr = c.k + p * hd * c.kind.size(), .kind = c.kind }, c.total * hd, hd, rows, s.kv_heads, hd, hd, p, theta);
        try o.kvWrite(at(v, (a + p) * kvw), c.v, rows, s.kv_heads, hd, c.total, p);
        const att = try o.arena.of(f32, rows * qw);
        try o.causalPrefill(q32, c, att, rows, p + rows, s.heads, scaleOf(hd), p);
        try o.headsRows(att, at(attn, (a + p) * qw), rows, s.heads, hd);
    }
}

/// `n` values of `src` into `dst` (same kind), as a one-row column copy.
fn copyRows(o: Ops, src: Tensor, dst: Tensor, n: usize) Error!void {
    try o.copyCols(src, n, 0, dst.ptr, 1, n);
}
