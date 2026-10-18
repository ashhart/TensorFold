//! Nemotron-H's forward on a HIP rank: a prompt's rows through every block by kind, the final norm, the MTP head.
const std = @import("std");
const hip = @import("hip");
const config = @import("../config.zig");
const model = @import("model.zig");
const pages = @import("pages.zig");
const mamba = @import("mamba.zig");
const moe = @import("moe.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const Kind = hip.ops.Kind;
const pops = hip.plan_ops;

pub const Error = hip.reduce.Error || mamba.Error || moe.Error || error{Cancelled};

pub fn traceLayer(trace: ?Trace, index: usize, x: Tensor, rows: usize) Error!void {
    if (trace) |t| t.layer(t.ctx, index, x, rows) catch |err| {
        return if (err == error.Cancelled) error.Cancelled else error.BadShape;
    };
}

/// A device buffer of `n` values of `kind` from the arena.
pub fn take(o: Ops, kind: Kind, n: usize) Error!Tensor {
    return .{ .ptr = try o.arena.take(n * kind.size()), .kind = kind };
}

/// `t` advanced by `n` values.
pub fn at(t: Tensor, n: usize) Tensor {
    return .{ .ptr = t.ptr + n * t.kind.size(), .kind = t.kind };
}

/// The attention scale, head_dim ** -0.5 in double, passed as fp32.
pub fn scaleOf(head_dim: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(head_dim), -0.5));
}

/// Where the blocks find a stream's state: the KV pool and the stream.
pub const Ctx = struct { pool: *const pages.Pool, stream: *pages.Stream, trace: ?Trace = null };

/// A check's view of the residual after each block (the layer, the rows, their count).
pub const Trace = struct { ctx: *anyopaque, layer: *const fn (*anyopaque, usize, Tensor, usize) anyerror!void };

/// Where attention keeps and reads keys: one stream's pages from `pos` (a prompt), or a round's plan (any streams).
pub const Where = union(enum) {
    paged: struct { c: Ops.Paged, pos: usize },
    plan: struct { p: pops.Plan, slot: usize, span: usize },
};

/// Attention on `rows` normed rows: q, k, v; k and v into the pages; the causal walk over them; o.
pub fn attention(
    o: Ops,
    act: Kind,
    d: model.Dims,
    w: model.Attention,
    where: Where,
    x: Tensor,
    rows: usize,
) Error!Tensor {
    const hd = d.head_dim;
    // q, k and v in one launch where a grouped tile takes them: each column's sum is the single product's
    var qkv: [3]Tensor = undefined;
    if (!try o.projectGroup(x, &.{ w.q, w.k, w.v }, rows, &qkv)) {
        for (&qkv, [_]model.Projection{ w.q, w.k, w.v }) |*t, p| t.* = try o.project(x, p, rows, false);
    }
    const q, const keys, const values = qkv;
    const n = rows * d.heads * hd;
    const q32 = try o.arena.of(f32, n);
    const att = try o.arena.of(f32, n);
    const flat = try take(o, act, n);
    switch (where) {
        .paged => |pg| {
            const c = pg.c;
            const pos = pg.pos;
            try o.pageWrite(keys.ptr, c.k, c.table, rows, d.kv_heads, hd, hd, d.kv_heads * hd, pos, c.count);
            try o.pageWrite(values.ptr, c.v, c.table, rows, d.kv_heads, hd, hd, d.kv_heads * hd, pos, c.count);
            // q (rows, heads, d) to (heads, rows, d) fp32: the prefill RoPE kernel with no rotary part is that copy
            try o.ropePrefill(q, .{ .ptr = q32, .kind = .f32 }, rows * hd, hd, rows, d.heads, hd, 0, pos, 1);
            try o.causalPaged(q32, c, att, rows, pos + rows, d.heads, scaleOf(hd), pos);
            // (heads, rows, d) fp32 back to (rows, heads, d) in act: the same copy with heads and rows swapped
            try o.ropePrefill(.{ .ptr = att, .kind = .f32 }, flat, d.heads * hd, hd, d.heads, rows, hd, 0, 0, 1);
        },
        .plan => |pl| {
            // each row's keys and values into its own stream's pages, then one query a row over them
            const k32 = try o.arena.of(f32, rows * d.kv_heads * hd);
            try o.cast(keys, .{ .ptr = k32, .kind = .f32 }, rows * d.kv_heads * hd);
            try pops.kvWrite(o, k32, values, pl.p, pl.slot, d.kv_heads, hd);
            try o.cast(q, .{ .ptr = q32, .kind = .f32 }, n);
            try pops.causal(o, q32, act, att, pl.p, pl.slot, d.heads, d.kv_heads, hd, pl.span, scaleOf(hd));
            try o.cast(.{ .ptr = att, .kind = .f32 }, flat, n);
        },
    }
    return o.project(flat, w.o, rows, w.o.partial);
}

/// Each block's place among its kind: attention layers' pool slots and Mamba layers' states.
pub fn slots(c: *const config.Config, out: []usize) []const usize {
    var counts: [3]usize = @splat(0);
    for (c.kinds[0..c.layers], out[0..c.layers]) |kind, *s| {
        s.* = counts[@backingInt(kind)];
        counts[@backingInt(kind)] += 1;
    }
    return out[0..c.layers];
}

/// A prompt's `rows` tokens (int32 `ids` on the device) at `pos` through every block: the final-normed rows.
pub fn run(
    ops: Ops,
    m: *const model.Model,
    place: []const usize,
    ctx: Ctx,
    ids: u64,
    rows: usize,
    pos: usize,
) Error!Tensor {
    var o = ops;
    // a prompt's rows run on the prefill kernels at any count, so their bits do not depend on where it is cut
    o.prefill = true;
    const d = m.dims;
    const x = try take(o, m.act, rows * d.hidden);
    try o.embedRows(m.embed, ids, rows, x);
    for (m.blocks, place, 0..) |b, slot, i| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const normed = try take(o, m.act, rows * d.hidden);
        try o.rms(x, b.norm, normed, rows, d.hidden, d.eps);
        const y = switch (b.mixer) {
            .mamba => |w| try mamba.mixer(o, d, m.act, w, ctx.stream.states[slot], normed, rows, null),
            .attention => |w| try attention(
                o,
                m.act,
                d,
                w,
                .{ .paged = .{ .c = ctx.pool.paged(slot, ctx.stream.table()), .pos = pos } },
                normed,
                rows,
            ),
            .moe => |w| try moe.layer(o, d, m.act, w, normed, rows),
        };
        try hip.reduce.residual(o, m.tp, x, y, rows * d.hidden);
        try traceLayer(ctx.trace, i, x, rows);
    }
    const out = try take(o, m.act, rows * d.hidden);
    try o.rms(x, m.norm_f, out, rows, d.hidden, d.eps);
    return out;
}

/// What the MTP head gives: the draft rows' logits, and its final-normed rows (a chained step's hidden input).
pub const Drafted = struct { logits: Tensor, normed: Tensor };

/// The MTP head over the next tokens' `ids` and the model's final-normed `hidden` rows: logits of the draft rows.
pub fn draft(o: Ops, m: *const model.Model, where: Where, ids: u64, hidden: Tensor, rows: usize) Error!Drafted {
    const h = m.mtp orelse return error.BadShape;
    const d = h.dims;
    const n = rows * d.hidden;
    const e = try take(o, m.act, n);
    try o.embedRows(m.embed, ids, rows, e);
    const en = try take(o, m.act, n);
    try o.rms(e, h.enorm, en, rows, d.hidden, d.eps);
    const hn = try take(o, m.act, n);
    try o.rms(hidden, h.hnorm, hn, rows, d.hidden, d.eps);
    // eh_proj over [enorm(e) | hnorm(h)] as its two halves along K, summed in fp32
    const ye = try o.project(en, h.fc_e, rows, true);
    const yh = try o.project(hn, h.fc_h, rows, true);
    try o.add(ye, yh, ye, n);
    const x = try take(o, m.act, n);
    try o.cast(ye, x, n);
    const normed = try take(o, m.act, n);
    try o.rms(x, h.attn_norm, normed, rows, d.hidden, d.eps);
    const y = try attention(o, m.act, d, h.attn, where, normed, rows);
    try hip.reduce.residual(o, null, x, y, n);
    try o.rms(x, h.moe_norm, normed, rows, d.hidden, d.eps);
    const z = try moe.layer(o, d, m.act, h.moe, normed, rows);
    try hip.reduce.residual(o, null, x, z, n);
    try o.rms(x, h.final_norm, normed, rows, d.hidden, d.eps);
    return .{ .logits = try o.project(normed, h.draft, rows, false), .normed = normed };
}

test "blocks take their places among their kind" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"), &why);
    var buf: [config.max_layers]usize = undefined;
    const p = slots(&c, &buf);
    // layers 0, 1, 2, 5 and 51: mamba 0, moe 0, mamba 1, the first attention layer, the last MoE (the 23rd)
    try std.testing.expectEqual(@as(usize, 0), p[0]);
    try std.testing.expectEqual(@as(usize, 0), p[1]);
    try std.testing.expectEqual(@as(usize, 1), p[2]);
    try std.testing.expectEqual(@as(usize, 0), p[5]);
    try std.testing.expectEqual(@as(usize, 22), p[51]);
}
