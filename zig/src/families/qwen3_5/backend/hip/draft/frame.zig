//! The drafter's launches (cuda_draft_frame.zig's design): an absorb's key ring rows, and one five-layer block frame.
const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const fwd = @import("../forward/forward.zig");
const config = @import("config.zig");
const weights = @import("weights.zig");
const context = @import("context.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const dops = hip.draft_ops;

/// The drafter's head width (qkRope rotates all of it).
const head = dops.head;

/// The scalars a frame or absorb uploads and a frame downloads, by byte offset in one pinned and one device buffer.
pub const Layout = struct {
    ids: usize, // u32 a frame row
    pos: usize, // i32 a frame row
    ks: usize, // i32 a draft row: the selector's top-k
    segs: usize, // a Segment a stream
    absorb: usize, // i32 an absorbed row's position, `window` rows
    top_ids: usize, // i32 (draft rows, topk), downloaded
    top_vals: usize, // 16-bit logits (draft rows, topk)
    hp: usize, // 16-bit selector rows (draft rows, rank)
    total: usize,

    pub fn of(c: config.Config, streams: usize) Layout {
        const rows = streams * c.block;
        const drafts = streams * (c.block - 1);
        const a = std.mem.alignForward;
        var l: Layout = undefined;
        l.ids = 0;
        l.pos = l.ids + 4 * rows;
        l.ks = l.pos + 4 * rows;
        l.segs = a(usize, l.ks + 4 * drafts, 8);
        l.absorb = l.segs + @sizeOf(context.Segment) * streams;
        l.top_ids = a(usize, l.absorb + 4 * c.window, 8);
        l.top_vals = l.top_ids + 4 * drafts * c.topk;
        l.hp = a(usize, l.top_vals + 2 * drafts * c.topk, 8);
        l.total = a(usize, l.hp + 2 * drafts * c.rank, 256);
        return l;
    }
};

/// What the launches read: the drafter, the target's model (embedding, head), the activation kind.
pub const Net = struct {
    c: config.Config,
    w: *const weights.Weights,
    m: *const view.Model,
    act: hip.ops.Kind,
};

/// A layer's key and value rings in a context's ring buffer.
pub fn rings(c: config.Config, ring: u64, layer: usize) [2]u64 {
    const half = c.ringRows() * c.kvWidth() * 2;
    const k = ring + layer * 2 * half;
    return .{ k, k + half };
}

/// What an absorb chunk reads: the target's tap ring, its rows, the chunk's first position and its rows' positions.
pub const Chunk = struct {
    taps: u64,
    tap_rows: usize,
    start: u64, // the gather's first ring row (rows index from it)
    rows: ?[]const u32, // a kept path's ring rows (start + rows[i]); null: start + i
    n: usize,
    put: u64, // the first absorbed position (ring slot and rope)
    pos: u64, // device i32 positions put + i
    ring: u64, // the context's ring
};

/// An absorb chunk: the taps through fc and the hidden norm, then every layer's keys (normed, rotated) and values.
pub fn absorb(o: Ops, net: Net, k: Chunk) !void {
    const c = net.c;
    const width = c.n_taps * c.hidden;
    const taps = try fwd.take(o, net.act, k.n * width);
    if (k.rows) |rs| {
        try dops.gatherRows(o, k.taps, k.tap_rows, k.start, rs, width, taps.ptr);
    } else try dops.gatherTaps(o, k.taps, k.tap_rows, k.start, k.n, width, taps.ptr);
    const th = try o.project(taps, net.w.fc, k.n, false);
    const thn = try fwd.take(o, net.act, k.n * c.hidden);
    try o.rms(th, net.w.hidden_norm, thn, k.n, c.hidden, c.eps);
    const kvw = c.kvWidth();
    for (net.w.layers, 0..) |l, i| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        var outs: [4]Tensor = undefined;
        const keys, const values = if (try o.projectGroup(thn, &.{ l.k, l.v }, k.n, &outs))
            .{ outs[0], outs[1] }
        else
            .{ try o.project(thn, l.k, k.n, false), try o.project(thn, l.v, k.n, false) };
        const k32 = try o.arena.of(f32, k.n * kvw);
        try o.qkRope(keys, kvw, head, l.k_norm, c.eps, k.n, c.kv_heads, head, head, c.theta, k.pos, k32, null, 0);
        const r = rings(c, k.ring, i);
        try dops.ringPut(o, k32, values, r[0], r[1], k.n, kvw, k.put, c.ringRows());
    }
}

/// The device addresses a frame leaves its drafts' candidates at.
pub const Out = struct { top_ids: u64, top_vals: u64, hp: u64 };

/// One block frame of `streams` streams over the uploaded scalars at `dev`: the draft rows' top-k and selector rows.
pub fn forward(o: Ops, net: Net, l: Layout, dev: u64, streams: usize) !Out {
    const c = net.c;
    const B = streams * c.block;
    const hd = c.hidden;
    const h = try fwd.take(o, net.act, B * hd);
    try o.embedRows(net.m.embed, dev + l.ids, B, h);
    const x = try fwd.take(o, net.act, B * hd);
    const h2 = try fwd.take(o, net.act, B * hd);
    const at: dops.Attention = .{
        .rows = B,
        .block_rows = c.block,
        .heads = c.heads,
        .kv_heads = c.kv_heads,
        .ring_rows = c.ringRows(),
        .window = c.window,
        .layer = 0,
    };
    for (net.w.layers, 0..) |w, i| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        try o.rms(h, w.input_norm, x, B, hd, c.eps);
        const dyn = try o.project(x, w.attn_conv, B, false);
        const xin = try fwd.take(o, net.act, B * hd);
        try dops.dynConv(o, x, dyn, w.attn_base, null, xin, B, hd, c.group, 0, c.block);
        var outs: [4]Tensor = undefined;
        const q, const k, const v = if (try o.projectGroup(xin, &.{ w.q, w.k, w.v }, B, &outs))
            .{ outs[0], outs[1], outs[2] }
        else
            .{
                try o.project(xin, w.q, B, false),
                try o.project(xin, w.k, B, false),
                try o.project(xin, w.v, B, false),
            };
        const q32 = try o.arena.of(f32, B * c.qWidth());
        const pos = dev + l.pos;
        try o.qkRope(q, c.qWidth(), head, w.q_norm, c.eps, B, c.heads, head, head, c.theta, pos, q32, null, 0);
        const k32 = try o.arena.of(f32, B * c.kvWidth());
        try o.qkRope(k, c.kvWidth(), head, w.k_norm, c.eps, B, c.kv_heads, head, head, c.theta, pos, k32, null, 0);
        const att = try fwd.take(o, net.act, B * c.qWidth());
        const parts = dops.Partials.at(try o.arena.take(dops.Partials.bytes(B, c.heads)), B, c.heads);
        var shape = at;
        shape.layer = i;
        // the ring holds absorbed rows only (block keys come from k32 and v); no mask: a row sees its whole block
        try dops.attention(o, q32, k32, v, null, dev + l.segs, shape, parts, att);
        const attn = try o.project(att, w.o, B, false);
        try dops.dynConv(o, attn, dyn, w.attn_base, h, h2, B, hd, c.group, 1, c.block);
        try o.rms(h2, w.post_norm, x, B, hd, c.eps);
        const mdyn = try o.project(x, w.mlp_conv, B, false);
        try dops.dynConv(o, x, mdyn, w.mlp_base, null, xin, B, hd, c.group, 0, c.block);
        const gate, const up = if (try o.projectGroup(xin, &.{ w.gate, w.up }, B, &outs))
            .{ outs[0], outs[1] }
        else
            .{ try o.project(xin, w.gate, B, false), try o.project(xin, w.up, B, false) };
        const act = try fwd.take(o, net.act, B * c.intermediate);
        try o.siluMul(gate, up, act, B * c.intermediate);
        const down = try o.project(act, w.down, B, false);
        try dops.dynConv(o, down, mdyn, w.mlp_base, h2, h, B, hd, c.group, 1, c.block);
    }
    // the draft rows: every stream's block but its pending row, final-normed, through the target's head
    const rows = streams * (c.block - 1);
    const masked = try fwd.take(o, net.act, rows * hd);
    try dops.maskRows(o, h, masked, streams, hd, c.block);
    try o.rms(masked, net.w.final_norm, x, rows, hd, c.eps);
    const logits = try o.project(x, net.m.head, rows, false);
    const top_ids = try o.arena.of(i32, rows * c.topk);
    const top_vals = try o.arena.of(u16, rows * c.topk);
    try o.topkRows(logits, rows, net.m.head.n, dev + l.ks, c.topk, top_ids, top_vals);
    const hp = try o.project(x, net.w.selector, rows, false);
    return .{ .top_ids = top_ids, .top_vals = top_vals, .hp = hp.ptr };
}

test "the scalars' layout keeps every region apart and aligned" {
    var c: config.Config = undefined;
    c.block = 8;
    c.window = 2048;
    c.topk = 16;
    c.rank = 256;
    const l = Layout.of(c, 3);
    try std.testing.expect(l.pos >= l.ids + 4 * 24 and l.ks >= l.pos + 4 * 24 and l.segs >= l.ks + 4 * 21);
    try std.testing.expect(l.absorb >= l.segs + 24 * 3 and l.top_ids >= l.absorb + 4 * 2048);
    try std.testing.expect(l.top_vals >= l.top_ids + 4 * 21 * 16 and l.hp >= l.top_vals + 2 * 21 * 16);
    try std.testing.expect(l.total >= l.hp + 2 * 21 * 256);
    for ([_]usize{ l.segs, l.top_ids, l.hp }) |o| try std.testing.expectEqual(@as(usize, 0), o % 8);
}
