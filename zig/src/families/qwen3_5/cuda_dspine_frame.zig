//! DSpine's draft frame: block-causal layers each closed by the code bridge, then the exact-sender table per row.
const std = @import("std");
const cuda = @import("cuda");
const dr = @import("lanes").drafter;
const draft = @import("cuda_draft.zig");
const frame = @import("cuda_draft_frame.zig");
const bridge = @import("cuda_dspine_weights.zig");
const Config = @import("cuda_draft_config.zig").Config;

const lp_words = 43;
pub const Fns = struct { bridge_read: cuda.Function, bridge_code: cuda.Function, bridge_message: cuda.Function, bridge_write: cuda.Function, senders: cuda.Function, sender_message: cuda.Function, table: cuda.Function, transpose: cuda.Function };

/// The bridge's weights and kernels beside the drafter's.
pub const Spine = struct {
    fns: Fns,
    w: bridge.Bridge,
};

pub fn init(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, c: Config, mod: cuda.Module) !*Spine {
    const s = try gpa.create(Spine);
    errdefer gpa.destroy(s);
    inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(s.fns, name) = try mod.function("tf_q35d_" ++ name);
    s.w = try bridge.load(gpa, io, e, dir, c, s.fns.transpose);
    return s;
}

pub fn deinit(gpa: std.mem.Allocator, s: *Spine) void {
    s.w.deinit();
    gpa.destroy(s);
}

/// A frame's bridge rows: rms (f32), read codes, codes, gates, sender terms, messages, writes, head rows and logits.
const Bufs = struct { rms: u64, p: u64, code: u64, q: u64, sv: u64, msg: u64, wm: u64, hx: u64, logits: u64, sids: u64, table: u64, end: u64 };

fn bufs(c: Config, streams: usize, base: u64) Bufs {
    const B = c.block * streams;
    const rows = (c.block - 1) * streams;
    const P = rows * c.topk;
    const W = @max(B, P);
    var at = base;
    const sizes = [_]usize{ B * 4, B * c.code * 2, W * c.code * 2, B * c.rank * 2, W * c.rank * 4, W * c.rank * 2, P * c.hidden * 2, 2 * rows * c.hidden * 2, 2 * rows * c.vocab * 2, P * 4, P * c.topk * 4 };
    var out: [sizes.len + 1]u64 = undefined;
    for (sizes, 0..) |n, i| {
        out[i] = at;
        at = std.mem.alignForward(u64, at + n, 256);
    }
    out[sizes.len] = at;
    return .{ .rms = out[0], .p = out[1], .code = out[2], .q = out[3], .sv = out[4], .msg = out[5], .wm = out[6], .hx = out[7], .logits = out[8], .sids = out[9], .table = out[10], .end = out[11] };
}

/// Scratch bytes of a frame of `streams` (none for DFlash2).
pub fn bytes(c: Config, streams: usize) usize {
    return if (c.kind == .dspine) bufs(c, streams, 0).end else 0;
}

fn int(x: usize) c_int {
    return @intCast(x);
}

/// The block's layers: attention as DFlash2's (causal here), a plain residual MLP, and the bridge after each.
fn layers(d: anytype, s: *Spine, b: Bufs, B: usize, rows: usize) !void {
    const c = d.c;
    const e = d.e;
    const w = &s.w;
    const sqrt_c: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(c.code))));
    try e.embed(d.ids, d.h, B, 0);
    try d.norm(d.h, d.w.layers[0].input_norm, d.x, B);
    for (d.w.layers, 0..) |l, i| {
        try d.projMany(&.{ l.q, l.k, l.v }, d.x, &.{ d.qraw, d.kraw, d.vb }, B);
        try frame.rope(d, d.qraw, l.q_norm, d.q, B, c.heads);
        try frame.rope(d, d.kraw, l.k_norm, d.kb, B, c.kv_heads);
        try frame.attention(d, B, i);
        try d.proj(l.o, d.att, d.o, B);
        try e.run(e.fns.norm, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ d.o, d.h, l.post_norm, d.x, c.eps, int(c.hidden), @as(u64, 0), d.xs });
        try d.projManySums(&.{ l.gate, l.up }, d.x, &.{ d.gate, d.up }, B, false);
        try e.run(e.fns.mlp_act, .{ (B * c.intermediate + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ d.gate, d.up, d.act, @as(c_longlong, @intCast(B * c.intermediate)), @as(u64, 0), int(c.intermediate), d.xs });
        try d.projAgain(l.down, d.act, d.h2, B);
        try e.run(s.fns.bridge_read, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ d.h2, d.o, d.xin, d.x, b.rms, int(c.hidden), c.bridge_eps });
        try d.proj(w.read[i], d.x, b.p, B);
        try e.run(s.fns.bridge_code, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ b.p, w.codebook, d.ids, b.code, int(c.code), int(c.block), sqrt_c, c.bridge_eps });
        try d.projMany(&.{ w.gate, w.pv }, b.code, &.{ b.q, b.sv }, B);
        try e.run(s.fns.bridge_message, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ b.q, b.sv, w.gate_b, b.msg, int(c.rank), int(c.block) });
        try d.proj(w.write[i], b.msg, d.o, B);
        const last = i + 1 == c.layers;
        const next = if (last) d.w.final_norm else d.w.layers[i + 1].input_norm;
        const xn = if (last) b.hx else d.x;
        try e.run(s.fns.bridge_write, .{ B, 1, 1 }, .{ 256, 1, 1 }, .{ d.xin, d.o, b.rms, next, d.h, xn, if (last) b.hx else @as(u64, 0), int(c.hidden), int(c.block), int(rows), c.eps });
    }
}

/// The second pass: each head row's candidates scored after each of the row before's (its anchor for the first).
fn table(d: anytype, s: *Spine, b: Bufs, rows: usize, logits: u64, spanned: bool) !void {
    const c = d.c;
    const e = d.e;
    const w = &s.w;
    const P = rows * c.topk;
    const sp = frame.split(spanned);
    const sqrt_c: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(c.code))));
    try e.run(s.fns.senders, .{ rows, 1, 1 }, .{ 32, 1, 1 }, .{ d.words, d.ids, b.sids, int(c.topk), int(c.block), int(lp_words), sp[0], sp[1] });
    try e.run(s.fns.bridge_code, .{ P, 1, 1 }, .{ 256, 1, 1 }, .{ @as(u64, 0), w.codebook, b.sids, b.code, int(c.code), int(1), sqrt_c, c.bridge_eps });
    try d.proj(w.pv, b.code, b.sv, P);
    try e.run(s.fns.sender_message, .{ P, 1, 1 }, .{ 256, 1, 1 }, .{ b.q, b.sv, w.gate_b, b.msg, int(c.rank), int(c.topk), int(c.block) });
    try d.proj(w.write[c.layers - 1], b.msg, b.wm, P);
    const cols = frame.columns(spanned, c.vocab);
    try e.run(s.fns.table, .{ P, 1, 1 }, .{ 256, 1, 1 }, .{ d.xin, b.rms, b.msg, b.wm, w.tg, logits, d.words, b.table, int(c.hidden), int(c.rank), int(c.topk), int(c.block), int(rows), int(cols), int(lp_words), sp[0], sp[1], c.eps });
}

/// One DSpine frame for `items`: block, bridge, head over the drafted rows and final_norm * raw rows, table, selection.
pub fn forward(d: anytype, s: *Spine, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    const c = d.c;
    const e = d.e;
    const B = items.len * c.block;
    const rows = items.len * (c.block - 1);
    const b = bufs(c, d.frame_streams, d.ds);
    const st = d.staged.bytes;
    try frame.stage(c, items, contexts, st);
    for (contexts) |ctx| {
        ctx.held_n = 0;
        ctx.grown = 0;
    }
    errdefer cuda.completion.drain(e.d, e.stream, "a failed DSpine frame did not drain before its pinned staging was reused");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.ids, st.ptr, B * 4, e.stream.handle), "cuMemcpyHtoDAsync");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.segments, st.ptr + frame.segments_at, items.len * @sizeOf(frame.Segment), e.stream.handle), "cuMemcpyHtoDAsync");
    try layers(d, s, b, B, rows);
    try e.run(e.fns.sums, .{ 2 * rows, (c.hidden / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ b.hx, d.xs, int(c.hidden), @as(u64, 0) });
    const spanned = try frame.logitsOf(d, contexts, b.hx, b.logits, 2 * rows);
    const logits = spanned orelse b.logits;
    try e.draws.topkRows(e.stream, logits, frame.columns(spanned != null, c.vocab), @intCast(c.topk), d.parts, d.words, rows);
    try table(d, s, b, rows, logits, spanned != null);
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + frame.words_at, d.words, rows * lp_words * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + frame.hp_at, b.table, rows * c.topk * c.topk * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    for (0..rows) |r| {
        const words = std.mem.bytesAsSlice(u32, st[frame.words_at + r * lp_words * 4 ..][0 .. lp_words * 4]);
        for (words[2..][0..c.topk]) |*id| id.* = if (spanned != null) frame.columnId(id.*) else id.*;
    }
    try frame.publish(d, items, contexts);
}
