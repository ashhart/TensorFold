//! One DFlash frame for all selected streams, with separate rings, positions, convolution boundaries and trees.
const std = @import("std");
const cuda = @import("cuda");
const dr = @import("lanes").drafter;
const draft = @import("cuda_draft.zig");
const tree = @import("cuda_draft_tree.zig");
pub const max_streams = @import("cuda_engine.zig").max_segs;
pub const max_rows = max_streams * 16;
pub const chunks = 9;
const lp_words = 43;
pub const Segment = extern struct { ring: u64, begin: i64, position: i64 };
pub const segments_at = max_rows * 4;
pub const words_at = segments_at + max_streams * @sizeOf(Segment);
pub const hp_at = words_at + max_rows * lp_words * 4;
pub fn stagedBytes(rank: usize) !usize {
    return std.math.add(usize, hp_at, try std.math.mul(usize, try std.math.mul(usize, max_rows, rank), 2));
}
pub fn stagedLen(rank: usize) usize {
    return stagedBytes(rank) catch unreachable;
}
pub fn segments(st: []u8) []Segment {
    return @alignCast(std.mem.bytesAsSlice(Segment, st[segments_at..words_at]));
}
pub fn stage(c: draft.Config, items: []const dr.Hold, contexts: []const *draft.Ctx, st: []u8) !void {
    if (items.len == 0 or items.len > max_streams or contexts.len != items.len or (c.block != 8 and c.block != 16) or st.len < try stagedBytes(c.rank) or @intFromPtr(st.ptr) % 8 != 0) return error.BadDraftFrame;
    for (items, contexts, 0..) |it, ctx, index| {
        if (it.position == 0 or it.position > std.math.maxInt(i64) - c.block or ctx.end != it.position - 1 or ctx.begin > ctx.end) return error.DraftContextBehind;
        // a chain drafts a row a depth; a grown tree's depth is its node budget (tree.nodes(), up to the window)
        const most = if (c.block == 16 and it.lanes == null) @max(c.block - 1, tree.nodes()) else c.block - 1;
        if (it.pending >= c.vocab or it.depth > most) return error.BadDraftFrame;
        for (contexts[0..index]) |prior| if (ctx == prior or ctx.ring.ptr == prior.ring.ptr) return error.OverlappingDraftContexts;
        if (it.lanes) |shape| {
            if (shape.parents.len > ctx.held.len or shape.parents.len != shape.depths.len or shape.parents.len != shape.ranks.len) return error.BadLanes;
            for (shape.parents, shape.depths, shape.ranks, 0..) |parent, depth, rank, row| {
                if (parent < -1 or parent >= @as(i32, @intCast(row)) or depth >= c.block - 1 or rank >= c.topk) return error.BadLanes;
            }
        }
    }
    const ids = std.mem.bytesAsSlice(u32, st[0 .. c.block * items.len * 4]);
    const sg = segments(st);
    for (items, contexts, 0..) |it, ctx, i| {
        ids[i * c.block] = it.pending;
        @memset(ids[i * c.block + 1 ..][0 .. c.block - 1], c.mask);
        sg[i] = .{ .ring = ctx.ring.ptr, .begin = @intCast(ctx.begin), .position = @intCast(ctx.end) };
    }
}
fn conv(d: anytype, x: u64, base: u64, res: u64, out: u64, rows: usize, branch: c_int) !void {
    try d.e.run(d.fns.dyn_conv_batch, .{ rows, (d.c.hidden + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ x, d.dyn, base, res, out, @as(c_int, @intCast(d.c.hidden)), @as(c_int, @intCast(d.c.group)), branch, @as(c_int, @intCast(d.c.block)) });
}
fn rope(d: anytype, x: u64, w: u64, out: u64, rows: usize, heads: usize) !void {
    try d.e.run(d.fns.norm_rope_batch, .{ rows * heads, 1, 1 }, .{ 32, 1, 1 }, .{ x, w, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(heads * 128)), d.segments, @as(c_int, @intCast(d.c.block)), d.c.eps, d.c.theta });
}
pub fn attention(d: anytype, rows: usize, layer: usize) !void {
    const c = d.c;
    try d.e.run(d.fns.attn_partial, .{ rows, c.heads, chunks }, .{ 128, 1, 1 }, .{ d.q, d.kb, d.vb, d.segments, @as(c_int, @intCast(c.block)), @as(c_int, @intCast(c.heads)), @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(c.ringRows())), @as(c_int, @intCast(c.window)), @as(c_int, @intCast(layer)), d.pm, d.pl, d.po });
    try d.e.run(d.fns.attn_merge, .{ rows, c.heads, 1 }, .{ 32, 1, 1 }, .{ d.pm, d.pl, d.po, @as(c_int, @intCast(c.heads)), d.att });
}
pub fn publish(d: anytype, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    defer d.selection_base = 0;
    for (items, contexts, 0..) |it, ctx, i| {
        d.selection_base = i * (d.c.block - 1);
        try d.selectHold(ctx, it);
    }
}
pub fn forward(d: anytype, items: []const dr.Hold, contexts: []const *draft.Ctx) !void {
    const c = d.c;
    const e = d.e;
    const B = items.len * c.block;
    const st = d.staged.bytes;
    try stage(c, items, contexts, st);
    for (contexts) |ctx| {
        ctx.held_n = 0;
        ctx.grown = 0;
    }
    errdefer cuda.completion.drain(e.d, e.stream, "a failed DFlash frame did not drain before its pinned staging was reused");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.ids, st.ptr, B * 4, e.stream.handle), "cuMemcpyHtoDAsync");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(d.segments, st.ptr + segments_at, items.len * @sizeOf(Segment), e.stream.handle), "cuMemcpyHtoDAsync");
    try e.embed(d.ids, d.h, B, 0);
    for (d.w.layers, 0..) |l, i| {
        try d.norm(d.h, l.input_norm, d.x, B);
        try d.projAgain(l.attn_conv, d.x, d.dyn, B);
        try conv(d, d.x, l.attn_base, 0, d.xin, B, 0);
        try d.projMany(&.{ l.q, l.k, l.v }, d.xin, &.{ d.qraw, d.kraw, d.vb }, B);
        try rope(d, d.qraw, l.q_norm, d.q, B, c.heads);
        try rope(d, d.kraw, l.k_norm, d.kb, B, c.kv_heads);
        try attention(d, B, i);
        try d.proj(l.o, d.att, d.o, B);
        try conv(d, d.o, l.attn_base, d.h, d.h2, B, 1);
        try d.norm(d.h2, l.post_norm, d.x, B);
        try d.projAgain(l.mlp_conv, d.x, d.dyn, B);
        try conv(d, d.x, l.mlp_base, 0, d.xin, B, 0);
        try d.proj(l.gate, d.xin, d.gate, B);
        try d.projAgain(l.up, d.xin, d.up, B);
        try e.run(e.fns.mlp_act, .{ (B * c.intermediate + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, .{ d.gate, d.up, d.act, @as(c_longlong, @intCast(B * c.intermediate)), @as(u64, 0), @as(c_int, @intCast(c.intermediate)) });
        try d.proj(l.down, d.act, d.o, B);
        try conv(d, d.o, l.mlp_base, d.h2, d.h, B, 1);
    }
    const rows = items.len * (c.block - 1);
    var masked = d.h + c.hidden * 2;
    if (items.len > 1) {
        try e.run(d.fns.mask_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.h, d.h2, @as(c_int, @intCast(c.hidden)), @as(c_int, @intCast(c.block)) });
        masked = d.h2;
    }
    try d.norm(masked, d.w.final_norm, d.x, rows);
    try draft.precision.project(e, e.head, d.x, d.xs, d.logits, rows, .proposal, draft.precision.dflash_projection_prompt_cutoff);
    try e.draws.topkRows(e.stream, d.logits, c.vocab, @intCast(c.topk), d.parts, d.words, rows);
    try d.proj(d.w.selector, d.x, d.hp, rows);
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + words_at, d.words, rows * lp_words * 4, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + hp_at, d.hp, rows * c.rank * 2, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    try publish(d, items, contexts);
}
