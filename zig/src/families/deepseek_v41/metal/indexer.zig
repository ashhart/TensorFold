//! An index source's selection for one launch: FP4 queries and weights, scores, candidate blocks, the top-512.
const std = @import("std");
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const scratch = @import("scratch.zig");
const pipelines = @import("pipelines.zig");
const segs_ = @import("segs.zig");

const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const Meta = scratch.Meta;
const g = pipelines.glue;
const cp = pipelines.compress;
const ix = pipelines.index;
const cand = pipelines.cand;
const cv = ctx_.cv;
const attn_max = pipelines.attn.max_selected;

/// FP4 queries and weights to sel [rows][512] (cmask at a candidate source): a slot table makes it a dispatch a stage.
pub fn select(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, ratio: u32, tab: ?Ref) !void {
    if (tab) |t| return selectAll(x, e, w, ratio, t);
    const c = x.c;
    const s = x.s;
    const l = w.layer;
    const qdim = c.index_heads * c.index_dim;
    var buf: [pipelines.max_rows]Ctx = undefined;
    for (x.parts(&buf)) |*px| {
        const keys = px.keys(ratio);
        if (keys > s.keys) return error.PastLimit;
        const vis = px.byRatio(Meta.vis, ratio);
        const scores = px.row(s.scores, s.keys * 4);
        const mask = px.row(s.cmask, s.blocks);
        if (keys > 0) {
            const src = px.slot.source(c, l).?;
            const masked = c.usesCandidates(l);
            try ix.encodeDecode(&x.p.index, e, cv(ix.Ref, px.row(s.ixqc, qdim / 2)), cv(ix.Ref, px.row(s.ixqs, qdim / 32)), cv(ix.Ref, px.row(s.ixw, c.index_heads * 2)), .{ .buf = src.keys.? }, cv(ix.Ref, scores), cv(ix.Ref, if (masked) mask else scores), .{
                .rows = px.rows,
                .keys = keys,
                .start = px.p0 + 1,
                .ratio = ratio,
                .mask_stride = if (masked) s.blocks else 0,
            });
            e.barrier();
            if (c.isCandidateSource(l)) {
                cand.encode(x.p.cand, e, scores, vis, px.row(s.bscore, s.blocks * 4), mask, px.rows, keys, s.blocks, c.candidate_blocks);
                e.barrier();
            }
        }
        g.topk512(x.p.topk512, e, scores, vis, px.row(s.sel, attn_max * 4), px.rows, @max(keys, 1));
    }
    e.barrier();
}

/// Every slot's rows at once, scores a row of s.keys: the scores, the candidate blocks, the top-512, barriers between.
fn selectAll(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, ratio: u32, tab: Ref) !void {
    const c = x.c;
    const s = x.s;
    const l = w.layer;
    var most: u32 = 0;
    for (x.segs) |sg| most = @max(most, (sg.p0 + sg.n) / ratio);
    if (most > s.keys) return error.PastLimit;
    const vis = x.byRatio(Meta.vis, ratio);
    const slots = segs_.rowSlots(x);
    if (most > 0) {
        const masked = c.usesCandidates(l);
        const call: ix.Call = .{ .rows = x.rows, .keys = most, .ratio = ratio, .mask_stride = if (masked) s.blocks else 0 };
        try ix.encodeDecodeSegs(&x.p.index, e, cv(ix.Ref, s.ixqc), cv(ix.Ref, s.ixqs), cv(ix.Ref, s.ixw), cv(ix.Ref, s.scores), cv(ix.Ref, if (masked) s.cmask else s.scores), cv(ix.Ref, tab), cv(ix.Ref, slots), s.keys, call);
        e.barrier();
        if (c.isCandidateSource(l)) {
            cand.encodeSegs(x.p.cand_segs, e, s.scores, vis, s.bscore, s.cmask, tab, slots, x.rows, s.keys, s.blocks, c.candidate_blocks);
            e.barrier();
        }
    }
    g.topk512(x.p.topk512, e, s.scores, vis, s.sel, x.rows, s.keys);
    e.barrier();
}
