//! A layer's CSA2 attention for one launch's rows, in model.py's order: q and window KV, compressor, indexer, output.
const std = @import("std");
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const scratch = @import("scratch.zig");
const pipelines = @import("pipelines.zig");
const indexer = @import("indexer.zig");

const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const Meta = scratch.Meta;
const g = pipelines.glue;
const cp = pipelines.compress;
const f = pipelines.fused;
const attn = pipelines.attn;
const cv = ctx_.cv;
const plus = ctx_.plus;
const whole = ctx_.whole;

/// hn and hq [rows][hidden] to aout [rows][hidden]: q, window key, compressor and indexer in barrier-separated waves.
pub fn encode(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const R = x.rows;
    const l = w.layer;
    const D = c.head_dim;
    const pos = x.meta(Meta.pos);
    const table = x.table(l);
    const ratio = c.compressRatio(l);
    const kv_src = ratio > 0 and c.isKvSource(l);
    const ix_src = ratio > 0 and c.isIndexSource(l);
    var buf: [pipelines.max_rows]Ctx = undefined;
    const segs = x.parts(&buf);
    // every reader of hq and hn: wq_a, wkv, the compressor's linears, the index weights
    try x.proj(e, s.hq, w.wq_a, s.qa, false);
    try x.proj(e, s.hq, w.wkv, s.kv, false);
    if (kv_src) compressIn(x, e, w, ratio);
    if (ix_src) g.mv(gp, e, s.hn, w.ix_wp.?, s.ixw, R, c.hidden, c.index_heads, false);
    e.barrier();
    // q_norm; the window key (kv_norm, RoPE, act_quant) into both mirror slots of the ring; the ratio-2 pool
    try f.normq(&x.p.fused, e, cv(f.Ref, s.qa), cv(f.Ref, w.q_norm), cv(f.Ref, s.qr), cv(f.Ref, s.qrq), R, c.q_lora, c.eps);
    for (segs) |*px| {
        const region = &px.slot.regions[l];
        const k: f.Kv = .{
            .kv = cv(f.Ref, px.row(s.kv, D * 2)),
            .w = cv(f.Ref, w.kv_norm),
            .kvn = cv(f.Ref, px.row(s.kvn, D * 2)),
            .kvr = cv(f.Ref, px.row(s.kvr, D * 2)),
            .kvq = cv(f.Ref, px.row(s.kvq, D * 2)),
            .ring = .{ .buf = region.window },
            .pos = cv(f.Ref, px.meta(Meta.pos)),
            .table = cv(f.Ref, table),
            .dest = cv(f.Ref, px.metaRow(Meta.wdest, 2)),
            .reps = 2,
        };
        try f.kv(&x.p.fused, e, k, px.rows, D, c.eps);
    }
    if (kv_src and ratio != 1) pool(x, e, w, segs);
    e.barrier();
    // wq_b and the index queries' wq_b on the q latent; the compressed latent's norm
    try x.proj(e, s.qrq, w.wq_b, s.q, false);
    if (ix_src) try x.proj(e, s.qrq, w.ix_wq_b.?, s.ixq, false);
    if (kv_src) g.rmsnorm(gp, e, s.lat, w.comp_norm.?, s.latn, R, D, c.eps);
    e.barrier();
    // RoPE on each head's last 64 dims, the index queries' RoPE; the latent's index key and its RoPE
    g.rope(gp, e, s.q, s.qrope, pos, table, R, c.heads, D);
    if (ix_src) g.rope(gp, e, s.ixq, s.ixqr, pos, table, R, c.index_heads, c.index_dim);
    const cpos = x.byRatio(Meta.cpos, if (ratio > 0) ratio else 1);
    if (kv_src) {
        g.mv(gp, e, s.latn, w.ix_wk.?, s.ik, R, D, c.index_dim, false);
        g.rope(gp, e, s.latn, s.latr, cpos, table, R, 1, D);
    }
    e.barrier();
    if (ix_src) {
        const qdim = c.index_heads * c.index_dim;
        cp.fp4pack(&x.p.compress, e, s.ixqr, s.ixqc, s.ixqs, R, .{ .blocks = qdim / 32, .in_stride = qdim, .code_row = qdim / 2, .scale_row = qdim / 32 });
    }
    if (kv_src) {
        g.rmsnorm(gp, e, s.ik, w.ix_k_norm.?, s.ikn, R, c.index_dim, c.eps);
        for (segs) |*px| {
            cp.fp4rt(&x.p.compress, e, px.row(s.latr, D * 2), whole(px.slot.regions[l].pool.?), px.byRatio(Meta.dest, ratio), px.rows, D);
        }
        e.barrier();
        g.rope(gp, e, s.ikn, s.ikr, cpos, table, R, 1, c.index_dim);
        e.barrier();
        for (segs) |*px| {
            const keys = whole(px.slot.regions[l].keys.?);
            const p: cp.Pack = .{ .blocks = c.index_dim / 32, .in_stride = c.index_dim, .code_row = 68, .scale_row = 68, .scale_off = c.index_dim / 2, .dest = px.byRatio(Meta.dest, ratio) };
            cp.fp4pack(&x.p.compress, e, px.row(s.ikr, c.index_dim * 2), keys, keys, px.rows, p);
        }
    }
    if (ix_src or kv_src) e.barrier();
    if (ix_src) try indexer.select(x, e, w, ratio);
    // sparse attention over the window and the selected compressed rows, each slot's rows over its own state
    for (segs) |*px| {
        const region = &px.slot.regions[l];
        const src = px.slot.source(c, l);
        const b: attn.Bind = .{
            .q = cv(attn.Ref, px.row(s.qrope, c.heads * D * 2)),
            .w = .{ .buf = region.window },
            .c = .{ .buf = if (src) |r| r.pool.? else region.window },
            .i = cv(attn.Ref, px.row(s.sel, attn.max_selected * 4)),
            .m = cv(attn.Ref, px.metaRow(if (ratio > 0) Meta.attn_comp else Meta.attn_swa, 8)),
            .sk = cv(attn.Ref, w.sink),
            .cs = cv(attn.Ref, table),
            .o = cv(attn.Ref, px.row(s.o, c.heads * D * 2)),
        };
        try attn.encodeDecode(&x.p.attn, e, b, .{ .rows = px.rows, .heads = c.heads, .out = .bf16 });
    }
    e.barrier();
    try output(x, e, w);
}

/// The KV source's compressor linears on hn (bf16 in the release): a bf16 latent at ratio 1, fp32 kv and gate above.
fn compressIn(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, ratio: u32) void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    if (ratio == 1) return g.mv(gp, e, s.hn, w.comp_wkv.?, s.lat, x.rows, c.hidden, c.head_dim, false);
    g.mv(gp, e, s.hn, w.comp_wkv.?, s.cf, x.rows, c.hidden, c.head_dim, true);
    g.mv(gp, e, s.hn, w.comp_wgate.?, s.cg, x.rows, c.hidden, c.head_dim, true);
}

/// The ratio-2 pool of each slot's rows over its carry into the latent, and the inputs kept to undo a draft row.
fn pool(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, segs: []const Ctx) void {
    const s = x.s;
    const gp = &x.p.glue;
    const D = x.c.head_dim;
    for (segs) |*px| {
        const region = &px.slot.regions[w.layer];
        const cf = px.row(s.cf, D * 4);
        const cg = px.row(s.cg, D * 4);
        cp.pool2(&x.p.compress, e, cf, cg, whole(region.carry.?), px.meta(Meta.pos), px.row(s.lat, D * 2), px.rows, D);
        // the inputs kept per row, so a rejected draft row's carry can be undone (kv.rewindCarry)
        if (region.inputs) |b| {
            g.scatter(gp, e, cf, whole(b), px.meta(Meta.iota), px.rows, D * 4, 1);
            g.scatter(gp, e, cg, plus(whole(b), pipelines.max_rows * D * 4), px.meta(Meta.iota), px.rows, D * 4, 1);
        }
    }
}

/// o [rows][heads][512]: act_quant per group, wo_a group by group (block-diagonal), act_quant, then wo_b.
pub fn output(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer) !void {
    const c = x.c;
    const s = x.s;
    const R = x.rows;
    const G = c.o_groups;
    const kg = c.heads / G * c.head_dim;
    const ol = c.o_lora;
    try f.actq(&x.p.fused, e, cv(f.Ref, s.o), cv(f.Ref, s.oq), g.Strides{ .k = kg, .nb = R, .na = G, .sa = kg, .sb = c.heads * c.head_dim, .ta = R * kg, .tb = kg });
    e.barrier();
    // wo_a's groups in one dispatch: group gi's input, weight rows, scales and output a fixed stride on
    const wg: weights.Fp8 = .{ .w = w.wo_a.w, .s = w.wo_a.s, .k = kg, .n = ol };
    try x.projBatch(e, s.oq, wg, s.oa, false, .{ .n = G, .xs = R * kg * 2, .ws = ol * kg, .ss = ol / 32 * (kg / 32), .ys = R * ol * 2 });
    e.barrier();
    try f.actq(&x.p.fused, e, cv(f.Ref, s.oa), cv(f.Ref, s.ob), g.Strides{ .k = ol, .nb = R, .na = G, .sa = R * ol, .sb = ol, .ta = ol, .tb = G * ol });
    e.barrier();
    try x.proj(e, s.ob, w.wo_b, s.aout, false);
}
