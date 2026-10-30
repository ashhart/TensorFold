//! The DSpark drafter's encoders: the main rows into the key rings, a draft block's stages, and its head.
const std = @import("std");
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const dw = @import("dspark_weights.zig");
const pipelines = @import("pipelines.zig");
const layer = @import("layer.zig");
const layer_attn = @import("attn.zig");
const moe = @import("moe.zig");
const forward = @import("forward.zig");

const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const g = pipelines.glue;
const cl = @import("calls.zig");
const attn = pipelines.attn;
const plus = ctx_.plus;

/// A key of rows (a block's or main rows): wkv, kv_norm, RoPE, act_quant of `hq` [rows][hidden], into pool rows `dest`.
pub fn keys(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, hq: Ref, ring: mtl.Buffer, pos: Ref, dest: Ref) !void {
    const c = x.c;
    const s = x.s;
    const R = x.rows;
    const D = c.head_dim;
    try x.proj(e, hq, w.wkv, s.kv, false);
    const f = pipelines.fused;
    const cv = ctx_.cv;
    const k: f.Kv = .{ .kv = cv(f.Ref, s.kv), .w = cv(f.Ref, w.kv_norm), .kvn = cv(f.Ref, s.kvn), .kvr = cv(f.Ref, s.kvr), .kvq = cv(f.Ref, s.kvq), .ring = .{ .buf = ring }, .pos = cv(f.Ref, pos), .table = cv(f.Ref, x.table(w.layer)), .dest = cv(f.Ref, dest), .reps = 1 };
    try cl.key(x.p, e, k, R, D, c.eps);
}

/// The main rows' shared part: act_quant (into `quant`), main_proj, main_norm and act_quant again, into s.hq.
pub fn mainRows(x: *const Ctx, e: mtl.ComputeEncoder, d: *const dw.Weights, taps: Ref, quant: Ref) !void {
    const c = x.c;
    const s = x.s;
    try cl.actqRows(x.p, e, taps, quant, x.rows, d.main_proj.k);
    try x.proj(e, quant, d.main_proj, s.h, false);
    try cl.normq(x.p, e, s.h, d.main_norm, s.hn, s.hq, x.rows, c.hidden, c.eps);
}

/// Main rows into every stage's ring: `mainRows`, then each stage's key of the result.
pub fn ingest(x: *const Ctx, e: mtl.ComputeEncoder, d: *const dw.Weights, taps: Ref, quant: Ref, rings: []const mtl.Buffer, pos: Ref, dest: Ref) !void {
    try mainRows(x, e, d, taps, quant);
    for (d.stages, rings) |*w, ring| try keys(x, e, w, x.s.hq, ring, pos, dest);
}

/// A draft launch's rows: their positions, and where their keys, attention records and selected keys are.
pub const Rows = struct {
    pos: Ref,
    dest: Ref,
    meta: Ref,
    sel: Ref,
};

/// A stage's attention over `hn` and `hq`: q and the block's keys, then each row over its slot's context and the block.
fn attention(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, ring: mtl.Buffer, rows: Rows) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const R = x.rows;
    const D = c.head_dim;
    const table = x.table(w.layer);
    try x.proj(e, s.hq, w.wq_a, s.qa, false);
    try cl.normq(x.p, e, s.qa, w.q_norm, s.qr, s.qrq, R, c.q_lora, c.eps);
    try x.proj(e, s.qrq, w.wq_b, s.q, false);
    g.rope(gp, e, s.q, s.qrope, rows.pos, table, R, c.heads, D);
    try keys(x, e, w, s.hq, ring, rows.pos, rows.dest);
    const b: attn.Bind = .{
        .q = ctx_.cv(attn.Ref, s.qrope),
        .w = .{ .buf = ring },
        .c = .{ .buf = ring },
        .i = ctx_.cv(attn.Ref, rows.sel),
        .m = ctx_.cv(attn.Ref, rows.meta),
        .sk = ctx_.cv(attn.Ref, w.sink),
        .cs = ctx_.cv(attn.Ref, table),
        .o = ctx_.cv(attn.Ref, s.o),
    };
    try attn.encodeDecode(&x.p.attn, e, b, .{ .rows = R, .heads = c.heads, .out = .bf16 });
    try layer_attn.output(x, e, w);
}

/// The route's shape of a stage: the drafter's own experts and picks.
pub fn shape(x: *const Ctx) !moe.Shape {
    return .{ .experts = x.c.dspark_experts, .topk = x.c.dspark_topk, .route = x.p.route_dspark orelse return error.NoDspark };
}

/// One stage as layer.encode runs a layer, with the draft attention and the drafter's MoE.
pub fn stage(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, io: layer.Io, ring: mtl.Buffer, rows: Rows) !void {
    const s = x.s;
    const ma = try layer.mixNorm(x, e, io.x, w.hc_attn, s.mix_a, io.pre, w.attn_norm, .{ s.h, s.hn, s.hq });
    e.barrier();
    layer.mixFin(x, e, ma);
    try attention(x, e, w, ring, rows);
    e.barrier();
    try layer.ffn(x, e, w, io, try shape(x));
}

/// The block's token ids into all four streams of streams[0] (the shared embedding), the pre mix one-hot.
pub fn embed(x: *const Ctx, e: mtl.ComputeEncoder, top: *const weights.Top, b: *const forward.Bufs) void {
    g.embed(&x.p.glue, e, top.embed, b.ids, b.streams[0], b.pres[0], x.rows, x.c.hidden);
}

/// The last stage's streams collapsed (hf, the confidence head's input), the drafter's norm, the shared head's logits.
pub fn head(x: *const Ctx, e: mtl.ComputeEncoder, top: *const weights.Top, d: *const dw.Weights, b: *const forward.Bufs) void {
    const c = x.c;
    const gp = &x.p.glue;
    const last = d.stages.len % 2;
    cl.hcpre(x.p, e, b.streams[last], b.pres[last], b.hf, x.rows, c.hidden);
    g.rmsnorm(gp, e, b.hf, d.norm, b.hfn, x.rows, c.hidden, c.eps);
    g.mv(gp, e, b.hfn, top.head, b.logits, x.rows, c.hidden, c.vocab, true);
}
