//! attn.zig's slot-bound stages for rows of several slots, one dispatch a stage over the layer's slot table.
const std = @import("std");
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const scratch = @import("scratch.zig");
const pipelines = @import("pipelines.zig");
const segs = @import("segs.zig");

const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const Meta = scratch.Meta;
const cp = pipelines.compress;
const f = pipelines.fused;
const attn = pipelines.attn;
const cv = ctx_.cv;

/// The window key of every row (kv_norm, RoPE, act_quant) into both mirror slots of its slot's ring.
pub fn kv(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, tab: Ref) !void {
    const s = x.s;
    const k: f.Kv = .{
        .kv = cv(f.Ref, s.kv),
        .w = cv(f.Ref, w.kv_norm),
        .kvn = cv(f.Ref, s.kvn),
        .kvr = cv(f.Ref, s.kvr),
        .kvq = cv(f.Ref, s.kvq),
        .ring = cv(f.Ref, s.kv),
        .pos = cv(f.Ref, x.meta(Meta.pos)),
        .table = cv(f.Ref, x.table(w.layer)),
        .dest = cv(f.Ref, x.metaRow(Meta.wdest, 2)),
        .reps = 2,
    };
    try f.kvSegs(&x.p.fused, e, k, cv(f.Ref, tab), cv(f.Ref, segs.rowSlots(x)), x.rows, x.c.head_dim, x.c.eps);
}

/// The ratio-2 pool of each slot's rows over its carry into the latent, and the inputs kept to undo a draft row.
pub fn pool(x: *const Ctx, e: mtl.ComputeEncoder, tab: Ref) void {
    const s = x.s;
    cp.pool2Segs(&x.p.compress, e, s.cf, s.cg, x.meta(Meta.pos), s.lat, tab, @intCast(x.segs.len), pipelines.max_rows, x.c.head_dim);
}

/// The compressed latent's FP4 round trip into each slot's compressed rows at the ratio's publish rows.
pub fn roundTrip(x: *const Ctx, e: mtl.ComputeEncoder, ratio: u32, tab: Ref) void {
    cp.fp4rtSegs(&x.p.compress, e, x.s.latr, x.byRatio(Meta.dest, ratio), tab, segs.rowSlots(x), x.rows, x.c.head_dim);
}

/// The roped index keys packed into each slot's index keys at the ratio's publish rows.
pub fn pack(x: *const Ctx, e: mtl.ComputeEncoder, ratio: u32, tab: Ref) void {
    const d = x.c.index_dim;
    const p: cp.Pack = .{ .blocks = d / 32, .in_stride = d, .code_row = 68, .scale_row = 68, .scale_off = d / 2, .dest = x.byRatio(Meta.dest, ratio) };
    cp.fp4packSegs(&x.p.compress, e, x.s.ikr, tab, segs.rowSlots(x), x.rows, p);
}

/// Sparse attention of every row over its slot's window and selected compressed rows.
pub fn attend(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, ratio: u32, tab: Ref) !void {
    const c = x.c;
    const s = x.s;
    const b: attn.Bind = .{
        .q = cv(attn.Ref, s.qrope),
        .w = cv(attn.Ref, s.qrope),
        .c = cv(attn.Ref, s.qrope),
        .i = cv(attn.Ref, s.sel),
        .m = cv(attn.Ref, x.metaRow(if (ratio > 0) Meta.attn_comp else Meta.attn_swa, 8)),
        .sk = cv(attn.Ref, w.sink),
        .cs = cv(attn.Ref, x.table(w.layer)),
        .o = cv(attn.Ref, s.o),
    };
    var most: u32 = 0;
    for (x.segs) |sg| most = @max(most, sg.n);
    const call: attn.Call = .{ .rows = x.rows, .heads = c.heads, .out = .bf16 };
    try attn.encodeDecodeSegs(&x.p.attn, e, b, cv(attn.Ref, tab), cv(attn.Ref, segs.rowSlots(x)), call, most <= attn.prefetch_rows);
}
