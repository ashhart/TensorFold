//! One decoder layer's decode step for up to 16 rows of one stream, and a stack of layers, encoded on Metal.
const std = @import("std");
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const pipelines = @import("pipelines.zig");
const attn = @import("attn.zig");
const moe = @import("moe.zig");

pub const Ctx = ctx_.Ctx;
pub const Ref = ctx_.Ref;
const cl = @import("calls.zig");

/// A layer's streams: x [rows][4][hidden] bf16 and the pre mix [rows][4] fp32 from the layer before, in and out.
pub const Io = struct { x: Ref, pre: Ref, x_out: Ref, pre_out: Ref };

/// The mHC mixer's constants of the config.
pub fn hc(x: *const Ctx) cl.Hc {
    return .{ .dim = x.c.hc_mult * x.c.hidden, .iters = x.c.hc_sinkhorn_iters, .eps = x.c.eps, .hc_eps = x.c.hc_eps, .alpha = x.c.hc_post_alpha };
}

/// A sublayer's input (h, hn, hq) beside its hc mixes' sums; returns the mixes for `mixFin` after the next barrier.
pub fn mixNorm(x: *const Ctx, e: mtl.ComputeEncoder, streams: Ref, w_hc: [3]Ref, mix: [3]Ref, pre: Ref, w_norm: Ref, out: [3]Ref) !pipelines.fused.Mix {
    const f = pipelines.fused;
    const cr = ctx_.cv;
    const h = hc(x);
    const m: f.Mix = .{
        .x = cr(f.Ref, streams),
        .w = .{ cr(f.Ref, w_hc[0]), cr(f.Ref, w_hc[1]), cr(f.Ref, w_hc[2]) },
        .out = .{ cr(f.Ref, mix[0]), cr(f.Ref, mix[1]), cr(f.Ref, mix[2]) },
        .part = cr(f.Ref, x.s.hcp),
        .h = .{ .dim = h.dim, .iters = h.iters, .eps = h.eps, .hc_eps = h.hc_eps, .alpha = h.alpha },
    };
    const n: f.Norm = .{ .x = cr(f.Ref, streams), .pre = cr(f.Ref, pre), .w = cr(f.Ref, w_norm), .h = cr(f.Ref, out[0]), .hn = cr(f.Ref, out[1]), .hq = cr(f.Ref, out[2]), .part = cr(f.Ref, x.s.nrp), .dim = x.c.hidden, .eps = x.c.eps };
    if (!wide(x.c.hidden)) {
        try f.mixNorm(&x.p.fused, e, m, n, x.rows);
        return m;
    }
    // the sums beside hc_pre, the norm and act_quant in one group a row (no wave between them)
    const o = pipelines.fold;
    f.mixSums(&x.p.fused, e, m, x.rows);
    try o.prenorm(&x.p.fold, e, cr(o.Ref, streams), cr(o.Ref, pre), cr(o.Ref, w_norm), cr(o.Ref, out[0]), cr(o.Ref, out[1]), cr(o.Ref, out[2]), x.rows, x.c.hidden, x.c.eps);
    return m;
}

/// Rows the one-launch prenorm takes: those fused.zig splits in two (wider than 2048), up to its group's row.
fn wide(dim: u32) bool {
    return dim > 2048 and dim <= pipelines.fold.max_dim and dim % 32 == 0;
}

/// The mixes `mixNorm` left (its sums are read here, so no other mix may run before this).
pub fn mixFin(x: *const Ctx, e: mtl.ComputeEncoder, m: pipelines.fused.Mix) void {
    pipelines.fused.mixFinish(&x.p.fused, e, m, x.rows);
}

/// hc pre, attention, hc post, hc pre, MoE, hc post (Engram is added before, by forward.zig); a barrier after each.
pub fn encode(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, io: Io) !void {
    const c = x.c;
    const s = x.s;
    const R = x.rows;
    if (R == 0 or R > pipelines.max_rows) return error.BadRows;
    var buf: [pipelines.max_rows]Ctx = undefined;
    for (x.parts(&buf)) |px| {
        if (px.p0 + px.rows > px.slot.limit or px.p0 + px.rows > px.tables.positions) return error.PastLimit;
    }
    const ma = try mixNorm(x, e, io.x, w.hc_attn, s.mix_a, io.pre, w.attn_norm, .{ s.h, s.hn, s.hq });
    e.barrier();
    mixFin(x, e, ma);
    try attn.encode(x, e, w);
    e.barrier();
    try ffn(x, e, w, io, .{ .experts = c.experts, .topk = c.topk, .route = x.p.route });
}

/// After the attention's barrier: hc post, hc pre, the MoE (of shape `sh`) and its hc post, then a barrier.
pub fn ffn(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, io: Io, sh: moe.Shape) !void {
    const s = x.s;
    const R = x.rows;
    const H = x.c.hidden;
    const j: moe.Join = .{ .res = s.x1, .post = s.mix_f[1], .comb = s.mix_f[2], .out = io.x_out };
    if (!wide(H)) {
        cl.hcpost(x.p, e, s.aout, io.x, s.mix_a[1], s.mix_a[2], s.x1, R, H);
        e.barrier();
        const mf = try mixNorm(x, e, s.x1, w.hc_ffn, .{ io.pre_out, s.mix_f[1], s.mix_f[2] }, s.mix_a[0], w.ffn_norm, .{ s.h2, s.hn2, s.hq2 });
        e.barrier();
        mixFin(x, e, mf);
        try moe.encodeWith(x, e, w, sh, j, null);
        return e.barrier();
    }
    // hc post folded into the norm's launch; the mixes' sums and their finish ride the MoE's first two waves
    const f = pipelines.fused;
    const o = pipelines.fold;
    const cr = ctx_.cv;
    const post: o.Post = .{ .branch = cr(o.Ref, s.aout), .post = cr(o.Ref, s.mix_a[1]), .comb = cr(o.Ref, s.mix_a[2]), .x1 = cr(o.Ref, s.x1) };
    try o.prenormPost(&x.p.fold, e, cr(o.Ref, io.x), cr(o.Ref, s.mix_a[0]), cr(o.Ref, w.ffn_norm), cr(o.Ref, s.h2), cr(o.Ref, s.hn2), cr(o.Ref, s.hq2), post, R, H, x.c.eps);
    e.barrier();
    const h = hc(x);
    const m: f.Mix = .{
        .x = cr(f.Ref, s.x1),
        .w = .{ cr(f.Ref, w.hc_ffn[0]), cr(f.Ref, w.hc_ffn[1]), cr(f.Ref, w.hc_ffn[2]) },
        .out = .{ cr(f.Ref, io.pre_out), cr(f.Ref, s.mix_f[1]), cr(f.Ref, s.mix_f[2]) },
        .part = cr(f.Ref, s.hcp),
        .h = .{ .dim = h.dim, .iters = h.iters, .eps = h.eps, .hc_eps = h.hc_eps, .alpha = h.alpha },
    };
    f.mixSums(&x.p.fused, e, m, R);
    try moe.encodeWith(x, e, w, sh, j, m);
    e.barrier();
}

/// Layers in order over ping-pong streams: layer i reads streams[i % 2] and writes streams[(i + 1) % 2].
pub fn encodeStack(x: *const Ctx, e: mtl.ComputeEncoder, layers: []const weights.Layer, streams: [2]Ref, pres: [2]Ref) !void {
    for (layers, 0..) |*w, i| {
        const a = i % 2;
        try encode(x, e, w, .{ .x = streams[a], .pre = pres[a], .x_out = streams[1 - a], .pre_out = pres[1 - a] });
    }
}
