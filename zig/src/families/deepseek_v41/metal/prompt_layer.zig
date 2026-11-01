//! One layer over a prompt chunk's rows: window keys from row k, the rest from row o, as layer.zig and attn.zig do.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const pipelines = @import("pipelines.zig");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const ps = @import("prompt_scratch.zig");

const Ref = ctx_.Ref;
const g = pipelines.glue;
const cl = @import("calls.zig");
const cp = pipelines.compress;
const attn = pipelines.attn;
const ix = pipelines.index;
const cand = pipelines.cand;
const ex = pipelines.experts;
const xm = pipelines.xmma;
const mma = pipelines.fp8_mma;
const cv = ctx_.cv;
const plus = ctx_.plus;
const whole = ctx_.whole;
const back = ps.back;

/// A chunk's shared state: row r of every buffer (scratch and streams) is position a + r, for positions [a, b).
pub const Px = struct {
    c: *const Config,
    p: *const pipelines.Pipelines,
    s: *const ps.Scratch,
    slot: *const kv.Slot,
    tables: *const rope.Tables,
    a: u32,
    b: u32,
    /// routed experts on the decode row kernels, expert_rows a launch (the reference for the grouped kernels)
    expert_loop: bool = false,
    /// a timing harness's cut: encode stops with `error.Stopped` once this stage is encoded
    stop: Stage = .none,

    /// Position `pos`'s row of a buffer of `bytes` a row.
    pub fn at(x: *const Px, r: Ref, pos: u32, bytes: usize) Ref {
        return plus(r, @as(usize, pos - x.a) * bytes);
    }

    /// Position `pos`'s record in the run starting at int `base` (a ratio's run: base + (ratio - 1) * rows).
    pub fn rec(x: *const Px, base: usize, pos: u32) Ref {
        return x.s.metaAt(base + pos - x.a);
    }

    pub fn byRatio(x: *const Px, base: usize, ratio: u32, pos: u32) Ref {
        return x.rec(base + (ratio - 1) * x.s.rows, pos);
    }

    pub fn table(x: *const Px, layer: u32) Ref {
        return .{ .buf = x.tables.of(if (x.c.compressRatio(layer) > 0) .compressed else .window) };
    }

    /// A dense FP8 projection by the prompt MMA tiles (each row's bits equal the decode row kernel's).
    fn proj(x: *const Px, e: mtl.ComputeEncoder, in: Ref, w: weights.Fp8, out: Ref, rows: u32) !void {
        try mma.encode(&x.p.mma, e, cv(mma.Ref, in), cv(mma.Ref, w.w), cv(mma.Ref, w.s), cv(mma.Ref, out), .{ .k = w.k, .n = w.n, .rows = rows, .out = .bf16 });
    }
};

/// The points a timing harness may cut a layer at; each is the end of the stage it names.
pub const Stage = enum { none, attn_in, kv, q, ix_proj, ix_score, ix_select, core, out, ffn_in, route, xe_plan, xe_up, xe_down, experts, shared };

fn cut(x: *const Px, st: Stage) !void {
    if (x.stop == st) return error.Stopped;
}

/// A layer's streams, row r at position a + r: x [4][hidden] bf16 and pre [4] fp32 in, and the same out.
pub const Io = struct { x: Ref, pre: Ref, x_out: Ref, pre_out: Ref };

/// Window keys of positions [k, b), everything else for [o, b) (k <= o); `compress`: the KV source's compressor too.
pub const Rows = struct { k: u32, o: u32, compress: bool = true };

fn hc(c: *const Config) cl.Hc {
    return .{ .dim = c.hc_mult * c.hidden, .iters = c.hc_sinkhorn_iters, .eps = c.eps, .hc_eps = c.hc_eps, .alpha = c.hc_post_alpha };
}

/// The attention input of rows [from, b): the layer's mixes, hc pre and the norm into hn.
fn attnInput(x: *const Px, e: mtl.ComputeEncoder, w: *const weights.Layer, io: Io, from: u32) void {
    const c = x.c;
    const s = x.s;
    const H = c.hidden;
    const n = x.b - from;
    const mix = [3]Ref{ x.at(s.mix_a[0], from, 16), x.at(s.mix_a[1], from, 16), x.at(s.mix_a[2], from, 64) };
    cl.hcmix(x.p, e, x.at(io.x, from, 8 * H), w.hc_attn, mix, n, hc(c));
    cl.hcpre(x.p, e, x.at(io.x, from, 8 * H), x.at(io.pre, from, 16), x.at(s.h, from, 2 * H), n, H);
    g.rmsnorm(&x.p.glue, e, x.at(s.h, from, 2 * H), w.attn_norm, x.at(s.hn, from, 2 * H), n, H, c.eps);
}

/// hc pre, attention, hc post, hc pre, MoE, hc post for rows [o, b); window keys for [k, b).
pub fn encode(x: *const Px, e: mtl.ComputeEncoder, w: *const weights.Layer, io: Io, rows: Rows) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const H = c.hidden;
    if (rows.k > rows.o or rows.o > x.b or rows.k < x.a or rows.k >= x.b) return error.BadRows;
    if (x.b > x.slot.limit or x.b > x.tables.positions) return error.PastLimit;
    attnInput(x, e, w, io, rows.k);
    try cut(x, .attn_in);
    try attention(x, e, w, rows);
    const o = rows.o;
    const n = x.b - o;
    if (n == 0) return;
    cl.hcpost(x.p, e, x.at(s.aout, o, 2 * H), x.at(io.x, o, 8 * H), x.at(s.mix_a[1], o, 16), x.at(s.mix_a[2], o, 64), x.at(s.x1, o, 8 * H), n, H);
    const mix = [3]Ref{ x.at(io.pre_out, o, 16), x.at(s.mix_f[1], o, 16), x.at(s.mix_f[2], o, 64) };
    cl.hcmix(x.p, e, x.at(s.x1, o, 8 * H), w.hc_ffn, mix, n, hc(c));
    cl.hcpre(x.p, e, x.at(s.x1, o, 8 * H), x.at(s.mix_a[0], o, 16), x.at(s.h2, o, 2 * H), n, H);
    g.rmsnorm(gp, e, x.at(s.h2, o, 2 * H), w.ffn_norm, x.at(s.hn2, o, 2 * H), n, H, c.eps);
    try cut(x, .ffn_in);
    try moe(x, e, w, o);
    cl.hcpost(x.p, e, x.at(s.mout, o, 2 * H), x.at(s.x1, o, 8 * H), x.at(s.mix_f[1], o, 16), x.at(s.mix_f[2], o, 64), x.at(io.x_out, o, 8 * H), n, H);
}

/// The KV source's publish for every chunk row: its attention input and compressor (pool rows and index keys).
pub fn publish(x: *const Px, e: mtl.ComputeEncoder, w: *const weights.Layer, io: Io) !void {
    const ratio = x.c.compressRatio(w.layer);
    if (!x.c.isKvSource(w.layer)) return error.NotKvSource;
    attnInput(x, e, w, io, x.a);
    try compress(x, e, w, ratio, x.a);
}

/// q for rows [o, b), window keys for [k, b) through the linear window, compressor, indexer, attention, output.
fn attention(x: *const Px, e: mtl.ComputeEncoder, w: *const weights.Layer, rows: Rows) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const l = w.layer;
    const D = c.head_dim;
    const H = c.hidden;
    const k = rows.k;
    const o = rows.o;
    const nk = x.b - k;
    const no = x.b - o;
    const table = x.table(l);
    const pos = x.s.rec.pos;
    try cl.actqRows(x.p, e, x.at(s.hn, k, 2 * H), x.at(s.hq, k, 2 * H), nk, H);
    // the window: the ring's positions before k, then the keys of [k, b), all at row back + t - a
    const region = &x.slot.regions[l];
    const nw = kv.windowSlots(c);
    const g0 = k -| back;
    if (k > g0) g.scatter(gp, e, plus(whole(region.window), @as(usize, g0 % nw) * D * 2), s.lin, s.metaAt(s.rec.iota + back + g0 - x.a), k - g0, D * 2, 1);
    try x.proj(e, x.at(s.hq, k, 2 * H), w.wkv, x.at(s.kv, k, 2 * D), nk);
    g.rmsnorm(gp, e, x.at(s.kv, k, 2 * D), w.kv_norm, x.at(s.kvn, k, 2 * D), nk, D, c.eps);
    g.rope(gp, e, x.at(s.kvn, k, 2 * D), x.at(s.kvr, k, 2 * D), x.rec(pos, k), table, nk, 1, D);
    const lin = plus(s.lin, @as(usize, back) * D * 2);
    try cl.actqRows(x.p, e, x.at(s.kvr, k, 2 * D), x.at(lin, k, 2 * D), nk, D);
    // the last window slots' worth of keys into both mirror slots of the ring
    const m = @min(nk, nw);
    g.scatter(gp, e, x.at(lin, x.b - m, 2 * D), whole(region.window), x.s.metaAt(s.rec.wdest + 2 * (x.b - m - x.a)), 2 * m, D * 2, 2);
    const ratio = c.compressRatio(l);
    if (ratio > 0 and c.isKvSource(l) and rows.compress) try compress(x, e, w, ratio, k);
    try cut(x, .kv);
    if (no == 0) return;
    // q: wq_a, q_norm, wq_b and RoPE on each head's last 64 dims
    try x.proj(e, x.at(s.hq, o, 2 * H), w.wq_a, x.at(s.qa, o, 2 * c.q_lora), no);
    try cl.normq(x.p, e, x.at(s.qa, o, 2 * c.q_lora), w.q_norm, x.at(s.qr, o, 2 * c.q_lora), x.at(s.qrq, o, 2 * c.q_lora), no, c.q_lora, c.eps);
    const QH = c.heads * D;
    try x.proj(e, x.at(s.qrq, o, 2 * c.q_lora), w.wq_b, x.at(s.q, o, 2 * QH), no);
    g.rope(gp, e, x.at(s.q, o, 2 * QH), x.at(s.qrope, o, 2 * QH), x.rec(pos, o), table, no, c.heads, D);
    try cut(x, .q);
    if (ratio > 0 and c.isIndexSource(l)) try indexer(x, e, w, ratio, o);
    const src = x.slot.source(c, l);
    const b: attn.Bind = .{
        .q = cv(attn.Ref, x.at(s.qrope, o, 2 * QH)),
        .w = cv(attn.Ref, s.lin),
        .c = .{ .buf = if (src) |r| r.pool.? else region.window },
        .i = cv(attn.Ref, x.at(s.sel, o, 4 * attn.max_selected)),
        .m = cv(attn.Ref, s.metaAt((if (ratio > 0) s.rec.comp else s.rec.swa) + 8 * (o - x.a))),
        .sk = cv(attn.Ref, w.sink),
        .cs = cv(attn.Ref, table),
        .o = cv(attn.Ref, x.at(s.o, o, 2 * QH)),
    };
    try attn.encodePrompt(&x.p.attn, e, b, .{ .rows = no, .heads = c.heads, .out = .bf16 });
    try cut(x, .core);
    try output(x, e, w, o);
}

/// The KV source's compressor over rows [from, b): pooled latents, the index keys and the FP4 pool rows.
fn compress(x: *const Px, e: mtl.ComputeEncoder, w: *const weights.Layer, ratio: u32, from: u32) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const n = x.b - from;
    const D = c.head_dim;
    const H = c.hidden;
    const region = &x.slot.regions[w.layer];
    const hn = x.at(s.hn, from, 2 * H);
    const lat = x.at(s.lat, from, 2 * D);
    if (ratio == 1) {
        g.mv(gp, e, hn, w.comp_wkv.?, lat, n, H, D, false);
    } else {
        g.mv(gp, e, hn, w.comp_wkv.?, x.at(s.cf, from, 4 * D), n, H, D, true);
        g.mv(gp, e, hn, w.comp_wgate.?, x.at(s.cg, from, 4 * D), n, H, D, true);
        cp.pool2(&x.p.compress, e, x.at(s.cf, from, 4 * D), x.at(s.cg, from, 4 * D), whole(region.carry.?), x.rec(s.rec.pos, from), lat, n, D);
    }
    const latn = x.at(s.latn, from, 2 * D);
    g.rmsnorm(gp, e, lat, w.comp_norm.?, latn, n, D, c.eps);
    const cpos = x.byRatio(s.rec.cpos, ratio, from);
    const dest = x.byRatio(s.rec.dest, ratio, from);
    const table = x.table(w.layer);
    const IXD = c.index_dim;
    g.mv(gp, e, latn, w.ix_wk.?, x.at(s.ik, from, 2 * IXD), n, D, IXD, false);
    g.rmsnorm(gp, e, x.at(s.ik, from, 2 * IXD), w.ix_k_norm.?, x.at(s.ikn, from, 2 * IXD), n, IXD, c.eps);
    g.rope(gp, e, x.at(s.ikn, from, 2 * IXD), x.at(s.ikr, from, 2 * IXD), cpos, table, n, 1, IXD);
    const keys = whole(region.keys.?);
    cp.fp4pack(&x.p.compress, e, x.at(s.ikr, from, 2 * IXD), keys, keys, n, .{ .blocks = IXD / 32, .in_stride = IXD, .code_row = 68, .scale_row = 68, .scale_off = IXD / 2, .dest = dest });
    g.rope(gp, e, latn, x.at(s.latr, from, 2 * D), cpos, table, n, 1, D);
    cp.fp4rt(&x.p.compress, e, x.at(s.latr, from, 2 * D), whole(region.pool.?), dest, n, D);
}

/// The selection of rows [o, b): FP4 queries and weights, then scores, candidates and the top-512 a few rows at a time.
fn indexer(x: *const Px, e: mtl.ComputeEncoder, w: *const weights.Layer, ratio: u32, o: u32) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const n = x.b - o;
    const IX = c.index_heads * c.index_dim;
    try x.proj(e, x.at(s.qrq, o, 2 * c.q_lora), w.ix_wq_b.?, x.at(s.ixq, o, 2 * IX), n);
    g.rope(gp, e, x.at(s.ixq, o, 2 * IX), x.at(s.ixqr, o, 2 * IX), x.rec(s.rec.pos, o), x.table(w.layer), n, c.index_heads, c.index_dim);
    cp.fp4pack(&x.p.compress, e, x.at(s.ixqr, o, 2 * IX), x.at(s.ixqc, o, IX / 2), x.at(s.ixqs, o, IX / 32), n, .{ .blocks = IX / 32, .in_stride = IX, .code_row = IX / 2, .scale_row = IX / 32 });
    g.mv(gp, e, x.at(s.hn, o, 2 * c.hidden), w.ix_wp.?, x.at(s.ixw, o, 2 * c.index_heads), n, c.hidden, c.index_heads, false);
    const src = x.slot.source(c, w.layer).?;
    const masked = c.usesCandidates(w.layer);
    try cut(x, .ix_proj);
    var t = o;
    while (t < x.b) : (t += s.ix_rows) {
        const rows = @min(s.ix_rows, x.b - t);
        const keys = (t + rows) / ratio;
        if (keys > s.keys) return error.PastLimit;
        const vis = x.byRatio(s.rec.vis, ratio, t);
        const cmask = x.at(s.cmask, t, s.blocks);
        if (keys > 0) {
            try ix.encode(&x.p.index, e, cv(ix.Ref, x.at(s.ixqc, t, IX / 2)), cv(ix.Ref, x.at(s.ixqs, t, IX / 32)), cv(ix.Ref, x.at(s.ixw, t, 2 * c.index_heads)), .{ .buf = src.keys.? }, cv(ix.Ref, s.scores), cv(ix.Ref, if (masked) cmask else s.scores), .{
                .rows = rows,
                .keys = keys,
                .start = t + 1,
                .ratio = ratio,
                .mask_stride = if (masked) s.blocks else 0,
            });
            if (c.isCandidateSource(w.layer)) cand.encode(x.p.cand, e, s.scores, vis, s.bscore, cmask, rows, keys, s.blocks, c.candidate_blocks);
        }
        if (x.stop != .ix_score) g.topk512(x.p.topk512, e, s.scores, vis, x.at(s.sel, t, 4 * attn.max_selected), rows, @max(keys, 1));
    }
    try cut(x, .ix_score);
    try cut(x, .ix_select);
}

/// o [rows][heads][512] of rows [o, b) to aout: act_quant per group, wo_a group by group, act_quant, then wo_b.
fn output(x: *const Px, e: mtl.ComputeEncoder, w: *const weights.Layer, o: u32) !void {
    const c = x.c;
    const s = x.s;
    const R = x.b - o;
    const G = c.o_groups;
    const kg = c.heads / G * c.head_dim;
    const ol = c.o_lora;
    try cl.actq(x.p, e, x.at(s.o, o, 2 * c.heads * c.head_dim), s.oq, .{ .k = kg, .nb = R, .na = G, .sa = kg, .sb = c.heads * c.head_dim, .ta = R * kg, .tb = kg });
    for (0..G) |gi| {
        const wg: weights.Fp8 = .{ .w = plus(w.wo_a.w, gi * ol * kg), .s = plus(w.wo_a.s, gi * ol / 32 * (kg / 32)), .k = kg, .n = ol };
        try x.proj(e, plus(s.oq, gi * R * kg * 2), wg, plus(s.oa, gi * R * ol * 2), R);
    }
    try cl.actq(x.p, e, s.oa, s.ob, .{ .k = ol, .nb = R, .na = G, .sa = R * ol, .sb = ol, .ta = ol, .tb = G * ol });
    try x.proj(e, s.ob, w.wo_b, x.at(s.aout, o, 2 * c.hidden), R);
    try cut(x, .out);
}

/// The MoE of rows [o, b): the route, the routed experts, the shared expert, the join.
fn moe(x: *const Px, e: mtl.ComputeEncoder, w: *const weights.Layer, o: u32) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const R = x.b - o;
    const H = c.hidden;
    const W = c.expert_width;
    const K = c.topk;
    const hn2 = x.at(s.hn2, o, 2 * H);
    g.mv(gp, e, hn2, w.gate_w, x.at(s.logits, o, 4 * c.experts), R, H, c.experts, true);
    g.route(x.p.route, e, .{ .logits = x.at(s.logits, o, 4 * c.experts), .bias = w.gate_b, .bias_vl = w.gate_bvl orelse w.gate_b, .image = x.at(s.image, o, 1), .picks = x.at(s.picks, o, 4 * K), .weights = x.at(s.wts, o, 4 * K) }, R, K, c.routed_scale);
    try cut(x, .route);
    ex.use(e, w.backing());
    const table: ex.Ref = .{ .buf = w.table };
    if (x.expert_loop) try expertsLoop(x, e, o, table) else try expertsMma(x, e, o, table);
    try cut(x, .experts);
    try cl.actqRows(x.p, e, hn2, x.at(s.hq2, o, 2 * H), R, H);
    try x.proj(e, x.at(s.hq2, o, 2 * H), w.shared[0], x.at(s.sg, o, 2 * W), R);
    try x.proj(e, x.at(s.hq2, o, 2 * H), w.shared[2], x.at(s.su, o, 2 * W), R);
    cl.swiglu(x.p, e, x.at(s.sg, o, 2 * W), x.at(s.su, o, 2 * W), x.at(s.sact, o, 2 * W), x.at(s.sactq, o, 2 * W), R * W, c.swiglu_limit);
    try x.proj(e, x.at(s.sactq, o, 2 * W), w.shared[1], x.at(s.sout, o, 2 * H), R);
    cl.join(x.p, e, x.at(s.yr, o, 4 * H), x.at(s.sout, o, 2 * H), x.at(s.mout, o, 2 * H), R * H);
    try cut(x, .shared);
}

/// Routed experts of rows [o, b) on the decode row kernels, ps.expert_rows a launch.
fn expertsLoop(x: *const Px, e: mtl.ComputeEncoder, o: u32, table: ex.Ref) !void {
    const c = x.c;
    const s = x.s;
    const H = c.hidden;
    const W = c.expert_width;
    const K = c.topk;
    var t = o;
    while (t < x.b) : (t += ps.expert_rows) {
        const call: ex.Call = .{ .rows = @min(ps.expert_rows, x.b - t), .k = H, .n = W, .experts = c.experts };
        try ex.check(call);
        const route: ex.Route = .{ .ids = cv(ex.Ref, x.at(s.picks, t, 4 * K)), .weights = cv(ex.Ref, x.at(s.wts, t, 4 * K)), .slots = cv(ex.Ref, s.plan[0]), .nslots = cv(ex.Ref, s.plan[1]), .pairs = cv(ex.Ref, s.plan[2]), .where = cv(ex.Ref, s.plan[3]) };
        ex.plan(&x.p.experts, e, route, call);
        ex.up(&x.p.experts, e, cv(ex.Ref, x.at(s.hn2, t, 2 * H)), table, route, cv(ex.Ref, s.act), call);
        ex.down(&x.p.experts, e, cv(ex.Ref, s.act), table, route, cv(ex.Ref, s.eout), call);
        ex.combine(&x.p.experts, e, cv(ex.Ref, s.eout), route, cv(ex.Ref, s.ones), cv(ex.Ref, x.at(s.yr, t, 4 * H)), true, call);
    }
}

/// Routed experts of rows [o, b) in one grouped MXFP4 pass: plan, tiles, act_quant, gate and up, down, combine.
fn expertsMma(x: *const Px, e: mtl.ComputeEncoder, o: u32, table: ex.Ref) !void {
    const c = x.c;
    const s = x.s;
    const H = c.hidden;
    const K = c.topk;
    const R = x.b - o;
    const call: xm.Call = .{ .rows = R, .k = H, .n = c.expert_width, .experts = c.experts };
    try xm.check(call);
    const route: ex.Route = .{ .ids = cv(ex.Ref, x.at(s.picks, o, 4 * K)), .weights = cv(ex.Ref, x.at(s.wts, o, 4 * K)), .slots = cv(ex.Ref, s.plan[0]), .nslots = cv(ex.Ref, s.plan[1]), .pairs = cv(ex.Ref, s.plan[2]), .where = cv(ex.Ref, s.plan[3]) };
    const dcall: ex.Call = .{ .rows = R, .k = H, .n = c.expert_width, .experts = c.experts };
    ex.plan(&x.p.experts, e, route, dcall);
    const xp = &x.p.xmma;
    xm.tiles(xp, e, route, cv(xm.Ref, s.xtiles));
    xm.quant(xp, e, cv(xm.Ref, x.at(s.hn2, o, 2 * H)), cv(xm.Ref, s.xcodes), cv(xm.Ref, s.xscales), call);
    try cut(x, .xe_plan);
    xm.up(xp, e, cv(xm.Ref, s.xcodes), cv(xm.Ref, s.xscales), table, route, cv(xm.Ref, s.xtiles), cv(xm.Ref, s.act), call);
    try cut(x, .xe_up);
    xm.down(xp, e, cv(xm.Ref, s.act), table, cv(xm.Ref, s.xtiles), cv(xm.Ref, s.eout), call);
    try cut(x, .xe_down);
    ex.combine(&x.p.experts, e, cv(ex.Ref, s.eout), route, cv(ex.Ref, s.ones), cv(ex.Ref, x.at(s.yr, o, 4 * H)), true, dcall);
}
