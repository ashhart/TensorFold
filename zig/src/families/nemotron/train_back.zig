//! Each kind of layer undone over a pass's sequences: its output's gradient, through its mixer, to its normed input.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const pre = @import("prefill.zig");
const pl = @import("prefill_launch.zig");
const ops = @import("train_ops.zig");
const adapters = @import("adapters.zig");
const Span = @import("living").train_mode.Span;
const Store = @import("train_prefix.zig").Store;

const Buffer = mtl.Buffer;
const At = pl.At;

/// The f32 gradients a backward works in: `rows` rows of a pass, `seq` of a sequence (its scan and attention).
pub const Bufs = struct {
    g: Buffer, // [rows, D]: the residual's gradient
    dx: Buffer, // [rows, D]: the gradient at a mixer's normed input
    dxa: Buffer, // [rows, max_rank]
    d_upr: Buffer, // [rows, F]
    du: Buffer, // [rows, F]
    dyp: Buffer, // [pairs, D], and 32 rows the experts' tiles read past the last pair
    dxp: Buffer, // [pairs, D]
    d_y1r: Buffer, // [pairs, W]
    du1: Buffer, // [pairs, W], likewise
    dyn: Buffer, // [rows, inner]
    ytot: Buffer, // [rows, inner]
    dytot: Buffer, // [rows, inner]
    dt: Buffer, // [rows, heads]
    ddt: Buffer, // [rows, heads]
    dact: Buffer, // [rows, conv]
    dproj: Buffer, // [rows, proj]
    ckpt: Buffer, // [heads, rows / 8 + 1, dh * n]: the scan's state every 8 rows
    dbp: Buffer, // [rows, heads, n]
    dcp: Buffer, // [rows, heads, n]
    q: Buffer, // bf16 [rows, heads * hd]
    k: Buffer, // bf16 [rows, kv * hd]
    v: Buffer, // bf16 [rows, kv * hd]
    d_o: Buffer, // [rows, heads * hd]
    dq: Buffer, // [rows, heads * hd]
    dk: Buffer, // [rows, kv * hd]
    dv: Buffer, // [rows, kv * hd]
    probs: Buffer, // [heads, rows, rows]
    dscores: Buffer, // [heads, rows, rows]

    const names = .{ "g", "dx", "dxa", "d_upr", "du", "dyp", "dxp", "d_y1r", "du1", "dyn", "ytot", "dytot", "dt", "ddt", "dact", "dproj", "ckpt", "dbp", "dcp", "q", "k", "v", "d_o", "dq", "dk", "dv", "probs", "dscores" };

    pub fn init(device: mtl.Device, c: cfg.Config, rows: usize, seq: usize) !Bufs {
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const d = c.hidden;
        const pairs = rows * c.top_k;
        const qw = c.heads * c.head_dim;
        const kw = c.kv_heads * c.head_dim;
        const state = c.mamba_head_dim * c.state;
        const bytes = [_]usize{
            rows * d * 4,                              rows * d * 4,                       rows * adapters.max_rank * 4,       rows * c.shared_width * 4,
            rows * c.shared_width * 4,                 (pairs + 32) * d * 4,               pairs * d * 4,                      pairs * c.expert_width * 4,
            (pairs + 32) * c.expert_width * 4,         rows * c.inner() * 4,               rows * c.inner() * 4,               rows * c.inner() * 4,
            rows * c.mamba_heads * 4,                  rows * c.mamba_heads * 4,           rows * c.convDim() * 4,             rows * c.projDim() * 4,
            c.mamba_heads * (seq / 8 + 1) * state * 4, rows * c.mamba_heads * c.state * 4, rows * c.mamba_heads * c.state * 4, rows * qw * 2,
            rows * kw * 2,                             rows * kw * 2,                      rows * qw * 4,                      rows * qw * 4,
            rows * kw * 4,                             rows * kw * 4,                      c.heads * seq * seq * 4,            c.heads * seq * seq * 4,
            16 * 16 * 4,
        };
        var b: Bufs = undefined;
        var made: usize = 0;
        errdefer inline for (names, 0..) |n, i| {
            if (i < made) @field(b, n).deinit();
        };
        inline for (names, 0..) |n, i| {
            @field(b, n) = try device.buffer(@max(bytes[i], 16), opts);
            made += 1;
        }
        return b;
    }

    pub fn deinit(b: *Bufs) void {
        inline for (names) |n| @field(b, n).deinit();
    }
};

/// What a layer's backward reads: its forward's scratch, the model, its change and where the open block starts.
pub const Layer = struct {
    o: ops.Ops,
    c: cfg.Config,
    s: *const pre.Scratch,
    p: *const pre.Prefill,
    w: *const wts.Weights,
    ad: wts.Adapter,
    gb: Buffer,
    first: usize,
    rows: usize,
    spans: []const Span, // the pass's sequences: each its own scan, conv and attention
    store: *const Store, // their kept prefixes' states
};

fn raw(t: [3]@import("../../core/checkpoint_metal.zig").Tensor) [3]At {
    return .{ pre.tensorAt(t[0]), pre.tensorAt(t[1]), pre.tensorAt(t[2]) };
}

/// The MoE layer: the shared expert and its change, then the routed experts on their frozen routes.
pub fn moe(x: Layer, b: *Bufs, i: usize) void {
    const c = x.c;
    const s = x.s;
    const o = x.o;
    const rows = x.rows;
    const m = x.p.layers[i].moe;
    const w = x.w.layers[i].moe;
    const f = c.shared_width;
    o.product(b.g, m.down, b.d_upr, rows, c.hidden, f, false);
    if (w.slide) |r| o.restBack(b.g, r, b.d_upr, rows, c.hidden, f);
    if (x.ad.rank > 0) o.adapterBack(x.ad, x.gb, x.first, b.dxa, s.upr, b.g, b.d_upr, rows);
    o.relu2Back(At.of(s.up), b.d_upr, b.du, rows * f);
    o.product(b.du, m.up, b.dx, rows, f, c.hidden, false);
    const pairs = rows * c.top_k;
    o.pairsIn(s.order, s.wt, b.g, b.dyp, pairs, c.hidden, c.top_k);
    o.experts(b.dyp, raw(w.fc2), s.offsets, b.d_y1r, pairs, c.hidden, c.expert_width, c.experts);
    o.relu2Back(At.of(s.y1), b.d_y1r, b.du1, pairs * c.expert_width);
    o.experts(b.du1, raw(w.fc1), s.offsets, b.dxp, pairs, c.expert_width, c.hidden, c.experts);
    o.pairsOut(s.inv, b.dxp, b.dx, rows, c.hidden, c.top_k);
    o.routeBack(s.logits, s.ids, s.xr, b.g, pre.tensorAt(w.gate), b.dx, rows, c.experts, c.top_k, c.hidden, c.routed_scaling);
}

/// The Mamba-2 layer: out_proj and its change, the gated norm, the scan, dt, the conv, in_proj.
pub fn mamba(x: Layer, b: *Bufs, i: usize) void {
    const c = x.c;
    const s = x.s;
    const o = x.o;
    const rows = x.rows;
    const m = x.p.layers[i].mamba;
    const norm = pre.tensorAt(x.w.layers[i].mamba.norm);
    const shape: ops.Shape = .{ .rows = @intCast(rows), .heads = @intCast(c.mamba_heads), .dh = @intCast(c.mamba_head_dim), .groups = @intCast(c.groups), .n = @intCast(c.state), .inner = @intCast(c.inner()), .conv = @intCast(c.convDim()), .proj = @intCast(c.projDim()) };
    const h = c.mamba_heads;
    o.product(b.g, m.out_proj, b.dyn, rows, c.hidden, c.inner(), false);
    if (x.w.layers[i].mamba.slide) |r| o.restBack(b.g, r, b.dyn, rows, c.hidden, c.inner());
    if (x.ad.rank > 0) o.adapterBack(x.ad, x.gb, x.first, b.dxa, s.ya, b.g, b.dyn, rows);
    const xd = c.inner();
    const cd = c.convDim();
    const pd = c.projDim();
    o.mixer("tf_train_dt", &.{ At.of(s.proj), m.dt_bias, At.of(b.dt) }, shape, .{ h, rows, 1 }, .{ 64, 1, 1 });
    // each sequence's scan from its kept prefix's state (else zero), its checkpoints kept until its backward read them
    for (x.spans) |sp| {
        var one = shape;
        one.rows = @intCast(sp.rows);
        const r = sp.at;
        const init = if (sp.skip > 0) x.store.ssmAt(sp.slot, x.p.index[i]) else At.of(b.ckpt);
        o.go("tf_train_ssm_fwd");
        for ([_]At{ At.of(s.act).plus(r * cd * 2), At.of(b.dt).plus(r * h * 4), m.a, m.d, At.of(b.ytot).plus(r * xd * 4), At.of(b.ckpt) }, 0..) |t, k| o.e.buf(t.b, t.off, k);
        o.e.bytes(one, 6);
        o.e.buf(init.b, init.off, 7);
        o.e.bytes(@as(u32, @intFromBool(sp.skip > 0)), 8);
        o.e.run(.{ 1024 * h, 1, 1 }, .{ 1024, 1, 1 });
        o.mixer2("tf_train_gate_back", &.{ At.of(b.ytot).plus(r * xd * 4), At.of(s.proj).plus(r * pd * 2), norm, At.of(b.dyn).plus(r * xd * 4), At.of(b.dytot).plus(r * xd * 4), At.of(b.dproj).plus(r * pd * 4) }, one, c.eps, .{ 256 * c.groups, sp.rows, 1 }, .{ 256, 1, 1 });
        o.mixer("tf_train_ssm_back", &.{ At.of(s.act).plus(r * cd * 2), At.of(b.dt).plus(r * h * 4), m.a, m.d, At.of(b.dytot).plus(r * xd * 4), At.of(b.ckpt), At.of(b.dact).plus(r * cd * 4), At.of(b.dbp).plus(r * h * c.state * 4), At.of(b.dcp).plus(r * h * c.state * 4), At.of(b.ddt).plus(r * h * 4) }, one, .{ 1024 * h, 1, 1 }, .{ 1024, 1, 1 });
    }
    o.mixer("tf_train_ssm_bc", &.{ At.of(b.dbp), At.of(b.dcp), At.of(b.dact) }, shape, .{ c.state, c.groups, rows }, .{ 128, 1, 1 });
    for (x.spans) |sp| {
        var one = shape;
        one.rows = @intCast(sp.rows);
        const r = sp.at;
        o.mixer2("tf_train_conv_back", &.{ At.of(b.dact).plus(r * cd * 4), At.of(s.conv).plus(r * cd * 2), m.conv_w, m.conv_b, At.of(b.dproj).plus(r * pd * 4) }, one, @as(u32, @intCast(c.conv_kernel)), .{ cd, sp.rows, 1 }, .{ 256, 1, 1 });
    }
    o.mixer("tf_train_dt_back", &.{ At.of(s.proj), m.dt_bias, At.of(b.ddt), At.of(b.dproj) }, shape, .{ h, rows, 1 }, .{ 64, 1, 1 });
    o.product(b.dproj, m.in_proj, b.dx, rows, c.projDim(), c.hidden, false);
}

/// The attention layer: o_proj and its change, causal softmax attention with shared kv heads, q, k and v.
pub fn attention(x: Layer, b: *Bufs, i: usize) void {
    const c = x.c;
    const s = x.s;
    const o = x.o;
    const rows = x.rows;
    const a = x.p.layers[i].attention;
    const qw = c.heads * c.head_dim;
    const kw = c.kv_heads * c.head_dim;
    const l: pl.Launch = .{ .k = x.p.k, .e = o.e };
    l.qmm(At.of(s.x), a.q, At.of(b.q), At.of(s.parts), rows, qw, c.hidden);
    l.qmm(At.of(s.x), a.k, At.of(b.k), At.of(s.parts), rows, kw, c.hidden);
    l.qmm(At.of(s.x), a.v, At.of(b.v), At.of(s.parts), rows, kw, c.hidden);
    o.product(b.g, a.o, b.d_o, rows, c.hidden, qw, false);
    if (x.w.layers[i].attention.slide) |r| o.restBack(b.g, r, b.d_o, rows, c.hidden, qw);
    if (x.ad.rank > 0) o.adapterBack(x.ad, x.gb, x.first, b.dxa, s.ya, b.g, b.d_o, rows);
    // each sequence's causal attention undone on its own rows, over its kept prefix's keys and its own
    for (x.spans) |sp| {
        const heads: ops.Heads = .{ .rows = @intCast(sp.rows), .heads = @intCast(c.heads), .kv_heads = @intCast(c.kv_heads), .dim = @intCast(c.head_dim), .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.head_dim)))), .prefix = @intCast(sp.skip) };
        const r = sp.at;
        const q = At.of(b.q).plus(r * qw * 2);
        const d_o = At.of(b.d_o).plus(r * qw * 4);
        const pk = if (sp.skip > 0) x.store.keysAt(sp.slot, x.p.index[i], false) else At.of(b.k);
        const pv = if (sp.skip > 0) x.store.keysAt(sp.slot, x.p.index[i], true) else At.of(b.v);
        o.go("tf_train_attn_q");
        for ([_]At{ q, At.of(b.k).plus(r * kw * 2), At.of(b.v).plus(r * kw * 2), d_o, At.of(b.probs), At.of(b.dscores), At.of(b.dq).plus(r * qw * 4) }, 0..) |t, k| o.e.buf(t.b, t.off, k);
        o.e.bytes(heads, 7);
        o.e.buf(pk.b, pk.off, 8);
        o.e.buf(pv.b, pv.off, 9);
        o.e.run(.{ 256 * c.heads, sp.rows, 1 }, .{ 256, 1, 1 });
        o.mixer("tf_train_attn_kv", &.{ q, d_o, At.of(b.probs), At.of(b.dscores), At.of(b.dk).plus(r * kw * 4), At.of(b.dv).plus(r * kw * 4) }, heads, .{ c.head_dim * c.kv_heads, sp.rows, 1 }, .{ c.head_dim, 1, 1 });
    }
    o.product(b.dq, a.q, b.dx, rows, qw, c.hidden, false);
    o.product(b.dk, a.k, b.dx, rows, kw, c.hidden, true);
    o.product(b.dv, a.v, b.dx, rows, kw, c.hidden, true);
}
