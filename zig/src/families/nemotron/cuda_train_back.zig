//! The CUDA trainer's backward (train.zig's on Metal): the loss back through the head and every layer to each change.
const std = @import("std");
const train = @import("cuda_train.zig");
const ops = @import("cuda_train_ops.zig");
const sites_mod = @import("cuda_sites.zig");
const Trainer = train.Trainer;
const Raw = train.Raw;
const Mat = train.Mat;

/// y [rows, m.k] (+)= x [rows, m.n] (fp32) times the matrix: its input's gradient from its output's.
fn product(t: *Trainer, x: u64, m: Mat, y: u64, rows: usize, add: bool) !void {
    const k = t.ops_();
    try k.narrow(x, t.xb, rows * m.n);
    const right = if (m.bf16) m.w else blk: {
        try k.dequant(.{ m.w, m.s, m.b }, t.wt, m.n * m.k);
        break :blk t.wt;
    };
    try k.gemm(t.xb, right, y, rows, m.k, m.n, m.n, m.k, m.k, add);
}

/// The open block's gradient from its site's input x (rows `stride` apart) and output gradient gy; dx gains all.
fn adapterBack(t: *Trainer, l: usize, x: u64, stride: usize, gy: u64, dx: u64, rows: usize) !void {
    const s = &t.sites;
    if (s.rank == 0) return;
    const site = &s.list[l];
    const k = t.ops_();
    const first = if (s.rank >= sites_mod.block) s.first() else 0;
    try k.loraIn(x, stride, site.a.dev, s.xa.dev, rows, site.in, s.rank, s.xn.dev);
    try k.gate(s.xa.dev, s.xn.dev, site.tau.dev, s.gates.dev, rows, s.rank, sites_mod.unit, &t.route, s.latch.dev, s.score.dev);
    try k.loraDb(s.xa.dev, gy, site.gb.dev, rows, site.out, s.rank, first, sites_mod.scale);
    try k.loraDxa(gy, site.b.dev, t.dxa, rows, site.out, s.rank, sites_mod.scale, s.gates.dev);
    try k.loraDx(t.dxa, site.a.dev, dx, rows, site.in, s.rank);
}

/// The loss back through the head and every layer to each change, Adam after when `adam`; a target step (`target`)
/// keeps the gradient at each nudged row of the residuals its layers left, and stops below the lowest of them.
pub fn backward(t: *Trainer, rows: usize, adam: bool, target: bool) !void {
    const c = t.c;
    const e = t.e;
    const k = t.ops_();
    const D: usize = c.hidden;
    try k.gemm(t.logits, t.head_t, t.dx, rows, D, c.vocab, c.vocab, D, D, false);
    try k.zero(t.g, rows * D);
    try k.rmsBack(t.savedAt(c.layers), e.w.norm_f, t.dx, t.g, rows, D, c.eps);
    var l = c.layers;
    while (l > 0) {
        l -= 1;
        switch (t.raw[l]) {
            .moe => |m| try moe(t, l, m, rows),
            .mamba => |m| try mamba(t, l, m, rows),
            .attention => |a| try attention(t, l, a, rows),
        }
        try k.rmsBack(t.savedAt(l), e.w.blocks[l].norm, t.dx, t.g, rows, D, c.eps);
        if (target and l > 0) {
            const n = t.nudge;
            for (n.layers, n.rows, 0..) |mask, row, i| if (mask >> @intCast(l - 1) & 1 != 0) {
                try e.ops().copy(t.push.dev() + (i * c.layers + l - 1) * D * 4, t.g + row * D * 4, D * 4);
            };
            if (l - 1 == @ctz(n.layers[0] | n.layers[1] | n.layers[2])) return;
        }
    }
    if (adam) try t.sites.adam(k);
}

/// The MoE layer: the shared expert and its change (at its slot's weight), the routed experts, the routing weights.
fn moe(t: *Trainer, l: usize, m: @FieldType(Raw, "moe"), rows: usize) !void {
    const c = t.c;
    const k = t.ops_();
    const kp = t.kept[l];
    const D: usize = c.hidden;
    const S = c.slots();
    const W = c.expert_width;
    const F = c.shared_width;
    try k.scaleRows(t.g, kp.wts, S, c.top_k, t.gs, rows, D);
    try product(t, t.gs, m.down, t.d_upr, rows, false);
    try adapterBack(t, l, kp.upr, F, t.gs, t.d_upr, rows);
    try k.relu2Back(kp.upr, t.d_upr, t.du, rows * F);
    try product(t, t.du, m.up, t.dx, rows, false);
    const count = rows * c.top_k;
    try k.sortPairs(kp.ids, rows, S, c.top_k, c.experts, t.order, t.inv, t.offsets);
    try k.pairsIn(t.order, kp.wts, t.g, t.dyp, D, S, count);
    try k.expertProducts(t.dyp, m.fc2, t.offsets, t.part, t.d_y1r, count, D, W, c.experts);
    try k.copyRows(kp.act, W, t.order, t.y1s, W, W, count);
    try k.relu2Back(t.y1s, t.d_y1r, t.du1, count * W);
    try k.expertProducts(t.du1, m.fc1, t.offsets, t.part, t.dxp, count, W, D, c.experts);
    try k.pairsOut(t.inv, t.dxp, t.dx, D, c.top_k, rows);
    const router = t.e.w.blocks[l].moe.router;
    try k.routerLogits(kp.x, router, t.lg, rows, c.experts, D);
    try k.routeBack(t.lg, kp.ids, S, kp.ys, t.g, router, t.dx, rows, c.experts, c.top_k, D, c.routed_scaling);
}

/// The Mamba-2 layer: out_proj and its change, the gated norm, the scan, dt, the conv, in_proj.
fn mamba(t: *Trainer, l: usize, m: @FieldType(Raw, "mamba"), rows: usize) !void {
    const c = t.c;
    const k = t.ops_();
    const kp = t.kept[l];
    const mw = t.e.w.blocks[l].mamba;
    const s: ops.Shape = .{ .rows = @intCast(rows), .heads = @intCast(c.mamba_heads), .dh = @intCast(c.mamba_head_dim), .groups = @intCast(c.groups), .n = @intCast(c.state), .inner = @intCast(c.inner()), .conv = @intCast(c.convDim()), .proj = @intCast(c.projDim()) };
    try product(t, t.g, m.out_proj, t.dyn, rows, false);
    try adapterBack(t, l, kp.ya, c.inner(), t.g, t.dyn, rows);
    try k.convPre(kp.proj, mw.conv_w, mw.conv_b, t.conv_out, t.act_m, s, c.conv_kernel);
    try k.dt(kp.proj, mw.dt_bias, t.dt, s);
    try k.ssmFwd(t.act_m, t.dt, mw.a, mw.d, t.ytot, t.ckpt, s);
    try k.gateBack(t.ytot, kp.proj, mw.gnorm, t.dyn, t.dytot, t.dproj, s, c.eps);
    try k.ssmBack(.{ t.act_m, t.dt, mw.a, mw.d, t.dytot, t.ckpt, t.states, t.dact, t.dbp, t.dcp, t.ddt }, s);
    try k.ssmBc(t.dbp, t.dcp, t.dact, s);
    try k.convBack(t.dact, t.conv_out, mw.conv_w, mw.conv_b, t.dproj, s, c.conv_kernel);
    try k.dtBack(kp.proj, mw.dt_bias, t.ddt, t.dproj, s);
    try product(t, t.dproj, m.in_proj, t.dx, rows, false);
}

/// The attention layer: o_proj and its change, causal softmax attention with shared kv heads, q, k and v.
fn attention(t: *Trainer, l: usize, a: @FieldType(Raw, "attention"), rows: usize) !void {
    const c = t.c;
    const k = t.ops_();
    const kp = t.kept[l];
    const qw = c.heads * c.head_dim;
    try t.e.ops().prefillDense(kp.x, t.e.w.blocks[l].attn.qkv, t.qkv, rows);
    try product(t, t.g, a.o, t.d_o, rows, false);
    try adapterBack(t, l, kp.ya, qw, t.g, t.d_o, rows);
    const h: ops.Heads = .{ .rows = @intCast(rows), .heads = @intCast(c.heads), .kv_heads = @intCast(c.kv_heads), .dim = @intCast(c.head_dim), .qkv = @intCast(c.qkvDim()), .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.head_dim)))) };
    try k.attention(t.qkv, t.d_o, t.probs, t.dscores, t.dq, t.dk, t.dv, h);
    try product(t, t.dq, a.q, t.dx, rows, false);
    try product(t, t.dk, a.k, t.dx, rows, true);
    try product(t, t.dv, a.v, t.dx, rows, true);
}
