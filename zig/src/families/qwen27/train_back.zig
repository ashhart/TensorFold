//! Loss input gradients through the native dense MLP, GDN and attention; BF16 nodes use straight-through gradients.
const std = @import("std");
const mtl = @import("metal");
const to = @import("train_ops.zig");
const Ref = to.Ref;
const plus = to.plus;
const wts = @import("gpu_weights.zig");
const train = @import("train.zig");

pub const Scratch = struct {
    mlp: Ref,
    gu: Ref,
    mix: Ref,
    dp: Ref,
    dact: Ref,
    gy: Ref,
    work: Ref,
    dqparts: Ref,
    dkparts: Ref,
    dg: Ref,
    db: Ref,
    dout: Ref,
    dpq: Ref,
    dpk: Ref,
    dv: Ref,
    dq: Ref,
    dk: Ref,
    probability: Ref,
    dscores: Ref,

    pub fn init(t: anytype) !Scratch {
        const g = t.g;
        const R = train.pack_rows;
        const M = train.max_rows;
        const vi = g.gdnValueDim();
        const conv = g.gdnQkvDim();
        const qi = g.heads * g.head_dim;
        const kv = g.kvDim();
        return .{
            .mlp = try t.alloc(R * g.intermediate * 4),
            .gu = try t.alloc(R * 2 * g.intermediate * 4),
            .mix = try t.alloc(R * @max(vi, qi) * 4),
            .dp = try t.alloc(M * (conv + vi + 2 * g.v_heads) * 4),
            .dact = try t.alloc(M * conv * 4),
            .gy = try t.alloc(M * vi * 4),
            .work = try t.alloc(g.v_heads * 4 * 4 * 32 * 128 * 4),
            .dqparts = try t.alloc(M * g.v_heads * 4 * 128 * 4),
            .dkparts = try t.alloc(M * g.v_heads * 4 * 128 * 4),
            .dg = try t.alloc(M * g.v_heads * 4 * 4),
            .db = try t.alloc(M * g.v_heads * 4 * 4),
            .dout = try t.alloc(M * qi * 4),
            .dpq = try t.alloc(M * 2 * qi * 4),
            .dpk = try t.alloc(M * kv * 4),
            .dv = try t.alloc(M * kv * 4),
            .dq = try t.alloc(M * qi * 4),
            .dk = try t.alloc(M * kv * 4),
            .probability = try t.alloc(g.heads * M * M * 4),
            .dscores = try t.alloc(g.heads * M * M * 4),
        };
    }
};

pub fn run(t: *train.Trainer, o: to.Ops) !void {
    const g = t.g;
    const R = t.rows;
    const D = g.hidden;
    try @import("train_schedule.zig").headBack(t, o);
    try o.zero(t.grad, R * D);
    try o.rmsBack(t.final_h, wts.ref(t.m.weights.norm), t.dy, t.grad, R, D, g.eps);
    var l = g.layers;
    while (l > t.low) {
        l -= 1;
        for (t.seqs[0..t.nseq], t.spans[0..t.nseq], 0..) |sq, sp, si| {
            for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| {
                if (mask >> @intCast(l) & 1 == 0) continue;
                try o.copy(plus(t.grad, sp.row(row) * D * 4), plus(t.push, t.vec(si, i, l) * 4), D * 4);
            }
        }
        if (l == t.low) return;
        const kp = t.tape[l];
        const w = t.w[l];
        const layer = t.m.weights.layers[l];
        try o.q4Back(t.grad, w.mlp.down, t.sc.mlp, R, false);
        try o.swigluBack(kp.gu, t.sc.mlp, t.sc.gu, R, g.intermediate);
        try o.q4Back(t.sc.gu, w.mlp.gu, t.dy, R, false);
        if (t.pocket_back) if (t.m.pockets) |p| if (p.layers[l]) |pk| if (pk.used > 0) {
            for (t.spans[0..t.nseq], t.seqs[0..t.nseq]) |sp, sq| {
                try o.go("tf_learn_pocket_back", &.{ plus(kp.y, sp.at * D * 2), .{ .buffer = pk.weights }, plus(t.grad, sp.at * D * 4), plus(t.dy, sp.at * D * 4) }, to.Shape{
                    .u = .{ t.from[l], @intCast(D), @intCast(pk.used), @intCast(pk.cap) },
                    .f = .{ sq.scale, 0, 0, 0 },
                }, .{ .width = sp.rows }, 256);
            }
        };
        try o.rmsBack(kp.h_mid, wts.ref(layer.post_norm), t.dy, t.grad, R, D, g.eps);
        if (w.gdn) |gl| {
            try o.q4Back(t.grad, gl.out, t.sc.mix, R, false);
            for (t.spans[0..t.nseq], 0..) |sp, si| try gdn(t, o, l, si, sp);
        } else {
            try o.q4Back(t.grad, w.attn.?.out, t.sc.mix, R, false);
            for (t.spans[0..t.nseq]) |sp| try attention(t, o, l, sp);
        }
        try o.rmsBack(kp.h_in, wts.ref(layer.norm), t.dy, t.grad, R, D, g.eps);
    }
}

fn gdn(t: *train.Trainer, o: to.Ops, l: usize, si: usize, sp: train.Span) !void {
    const g = t.g;
    const w = t.w[l].gdn.?;
    const k = t.tape[l];
    const s = t.sc;
    const vi = g.gdnValueDim();
    const conv = g.gdnQkvDim();
    const ki = g.k_heads * 128;
    const proj = conv + vi + 2 * g.v_heads;
    const rows = sp.rows;
    const shape: to.GdnShape = .{ .rows = @intCast(rows), .nk = @intCast(g.k_heads), .nv = @intCast(g.v_heads), .conv = @intCast(conv), .proj = @intCast(proj) };
    const p = plus(k.proj, sp.at * proj * 2);
    const act = plus(k.act, sp.at * conv * 2);
    const q = plus(k.q, sp.at * ki * 4);
    const key = plus(k.k, sp.at * ki * 4);
    const decay = plus(k.decay, sp.at * g.v_heads * 4);
    const beta = plus(k.beta, sp.at * g.v_heads * 4);
    try o.zero(s.dp, rows * proj);
    try o.go("tf_learn_gdn_act", &.{ p, w.conv, act }, to.Shape{ .u = .{ @intCast(rows), @intCast(2 * ki), @intCast(proj), @intCast(conv) } }, .{ .width = (2 * ki + 255) / 256, .height = rows }, 256);
    try o.go("tf_gdn_norm_back", &.{ plus(k.recur, sp.at * vi * 2), p, w.norm, plus(s.mix, sp.at * vi * 4), s.gy, s.dp }, extern struct { s: to.GdnShape, eps: f32 }{ .s = shape, .eps = g.eps }, .{ .width = rows, .height = g.v_heads }, 128);
    const checkpoints = plus(k.checkpoints, t.cp_first[si] * g.v_heads * 128 * 128 * 4);
    try o.go("tf_gdn_scan_back", &.{ act, q, key, decay, beta, checkpoints, s.gy, s.work, s.dqparts, s.dkparts, s.dact, s.dg, s.db }, extern struct { s: to.GdnShape, from: i32 }{ .s = shape, .from = 0 }, .{ .width = g.v_heads * 4 }, 256);
    try o.go("tf_gdn_qk_back", &.{ act, s.dqparts, s.dkparts, s.dact }, shape, .{ .width = rows, .height = g.k_heads }, 128);
    try o.go("tf_gdn_gates_back", &.{ p, w.a_log, w.dt, decay, s.dg, s.db, s.dp }, shape, .{ .width = rows }, 256);
    // Fresh sequences have no incoming conv history; the zero slab is deliberately valid device storage.
    try o.go("tf_gdn_conv_back", &.{ p, t.zero_history, w.conv, s.dact, s.dp }, shape, .{ .width = (conv + 255) / 256, .height = rows }, 256);
    try o.q4Back(s.dp, w.input, plus(t.dy, sp.at * g.hidden * 4), rows, false);
}

fn attention(t: *train.Trainer, o: to.Ops, l: usize, sp: train.Span) !void {
    const g = t.g;
    const w = t.w[l].attn.?;
    const k = t.tape[l];
    const s = t.sc;
    const rows = sp.rows;
    const qi = g.heads * g.head_dim;
    const kv = g.kvDim();
    const pq = plus(k.pq, sp.at * 2 * qi * 2);
    const pk = plus(k.pk, sp.at * kv * 2);
    const qkv = plus(k.qkv, sp.at * (qi + 2 * kv) * 2);
    try o.go("tf_attn_gate_back", &.{ plus(s.mix, sp.at * qi * 4), plus(k.gated, sp.at * qi * 2), pq, s.dout, s.dpq }, extern struct { heads: i32, dim: i32 }{ .heads = @intCast(g.heads), .dim = @intCast(g.head_dim) }, .{ .width = (qi + 255) / 256, .height = rows }, 256);
    const heads: to.Heads = .{ .rows = @intCast(rows), .heads = @intCast(g.heads), .kv_heads = @intCast(g.kv_heads), .dim = @intCast(g.head_dim), .qkv = @intCast(qi + 2 * kv), .scale = 1 / @sqrt(@as(f32, @floatFromInt(g.head_dim))) };
    try o.go("tf_attn_q", &.{ qkv, s.dout, s.probability, s.dscores, s.dq, t.zero_history, t.zero_history }, heads, .{ .width = g.heads, .height = rows }, 256);
    try o.linear("tf_learn_attn_kv_back", &.{ qkv, s.dout, s.probability, s.dscores, s.dk, s.dv }, .{ .u = .{ @intCast(rows), @intCast(g.heads), @intCast(g.kv_heads), @intCast(g.head_dim) }, .f = .{ heads.scale, 0, 0, 0 } }, rows * kv);
    try o.linear("tf_learn_attn_gate_grad", &.{ plus(s.mix, sp.at * qi * 4), pq, qkv, s.probability, s.dpq }, .{ .u = .{ @intCast(rows), @intCast(g.heads), @intCast(g.kv_heads), @intCast(g.head_dim) } }, rows * qi);
    const Params = extern struct { pos0: i32 = 0, heads: i32, kv_heads: i32, dim: i32, half: i32, eps: f32 };
    try o.go("tf_attn_prep_back", &.{ pq, pk, w.qs, w.ks, w.inv, s.dq, s.dk, s.dpq, s.dpk }, Params{ .heads = @intCast(g.heads), .kv_heads = @intCast(g.kv_heads), .dim = @intCast(g.head_dim), .half = @intCast(g.rope_dims / 2), .eps = g.eps }, .{ .width = g.heads + g.kv_heads, .height = rows }, g.head_dim);
    const dest = plus(t.dy, sp.at * g.hidden * 4);
    try o.q4Back(s.dpq, w.q, dest, rows, false);
    try o.q4Back(s.dpk, w.k, dest, rows, true);
    try o.q4Back(s.dv, w.v, dest, rows, true);
}
