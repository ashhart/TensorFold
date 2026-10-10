//! A lane round's forward: every stream's window in one pass over the round's device plan, each row bitwise serial.

const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const fwd = @import("forward.zig");
const moe = @import("moe.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const pops = hip.plan_ops;

/// A round's launches: the device plan, its token ids, the keys the attention walk covers, each layer's kept inputs.
pub const Round = struct { plan: pops.Plan, tokens: u64, span: usize, inputs: []const [2]u64 };

/// Whether this build and model run lane rounds: the merged decode launches, and shapes the plan kernels take.
pub fn supported(o: Ops, m: *const view.Model) bool {
    const s = m.spec;
    return o.fused() and s.key_dim <= 1024 and s.conv <= 8 and (s.key_dim == 128 or s.key_dim == 16) and s.value_dim % 8 == 0 and s.head_dim <= 256 and m.act != .f32;
}

/// A linear layer's inputs a keep replays, `rows` rows: the conv inputs, then k, v, gate and beta; zero for attention.
pub fn kept(o: Ops, m: *const view.Model, rows: usize, table: [][2]u64) !void {
    const s = m.spec;
    for (table, 0..) |*t, index| {
        t.* = .{ 0, 0 };
        if (s.full(index)) continue;
        t[0] = try o.arena.of(f32, rows * view.convChannels(s));
        t[1] = try o.arena.of(f32, rows * (s.keyWidth() + s.valueWidth() + 2 * s.value_heads));
    }
}

/// The final-normed rows of the round, in order (rows, hidden), padding rows included.
pub fn forward(ops: Ops, m: *const view.Model, r: Round, trace: ?fwd.Trace) fwd.Error!Tensor {
    // a round's rows keep their decode kernels however many share it
    var o = ops;
    o.window = true;
    const s = m.spec;
    const total = r.plan.rows;
    const x = try fwd.take(o, m.act, total * s.hidden);
    try o.embedRows(m.embed, r.tokens, total, x);
    const normed = try fwd.take(o, m.act, total * s.hidden);
    const out = try fwd.take(o, m.act, total * s.hidden);
    const eps: f32 = @floatCast(s.eps);
    try o.rms(x, inputNorm(m.layers[0]), normed, total, s.hidden, eps);
    for (m.layers, 0..) |layer, index| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        const post_norm, const mlp = switch (layer) {
            .full => |f| .{ f.post_norm, f.mlp },
            .linear => |l| .{ l.post_norm, l.mlp },
        };
        const y = switch (layer) {
            .full => |f| try attentionRows(o, m, f, r, index, normed),
            .linear => |l| try linearRows(o, m, l, r, index, normed),
        };
        // the launches of a layer's two tails merge when the rows stay on this rank
        const merge = o.fused() and y.kind == m.act;
        const last = index + 1 == m.layers.len;
        const next_norm = if (last) m.final_norm else inputNorm(m.layers[index + 1]);
        const dest = if (last) out else normed;
        if (merge) {
            try o.addRms(x, y, post_norm, normed, total, s.hidden, eps);
        } else {
            try fwd.residual(o, m, x, y, total * s.hidden);
            try o.rms(x, post_norm, normed, total, s.hidden, eps);
        }
        switch (mlp) {
            .moe => |rt| if (merge and rt.remap == 0) {
                const parts = try moe.parts(o, m, rt, normed, total);
                try o.moeTail(x, parts.y, parts.wts, next_norm, dest, total, parts.slots, s.hidden, eps);
            } else {
                const z = try fwd.mlpRows(o, m, mlp, normed, total, true);
                try fwd.residual(o, m, x, z, total * s.hidden);
                try o.rms(x, next_norm, dest, total, s.hidden, eps);
            },
            .dense => {
                const z = try fwd.mlpRows(o, m, mlp, normed, total, true);
                if (merge and z.kind == m.act) {
                    try o.addRms(x, z, next_norm, dest, total, s.hidden, eps);
                } else {
                    try fwd.residual(o, m, x, z, total * s.hidden);
                    try o.rms(x, next_norm, dest, total, s.hidden, eps);
                }
            },
        }
        if (trace) |t| t.layer(t.ctx, index, x, total) catch return error.HipFailed;
    }
    return out;
}

fn inputNorm(layer: view.Layer) u64 {
    return switch (layer) {
        .full => |f| f.input_norm,
        .linear => |l| l.input_norm,
    };
}

/// Keeps the slots `counts` lists (negative: none): linear layers replay their kept rows, `hidden` gets the last.
pub fn keep(o: Ops, m: *const view.Model, p: pops.Plan, counts: u64, hidden: u64) hip.ops.Error!void {
    const s = m.spec;
    try pops.keep(o, p, .{
        .keep = counts,
        .hidden = hidden,
        .hidden_words = s.hidden * m.act.size() / 4,
        .channels = view.convChannels(s),
        .taps = s.conv - 1,
        .layers = s.n_layers,
    });
    try pops.gdnReplay(o, counts, p, s.key_heads, s.value_heads, s.key_dim, s.value_dim, s.n_layers);
}

fn attentionRows(o: Ops, m: *const view.Model, f: view.Full, r: Round, index: usize, x: Tensor) fwd.Error!Tensor {
    const s = m.spec;
    const hd = s.head_dim;
    const total = r.plan.rows;
    var outs: [4]Tensor = undefined;
    const qg, const keys, const values = if (try o.projectGroup(x, &.{ f.q, f.k, f.v }, total, &outs))
        .{ outs[0], outs[1], outs[2] }
    else
        .{ try o.project(x, f.q, total, false), try o.project(x, f.k, total, false), try o.project(x, f.v, total, false) };
    const q_rows = total * s.heads;
    const eps: f32 = @floatCast(s.eps);
    const theta: f32 = @floatCast(s.rope_theta);
    // q and k normed, rotated and rounded in one launch; k then goes with v into the slots' caches
    const q32 = try o.arena.of(f32, q_rows * hd);
    try o.qkRope(qg, s.heads * 2 * hd, 2 * hd, f.q_norm, eps, total, s.heads, hd, s.rotary_dim, theta, r.plan.args.pos, q32, null, 0);
    const k32 = try o.arena.of(f32, total * s.kv_heads * hd);
    try o.qkRope(keys, s.kv_heads * hd, hd, f.k_norm, eps, total, s.kv_heads, hd, s.rotary_dim, theta, r.plan.args.pos, k32, null, 0);
    try pops.kvWrite(o, k32, values, r.plan, index, s.kv_heads, hd);
    const att = try o.arena.of(f32, q_rows * hd);
    try pops.causal(o, q32, m.act, att, r.plan, index, s.heads, s.kv_heads, hd, r.span, fwd.scaleOf(hd));
    const gated = try fwd.take(o, m.act, q_rows * hd);
    try o.attnGate(att, qg, gated, total, s.heads, hd, true);
    return o.project(gated, f.o, total, f.o.partial);
}

fn linearRows(o: Ops, m: *const view.Model, l: view.Linear, r: Round, index: usize, x: Tensor) fwd.Error!Tensor {
    const s = m.spec;
    const total = r.plan.rows;
    var outs: [4]Tensor = undefined;
    const qkv, const z, const a, const b = if (try o.projectGroup(x, &.{ l.qkv, l.z, l.a, l.b }, total, &outs))
        .{ outs[0], outs[1], outs[2], outs[3] }
    else
        .{ try o.project(x, l.qkv, total, false), try o.project(x, l.z, total, false), try o.project(x, l.a, total, false), try o.project(x, l.b, total, false) };
    const ch = view.convChannels(s);
    const y = try o.arena.of(f32, total * s.valueWidth());
    // the k, v, gate and beta a keep replays live in the round's kept inputs
    const kw = s.keyWidth();
    const k_kept = r.inputs[index][1];
    const vv = k_kept + 4 * total * kw;
    const gate = vv + 4 * total * s.valueWidth();
    const beta = gate + 4 * total * s.value_heads;
    // the cast, the conv, the split of its output and, for heads of 128, the q and k norms and the gate in one launch
    const qn = try o.arena.of(f32, total * kw);
    const heads128 = s.key_dim == 128 and kw % 128 == 0;
    const kn = if (heads128) k_kept else try o.arena.of(f32, total * kw);
    const norm: ?pops.Norm = if (heads128) .{ .q = m.qk.q_weight, .k = m.qk.k_weight, .eps = m.qk.eps } else null;
    const gates: pops.Gates = .{ .a = a, .b = b, .a_log = l.a_log, .dt_bias = l.dt_bias, .gate = gate, .beta = beta, .count = total * s.value_heads, .heads = s.value_heads };
    try pops.convSplit(o, qkv, l.conv, qn, kn, vv, r.plan, index, ch, s.conv, kw, s.valueWidth(), norm, if (heads128) gates else null);
    var q = qn;
    var k = kn;
    if (!heads128) {
        const qc = try o.arena.of(f32, total * kw);
        const kc = k_kept;
        try o.rms2(qn, m.qk.q_weight, qc, kn, m.qk.k_weight, kc, total * s.key_heads, s.key_dim, m.qk.eps);
        try o.gdnGate(a, b, l.a_log, l.dt_bias, gate, beta, total * s.value_heads, s.value_heads);
        q = qc;
        k = kc;
    }
    try pops.gatedDelta(o, q, k, vv, gate, beta, y, r.plan, index, s.key_heads, s.value_heads, s.key_dim, s.value_dim);
    return fwd.gatedOut(o, m, l, y, z, total);
}
