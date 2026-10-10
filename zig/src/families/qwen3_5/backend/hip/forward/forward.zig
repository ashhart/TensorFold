//! A prompt span through every layer in steps of SPAN rows, on the prefill formulas.

const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const state = @import("state.zig");
const moe = @import("moe.zig");
const reduce = @import("reduce.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const Kind = hip.ops.Kind;

/// Rows of one prefill step (qwen_math.SPAN).
pub const SPAN = 2048;

pub const Error = reduce.Error || error{Cancelled};

pub const residual = reduce.residual;

/// A device buffer of `n` values of `kind` from the arena.
pub fn take(o: Ops, kind: Kind, n: usize) Error!Tensor {
    return .{ .ptr = try o.arena.take(n * kind.size()), .kind = kind };
}

/// `t` advanced by `n` values.
pub fn at(t: Tensor, n: usize) Tensor {
    return .{ .ptr = t.ptr + n * t.kind.size(), .kind = t.kind };
}

/// Called after each layer with the residual rows (a check reads them; the forward does not wait for it).
pub const Trace = struct {
    ctx: *anyopaque,
    layer: *const fn (ctx: *anyopaque, index: usize, x: Tensor, rows: usize) anyerror!void,
};

/// The span's final-normed rows (len, hidden) in the activation dtype; `ids` are int32 token ids on the device.
pub fn span(ops: Ops, m: *const view.Model, caches: *state.Caches, ids: u64, len: usize, pos0: usize, trace: ?Trace) Error!Tensor {
    // prefill's kernels at any row count: a prompt's rows have the same bits fresh, resumed or in steps
    var o = ops;
    o.prefill = true;
    const s = m.spec;
    const x = try take(o, m.act, len * s.hidden);
    var start: usize = 0;
    while (start < len) : (start += SPAN) {
        const rows = @min(SPAN, len - start);
        try o.embedRows(m.embed, ids + start * 4, rows, at(x, start * s.hidden));
    }
    for (m.layers, 0..) |layer, index| {
        start = 0;
        while (start < len) : (start += SPAN) {
            const rows = @min(SPAN, len - start);
            const mark = o.arena.mark();
            defer o.arena.release(mark);
            const xs = at(x, start * s.hidden);
            const input_norm, const post_norm, const mlp = switch (layer) {
                .full => |f| .{ f.input_norm, f.post_norm, f.mlp },
                .linear => |l| .{ l.input_norm, l.post_norm, l.mlp },
            };
            const normed = try take(o, m.act, rows * s.hidden);
            try o.rms(xs, input_norm, normed, rows, s.hidden, @floatCast(s.eps));
            const y = switch (layer) {
                .full => |f| try attention(o, m, f, caches, index, normed, rows, pos0 + start),
                .linear => |l| try linear(o, m, l, caches, index, normed, rows),
            };
            try residual(o, m, xs, y, rows * s.hidden);
            try o.rms(xs, post_norm, normed, rows, s.hidden, @floatCast(s.eps));
            const z = try mlpRows(o, m, mlp, normed, rows, true);
            try residual(o, m, xs, z, rows * s.hidden);
        }
        if (trace) |t| t.layer(t.ctx, index, x, len) catch |err| return if (err == error.Cancelled) error.Cancelled else error.HipFailed;
    }
    const out = try take(o, m.act, len * s.hidden);
    try o.rms(x, m.final_norm, out, len, s.hidden, @floatCast(s.eps));
    return out;
}

/// The attention scale, head_dim ** -0.5 in double as Python has it, passed as fp32.
pub fn scaleOf(head_dim: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(head_dim), -0.5));
}

/// Full attention on `rows` normed rows at `pos`: q and its gate, k, v, norms, RoPE, cache write, prefill tile, o.
fn attention(o: Ops, m: *const view.Model, f: view.Full, caches: *state.Caches, index: usize, x: Tensor, rows: usize, pos: usize) Error!Tensor {
    const s = m.spec;
    const hd = s.head_dim;
    const qg = try o.project(x, f.q, rows, false);
    const keys = try o.project(x, f.k, rows, false);
    const values = try o.project(x, f.v, rows, false);
    const q_rows = rows * s.heads;
    const qc = try take(o, m.act, q_rows * hd);
    try o.copyCols(qg, 2 * hd, 0, qc.ptr, q_rows, hd);
    const qn = try take(o, m.act, q_rows * hd);
    try o.rms(qc, f.q_norm, qn, q_rows, hd, @floatCast(s.eps));
    const kn = try take(o, m.act, rows * s.kv_heads * hd);
    try o.rms(keys, f.k_norm, kn, rows * s.kv_heads, hd, @floatCast(s.eps));
    const theta: f32 = @floatCast(s.rope_theta);
    const q32 = try o.arena.of(f32, q_rows * hd);
    // q rounded to the activation dtype by apply_rope, then widened for the kernel: (heads, rows, d) fp32
    try o.ropePrefill(qn, .{ .ptr = q32, .kind = .f32 }, rows * hd, hd, rows, s.heads, hd, s.rotary_dim, pos, theta);
    const c = caches.paged(m, index);
    // k rotated and rounded to the activation dtype as (kv_heads, rows, d), then with v into the pages from slot pos
    const kr = try take(o, m.act, rows * s.kv_heads * hd);
    try o.ropePrefill(kn, kr, rows * hd, hd, rows, s.kv_heads, hd, s.rotary_dim, pos, theta);
    try o.pageWrite(kr.ptr, c.k, c.table, rows, s.kv_heads, hd, rows * hd, hd, pos, c.count);
    try o.pageWrite(values.ptr, c.v, c.table, rows, s.kv_heads, hd, hd, s.kv_heads * hd, pos, c.count);
    const att = try o.arena.of(f32, q_rows * hd);
    try o.causalPaged(q32, c, att, rows, pos + rows, s.heads, scaleOf(hd), pos);
    const gated = try take(o, m.act, q_rows * hd);
    try o.attnGate(att, qg, gated, rows, s.heads, hd, false);
    return o.project(gated, f.o, rows, f.o.partial);
}

/// Linear attention: qkv, z, a, b, the conv over [state | rows], norms, gate, recurrence and the gated norm.
fn linear(o: Ops, m: *const view.Model, l: view.Linear, caches: *state.Caches, index: usize, x: Tensor, rows: usize) Error!Tensor {
    const s = m.spec;
    const qkv = try o.project(x, l.qkv, rows, false);
    const z = try o.project(x, l.z, rows, false);
    const a = try o.project(x, l.a, rows, false);
    const b = try o.project(x, l.b, rows, false);
    const cache = caches.layers[index].linear;
    const ch = view.convChannels(s);
    const mixed = try o.arena.of(f32, rows * ch);
    // the new window lands beside the old one (every row reads the old), then replaces it
    const conv_bytes = (s.conv - 1) * ch * 4;
    const window = try o.arena.take(conv_bytes);
    try o.convPrefill(qkv, l.conv, cache.conv.base(), mixed, window, rows, ch, s.conv);
    try hip.raw.copy(cache.conv, 0, window, conv_bytes, o.stream);
    const q, const k, const v = try splitQkv(o, m, .{ .ptr = mixed, .kind = .f32 }, rows);
    const gate = try o.arena.of(f32, rows * s.value_heads);
    const beta = try o.arena.of(f32, rows * s.value_heads);
    try o.gdnGatePrefill(a, b, l.a_log, l.dt_bias, gate, beta, rows, s.value_heads);
    const y = try o.arena.of(f32, rows * s.valueWidth());
    try o.gatedDelta(q, k, v, gate, beta, cache.state.base(), y, rows, s.key_heads, s.value_heads, s.key_dim, s.value_dim, null);
    return gatedOut(o, m, l, y, z, rows);
}

/// The conv output's q and k, normalized (fp32, contiguous), and v (fp32, contiguous).
pub fn splitQkv(o: Ops, m: *const view.Model, mixed: Tensor, rows: usize) Error!struct { u64, u64, u64 } {
    const s = m.spec;
    const kw = s.keyWidth();
    const ch = view.convChannels(s);
    const qc = try o.arena.of(f32, rows * kw);
    const kc = try o.arena.of(f32, rows * kw);
    const v = try o.arena.of(f32, rows * s.valueWidth());
    try o.copyCols(mixed, ch, 0, qc, rows, kw);
    try o.copyCols(mixed, ch, kw, kc, rows, kw);
    try o.copyCols(mixed, ch, 2 * kw, v, rows, s.valueWidth());
    const q = try o.arena.of(f32, rows * kw);
    const k = try o.arena.of(f32, rows * kw);
    const heads = rows * s.key_heads;
    try o.rms(.{ .ptr = qc, .kind = .f32 }, m.qk.q_weight, .{ .ptr = q, .kind = .f32 }, heads, s.key_dim, m.qk.eps);
    try o.rms(.{ .ptr = kc, .kind = .f32 }, m.qk.k_weight, .{ .ptr = k, .kind = .f32 }, heads, s.key_dim, m.qk.eps);
    return .{ q, k, v };
}

/// rms_norm(y, gnorm) * silu(z), rounded to the activation dtype, then the out projection.
pub fn gatedOut(o: Ops, m: *const view.Model, l: view.Linear, y: u64, z: Tensor, rows: usize) Error!Tensor {
    const s = m.spec;
    const n = rows * s.valueWidth();
    const out = try take(o, m.act, n);
    if (o.fused() and s.value_dim <= 1024 and z.kind == m.act) {
        try o.gnormOut(y, l.gnorm, z, out, rows * s.value_heads, s.value_dim, @floatCast(s.eps));
    } else {
        const yn = try o.arena.of(f32, n);
        try o.rms(.{ .ptr = y, .kind = .f32 }, l.gnorm, .{ .ptr = yn, .kind = .f32 }, rows * s.value_heads, s.value_dim, @floatCast(s.eps));
        try o.gnormSilu(yn, z, out, n);
    }
    return o.project(out, l.out, rows, l.out.partial);
}

/// _mlp: gate and up, silu(gate) * up, down; or the routed experts (`prefill` is the span's length above one).
pub fn mlpRows(o: Ops, m: *const view.Model, mlp: view.Mlp, x: Tensor, rows: usize, prefill: bool) Error!Tensor {
    switch (mlp) {
        .moe => |r| return moe.run(o, m, r, x, rows, prefill and rows > 1),
        .dense => |d| {
            var outs: [4]Tensor = undefined;
            const gate, const up = if (try o.projectGroup(x, &.{ d.gate, d.up }, rows, &outs))
                .{ outs[0], outs[1] }
            else
                .{ try o.project(x, d.gate, rows, false), try o.project(x, d.up, rows, false) };
            const act = try take(o, m.act, rows * d.gate.n);
            try o.siluMul(gate, up, act, rows * d.gate.n);
            return o.project(act, d.down, rows, d.down.partial);
        },
    }
}
