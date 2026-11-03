//! Native forward values retained for the learner; all offsets are packed pass rows.
const std = @import("std");
const mtl = @import("metal");
const to = @import("train_ops.zig");
const hooks = @import("train_hooks.zig");
pub const Ref = @import("projection.zig").Ref;
pub const Layer = struct {
    y: Ref = undefined,
    h_in: Ref = undefined,
    h_mid: Ref = undefined,
    gu: Ref = undefined,
    proj: Ref = undefined,
    act: Ref = undefined,
    q: Ref = undefined,
    k: Ref = undefined,
    decay: Ref = undefined,
    beta: Ref = undefined,
    recur: Ref = undefined,
    checkpoints: Ref = undefined,
    pq: Ref = undefined,
    pk: Ref = undefined,
    pv: Ref = undefined,
    qkv: Ref = undefined,
    gated: Ref = undefined,
};

pub fn save(ctx: *anyopaque, e: mtl.ComputeEncoder, point: hooks.Point, l: usize, refs: []const Ref, rows: u32) !void {
    const t: *@import("train.zig").Trainer = @ptrCast(@alignCast(ctx));
    if (rows != 1) return error.BadLearningRows;
    const g = t.g;
    const D = g.hidden;
    const o = to.Ops{ .kernels = &t.k, .e = e };
    const at = t.hook_at;
    if (point == .final) {
        if (t.mode != .keys) try copy(o, refs[0], t.final_h, at, D * 2);
        return;
    }
    if (point == .logits) {
        if (t.mode != .keys) try copy(o, refs[0], t.logits, at, g.vocab * 2);
        return;
    }
    const kept = t.tape[l];
    // Every layer's postnorm is needed by the pocket fitter, even below the backwards limit.
    if (point == .post) try copy(o, refs[1], kept.y, at, D * 2);
    if (t.mode != .target or l < t.low) return;
    const conv = g.gdnQkvDim();
    const vi = g.gdnValueDim();
    const ki = g.k_heads * g.dk;
    const proj = conv + vi + 2 * g.v_heads;
    switch (point) {
        .input => {
            try copy(o, refs[0], kept.h_in, at, D * 2);
            if (g.kind(l) == .linear and t.local_row % 4 == 0) {
                const state = try t.runner.gdn.committedSpan(t.runner.linear_index[l], t.slot, true);
                try o.copy(.{ .buffer = state.buffer, .offset = state.offset }, to.plus(kept.checkpoints, (t.cp_first[t.slot] + t.local_row / 4) * state.bytes), state.bytes);
            }
        },
        .post => try copy(o, refs[0], kept.h_mid, at, D * 2),
        .mlp => try copy(o, refs[0], kept.gu, at, 2 * g.intermediate * 2),
        .gdn => {
            const dest = to.plus(kept.proj, at * proj * 2);
            try o.copy(refs[0], dest, conv * 2);
            try o.copy(refs[1], to.plus(dest, conv * 2), (vi + 2 * g.v_heads) * 2);
        },
        .gdn_state => {
            try o.widen(refs[0], to.plus(kept.q, at * ki * 4), ki);
            try o.widen(refs[1], to.plus(kept.k, at * ki * 4), ki);
            try o.copy(refs[2], to.plus(kept.act, (at * conv + 2 * ki) * 2), vi * 2);
            try copy(o, refs[3], kept.decay, at, g.v_heads * 4);
            try o.widen(refs[4], to.plus(kept.beta, at * g.v_heads * 4), g.v_heads);
            try copy(o, refs[5], kept.recur, at, vi * 2);
        },
        .attention => {
            try copy(o, refs[0], kept.pq, at, g.qDim() * 2);
            try copy(o, refs[1], kept.pk, at, g.kvDim() * 2);
            try copy(o, refs[2], kept.pv, at, g.kvDim() * 2);
        },
        .prepared_attention => {
            const qi = g.heads * g.head_dim;
            const kv = g.kvDim();
            const dest = to.plus(kept.qkv, at * (qi + 2 * kv) * 2);
            try o.copy(refs[0], dest, qi * 2);
            try o.copy(refs[1], to.plus(dest, qi * 2), kv * 2);
            try o.copy(refs[2], to.plus(dest, (qi + kv) * 2), kv * 2);
        },
        .gated => if (g.kind(l) == .attention) {
            try copy(o, refs[0], kept.gated, at, g.heads * g.head_dim * 2);
        },
        .final, .logits => unreachable,
    }
}
fn copy(o: to.Ops, src: Ref, dest: Ref, row: usize, width: usize) !void {
    try o.copy(src, to.plus(dest, row * width), width);
}

pub fn carve(t: anytype) !void {
    const g = t.g;
    const R = @import("train.zig").pack_rows;
    t.tape = try t.gpa.alloc(Layer, g.layers);
    @memset(t.tape, .{});
    for (t.tape, 0..) |*k, l| {
        k.y = try t.alloc(R * g.hidden * 2);
        if (l < @import("train.zig").keepLow(g.layers)) continue;
        k.h_in = try t.alloc(R * g.hidden * 2);
        k.h_mid = try t.alloc(R * g.hidden * 2);
        k.gu = try t.alloc(R * 2 * g.intermediate * 2);
        if (g.kind(l) == .linear) {
            const conv = g.gdnQkvDim();
            k.proj = try t.alloc(R * (conv + g.gdnValueDim() + 2 * g.v_heads) * 2);
            k.act = try t.alloc(R * conv * 2);
            k.q = try t.alloc(R * g.k_heads * g.dk * 4);
            k.k = try t.alloc(R * g.k_heads * g.dk * 4);
            k.decay = try t.alloc(R * g.v_heads * 4);
            k.beta = try t.alloc(R * g.v_heads * 4);
            k.recur = try t.alloc(R * g.gdnValueDim() * 2);
            k.checkpoints = try t.alloc((R / 4 + @import("train.zig").max_seqs) * g.v_heads * g.dv * g.dk * 4);
        } else {
            k.pq = try t.alloc(R * g.qDim() * 2);
            k.pk = try t.alloc(R * g.kvDim() * 2);
            k.pv = try t.alloc(R * g.kvDim() * 2);
            k.qkv = try t.alloc(R * (g.heads * g.head_dim + 2 * g.kvDim()) * 2);
            k.gated = try t.alloc(R * g.heads * g.head_dim * 2);
        }
    }
}
