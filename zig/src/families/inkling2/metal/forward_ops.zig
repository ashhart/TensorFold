//! Stateless layer operations and dense/MOE/gate dispatch helpers for Inkling 2 Mini forward.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const experts = @import("experts.zig");
const loop = @import("loop.zig");
const dense_mma = @import("dense.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const Model = model_file.Model;

pub const Ref = state.Ref;
pub const RArgs = extern struct { count: u32, boundary: u32 = 0 };
pub const NormArgs = extern struct { rows: u32, width: u32, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 };
pub const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
pub const DenseStep = struct { variant: usize, rows: usize };

pub fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
pub fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
pub fn row(buf: mtl.Buffer, i: usize, row_bytes: usize) Ref {
    return at(buf, i * row_bytes);
}
pub fn a(x_: Ref) attention.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}
pub fn x(x_: Ref) experts.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}
pub fn l(x_: Ref) loop.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}
pub fn arRow(s: *state.Scratch, role: usize, rows: usize, i: usize, gated: bool) Ref {
    return at(if (gated) s.gated else s.ar, (role * rows + i) * state.hidden * 4);
}
pub fn messageRow(s: *state.Scratch, core: usize, role: usize, i: usize) Ref {
    return at(s.messages, ((i * state.message_entries + core * 4 + role) * state.hidden) * 4);
}
fn gateNorm(s: *state.Scratch, role: usize, rows: usize) Ref {
    return at(s.gate_norm, role * rows * 2 * state.hidden * 2);
}
fn gateZeta(s: *state.Scratch, role: usize, rows: usize) Ref {
    return at(s.gate_zeta, role * rows * 64 * 4);
}

/// One controller over every row: m's rows are a token's message entries apart; the bits equal per-row `loop.gates`.
pub fn gates(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, h: Ref, msg: Ref, c: model_file.Controller, out: Ref, rows: usize) !void {
    const stride: u32 = state.message_entries * state.hidden;
    // eight rows or more: simdgroup matrices, every sum in the per-row order (#4756)
    if (m.loop.gates_mma != null and rows >= loop.mma_rows) return loop.gatesRows(&m.loop, e, l(h), l(msg), stride, l(r(c.down)), l(r(c.forget)), l(r(c.gain)), l(out), @intCast(rows));
    // under two rows (decode's one), main's call: one launch a row
    if (!loop.plan.splitRows(rows)) {
        for (0..rows) |_| loop.gates(&m.loop, e, l(h), l(msg), l(r(c.down)), l(r(c.forget)), l(r(c.gain)), l(out), 1);
        return;
    }
    loop.gatesSplit(&m.loop, e, l(h), l(msg), stride, l(r(c.down)), l(r(c.forget)), l(r(c.gain)), l(out), l(at(s.gate_norm, 0)), l(at(s.gate_zeta, 0)), @intCast(rows));
}

pub fn runtime(m: *const Model, e: mtl.ComputeEncoder, op: model_file.RuntimeOp, refs: []const Ref, count: usize, boundary: usize) void {
    e.setPipeline(m.runtime[@backingInt(op)]);
    for (refs, 0..) |r_, i| e.setBuffer(r_.buf, r_.off, i);
    e.setValue(RArgs{ .count = @intCast(count), .boundary = @intCast(boundary) }, refs.len);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn copyBf16(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, count: usize) void {
    runtime(m, e, .copy_bf16, &.{ from, to }, count, 0);
}
pub fn captureTap(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, rows: usize, channel: usize) void {
    std.debug.assert(channel < state.feature_taps);
    runtime(m, e, .capture_tap, &.{ from, to }, rows * state.hidden, channel);
}
pub fn widen(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, count: usize) void {
    runtime(m, e, .widen, &.{ from, to }, count, 0);
}

pub fn norm(m: *const Model, e: mtl.ComputeEncoder, input: Ref, weight: ck.Tensor, out: Ref, rows: usize) void {
    e.setPipeline(m.norm);
    e.setBuffer(input.buf, input.off, 0);
    e.setBuffer(weight.buffer, weight.offset, 1);
    e.setValue(NormArgs{ .rows = @intCast(rows), .width = state.hidden, .eps = 1e-6 }, 2);
    e.setBuffer(out.buf, out.off, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

/// Ceiling-power-of-two matvec variant for `left` rows, reading weights once per <=32 rows.
pub fn denseStep(left: usize) DenseStep {
    std.debug.assert(left > 0);
    const take = @min(left, 32);
    const variant: usize = if (take <= 1) 0 else if (take <= 2) 1 else if (take <= 4) 2 else if (take <= 8) 3 else if (take <= 16) 4 else 5;
    return .{ .variant = variant, .rows = take };
}

/// Every projection uses the shared BF16 matvec's FP32 reduction and one BF16 store.
pub fn dense(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    encodeDense(mtl.ComputeEncoder, m, e, weight, input, out, rows);
}

/// Dispatch helper parameterized on encoder so host tests record exact launch counts and arguments.
pub fn encodeDense(comptime Encoder: type, m: anytype, e: Encoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    const k = weight.shape[weight.rank - 1];
    const n = weight.shape[weight.rank - 2];
    var base: usize = 0;
    while (base < rows) {
        const step = denseStep(rows - base);
        e.setPipeline(m.dense[step.variant]);
        e.setBuffer(input.buf, input.off + base * k * 2, 0);
        e.setBuffer(weight.buffer, weight.offset, 1);
        e.setValue(DenseArgs{ .rows = @intCast(step.rows), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
        e.setBuffer(out.buf, out.off + base * n * 2, 3);
        e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        base += step.rows;
    }
}

/// A visit's projection: `dense`, or its sums on simdgroup matrices for many rows where the GPU has no tensor units.
pub fn project(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    if (m.dense_mma) |mma| {
        const n: u32 = @intCast(weight.shape[weight.rank - 2]);
        const call = dense_mma.Call{ .rows = @intCast(rows), .n = n, .k = @intCast(weight.shape[weight.rank - 1]), .ldy = n };
        if (dense_mma.Mma.serves(call)) return mma.launch(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = weight.buffer, .off = weight.offset }, .{ .buf = out.buf, .off = out.off }, call);
    }
    dense(m, e, weight, input, out, rows);
}

pub fn bankAndReset(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, rows: usize) void {
    for (0..rows) |i| {
        // The second completed block of the previous pass is B12; its final p is P18.
        copyBf16(m, e, at(s.blocks, (i * state.block_slots + 2) * state.hidden * 2), row(s.bank0, i, state.hidden * 2), state.hidden);
    }
    copyBf16(m, e, at(s.p, 0), at(s.bank1, 0), rows * state.hidden);
    e.barrier();
    loop.crossPassMessageAttention(&m.loop, e, l(at(s.bank0, 0)), l(at(s.bank1, 0)), l(r(m.weights.cross_queries)), l(r(m.weights.cross_kv_queries)), l(at(s.messages, 0)), @intCast(rows));
    // The prelude p is restored, but every visit's KV and convolution state remains live.
    copyBf16(m, e, at(s.prelude, 0), at(s.p, 0), rows * state.hidden);
    e.barrier();
}

pub fn gateRoles(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, lp: *const layer_plan.Layer, w: *const model_file.LayerWeights, core: usize, rows: usize) !void {
    const controllers = w.controllers.?;
    const stride: u32 = state.message_entries * state.hidden;
    if (m.loop.gates_mma != null and rows >= loop.mma_rows) {
        for (0..3) |role| {
            const c = controllers[role];
            const source_role = if (lp.kind == .local) 0 else role;
            try loop.gatesRows(&m.loop, e, l(arRow(s, source_role, rows, 0, false)), l(messageRow(s, core, role, 0)), stride, l(r(c.down)), l(r(c.forget)), l(r(c.gain)), l(arRow(s, role, rows, 0, true)), @intCast(rows));
        }
        e.barrier();
        return;
    }
    if (!loop.plan.splitRows(rows)) {
        for (0..3) |role| {
            const c = controllers[role];
            const source_role = if (lp.kind == .local) 0 else role;
            for (0..rows) |_| loop.gates(&m.loop, e, l(arRow(s, source_role, rows, 0, false)), l(messageRow(s, core, role, 0)), l(r(c.down)), l(r(c.forget)), l(r(c.gain)), l(arRow(s, role, rows, 0, true)), 1);
        }
        e.barrier();
        return;
    }
    for (0..3) |role| {
        const source_role = if (lp.kind == .local) 0 else role;
        loop.gatesSplitNorm(&m.loop, e, l(arRow(s, source_role, rows, 0, false)), l(messageRow(s, core, role, 0)), stride, l(gateNorm(s, role, rows)), @intCast(rows));
    }
    e.barrier();
    for (0..3) |role| loop.gatesSplitDown(&m.loop, e, l(gateNorm(s, role, rows)), stride, l(r(controllers[role].down)), l(gateZeta(s, role, rows)), @intCast(rows));
    e.barrier();
    for (0..3) |role| {
        const c = controllers[role];
        const source_role = if (lp.kind == .local) 0 else role;
        loop.gatesSplitApply(&m.loop, e, l(arRow(s, source_role, rows, 0, false)), l(messageRow(s, core, role, 0)), stride, l(gateZeta(s, role, rows)), l(r(c.forget)), l(r(c.gain)), l(arRow(s, role, rows, 0, true)), @intCast(rows));
    }
    e.barrier();
}

pub fn prepareAttentionInputs(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, rows: usize, pre_fused: bool) !void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    if (v.layer != 0 and !pre_fused) {
        const q = m.weights.attn_queries[v.layer - 1];
        for (0..lp.attn_queries) |role| {
            const query = at(q.buffer, q.offset + role * state.hidden * 4);
            for (0..rows) |i| glue.cachedAr(&m.glue, e, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), at(s.block_inv_rms, i * state.block_slots * 4), query, arRow(s, role, rows, i, false), lp.attn_sources, 1);
        }
        e.barrier();
    }
    const gated = v.hasMessages();
    if (gated) try gateRoles(m, e, s, lp, w, v.layer - 6, rows);
    // An ungated local visit has exactly one AR result and one norm result; Q/K/V alias.
    if (!gated and lp.kind == .local) {
        norm(m, e, arRow(s, 0, rows, 0, false), w.attn.q_norm, at(s.q_norm, 0), rows);
    } else {
        norm(m, e, arRow(s, 0, rows, 0, gated), w.attn.q_norm, at(s.q_norm, 0), rows);
        norm(m, e, arRow(s, 1, rows, 0, gated), w.attn.k_norm, at(s.k_norm, 0), rows);
        norm(m, e, arRow(s, 2, rows, 0, gated), w.attn.v_norm, at(s.v_norm, 0), rows);
    }
    e.barrier();
}

pub fn routedExperts(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, layer: usize, rows: usize) !void {
    const w = &m.weights.layers[layer];
    const precision: experts.Format = switch (m.weight_plan.at(@intCast(layer)).precision) {
        .nvfp4 => .nvfp4,
        .mxfp8 => .mxfp8,
        .exl3 => return error.UnsupportedInkling2ExpertFormat,
    };
    const gu_call = experts.Call{ .rows = @intCast(rows), .k = state.hidden, .n = state.intermediate, .format = precision };
    const down_call = experts.Call{ .rows = @intCast(rows), .k = state.intermediate, .n = state.hidden, .format = precision };
    if (m.experts.mma != null and rows >= experts.mma_rows) {
        // A row's sums are the same on either path, so a chunk or window takes the one that fits its width.
        const nv = precision == .nvfp4;
        const gu_globals: ?experts.Globals = if (nv) .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) } else null;
        const down_globals: ?experts.Globals = if (nv) .{ .amax = x(r(w.down.amax.?)), .denominator = x(r(w.down.denominator.?)) } else null;
        try experts.group(&m.experts, e, x(at(s.route_ids, 0)), gu_call);
        e.barrier();
        try experts.mma(&m.experts, e, false, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), gu_globals, x(at(s.gu, 0)), gu_call);
        e.barrier();
        try experts.mma(&m.experts, e, true, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), down_globals, x(at(s.down, 0)), down_call);
        e.barrier();
    } else if (rows >= 2) {
        var row0: usize = 0;
        while (row0 < rows) {
            const take: usize = @min(rows - row0, experts.max_grouped_rows);
            const chunk_gu = experts.Call{ .rows = @intCast(take), .k = state.hidden, .n = state.intermediate, .format = precision };
            const chunk_down = experts.Call{ .rows = @intCast(take), .k = state.intermediate, .n = state.hidden, .format = precision };
            const x_in = x(row(s.mlp_normed, row0, state.hidden * 2));
            const ids_in = x(row(s.route_ids, row0, state.experts_per_token * 4));
            const gu_out = x(row(s.gu, row0, state.experts_per_token * state.intermediate * 2));
            const down_out = x(row(s.down, row0, state.experts_per_token * state.hidden * 2));
            try experts.groupBuckets(&m.experts, e, ids_in, x(at(s.expert_group, 0)), chunk_gu);
            e.barrier();
            if (precision == .nvfp4) {
                try experts.guGrouped(&m.experts, e, x_in, x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.expert_group, 0)), gu_out, chunk_gu);
                e.barrier();
                try experts.nvfp4DownGrouped(&m.experts, e, gu_out, x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.expert_group, 0)), down_out, chunk_down);
            } else {
                try experts.guGrouped(&m.experts, e, x_in, x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.expert_group, 0)), gu_out, chunk_gu);
                e.barrier();
                try experts.mxfp8DownGrouped(&m.experts, e, gu_out, x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.expert_group, 0)), down_out, chunk_down);
            }
            e.barrier();
            row0 += take;
        }
    } else {
        if (precision == .nvfp4) {
            try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
        } else {
            try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
        }
        e.barrier();
        switch (precision) {
            .nvfp4 => try experts.nvfp4Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
            .mxfp8 => try experts.mxfp8Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
        }
        e.barrier();
    }
    try experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
    e.barrier();
}
