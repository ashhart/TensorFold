//! Inkling 2 Mini per-visit layer execution and shared dense/norm/runtime encoders.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const experts = @import("experts.zig");
const loop = @import("loop.zig");
const dense_fuse = @import("dense_fuse.zig");
const mv_mma = @import("../../deepseek_v41/metal/mv_mma.zig");
const state = @import("state.zig");
const conv = @import("forward_conv.zig");
const model_file = @import("model.zig");

const Model = model_file.Model;
pub const Ref = state.Ref;
pub const Segment = conv.Segment;
pub const Meta = conv.Meta;
pub const declareConvResources = conv.declareConvResources;
pub const prepareConvTables = conv.prepareConvTables;
pub const commitHistories = conv.commitHistories;

const RArgs = extern struct { count: u32, boundary: u32 = 0 };
const NormArgs = extern struct { rows: u32, width: u32, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 };
const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

pub fn df(x_: Ref) dense_fuse.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}

pub fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

pub fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}

pub fn row(buf: mtl.Buffer, i: usize, row_bytes: usize) Ref {
    return at(buf, i * row_bytes);
}

fn a(x_: Ref) attention.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}

fn x(x_: Ref) experts.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}

fn l(x_: Ref) loop.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}

fn arRow(s: *state.Scratch, role: usize, rows: usize, i: usize, gated: bool) Ref {
    return at(if (gated) s.gated else s.ar, (role * rows + i) * state.hidden * 4);
}

fn messageRow(s: *state.Scratch, core: usize, role: usize, i: usize) Ref {
    return at(s.messages, ((i * state.message_entries + core * 4 + role) * state.hidden) * 4);
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

/// Every projection uses the shared BF16 matvec's FP32 reduction and one BF16 store.
pub fn dense(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    const k = weight.shape[weight.rank - 1];
    const n = weight.shape[weight.rank - 2];
    if (rows > 32 and k >= 32) {
        mv_mma.encode(&m.dense_mma, e, .{ .buf = input.buf, .off = input.off }, .{ .buf = weight.buffer, .off = weight.offset }, .{ .buf = out.buf, .off = out.off }, .{ .k = @intCast(k), .n = @intCast(n), .rows = @intCast(rows), .out = .bf16 }) catch unreachable;
        return;
    }
    var base: usize = 0;
    while (base < rows) {
        const left = rows - base;
        // One launch of the next size up: the kernel stores only `take` rows, and reads one pass of the weights.
        const take: usize = @min(left, 32);
        const variant: usize = if (take > 16) 5 else if (take > 8) 4 else if (take > 4) 3 else if (take > 2) 2 else if (take > 1) 1 else 0;
        e.setPipeline(m.dense[variant]);
        e.setBuffer(input.buf, input.off + base * k * 2, 0);
        e.setBuffer(weight.buffer, weight.offset, 1);
        e.setValue(DenseArgs{ .rows = @intCast(take), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
        e.setBuffer(out.buf, out.off + base * n * 2, 3);
        e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        base += take;
    }
}

/// One row because M2's AR ABI is [sources,H] -> [queries,H]. Batch composition never mixes rows.
fn residual(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, queries: ck.Tensor, sources: usize, rows: usize) void {
    glue.arRows(&m.glue, e, at(s.blocks, 0), at(s.partial, 0), r(queries), at(s.mlp_ar, 0), @intCast(sources), 1, @intCast(rows));
}

/// Ends with a barrier: the cross-pass messages and the restored p are written.
fn bankAndReset(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, rows: usize) void {
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

/// Ends with a barrier: the norms are written.
fn prepareAttentionInputs(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, rows: usize) void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    if (v.layer == 0) {
        widen(m, e, at(s.p, 0), at(s.ar, 0), rows * state.hidden);
    } else {
        const q = m.weights.attn_queries[v.layer - 1];
        glue.arRowsBf16(&m.glue, e, at(s.blocks, 0), at(s.p, 0), r(q), at(s.ar, 0), lp.attn_sources, @intCast(lp.attn_queries), @intCast(rows));
    }
    e.barrier();
    const gated = v.hasMessages();
    if (gated) {
        const controllers = w.controllers.?;
        const core = v.layer - 6;
        var roles: [3]loop.Gate = undefined;
        for (&roles, 0..) |*g, role| {
            const c = controllers[role];
            const source_role = if (lp.kind == .local) 0 else role;
            g.* = .{ .h = l(arRow(s, source_role, rows, 0, false)), .m = l(messageRow(s, core, role, 0)), .down = l(r(c.down)), .forget = l(r(c.forget)), .gain = l(r(c.gain)), .out = l(arRow(s, role, rows, 0, true)) };
        }
        loop.gatesEach(&m.loop, e, &roles, @intCast(rows), state.message_entries * state.hidden);
        e.barrier();
    }
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

/// One stream's rows of a visit's attention.
const Attend = struct {
    cache: *state.Cache,
    base: usize,
    rows: usize,
    rr: ap.Rows,
    layer: ap.Layer,
    meta: attention.RowMeta,
    kv: attention.Cache,
};

/// Every stream's attention a step at a time: a step's launches run side by side, and a barrier ends each step.
fn attendSegments(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment, meta: ?Ref) !void {
    const w = &m.weights.layers[v.layer];
    const prep = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
    var items: [state.max_segments]Attend = undefined;
    var base: usize = 0;
    for (segments, items[0..segments.len], 0..) |seg, *it, si| {
        const cache = seg.cache;
        const layer0 = cache.attentionLayer(&m.weight_plan, v.cache_slot, seg.rows);
        it.* = .{
            .cache = cache,
            .base = base,
            .rows = seg.rows,
            .rr = .{ .rows = @intCast(seg.rows), .heads = 24, .kv_heads = 8, .last_pos = @intCast(cache.len + seg.rows - 1), .streams = 1 },
            .layer = .{ .window = layer0.window, .ring = @intCast(cache.ring(v.cache_slot)), .chunk = layer0.chunk, .tau = layer0.tau },
            .meta = if (meta) |mref| .{
                .pos = a(.{ .buf = mref.buf, .off = mref.off + @offsetOf(Meta, "pos") }),
                .seg_of = a(.{ .buf = mref.buf, .off = mref.off + @offsetOf(Meta, "seg_of") }),
                .segs = a(.{ .buf = mref.buf, .off = mref.off + @offsetOf(Meta, "seg") }),
            } else .{ .pos = a(at(s.pos, base * 4)), .seg_of = a(at(s.seg_of, base * 4)), .segs = a(at(s.segs, si * @sizeOf(ap.Seg))) },
            .kv = .{ .k = a(cache.keys(v.cache_slot)), .v = a(cache.values(v.cache_slot)) },
        };
        base += seg.rows;
    }
    const all = items[0..segments.len];
    const scratch_stage = segments.len == 1 and segments[0].cache.pending == null;
    if (segments.len > 1 or base <= state.window_rows) {
        const segs_ref = conv.convSegsRef(s, meta, v.cache_slot, 0);
        switch (w.attn.k_conv) {
            .static => |k_w| {
                conv.stageConvStatic(m, e, segs_ref, r(k_w), r(w.attn.v_conv.static), segments.len, state.kv_width, 2);
                e.barrier();
            },
            .dynamic => |k_dyn| {
                conv.stageConvSegs(m, e, segs_ref, segments.len, state.kv_width, 2);
                e.barrier();
                const v_dyn = w.attn.v_conv.dynamic;
                glue.convDynamicSegs(&m.glue, e, conv.convDynSegsRef(s, meta, v.cache_slot, 0), r(k_dyn.static_w), r(k_dyn.projection), @intCast(segments.len), @intCast(base), state.kv_width);
                glue.convDynamicSegs(&m.glue, e, conv.convDynSegsRef(s, meta, v.cache_slot, 1), r(v_dyn.static_w), r(v_dyn.projection), @intCast(segments.len), @intCast(base), state.kv_width);
                e.barrier();
            },
        }
    } else {
        for (all) |it| conv.stageConvKv(m, e, s, it.cache, v, it.base, it.rows, scratch_stage);
        e.barrier();
        for (all) |it| for ([_]bool{ true, false }) |key| conv.convInputForAttention(m, e, s, it.cache, v, it.base, it.rows, scratch_stage, key);
        e.barrier();
    }
    for (all) |it| {
        const b = it.base;
        attention.prep(&m.attention, e, it.rr, it.layer, prep, it.meta, .{ a(at(s.q_raw, b * state.query_width * 2)), a(at(s.k_conv, b * state.kv_width * 2)), a(at(s.v_conv, b * state.kv_width * 2)), a(at(s.rel_raw, b * state.rel_width * 2)) }, .{ a(r(w.attn.q_scale)), a(r(w.attn.rel_scale)) }, a(at(s.q_prep, b * state.query_width * 2)), a(at(s.rel_prep, b * state.rel_width * 2)), it.kv);
    }
    e.barrier();
    for (all) |it| attention.relTable(&m.attention, e, it.rr, a(at(s.rel_prep, it.base * state.rel_width * 2)), a(r(w.attn.rel_projection)), a(at(s.rel_tab, it.base * 24 * ap.extent * 2)));
    e.barrier();
    const stride = s.attn_stride;
    if (all.len == 1) {
        const it = all[0];
        const b = it.base;
        try attention.attend(&m.attention, e, it.rr, it.layer, stride, 0, it.meta, a(at(s.q_prep, b * state.query_width * 2)), it.kv, a(at(s.rel_tab, b * 24 * ap.extent * 2)), a(at(s.part_o, b * 24 * @as(usize, stride) * ap.head_dim * 4)), a(at(s.part_ml, b * 24 * @as(usize, stride) * 8)), a(at(s.sink, b * 24 * 2)), a(at(s.gate, b * state.query_width * 2)), a(at(s.att, b * state.query_width * 2)));
        e.barrier();
        return;
    }
    for (all) |it| {
        const b = it.base;
        try attention.partials(&m.attention, e, it.rr, it.layer, stride, 0, it.meta, a(at(s.q_prep, b * state.query_width * 2)), it.kv, a(at(s.rel_tab, b * 24 * ap.extent * 2)), a(at(s.part_o, b * 24 * @as(usize, stride) * ap.head_dim * 4)), a(at(s.part_ml, b * 24 * @as(usize, stride) * 8)));
    }
    e.barrier();
    for (all) |it| {
        const b = it.base;
        try attention.merge(&m.attention, e, it.rr, it.layer, stride, 0, it.meta, a(at(s.part_o, b * 24 * @as(usize, stride) * ap.head_dim * 4)), a(at(s.part_ml, b * 24 * @as(usize, stride) * 8)), a(at(s.sink, b * 24 * 2)), a(at(s.gate, b * state.query_width * 2)), a(at(s.att, b * state.query_width * 2)));
    }
    e.barrier();
}

/// The combine's output is written by the last launch: the caller's barrier follows.
fn routedExperts(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, layer: usize, rows: usize) !void {
    const w = &m.weights.layers[layer];
    const precision: experts.Format = switch (m.weight_plan.at(@intCast(layer)).precision) {
        .nvfp4 => .nvfp4,
        .mxfp8 => .mxfp8,
    };
    const gu_call = experts.Call{ .rows = @intCast(rows), .k = state.hidden, .n = state.intermediate, .format = precision };
    const down_call = experts.Call{ .rows = @intCast(rows), .k = state.intermediate, .n = state.hidden, .format = precision };
    if (m.experts.units != null) {
        // On the tensor units a row's sums are the same at every width, so every chunk and window takes this path.
        try experts.group(&m.experts, e, x(at(s.route_ids, 0)), gu_call);
        e.barrier();
        if (precision == .nvfp4) {
            try experts.units(&m.experts, e, false, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.gu, 0)), gu_call);
            e.barrier();
            try experts.units(&m.experts, e, true, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), .{ .amax = x(r(w.down.amax.?)), .denominator = x(r(w.down.denominator.?)) }, x(at(s.down, 0)), down_call);
        } else {
            try experts.units(&m.experts, e, false, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.gu, 0)), gu_call);
            e.barrier();
            try experts.units(&m.experts, e, true, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), null, x(at(s.down, 0)), down_call);
        }
        e.barrier();
        return experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
    }
    if (rows >= experts.grouped_rows) {
        try experts.group(&m.experts, e, x(at(s.route_ids, 0)), gu_call);
        e.barrier();
        if (precision == .nvfp4) {
            try experts.guGrouped(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.gu, 0)), gu_call);
            e.barrier();
            try experts.downGrouped(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), .{ .amax = x(r(w.down.amax.?)), .denominator = x(r(w.down.denominator.?)) }, x(at(s.down, 0)), down_call);
            e.barrier();
        } else {
            try experts.guGrouped(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.gu, 0)), gu_call);
            e.barrier();
            try experts.downGrouped(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), null, x(at(s.down, 0)), down_call);
            e.barrier();
        }
        return experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
    }
    if (precision == .nvfp4) {
        try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
        e.barrier();
    } else {
        try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
        e.barrier();
    }
    switch (precision) {
        .nvfp4 => try experts.nvfp4Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
        .mxfp8 => try experts.mxfp8Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
    }
    e.barrier();
    try experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
}

/// A visit's launches in dependency order; a barrier follows every step a later launch reads.
pub fn visit(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment, rows: usize, taps: ?state.Ref, meta: ?Ref) !void {
    const lp = m.weight_plan.at(v.layer);
    const gated = v.hasMessages();
    if (taps) |to| {
        const channel: ?usize = switch (v.cache_slot) {
            5 => 0,
            23 => 1,
            41 => 2,
            else => null,
        };
        if (channel) |ch| captureTap(m, e, at(s.p, 0), to, rows, ch);
    }
    if (v.pass > 1 and v.layer == 6) bankAndReset(m, e, s, rows);
    const w = &m.weights.layers[v.layer];
    prepareAttentionInputs(m, e, s, v, rows);
    const knorm = if (gated or lp.kind == .global) at(s.k_norm, 0) else at(s.q_norm, 0);
    const vnorm = if (gated or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    if (rows <= 32) {
        try dense_fuse.encode6(
            &m.dense_fuse,
            e,
            df(at(s.q_norm, 0)),
            df(knorm),
            df(vnorm),
            .{ .w = df(r(w.attn.wq)), .y = df(at(s.q_raw, 0)), .n = state.query_width },
            .{ .w = df(r(w.attn.wk)), .y = df(at(s.k_raw, 0)), .n = state.kv_width },
            .{ .w = df(r(w.attn.wv)), .y = df(at(s.v_raw, 0)), .n = state.kv_width },
            .{ .w = df(r(w.attn.wr)), .y = df(at(s.rel_raw, 0)), .n = state.rel_width },
            .{ .w = df(r(w.attn.sink)), .y = df(at(s.sink, 0)), .n = 24 },
            .{ .w = df(r(w.attn.gate)), .y = df(at(s.gate, 0)), .n = state.query_width },
            @intCast(rows),
            state.hidden,
        );
    } else {
        dense(m, e, w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
        dense(m, e, w.attn.wk, knorm, at(s.k_raw, 0), rows);
        dense(m, e, w.attn.wv, vnorm, at(s.v_raw, 0), rows);
        dense(m, e, w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
        dense(m, e, w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
        dense(m, e, w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
    }
    e.barrier();
    try attendSegments(m, e, s, v, segments, meta);
    dense(m, e, w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
    e.barrier();
    conv.postConv(m, e, s, segments, meta, v.cache_slot, 2, at(s.attn_dense, 0), w.attn_conv, at(s.attn_conv, 0));
    runtime(m, e, .attention_partial, &.{ at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.partial, 0) }, rows * state.hidden, if (lp.boundary) lp.blocks_in + 1 else 0);
    e.barrier();
    residual(m, e, s, m.weights.mlp_queries[v.layer], lp.mlp_sources, rows);
    e.barrier();
    if (gated) {
        const c = w.controllers.?[3];
        const core = v.layer - 6;
        loop.gatesStrided(&m.loop, e, l(at(s.mlp_ar, 0)), l(messageRow(s, core, 3, 0)), l(r(c.down)), l(r(c.forget)), l(r(c.gain)), l(at(s.mlp_gated, 0)), @intCast(rows), state.message_entries * state.hidden);
        e.barrier();
        norm(m, e, at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    } else norm(m, e, at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    e.barrier();
    glue.router(&m.glue, e, at(s.mlp_normed, 0), r(w.router_weight), r(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
    e.barrier();
    try routedExperts(m, e, s, v.layer, rows);
    e.barrier();
    conv.postConv(m, e, s, segments, meta, v.cache_slot, 3, at(s.mlp_out, 0), w.mlp_conv, at(s.mlp_conv, 0));
    runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, rows * state.hidden, 0);
    e.barrier();
    if (v.cache_slot == 5) copyBf16(m, e, at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
}
