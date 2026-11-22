//! Mini's whole 48-visit forward, encoded entirely on Metal for prompt chunks and decode.
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
const model_file = @import("model.zig");
const Model = model_file.Model;

pub const Ref = state.Ref;
pub const Segment = struct { cache: *state.Cache, rows: usize };
pub const Options = struct {
    all: bool = false,
    ids: ?Ref = null,
    flight: ?*mtl.CommandBuffer = null,
    taps: ?state.Ref = null,
};

pub const FinalPlan = struct { work_rows: usize, logits_rows: usize, logits_row: usize, capture_rows: usize };

/// Features normalize all rows; without full logits, only the last unembeds. taps=null preserves the original plan.
pub fn finalPlan(rows: usize, all: bool, taps: bool) FinalPlan {
    std.debug.assert(rows > 0);
    return .{
        .work_rows = if (all or taps) rows else 1,
        .logits_rows = if (all) rows else 1,
        .logits_row = if (!all and taps) rows - 1 else 0,
        .capture_rows = if (taps) rows else 0,
    };
}

const RArgs = extern struct { count: u32, boundary: u32 = 0 };
const StageConvArgs = extern struct { rows: u32, width: u32, commit: u32, pairs: u32 };
const CommitHistoriesArgs = extern struct { offset: u32, slots: u32, tap_rows: u32 };
const NormArgs = extern struct { rows: u32, width: u32, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 };
const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

fn df(x_: Ref) dense_fuse.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}

fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
fn row(buf: mtl.Buffer, i: usize, row_bytes: usize) Ref {
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

fn runtime(m: *const Model, e: mtl.ComputeEncoder, op: model_file.RuntimeOp, refs: []const Ref, count: usize, boundary: usize) void {
    e.setPipeline(m.runtime[@backingInt(op)]);
    for (refs, 0..) |r_, i| e.setBuffer(r_.buf, r_.off, i);
    e.setValue(RArgs{ .count = @intCast(count), .boundary = @intCast(boundary) }, refs.len);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}

fn copyBf16(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, count: usize) void {
    runtime(m, e, .copy_bf16, &.{ from, to }, count, 0);
}
fn captureTap(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, rows: usize, channel: usize) void {
    std.debug.assert(channel < state.feature_taps);
    runtime(m, e, .capture_tap, &.{ from, to }, rows * state.hidden, channel);
}
fn widen(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, count: usize) void {
    runtime(m, e, .widen, &.{ from, to }, count, 0);
}

fn norm(m: *const Model, e: mtl.ComputeEncoder, input: Ref, weight: ck.Tensor, out: Ref, rows: usize) void {
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
    var base: usize = 0;
    while (base < rows) {
        const take = @min(rows - base, 64);
        loop.crossPassMessageAttention(&m.loop, e, l(row(s.bank0, base, state.hidden * 2)), l(row(s.bank1, base, state.hidden * 2)), l(r(m.weights.cross_queries)), l(r(m.weights.cross_kv_queries)), l(row(s.messages, base, state.message_entries * state.hidden * 4)), @intCast(take));
        base += take;
    }
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
        glue.arRows(&m.glue, e, at(s.blocks, 0), at(s.partial, 0), r(q), at(s.ar, 0), lp.attn_sources, @intCast(lp.attn_queries), @intCast(rows));
    }
    e.barrier();
    const gated = v.hasMessages();
    if (gated) {
        const controllers = w.controllers.?;
        const core = v.layer - 6;
        var base: usize = 0;
        while (base < rows) {
            const take = @min(rows - base, 64);
            var roles: [3]loop.Gate = undefined;
            for (&roles, 0..) |*g, role| {
                const c = controllers[role];
                const source_role = if (lp.kind == .local) 0 else role;
                g.* = .{ .h = l(arRow(s, source_role, rows, base, false)), .m = l(messageRow(s, core, role, base)), .down = l(r(c.down)), .forget = l(r(c.forget)), .gain = l(r(c.gain)), .out = l(arRow(s, role, rows, base, true)) };
            }
            loop.gatesEach(&m.loop, e, &roles, @intCast(take), state.message_entries * state.hidden);
            e.barrier();
            base += take;
        }
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

fn stageConv(m: *const Model, e: mtl.ComputeEncoder, hist0: Ref, raw0: Ref, staged0: Ref, hist1: Ref, raw1: Ref, staged1: Ref, rows: usize, width: usize, commit: bool, pairs: usize) void {
    std.debug.assert(pairs == 1 or pairs == 2);
    e.setPipeline(m.runtime[@backingInt(model_file.RuntimeOp.stage_conv)]);
    e.setBuffer(hist0.buf, hist0.off, 0);
    e.setBuffer(raw0.buf, raw0.off, 1);
    e.setBuffer(staged0.buf, staged0.off, 2);
    e.setBuffer(hist1.buf, hist1.off, 3);
    e.setBuffer(raw1.buf, raw1.off, 4);
    e.setBuffer(staged1.buf, staged1.off, 5);
    e.setValue(StageConvArgs{ .rows = @intCast(rows), .width = @intCast(width), .commit = @intFromBool(commit), .pairs = @intCast(pairs) }, 6);
    e.dispatchThreads(mtl.Size.of(pairs * width, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn commitHistories(m: *const Model, e: mtl.ComputeEncoder, cache: *const state.Cache, offset: usize) void {
    e.setPipeline(m.runtime[@backingInt(model_file.RuntimeOp.commit_histories)]);
    e.setBuffer(cache.taps, 0, 0);
    e.setBuffer(cache.conv, 0, 1);
    e.setValue(CommitHistoriesArgs{ .offset = @intCast(offset), .slots = state.visit_slots, .tap_rows = 3 + state.window_rows }, 2);
    e.dispatchThreads(mtl.Size.of(state.visit_slots * 3 * state.conv_row_values, 1, 1), mtl.Size.of(256, 1, 1));
}

/// Every stream's static convolution of `raw` into `out`: the taps staged, a barrier, the convolutions, a barrier.
fn postConv(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, segments: []const Segment, slot: usize, conv_state: usize, raw: Ref, weight: ck.Tensor, out: Ref) void {
    const channels = state.convWidth(conv_state);
    var base: usize = 0;
    for (segments) |seg| {
        const taps = if (segments.len == 1 and seg.cache.pending == null) at(s.conv_stage, 0) else seg.cache.staged(slot, conv_state, 0);
        const h = seg.cache.history(slot, conv_state);
        const raw_ref = at(raw.buf, raw.off + base * channels * 2);
        stageConv(m, e, h, raw_ref, taps, h, raw_ref, taps, seg.rows, channels, seg.cache.pending == null, 1);
        base += seg.rows;
    }
    e.barrier();
    base = 0;
    for (segments) |seg| {
        const taps = if (segments.len == 1 and seg.cache.pending == null) at(s.conv_stage, 0) else seg.cache.staged(slot, conv_state, 0);
        glue.convStatic(&m.glue, e, taps, r(weight), at(out.buf, out.off + base * channels * 2), @intCast(seg.rows), @intCast(channels));
        base += seg.rows;
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
fn attendSegments(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment) !void {
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
            .meta = .{ .pos = a(at(s.pos, base * 4)), .seg_of = a(at(s.seg_of, base * 4)), .segs = a(at(s.segs, si * @sizeOf(ap.Seg))) },
            .kv = .{ .k = a(cache.keys(v.cache_slot)), .v = a(cache.values(v.cache_slot)) },
        };
        base += seg.rows;
    }
    const all = items[0..segments.len];
    const scratch_stage = segments.len == 1 and segments[0].cache.pending == null;
    for (all) |it| stageConvKv(m, e, s, it.cache, v, it.base, it.rows, scratch_stage);
    e.barrier();
    for (all) |it| for ([_]bool{ true, false }) |key| convInputForAttention(m, e, s, it.cache, v, it.base, it.rows, scratch_stage, key);
    e.barrier();
    for (all) |it| {
        const b = it.base;
        attention.prep(&m.attention, e, it.rr, it.layer, prep, it.meta, .{ a(at(s.q_raw, b * state.query_width * 2)), a(at(s.k_conv, b * state.kv_width * 2)), a(at(s.v_conv, b * state.kv_width * 2)), a(at(s.rel_raw, b * state.rel_width * 2)) }, .{ a(r(w.attn.q_scale)), a(r(w.attn.rel_scale)) }, a(at(s.q_prep, b * state.query_width * 2)), a(at(s.rel_prep, b * state.rel_width * 2)), it.kv);
    }
    e.barrier();
    for (all) |it| attention.relTable(&m.attention, e, it.rr, a(at(s.rel_prep, it.base * state.rel_width * 2)), a(r(w.attn.rel_projection)), a(at(s.rel_tab, it.base * 24 * ap.extent * 2)));
    e.barrier();
    for (all) |it| {
        const b = it.base;
        try attention.partials(&m.attention, e, it.rr, it.layer, state.max_attention_stride, 0, it.meta, a(at(s.q_prep, b * state.query_width * 2)), it.kv, a(at(s.rel_tab, b * 24 * ap.extent * 2)), a(at(s.part_o, b * 24 * state.max_attention_stride * ap.head_dim * 4)), a(at(s.part_ml, b * 24 * state.max_attention_stride * 8)));
    }
    e.barrier();
    for (all) |it| {
        const b = it.base;
        try attention.merge(&m.attention, e, it.rr, it.layer, state.max_attention_stride, 0, it.meta, a(at(s.part_o, b * 24 * state.max_attention_stride * ap.head_dim * 4)), a(at(s.part_ml, b * 24 * state.max_attention_stride * 8)), a(at(s.sink, b * 24 * 2)), a(at(s.gate, b * state.query_width * 2)), a(at(s.att, b * state.query_width * 2)));
    }
    e.barrier();
}

const ConvInput = struct { raw: Ref, projected: Ref, normalized: Ref, taps: Ref, state: usize };

fn convInput(s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, local_ungated: bool, base: usize, rows: usize, scratch_stage: bool, key: bool) ConvInput {
    const cs: usize = if (key) 0 else 1;
    const normalized = if (key) at(s.k_norm, 0) else if (local_ungated) at(s.q_norm, 0) else at(s.v_norm, 0);
    const taps = if (scratch_stage) at(s.conv_stage, if (key) 0 else (3 + rows) * state.kv_width * 2) else cache.staged(v.cache_slot, cs, 0);
    return .{
        .raw = if (key) at(s.k_conv, base * state.kv_width * 2) else at(s.v_conv, base * state.kv_width * 2),
        .projected = if (key) at(s.k_raw, base * state.kv_width * 2) else at(s.v_raw, base * state.kv_width * 2),
        .normalized = at(normalized.buf, normalized.off + base * state.hidden * 2),
        .taps = taps,
        .state = cs,
    };
}

/// Stage both K and V convolution histories and new rows in one launch; updates history in place when not pending.
fn stageConvKv(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, scratch_stage: bool) void {
    const local_ungated = !v.hasMessages() and m.weight_plan.at(v.layer).kind != .global;
    const kc = convInput(s, cache, v, local_ungated, base, rows, scratch_stage, true);
    const vc = convInput(s, cache, v, local_ungated, base, rows, scratch_stage, false);
    stageConv(m, e, cache.history(v.cache_slot, 0), kc.projected, kc.taps, cache.history(v.cache_slot, 1), vc.projected, vc.taps, rows, state.kv_width, cache.pending == null, 2);
}

fn convInputForAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, scratch_stage: bool, key: bool) void {
    const w = &m.weights.layers[v.layer];
    const c = convInput(s, cache, v, !v.hasMessages() and m.weight_plan.at(v.layer).kind != .global, base, rows, scratch_stage, key);
    switch (if (key) w.attn.k_conv else w.attn.v_conv) {
        .static => |weight| glue.convStatic(&m.glue, e, c.taps, r(weight), c.raw, @intCast(rows), state.kv_width),
        .dynamic => |d| glue.convDynamic(&m.glue, e, c.taps, c.normalized, r(d.static_w), r(d.projection), c.raw, @intCast(rows), state.kv_width),
    }
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
fn visit(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment, rows: usize, taps: ?state.Ref) !void {
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
    // The running BF16 p is the partial AR source before attention replaces it with t.
    widen(m, e, at(s.p, 0), at(s.partial, 0), rows * state.hidden);
    e.barrier();
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
    try attendSegments(m, e, s, v, segments);
    dense(m, e, w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
    e.barrier();
    postConv(m, e, s, segments, v.cache_slot, 2, at(s.attn_dense, 0), w.attn_conv, at(s.attn_conv, 0));
    runtime(m, e, .attention_partial, &.{ at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.partial, 0) }, rows * state.hidden, if (lp.boundary) lp.blocks_in + 1 else 0);
    e.barrier();
    residual(m, e, s, m.weights.mlp_queries[v.layer], lp.mlp_sources, rows);
    e.barrier();
    if (gated) {
        const c = w.controllers.?[3];
        const core = v.layer - 6;
        var base: usize = 0;
        while (base < rows) {
            const take = @min(rows - base, 64);
            loop.gatesStrided(&m.loop, e, l(row(s.mlp_ar, base, state.hidden * 4)), l(messageRow(s, core, 3, base)), l(r(c.down)), l(r(c.forget)), l(r(c.gain)), l(row(s.mlp_gated, base, state.hidden * 4)), @intCast(take), state.message_entries * state.hidden);
            e.barrier();
            base += take;
        }
        norm(m, e, at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    } else norm(m, e, at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    e.barrier();
    var rbase: usize = 0;
    while (rbase < rows) {
        const take = @min(rows - rbase, 64);
        glue.router(&m.glue, e, row(s.mlp_normed, rbase, state.hidden * 2), r(w.router_weight), r(w.router_bias), row(s.route_ids, rbase, state.experts_per_token * 4), row(s.route_weights, rbase, state.experts_per_token * 4), @intCast(take));
        e.barrier();
        rbase += take;
    }
    try routedExperts(m, e, s, v.layer, rows);
    e.barrier();
    postConv(m, e, s, segments, v.cache_slot, 3, at(s.mlp_out, 0), w.mlp_conv, at(s.mlp_conv, 0));
    runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, rows * state.hidden, 0);
    e.barrier();
    if (v.cache_slot == 5) copyBf16(m, e, at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
}

fn commitFullHistories(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, segments: []const Segment) void {
    _ = s;
    for (segments) |seg| {
        if (seg.cache.pending != null) commitHistories(m, e, seg.cache, seg.rows);
    }
}

pub fn run(m: *const Model, s: *state.Scratch, cache: *state.Cache, ids: []const u32, options: Options) !void {
    return runShared(m, s, &.{.{ .cache = cache, .rows = ids.len }}, ids, options);
}

pub fn runShared(m: *const Model, s: *state.Scratch, segments: []const Segment, ids: []const u32, options: Options) !void {
    const rows = ids.len;
    if (rows == 0 or rows > s.rows or segments.len == 0 or segments.len > state.max_segments or (!options.all and segments.len != 1) or (options.all and rows > state.batch_rows)) return error.Inkling2RowsExceeded;
    var total: usize = 0;
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        if (seg.rows == 0 or seg.rows > state.max_rows or seg.cache.len + seg.rows > seg.cache.capacity) return error.Inkling2ContextFull;
        if (seg.rows > state.window_rows and seg.cache.pending != null) return error.Inkling2CacheTransaction;
        if (segments.len > 1 and seg.rows > state.window_rows) return error.Inkling2RowsExceeded;
        if (seg.cache.pending) |pending| if (pending != seg.rows) return error.Inkling2CacheTransaction;
        total += seg.rows;
        for (0..seg.rows) |i| {
            s.pos.slice(i32, rows)[base + i] = @intCast(seg.cache.len + i);
            s.seg_of.slice(i32, rows)[base + i] = 0;
        }
        s.segs.slice(ap.Seg, state.max_segments)[si] = .{ .row0 = 0, .n = @intCast(seg.rows), .p0 = @intCast(seg.cache.len), .off = 0 };
        base += seg.rows;
    }
    if (total != rows) return error.Inkling2Segments;
    if (options.ids == null) {
        for (ids) |id| if (id >= m.config.text.vocab) return error.Inkling2Token;
        @memcpy(s.ids.slice(u32, rows), ids);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var cb = m.queue.commandBuffer();
    var e = cb.compute(.concurrent);
    // Earlier parts of this forward, already committed: the GPU runs them while the host encodes the next.
    var parts: [cuts.len]mtl.CommandBuffer = undefined;
    var n_parts: usize = 0;
    defer for (parts[0..n_parts]) |part| part.release();
    errdefer if (n_parts > 0) parts[n_parts - 1].wait();
    glue.embedNorm(&m.glue, e, options.ids orelse at(s.ids, 0), r(m.weights.embed), r(m.weights.embed_norm), at(s.p, 0), @intCast(rows));
    e.barrier();
    for (m.weight_plan.visits[0..m.weight_plan.n_visits], 0..) |*v, vi| {
        try visit(m, e, s, v, segments, rows, options.taps);
        if (n_parts < cuts.len and vi + 1 == cuts[n_parts]) {
            e.end();
            cb.commit();
            parts[n_parts] = cb.retain();
            n_parts += 1;
            cb = m.queue.commandBuffer();
            e = cb.compute(.concurrent);
        }
    }
    widen(m, e, at(s.p, 0), at(s.partial, 0), rows * state.hidden);
    e.barrier();
    const final = finalPlan(rows, options.all, options.taps != null);
    if (final.work_rows == rows) {
        glue.arRows(&m.glue, e, at(s.blocks, 0), at(s.partial, 0), r(m.weights.output_query), at(s.mlp_ar, 0), m.weight_plan.final_sources, 1, @intCast(rows));
    } else {
        const i = rows - 1;
        glue.ar(&m.glue, e, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), r(m.weights.output_query), at(s.mlp_ar, 0), m.weight_plan.final_sources, 1);
    }
    e.barrier();
    norm(m, e, at(s.mlp_ar, 0), m.weights.norm, at(s.mlp_normed, 0), final.work_rows);
    e.barrier();
    if (options.taps) |to| captureTap(m, e, at(s.mlp_normed, 0), to, final.capture_rows, 3);
    const logits = if (options.all) at(s.logits_all, 0) else at(segments[0].cache.logits, 0);
    glue.logits(&m.glue, e, at(s.mlp_normed, final.logits_row * state.hidden * 2), r(m.weights.unembed), logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab);
    commitFullHistories(m, e, s, segments);
    e.end();
    cb.commit();
    if (options.flight) |flight| {
        flight.* = cb.retain();
        return;
    }
    cb.wait();
    for (parts[0..n_parts]) |part| if (part.failure()) |msg| {
        std.log.err("inkling2_loop forward: {s}", .{msg});
        return error.GpuFailed;
    };
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop forward: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// Visits after which a forward's command buffer is committed and the next begun.
const cuts = [_]usize{ 2, 12, 24, 36 };
