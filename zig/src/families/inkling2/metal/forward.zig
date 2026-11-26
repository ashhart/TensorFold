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
const dense_mma = @import("dense.zig");
const tree = @import("tree.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const Model = model_file.Model;

pub const Ref = state.Ref;
pub const Segment = struct { cache: *state.Cache, rows: usize, parents: ?[]const i32 = null };
const SegMeta = struct { is_tree: bool = false, max_depth: u32 = 0 };
/// The BF16 rows [off, off + rows * hidden * 2) of `buf` replace the normed embeddings of chunk rows [row, row + rows).
pub const ImageRows = struct { row: u32, rows: u32, buf: mtl.Buffer, off: usize = 0 };
pub const Options = struct {
    all: bool = false,
    ids: ?Ref = null,
    flight: ?*mtl.CommandBuffer = null,
    taps: ?state.Ref = null,
    images: []const ImageRows = &.{},
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
const NormArgs = extern struct { rows: u32, width: u32, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 };
pub const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

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

/// One controller over every row: m's rows are a token's message entries apart; the bits equal per-row `loop.gates`.
fn gates(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, h: Ref, msg: Ref, c: model_file.Controller, out: Ref, rows: usize) !void {
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

pub const DenseStep = struct { variant: usize, rows: usize };

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
fn project(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    if (m.dense_mma) |mma| {
        const n: u32 = @intCast(weight.shape[weight.rank - 2]);
        const call = dense_mma.Call{ .rows = @intCast(rows), .n = n, .k = @intCast(weight.shape[weight.rank - 1]), .ldy = n };
        if (dense_mma.Mma.serves(call)) return mma.launch(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = weight.buffer, .off = weight.offset }, .{ .buf = out.buf, .off = out.off }, call);
    }
    dense(m, e, weight, input, out, rows);
}

pub fn canPreFuse(v: *const layer_plan.Visit) bool {
    return v.layer != 0 and !(v.pass > 1 and v.layer == 6);
}

fn bankAndReset(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, rows: usize) void {
    for (0..rows) |i| {
        // The second completed block of the previous pass is B12; its final p is P18.
        copyBf16(m, e, at(s.blocks, (i * state.block_slots + 2) * state.hidden * 2), row(s.bank0, i, state.hidden * 2), state.hidden);
    }
    copyBf16(m, e, at(s.p, 0), at(s.bank1, 0), rows * state.hidden);
    loop.crossPassMessageAttention(&m.loop, e, l(at(s.bank0, 0)), l(at(s.bank1, 0)), l(r(m.weights.cross_queries)), l(r(m.weights.cross_kv_queries)), l(at(s.messages, 0)), @intCast(rows));
    // The prelude p is restored, but every visit's KV and convolution state remains live.
    copyBf16(m, e, at(s.prelude, 0), at(s.p, 0), rows * state.hidden);
}

fn prepareAttentionInputs(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, rows: usize, pre_fused: bool) !void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    if (v.layer == 0) {
        widen(m, e, at(s.p, 0), at(s.ar, 0), rows * state.hidden);
    } else if (!pre_fused) {
        const q = m.weights.attn_queries[v.layer - 1];
        for (0..lp.attn_queries) |role| {
            const query = at(q.buffer, q.offset + role * state.hidden * 4);
            for (0..rows) |i| glue.cachedAr(&m.glue, e, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), at(s.block_inv_rms, i * state.block_slots * 4), query, arRow(s, role, rows, i, false), lp.attn_sources, 1);
        }
    }
    if (v.hasMessages()) {
        const controllers = w.controllers.?;
        const core = v.layer - 6;
        for (0..3) |role| {
            const c = controllers[role];
            const source_role = if (lp.kind == .local) 0 else role;
            try gates(m, e, s, arRow(s, source_role, rows, 0, false), messageRow(s, core, role, 0), c, arRow(s, role, rows, 0, true), rows);
        }
    }
    const gated = v.hasMessages();
    // An ungated local visit has exactly one AR result and one norm result; Q/K/V alias.
    if (!gated and lp.kind == .local) {
        norm(m, e, arRow(s, 0, rows, 0, false), w.attn.q_norm, at(s.q_norm, 0), rows);
    } else {
        norm(m, e, arRow(s, 0, rows, 0, gated), w.attn.q_norm, at(s.q_norm, 0), rows);
        norm(m, e, arRow(s, 1, rows, 0, gated), w.attn.k_norm, at(s.k_norm, 0), rows);
        norm(m, e, arRow(s, 2, rows, 0, gated), w.attn.v_norm, at(s.v_norm, 0), rows);
    }
}

fn staticPostConv(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, slot: usize, conv_state: usize, raw: Ref, weight: ck.Tensor, out: Ref, base: usize, rows: usize, is_tree: bool) void {
    const channels = state.convWidth(conv_state);
    const taps = cache.staged(slot, conv_state, 0);
    copyBf16(m, e, cache.history(slot, conv_state), taps, 3 * channels);
    copyBf16(m, e, at(raw.buf, raw.off + base * channels * 2), cache.staged(slot, conv_state, 3), rows * channels);
    if (is_tree) {
        tree.convStatic(&m.tree, e, taps, r(weight), at(s.conv_src, base * 4 * 4), at(out.buf, out.off + base * channels * 2), @intCast(rows), @intCast(channels));
    } else {
        glue.convStatic(&m.glue, e, taps, r(weight), at(out.buf, out.off + base * channels * 2), @intCast(rows), @intCast(channels));
    }
}

fn attendSegments(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment, seg_meta: []const SegMeta) !void {
    const prep = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
    const proj_ref = a(r(m.weights.layers[v.layer].attn.rel_projection));
    const scale_refs = [2]attention.Ref{ a(r(m.weights.layers[v.layer].attn.q_scale)), a(r(m.weights.layers[v.layer].attn.rel_scale)) };
    var specs: [state.max_segments]ap.BatchedSegSpec = undefined;
    for (segments, seg_meta, 0..) |seg, sm, si| {
        const layer0 = seg.cache.attentionLayer(&m.weight_plan, v.cache_slot, seg.rows);
        const layer = ap.Layer{ .window = layer0.window, .ring = @intCast(seg.cache.ring(v.cache_slot)), .chunk = layer0.chunk, .tau = layer0.tau };
        const last_pos: u32 = @intCast(seg.cache.len + (if (sm.is_tree) sm.max_depth else seg.rows - 1));
        specs[si] = .{ .rows = @intCast(seg.rows), .last_pos = last_pos, .layer = layer, .is_tree = sm.is_tree };
    }
    if (ap.planBatched(specs[0..segments.len], 24, 8, state.max_attention_stride, 0)) |batch| {
        var caches: [state.max_segments]attention.BatchedCache = undefined;
        var base: usize = 0;
        for (segments, seg_meta, 0..) |seg, sm, si| {
            try convInputForAttention(m, e, s, seg.cache, v, base, seg.rows, true, sm.is_tree);
            try convInputForAttention(m, e, s, seg.cache, v, base, seg.rows, false, sm.is_tree);
            caches[si] = .{
                .k = a(seg.cache.keys(v.cache_slot)),
                .v = a(seg.cache.values(v.cache_slot)),
                .row0 = @intCast(base),
                .n = @intCast(seg.rows),
                .p0 = @intCast(seg.cache.len),
                .ring = specs[si].layer.ring,
            };
            base += seg.rows;
        }
        const rr = ap.Rows{ .rows = batch.total_rows, .heads = 24, .kv_heads = 8, .last_pos = 0, .streams = @intCast(segments.len) };
        const raw_refs = [4]attention.Ref{ a(at(s.q_raw, 0)), a(at(s.k_conv, 0)), a(at(s.v_conv, 0)), a(at(s.rel_raw, 0)) };
        try attention.prepBatched(&m.attention, e, rr, batch.tau, prep, a(at(s.pos, 0)), a(at(s.seg_of, 0)), caches[0..segments.len], raw_refs, scale_refs, a(at(s.q_prep, 0)), a(at(s.rel_prep, 0)));
        e.barrier();
        try attention.attendBatched(&m.attention, e, batch, 24, 8, a(at(s.pos, 0)), a(at(s.seg_of, 0)), caches[0..segments.len], a(at(s.q_prep, 0)), a(at(s.rel_prep, 0)), proj_ref, a(at(s.part_o, 0)), a(at(s.part_ml, 0)), a(at(s.sink, 0)), a(at(s.gate, 0)), a(at(s.att, 0)));
        return;
    }
    var base: usize = 0;
    for (segments, seg_meta, 0..) |seg, sm, si| {
        const rows = seg.rows;
        const cache = seg.cache;
        const layer = specs[si].layer;
        const start = cache.len;
        const last_pos = specs[si].last_pos;
        const rr = ap.Rows{ .rows = @intCast(rows), .heads = 24, .kv_heads = 8, .last_pos = last_pos, .streams = 1 };
        const meta = attention.RowMeta{
            .pos = a(at(s.pos, base * 4)),
            .seg_of = a(at(s.seg_of, base * 4)),
            .segs = a(at(s.segs, 0)),
            .prompt_segs = a(at(s.segs, si * @sizeOf(ap.Seg))),
        };
        const kv = attention.Cache{ .k = a(cache.keys(v.cache_slot)), .v = a(cache.values(v.cache_slot)) };
        try convInputForAttention(m, e, s, cache, v, base, rows, true, sm.is_tree);
        try convInputForAttention(m, e, s, cache, v, base, rows, false, sm.is_tree);
        const rel_ref = a(at(s.rel_prep, base * state.rel_width * 2));
        const q_ref = a(at(s.q_prep, base * state.query_width * 2));
        const po_ref = a(at(s.part_o, base * 24 * state.max_attention_stride * ap.head_dim * 4));
        const pml_ref = a(at(s.part_ml, base * 24 * state.max_attention_stride * 8));
        const sink_ref = a(at(s.sink, base * 24 * 2));
        const gate_ref = a(at(s.gate, base * state.query_width * 2));
        const att_ref = a(at(s.att, base * state.query_width * 2));
        const raw_refs = [4]attention.Ref{ a(at(s.q_raw, base * state.query_width * 2)), a(at(s.k_conv, base * state.kv_width * 2)), a(at(s.v_conv, base * state.kv_width * 2)), a(at(s.rel_raw, base * state.rel_width * 2)) };
        if (sm.is_tree) {
            attention.prepTree(&m.attention, e, rr, layer, prep, meta, raw_refs, scale_refs, q_ref, rel_ref, kv);
            try attention.attendTree(&m.attention, e, rr, layer, state.max_attention_stride, @intCast(start), meta, a(at(s.tree_paths, base * state.window_rows * 4)), a(at(s.tree_tail_o, base * 2 * 24 * ap.head_dim * 4)), a(at(s.tree_tail_ml, base * 24 * 16)), q_ref, kv, rel_ref, proj_ref, po_ref, pml_ref, sink_ref, gate_ref, att_ref);
        } else {
            const stride = ap.slotsFor(layer, @as(usize, last_pos) + 1);
            if (stride > state.max_attention_stride) return error.Inkling2Partials;
            attention.prep(&m.attention, e, rr, layer, prep, meta, raw_refs, scale_refs, q_ref, rel_ref, kv);
            try attention.attend(&m.attention, e, rr, layer, stride, 0, meta, q_ref, kv, rel_ref, proj_ref, po_ref, pml_ref, sink_ref, gate_ref, att_ref);
        }
        base += rows;
    }
}

fn convInputForAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, key: bool, is_tree: bool) !void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    const raw = if (key) at(s.k_conv, base * state.kv_width * 2) else at(s.v_conv, base * state.kv_width * 2);
    const projected = if (key) at(s.k_raw, base * state.kv_width * 2) else at(s.v_raw, base * state.kv_width * 2);
    const normalized = if (key) at(s.k_norm, 0) else if (v.hasMessages() or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    const taps = cache.staged(v.cache_slot, if (key) 0 else 1, 0);
    copyBf16(m, e, cache.history(v.cache_slot, if (key) 0 else 1), taps, 3 * state.kv_width);
    copyBf16(m, e, projected, cache.staged(v.cache_slot, if (key) 0 else 1, 3), rows * state.kv_width);
    if (is_tree) {
        const src = at(s.conv_src, base * 4 * 4);
        switch (if (key) w.attn.k_conv else w.attn.v_conv) {
            .static => |weight| tree.convStatic(&m.tree, e, taps, r(weight), src, raw, @intCast(rows), state.kv_width),
            .dynamic => |d| tree.convDynamic(&m.tree, e, taps, at(normalized.buf, normalized.off + base * state.hidden * 2), r(d.static_w), r(d.projection), src, raw, @intCast(rows), state.kv_width),
        }
    } else switch (if (key) w.attn.k_conv else w.attn.v_conv) {
        .static => |weight| glue.convStatic(&m.glue, e, taps, r(weight), raw, @intCast(rows), state.kv_width),
        .dynamic => |d| glue.convDynamic(&m.glue, e, taps, at(normalized.buf, normalized.off + base * state.hidden * 2), r(d.static_w), r(d.projection), raw, @intCast(rows), state.kv_width),
    }
}

fn routedExperts(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, layer: usize, rows: usize) !void {
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
        try experts.mma(&m.experts, e, false, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), gu_globals, x(at(s.gu, 0)), gu_call);
        try experts.mma(&m.experts, e, true, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), down_globals, x(at(s.down, 0)), down_call);
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
            if (precision == .nvfp4) {
                try experts.guGrouped(&m.experts, e, x_in, x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.expert_group, 0)), gu_out, chunk_gu);
                try experts.nvfp4DownGrouped(&m.experts, e, gu_out, x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.expert_group, 0)), down_out, chunk_down);
            } else {
                try experts.guGrouped(&m.experts, e, x_in, x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.expert_group, 0)), gu_out, chunk_gu);
                try experts.mxfp8DownGrouped(&m.experts, e, gu_out, x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.expert_group, 0)), down_out, chunk_down);
            }
            row0 += take;
        }
    } else {
        if (precision == .nvfp4) {
            try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
        } else {
            try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
        }
        switch (precision) {
            .nvfp4 => try experts.nvfp4Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
            .mxfp8 => try experts.mxfp8Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
        }
    }
    try experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
}

fn visit(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, prev_fused: bool, next_v: ?*const layer_plan.Visit, final_rows: usize, segments: []const Segment, seg_meta: []const SegMeta, rows: usize, taps: ?state.Ref) !void {
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
    if (v.layer != 0 and !prev_fused) widen(m, e, at(s.p, 0), at(s.partial, 0), rows * state.hidden);
    try prepareAttentionInputs(m, e, s, v, rows, prev_fused);
    const knorm = if (gated or lp.kind == .global) at(s.k_norm, 0) else at(s.q_norm, 0);
    const vnorm = if (gated or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    project(m, e, w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
    project(m, e, w.attn.wk, knorm, at(s.k_raw, 0), rows);
    project(m, e, w.attn.wv, vnorm, at(s.v_raw, 0), rows);
    project(m, e, w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
    project(m, e, w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
    project(m, e, w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
    try attendSegments(m, e, s, v, segments, seg_meta);
    project(m, e, w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
    var base: usize = 0;
    for (segments, seg_meta) |seg, sm| {
        staticPostConv(m, e, s, seg.cache, v.cache_slot, 2, at(s.attn_dense, 0), w.attn_conv, at(s.attn_conv, 0), base, seg.rows, sm.is_tree);
        base += seg.rows;
    }
    glue.attnPartialAr(&m.glue, e, at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), at(s.partial, 0), r(m.weights.mlp_queries[v.layer]), at(s.mlp_ar, 0), @intCast(rows), lp.mlp_sources, if (lp.boundary) lp.blocks_in + 1 else 0);
    if (gated) {
        const c = w.controllers.?[3];
        const core = v.layer - 6;
        try gates(m, e, s, at(s.mlp_ar, 0), messageRow(s, core, 3, 0), c, at(s.mlp_gated, 0), rows);
        norm(m, e, at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    } else norm(m, e, at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    if (loop.plan.splitRows(rows))
        glue.routerSplit(&m.glue, e, at(s.mlp_normed, 0), r(w.router_weight), r(w.router_bias), at(s.route_scores, 0), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows))
    else
        glue.router(&m.glue, e, at(s.mlp_normed, 0), r(w.router_weight), r(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
    try routedExperts(m, e, s, v.layer, rows);
    base = 0;
    for (segments, seg_meta) |seg, sm| {
        staticPostConv(m, e, s, seg.cache, v.cache_slot, 3, at(s.mlp_out, 0), w.mlp_conv, at(s.mlp_conv, 0), base, seg.rows, sm.is_tree);
        base += seg.rows;
    }
    if (next_v) |nv| {
        if (canPreFuse(nv)) {
            const lp_next = m.weight_plan.at(nv.layer);
            glue.mlpPartialAr(&m.glue, e, at(s.partial, 0), at(s.mlp_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), r(m.weights.attn_queries[nv.layer - 1]), at(s.p, 0), at(s.partial, 0), at(s.ar, 0), @intCast(rows), lp_next.attn_sources, lp_next.attn_queries);
        } else {
            runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, rows * state.hidden, 0);
        }
        if (v.cache_slot == 5) copyBf16(m, e, at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
    } else if (final_rows == rows) {
        glue.mlpPartialAr(&m.glue, e, at(s.partial, 0), at(s.mlp_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), r(m.weights.output_query), at(s.p, 0), at(s.partial, 0), at(s.mlp_ar, 0), @intCast(rows), m.weight_plan.final_sources, 1);
    } else {
        const last = rows - 1;
        runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, last * state.hidden, 0);
        glue.mlpPartialAr(&m.glue, e, row(s.partial, last, state.hidden * 4), row(s.mlp_conv, last, state.hidden * 2), at(s.blocks, last * state.block_slots * state.hidden * 2), at(s.block_inv_rms, last * state.block_slots * 4), r(m.weights.output_query), row(s.p, last, state.hidden * 2), row(s.partial, last, state.hidden * 4), at(s.mlp_ar, 0), 1, m.weight_plan.final_sources, 1);
    }
}

fn commitFullHistories(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, segments: []const Segment) void {
    _ = s;
    for (segments) |seg| for (0..state.visit_slots) |slot| for (0..state.conv_states) |cs| {
        copyBf16(m, e, seg.cache.staged(slot, cs, seg.rows), seg.cache.history(slot, cs), 3 * state.convWidth(cs));
    };
}

pub fn run(m: *const Model, s: *state.Scratch, cache: *state.Cache, ids: []const u32, options: Options) !void {
    return runShared(m, s, &.{.{ .cache = cache, .rows = ids.len }}, ids, options);
}

/// Image rows ride one segment's chunk, fit their buffer, and replace only placeholder tokens.
fn checkImages(m: *const Model, ids: []const u32, segments: usize, options: Options) !void {
    if (options.images.len == 0) return;
    if (segments != 1 or options.ids != null) return error.Inkling2ImageRows;
    for (options.images) |im| {
        const end = @as(usize, im.row) + im.rows;
        if (im.rows == 0 or end > ids.len or im.off + @as(usize, im.rows) * state.hidden * 2 > im.buf.length()) return error.Inkling2ImageRows;
        for (ids[im.row..end]) |id| if (id != m.config.image_token) return error.Inkling2ImageRows;
    }
}

pub fn runShared(m: *const Model, s: *state.Scratch, segments: []const Segment, ids: []const u32, options: Options) !void {
    const rows = ids.len;
    if (rows == 0 or rows > s.rows or segments.len == 0 or segments.len > state.max_segments or (!options.all and segments.len != 1)) return error.Inkling2RowsExceeded;
    var seg_meta: [state.max_segments]SegMeta = undefined;
    var total: usize = 0;
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        if (seg.rows == 0 or seg.rows > state.max_rows or seg.cache.len + seg.rows > seg.cache.capacity) return error.Inkling2ContextFull;
        if (seg.rows > state.window_rows and seg.cache.pending != null) return error.Inkling2CacheTransaction;
        if (seg.cache.pending) |pending| if (pending != seg.rows) return error.Inkling2CacheTransaction;
        total += seg.rows;
        if (seg.parents) |ps| {
            if (ps.len != seg.rows or seg.rows > state.window_rows) return error.Inkling2RowsExceeded;
            var depths: [state.window_rows]u32 = undefined;
            try state.validateParents(ps, depths[0..seg.rows]);
            const is_tree = !state.isChain(ps);
            var max_depth: u32 = 0;
            for (0..seg.rows) |i| {
                const d = if (is_tree) depths[i] else @as(u32, @intCast(i));
                max_depth = @max(max_depth, d);
                s.pos.slice(i32, rows)[base + i] = @intCast(seg.cache.len + d);
                s.seg_of.slice(i32, rows)[base + i] = @intCast(si);
            }
            seg_meta[si] = .{ .is_tree = is_tree, .max_depth = max_depth };
            if (is_tree) {
                if (rows > state.batch_rows) return error.Inkling2RowsExceeded;
                state.buildConvSrc(ps, s.conv_src.slice(u32, rows * 4)[base * 4 ..][0 .. seg.rows * 4]);
                state.buildTreePaths(ps, depths[0..seg.rows], s.tree_paths.slice(u32, rows * state.window_rows)[base * state.window_rows ..][0 .. seg.rows * state.window_rows]);
            }
        } else {
            seg_meta[si] = .{ .is_tree = false, .max_depth = @intCast(seg.rows - 1) };
            for (0..seg.rows) |i| {
                s.pos.slice(i32, rows)[base + i] = @intCast(seg.cache.len + i);
                s.seg_of.slice(i32, rows)[base + i] = @intCast(si);
            }
        }
        s.segs.slice(ap.Seg, state.max_segments)[si] = .{ .row0 = 0, .n = @intCast(seg.rows), .p0 = @intCast(seg.cache.len), .off = 0 };
        base += seg.rows;
    }
    if (total != rows) return error.Inkling2Segments;
    if (options.ids == null) {
        for (ids) |id| if (id >= m.config.text.vocab) return error.Inkling2Token;
        @memcpy(s.ids.slice(u32, rows), ids);
    }
    try checkImages(m, ids, segments.len, options);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    glue.embedNorm(&m.glue, e, options.ids orelse at(s.ids, 0), r(m.weights.embed), r(m.weights.embed_norm), at(s.p, 0), @intCast(rows));
    for (options.images) |im| copyBf16(m, e, at(im.buf, im.off), at(s.p, im.row * state.hidden * 2), im.rows * state.hidden);
    const visits = m.weight_plan.visits[0..m.weight_plan.n_visits];
    const final = finalPlan(rows, options.all, options.taps != null);
    for (visits, 0..) |*v, vi| {
        const prev_fused = vi > 0 and canPreFuse(v);
        const next_v: ?*const layer_plan.Visit = if (vi + 1 < visits.len) &visits[vi + 1] else null;
        try visit(m, e, s, v, prev_fused, next_v, final.work_rows, segments, seg_meta[0..segments.len], rows, options.taps);
    }
    norm(m, e, at(s.mlp_ar, 0), m.weights.norm, at(s.mlp_normed, 0), final.work_rows);
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
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop forward: {s}", .{msg});
        return error.GpuFailed;
    }
}
