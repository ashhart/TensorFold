//! Mini's whole 48-visit forward, encoded entirely on Metal for prompt chunks and decode.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const loop = @import("loop.zig");
const compute = @import("forward_compute.zig");
const affine4 = @import("affine4.zig");
const tree = @import("tree.zig");
const conv = @import("forward_conv.zig");
const state = @import("state.zig");
const sample = @import("sample.zig");
const model_file = @import("model.zig");
const Model = model_file.Model;

pub const Ref = state.Ref;
pub const RowMeta = attention.RowMeta;
pub const Draw = sample.Draw;
pub const DrawMode = sample.DrawMode;
pub const drawMode = sample.drawMode;
pub const Segment = struct { cache: *state.Cache, rows: usize, parents: ?[]const i32 = null };
const SegMeta = struct { is_tree: bool = false, max_depth: u32 = 0 };
/// The BF16 rows [off, off + rows * hidden * 2) of `buf` replace the normed embeddings of chunk rows [row, row + rows).
pub const ImageRows = struct { row: u32, rows: u32, buf: mtl.Buffer, off: usize = 0 };
pub const Options = struct {
    all: bool = false,
    ids: ?Ref = null,
    meta: ?attention.RowMeta = null,
    draw: ?Draw = null,
    draws: []const ?Draw = &.{},
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
pub const DenseArgs = compute.DenseArgs;
pub const DenseStep = compute.DenseStep;
pub const denseStep = compute.denseStep;
pub const dense = compute.dense;
pub const encodeDense = compute.encodeDense;
const project = compute.project;
const routedExperts = compute.routedExperts;

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

pub fn canPreFuse(v: *const layer_plan.Visit) bool {
    return v.layer != 0 and !(v.pass > 1 and v.layer == 6);
}

pub fn canPreFuseNorm(plan: *const layer_plan.Plan, v: *const layer_plan.Visit) bool {
    return canPreFuse(v) and !v.hasMessages() and plan.at(v.layer).kind == .local;
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

fn prepareAttentionInputs(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, rows: usize, pre_fused: bool, pre_fused_norm: bool) !void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    if (v.layer == 0) {
        glue.normBf16(&m.glue, e, at(s.p, 0), r(w.attn.q_norm), at(s.q_norm, 0), @intCast(rows));
        return;
    }
    if (pre_fused_norm) return;
    if (!pre_fused) {
        const q = m.weights.attn_queries[v.layer - 1];
        for (0..lp.attn_queries) |role| {
            const query = at(q.buffer, q.offset + role * state.hidden * 4);
            for (0..rows) |i| glue.cachedAr(&m.glue, e, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), at(s.block_inv_rms, i * state.block_slots * 4), query, arRow(s, role, rows, i, false), lp.attn_sources, 1);
        }
    }
    if (v.hasMessages()) {
        const controllers = w.controllers.?;
        const core = v.layer - 6;
        if (rows == 1 or (m.loop.gates_mma != null and rows >= loop.mma_rows)) {
            for (0..3) |role| {
                const c = controllers[role];
                const source_role = if (lp.kind == .local) 0 else role;
                try gates(m, e, s, arRow(s, source_role, rows, 0, false), messageRow(s, core, role, 0), c, arRow(s, role, rows, 0, true), rows);
            }
        } else {
            var roles: [3]loop.Gate = undefined;
            for (&roles, 0..) |*g, role| {
                const c = controllers[role];
                const source_role = if (lp.kind == .local) 0 else role;
                g.* = .{ .h = l(arRow(s, source_role, rows, 0, false)), .m = l(messageRow(s, core, role, 0)), .down = l(r(c.down)), .forget = l(r(c.forget)), .gain = l(r(c.gain)), .out = l(arRow(s, role, rows, 0, true)) };
            }
            loop.gatesEach(&m.loop, e, &roles, @intCast(rows), state.message_entries * state.hidden);
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

fn fusedConv(segments: []const Segment, meta: []const SegMeta) bool {
    for (meta) |m| if (m.is_tree) return false;
    return segments.len > 1 or segments[0].rows <= state.window_rows;
}

fn postConv(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, segments: []const Segment, meta: []const SegMeta, slot: usize, cs: usize, raw: Ref, weight: ck.Tensor, out: Ref) void {
    if (fusedConv(segments, meta)) return conv.postConv(m, e, s, segments, slot, cs, raw, weight, out);
    var base: usize = 0;
    for (segments, meta) |seg, sm| {
        staticPostConv(m, e, s, seg.cache, slot, cs, raw, weight, out, base, seg.rows, sm.is_tree);
        base += seg.rows;
    }
}

fn stageConvInput(m: *const Model, e: mtl.ComputeEncoder, cache: *state.Cache, slot: usize, conv_state: usize, raw: Ref, base: usize, rows: usize) Ref {
    const channels = state.convWidth(conv_state);
    const taps = cache.staged(slot, conv_state, 0);
    copyBf16(m, e, cache.history(slot, conv_state), taps, 3 * channels);
    copyBf16(m, e, at(raw.buf, raw.off + base * channels * 2), cache.staged(slot, conv_state, 3), rows * channels);
    return taps;
}

fn staticPostConv(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, slot: usize, conv_state: usize, raw: Ref, weight: ck.Tensor, out: Ref, base: usize, rows: usize, is_tree: bool) void {
    const channels = state.convWidth(conv_state);
    const taps = stageConvInput(m, e, cache, slot, conv_state, raw, base, rows);
    if (is_tree) {
        tree.convStatic(&m.tree, e, taps, r(weight), at(s.conv_src, base * 4 * 4), at(out.buf, out.off + base * channels * 2), @intCast(rows), @intCast(channels));
    } else {
        glue.convStatic(&m.glue, e, taps, r(weight), at(out.buf, out.off + base * channels * 2), @intCast(rows), @intCast(channels));
    }
}

fn attendSegments(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment, seg_meta: []const SegMeta, meta_override: ?attention.RowMeta) !void {
    const fused = fusedConv(segments, seg_meta);
    if (fused) conv.fusedAttention(m, e, s, v, segments);
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
            if (!fused) convInputsForAttention(m, e, s, seg.cache, v, base, seg.rows, sm.is_tree);
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
        const meta = meta_override orelse attention.RowMeta{
            .pos = a(at(s.pos, base * 4)),
            .seg_of = a(at(s.seg_of, base * 4)),
            .segs = a(at(s.segs, 0)),
            .prompt_segs = a(at(s.segs, si * @sizeOf(ap.Seg))),
        };
        const kv = attention.Cache{ .k = a(cache.keys(v.cache_slot)), .v = a(cache.values(v.cache_slot)) };
        if (!fused) convInputsForAttention(m, e, s, cache, v, base, rows, sm.is_tree);
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

fn convInputsForAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, is_tree: bool) void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    const k_taps = stageConvInput(m, e, cache, v.cache_slot, 0, at(s.k_raw, 0), base, rows);
    const v_taps = stageConvInput(m, e, cache, v.cache_slot, 1, at(s.v_raw, 0), base, rows);
    const k_out = at(s.k_conv, base * state.kv_width * 2);
    const v_out = at(s.v_conv, base * state.kv_width * 2);
    const k_in = at(s.k_norm, base * state.hidden * 2);
    const v_norm = if (v.hasMessages() or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    const v_in = at(v_norm.buf, v_norm.off + base * state.hidden * 2);
    if (is_tree) {
        const src = at(s.conv_src, base * 4 * 4);
        switch (w.attn.k_conv) {
            .static => |kw| tree.convStatic(&m.tree, e, k_taps, r(kw), src, k_out, @intCast(rows), state.kv_width),
            .dynamic => |kd| tree.convDynamic(&m.tree, e, k_taps, k_in, r(kd.static_w), r(kd.projection), src, k_out, @intCast(rows), state.kv_width),
        }
        switch (w.attn.v_conv) {
            .static => |vw| tree.convStatic(&m.tree, e, v_taps, r(vw), src, v_out, @intCast(rows), state.kv_width),
            .dynamic => |vd| tree.convDynamic(&m.tree, e, v_taps, v_in, r(vd.static_w), r(vd.projection), src, v_out, @intCast(rows), state.kv_width),
        }
        return;
    }
    switch (w.attn.k_conv) {
        .static => |kw| switch (w.attn.v_conv) {
            .static => |vw| glue.convStaticKv(&m.glue, e, k_taps, r(kw), k_out, v_taps, r(vw), v_out, @intCast(rows), state.kv_width),
            .dynamic => |vd| {
                glue.convStatic(&m.glue, e, k_taps, r(kw), k_out, @intCast(rows), state.kv_width);
                glue.convDynamic(&m.glue, e, v_taps, v_in, r(vd.static_w), r(vd.projection), v_out, @intCast(rows), state.kv_width);
            },
        },
        .dynamic => |kd| switch (w.attn.v_conv) {
            .dynamic => |vd| glue.convDynamicKv(&m.glue, e, k_taps, k_in, r(kd.static_w), r(kd.projection), k_out, v_taps, v_in, r(vd.static_w), r(vd.projection), v_out, @intCast(rows), state.kv_width),
            .static => |vw| {
                glue.convDynamic(&m.glue, e, k_taps, k_in, r(kd.static_w), r(kd.projection), k_out, @intCast(rows), state.kv_width);
                glue.convStatic(&m.glue, e, v_taps, r(vw), v_out, @intCast(rows), state.kv_width);
            },
        },
    }
}

fn visit(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, prev_fused: bool, prev_fused_norm: bool, next_v: ?*const layer_plan.Visit, final_rows: usize, segments: []const Segment, seg_meta: []const SegMeta, rows: usize, taps: ?state.Ref, meta_override: ?attention.RowMeta) !void {
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
    try prepareAttentionInputs(m, e, s, v, rows, prev_fused, prev_fused_norm);
    const knorm = if (gated or lp.kind == .global) at(s.k_norm, 0) else at(s.q_norm, 0);
    const vnorm = if (gated or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    project(m, e, w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
    project(m, e, w.attn.wk, knorm, at(s.k_raw, 0), rows);
    project(m, e, w.attn.wv, vnorm, at(s.v_raw, 0), rows);
    project(m, e, w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
    project(m, e, w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
    project(m, e, w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
    try attendSegments(m, e, s, v, segments, seg_meta, meta_override);
    project(m, e, w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
    postConv(m, e, s, segments, seg_meta, v.cache_slot, 2, at(s.attn_dense, 0), w.attn_conv, at(s.attn_conv, 0));
    const boundary: u32 = if (lp.boundary) lp.blocks_in + 1 else 0;
    if (gated) {
        glue.attnPartialAr(&m.glue, e, at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), at(s.partial, 0), r(m.weights.mlp_queries[v.layer]), at(s.mlp_ar, 0), @intCast(rows), lp.mlp_sources, boundary);
        const c = w.controllers.?[3];
        const core = v.layer - 6;
        try gates(m, e, s, at(s.mlp_ar, 0), messageRow(s, core, 3, 0), c, at(s.mlp_gated, 0), rows);
        norm(m, e, at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    } else glue.attnPartialArNorm(&m.glue, e, at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), at(s.partial, 0), r(m.weights.mlp_queries[v.layer]), r(w.mlp_norm), at(s.mlp_normed, 0), @intCast(rows), lp.mlp_sources, boundary);
    if (loop.plan.splitRows(rows))
        glue.routerSplit(&m.glue, e, at(s.mlp_normed, 0), r(w.router_weight), r(w.router_bias), at(s.route_scores, 0), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows))
    else
        glue.router(&m.glue, e, at(s.mlp_normed, 0), r(w.router_weight), r(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
    try routedExperts(m, e, s, v.layer, rows);
    postConv(m, e, s, segments, seg_meta, v.cache_slot, 3, at(s.mlp_out, 0), w.mlp_conv, at(s.mlp_conv, 0));
    if (next_v) |nv| {
        if (canPreFuseNorm(&m.weight_plan, nv)) {
            const lp_next = m.weight_plan.at(nv.layer);
            glue.mlpPartialArNorm(&m.glue, e, at(s.partial, 0), at(s.mlp_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), r(m.weights.attn_queries[nv.layer - 1]), r(m.weights.layers[nv.layer].attn.q_norm), at(s.p, 0), at(s.partial, 0), at(s.q_norm, 0), @intCast(rows), lp_next.attn_sources);
        } else if (canPreFuse(nv)) {
            const lp_next = m.weight_plan.at(nv.layer);
            glue.mlpPartialAr(&m.glue, e, at(s.partial, 0), at(s.mlp_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), r(m.weights.attn_queries[nv.layer - 1]), at(s.p, 0), at(s.partial, 0), at(s.ar, 0), @intCast(rows), lp_next.attn_sources, lp_next.attn_queries);
        } else {
            runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, rows * state.hidden, 0);
        }
        if (v.cache_slot == 5) copyBf16(m, e, at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
    } else if (final_rows == rows) {
        glue.mlpPartialArNorm(&m.glue, e, at(s.partial, 0), at(s.mlp_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), r(m.weights.output_query), r(m.weights.norm), at(s.p, 0), at(s.partial, 0), at(s.mlp_normed, 0), @intCast(rows), m.weight_plan.final_sources);
    } else {
        const last = rows - 1;
        runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, last * state.hidden, 0);
        glue.mlpPartialArNorm(&m.glue, e, row(s.partial, last, state.hidden * 4), row(s.mlp_conv, last, state.hidden * 2), at(s.blocks, last * state.block_slots * state.hidden * 2), at(s.block_inv_rms, last * state.block_slots * 4), r(m.weights.output_query), r(m.weights.norm), row(s.p, last, state.hidden * 2), row(s.partial, last, state.hidden * 4), at(s.mlp_normed, 0), 1, m.weight_plan.final_sources);
    }
}

fn commitFullHistories(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, segments: []const Segment, meta: []const SegMeta) void {
    _ = s;
    if (fusedConv(segments, meta)) {
        for (segments) |seg| if (seg.cache.pending != null) conv.commitHistories(m, e, seg.cache, seg.rows);
    } else for (segments) |seg| for (0..state.visit_slots) |slot| for (0..state.conv_states) |cs| copyBf16(m, e, seg.cache.staged(slot, cs, seg.rows), seg.cache.history(slot, cs), 3 * state.convWidth(cs));
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
    const final = try sample.validatedFinalPlan(finalPlan(rows, options.all, options.taps != null), options.meta != null and (segments.len != 1 or rows != 1 or segments[0].parents != null), options.draw, options.draws, options.all, segments.len);
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
            s.segs.slice(ap.Seg, state.max_segments)[si] = .{ .row0 = 0, .n = @intCast(seg.rows), .p0 = @intCast(seg.cache.len), .off = 0 };
        } else {
            seg_meta[si] = .{ .is_tree = false, .max_depth = @intCast(seg.rows - 1) };
            if (options.meta == null) for (0..seg.rows) |i| {
                s.pos.slice(i32, rows)[base + i] = @intCast(seg.cache.len + i);
                s.seg_of.slice(i32, rows)[base + i] = @intCast(si);
            };
            if (options.meta == null) s.segs.slice(ap.Seg, state.max_segments)[si] = .{ .row0 = 0, .n = @intCast(seg.rows), .p0 = @intCast(seg.cache.len), .off = 0 };
        }
        base += seg.rows;
    }
    if (total != rows) return error.Inkling2Segments;
    if (options.ids == null) {
        for (ids) |id| if (id >= m.config.text.vocab) return error.Inkling2Token;
        @memcpy(s.ids.slice(u32, rows), ids);
    }
    try checkImages(m, ids, segments.len, options);
    const use_fused_conv = fusedConv(segments, seg_meta[0..segments.len]);
    if (use_fused_conv) conv.prepareConvTables(m, s, segments);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    var need_backup = false;
    for (segments) |seg| if (seg.cache.pending != null and !seg.cache.conv_backed_up) {
        need_backup = true;
    };
    if (need_backup) {
        const blit = cb.blit();
        for (segments) |seg| if (seg.cache.pending != null and !seg.cache.conv_backed_up) seg.cache.backupBlit(blit);
        blit.end();
    }
    const e = cb.compute(.serial);
    if (use_fused_conv) conv.declareConvResources(e, s, segments);
    switch (m.weights.embed) {
        .bf16 => |w| glue.embedNorm(&m.glue, e, options.ids orelse at(s.ids, 0), r(w), r(m.weights.embed_norm), at(s.p, 0), @intCast(rows)),
        .affine4 => |q| affine4.embedNorm(&m.affine4.?, e, options.ids orelse at(s.ids, 0), q, r(m.weights.embed_norm), at(s.p, 0), @intCast(rows)),
    }
    for (options.images) |im| copyBf16(m, e, at(im.buf, im.off), at(s.p, im.row * state.hidden * 2), im.rows * state.hidden);
    const visits = m.weight_plan.visits[0..m.weight_plan.n_visits];
    for (visits, 0..) |*v, vi| {
        const prev_fused = vi > 0 and canPreFuse(v);
        const prev_fused_norm = vi > 0 and canPreFuseNorm(&m.weight_plan, v);
        const next_v: ?*const layer_plan.Visit = if (vi + 1 < visits.len) &visits[vi + 1] else null;
        try visit(m, e, s, v, prev_fused, prev_fused_norm, next_v, final.work_rows, segments, seg_meta[0..segments.len], rows, options.taps, options.meta);
    }
    if (options.taps) |to| captureTap(m, e, at(s.mlp_normed, 0), to, final.capture_rows, 3);
    const logits = if (options.all) at(s.logits_all, 0) else at(segments[0].cache.logits, 0);
    switch (m.weights.unembed) {
        .bf16 => |w| glue.logits(&m.glue, e, at(s.mlp_normed, final.logits_row * state.hidden * 2), r(w), logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab),
        .affine4 => |q| affine4.logits(&m.affine4.?, e, at(s.mlp_normed, final.logits_row * state.hidden * 2), q, logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab),
    }
    sample.encodeForwardDraw(m, e, s, logits, options.draw, options.draws, m.config.text.unpadded_vocab);
    commitFullHistories(m, e, s, segments, seg_meta[0..segments.len]);
    e.end();
    cb.commit();
    for (segments) |seg| if (seg.cache.pending != null) {
        seg.cache.conv_backed_up = true;
    };
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

/// Single-row pre-attention visit stage for the CPU-affine4 serial reference (zero runtime cost on forward).
pub fn serialVisitPreAttn(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, prev_fused: bool, taps: ?state.Ref) !void {
    if (taps) |to| {
        const channel: ?usize = switch (v.cache_slot) {
            5 => 0,
            23 => 1,
            41 => 2,
            else => null,
        };
        if (channel) |ch| captureTap(m, e, at(s.p, 0), to, 1, ch);
    }
    if (v.pass > 1 and v.layer == 6) bankAndReset(m, e, s, 1);
    if (v.layer != 0 and !prev_fused) widen(m, e, at(s.p, 0), at(s.partial, 0), state.hidden);
    try prepareAttentionInputs(m, e, s, v, 1, prev_fused, false);
}

/// Single-row attention stage for the CPU-affine4 serial reference.
pub fn serialVisitAttend(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, cache: *state.Cache) !void {
    const segs = [_]Segment{.{ .cache = cache, .rows = 1 }};
    const meta = [_]SegMeta{.{ .is_tree = false, .max_depth = 0 }};
    conv.declareConvResources(e, s, &segs);
    try attendSegments(m, e, s, v, &segs, &meta, null);
}

/// Single-row post-attention conv, residual, router, MoE, and loop stage for the CPU-affine4 serial reference.
pub fn serialVisitPostAttn(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, next_v: ?*const layer_plan.Visit, cache: *state.Cache) !void {
    const lp = m.weight_plan.at(v.layer);
    const gated = v.hasMessages();
    const w = &m.weights.layers[v.layer];
    staticPostConv(m, e, s, cache, v.cache_slot, 2, at(s.attn_dense, 0), w.attn_conv, at(s.attn_conv, 0), 0, 1, false);
    glue.attnPartialAr(&m.glue, e, at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), at(s.partial, 0), r(m.weights.mlp_queries[v.layer]), at(s.mlp_ar, 0), 1, lp.mlp_sources, if (lp.boundary) lp.blocks_in + 1 else 0);
    if (gated) {
        try gates(m, e, s, at(s.mlp_ar, 0), messageRow(s, v.layer - 6, 3, 0), w.controllers.?[3], at(s.mlp_gated, 0), 1);
        norm(m, e, at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), 1);
    } else norm(m, e, at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), 1);
    glue.router(&m.glue, e, at(s.mlp_normed, 0), r(w.router_weight), r(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), 1);
    try routedExperts(m, e, s, v.layer, 1);
    staticPostConv(m, e, s, cache, v.cache_slot, 3, at(s.mlp_out, 0), w.mlp_conv, at(s.mlp_conv, 0), 0, 1, false);
    if (next_v) |nv| {
        if (canPreFuse(nv)) {
            const lp_next = m.weight_plan.at(nv.layer);
            glue.mlpPartialAr(&m.glue, e, at(s.partial, 0), at(s.mlp_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), r(m.weights.attn_queries[nv.layer - 1]), at(s.p, 0), at(s.partial, 0), at(s.ar, 0), 1, lp_next.attn_sources, lp_next.attn_queries);
        } else runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, state.hidden, 0);
        if (v.cache_slot == 5) copyBf16(m, e, at(s.p, 0), at(s.prelude, 0), state.hidden);
    } else {
        glue.mlpPartialAr(&m.glue, e, at(s.partial, 0), at(s.mlp_conv, 0), at(s.blocks, 0), at(s.block_inv_rms, 0), r(m.weights.output_query), at(s.p, 0), at(s.partial, 0), at(s.mlp_ar, 0), 1, m.weight_plan.final_sources, 1);
    }
}

/// Single-row final RMSNorm, tap 3 capture, and conv history commit for the CPU-affine4 serial reference.
pub fn serialFinalNormAndCommit(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, taps: ?state.Ref) void {
    norm(m, e, at(s.mlp_ar, 0), m.weights.norm, at(s.mlp_normed, 0), 1);
    if (taps) |to| captureTap(m, e, at(s.mlp_normed, 0), to, 1, 3);
    const serial_meta = [_]SegMeta{.{ .is_tree = false, .max_depth = 0 }};
    commitFullHistories(m, e, s, &.{.{ .cache = cache, .rows = 1 }}, &serial_meta);
    // Fused attention committed K/V states 0 and 1; static states 2 and 3 still own their staged rows.
    for (0..state.visit_slots) |slot| for (2..state.conv_states) |cs| {
        copyBf16(m, e, cache.staged(slot, cs, 1), cache.history(slot, cs), 3 * state.convWidth(cs));
    };
}
