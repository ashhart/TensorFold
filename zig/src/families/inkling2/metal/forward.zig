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
const NormArgs = extern struct { rows: u32, width: u32, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 };
const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

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
fn gates(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, h: Ref, msg: Ref, c: model_file.Controller, out: Ref, rows: usize) void {
    const stride: u32 = state.message_entries * state.hidden;
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

/// Every projection uses the shared BF16 matvec's FP32 reduction and one BF16 store.
pub fn dense(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    const k = weight.shape[weight.rank - 1];
    const n = weight.shape[weight.rank - 2];
    var base: usize = 0;
    while (base < rows) {
        const left = rows - base;
        const take: usize = if (left >= 32) 32 else if (left >= 16) 16 else if (left >= 8) 8 else if (left >= 4) 4 else if (left >= 2) 2 else 1;
        const variant: usize = switch (take) {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            32 => 5,
            else => unreachable,
        };
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
    for (0..rows) |i| glue.ar(&m.glue, e, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), r(queries), row(s.mlp_ar, i, state.hidden * 4), @intCast(sources), 1);
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

fn prepareAttentionInputs(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, rows: usize) void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    if (v.layer == 0) {
        widen(m, e, at(s.p, 0), at(s.ar, 0), rows * state.hidden);
    } else {
        const q = m.weights.attn_queries[v.layer - 1];
        for (0..lp.attn_queries) |role| {
            const query = at(q.buffer, q.offset + role * state.hidden * 4);
            for (0..rows) |i| glue.ar(&m.glue, e, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), query, arRow(s, role, rows, i, false), lp.attn_sources, 1);
        }
    }
    if (v.hasMessages()) {
        const controllers = w.controllers.?;
        const core = v.layer - 6;
        for (0..3) |role| {
            const c = controllers[role];
            const source_role = if (lp.kind == .local) 0 else role;
            gates(m, e, s, arRow(s, source_role, rows, 0, false), messageRow(s, core, role, 0), c, arRow(s, role, rows, 0, true), rows);
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

fn staticPostConv(m: *const Model, e: mtl.ComputeEncoder, cache: *state.Cache, slot: usize, conv_state: usize, raw: Ref, weight: ck.Tensor, out: Ref, base: usize, rows: usize) void {
    const channels = state.convWidth(conv_state);
    const taps = cache.staged(slot, conv_state, 0);
    copyBf16(m, e, cache.history(slot, conv_state), taps, 3 * channels);
    copyBf16(m, e, at(raw.buf, raw.off + base * channels * 2), cache.staged(slot, conv_state, 3), rows * channels);
    glue.convStatic(&m.glue, e, taps, r(weight), at(out.buf, out.off + base * channels * 2), @intCast(rows), @intCast(channels));
}

fn attendSegments(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment) !void {
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        const rows = seg.rows;
        const cache = seg.cache;
        const layer0 = cache.attentionLayer(&m.weight_plan, v.cache_slot, rows);
        const layer = ap.Layer{ .window = layer0.window, .ring = @intCast(cache.ring(v.cache_slot)), .chunk = layer0.chunk, .tau = layer0.tau };
        const start = cache.len;
        const rr = ap.Rows{ .rows = @intCast(rows), .heads = 24, .kv_heads = 8, .last_pos = @intCast(start + rows - 1), .streams = 1 };
        const prep = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
        const meta = attention.RowMeta{
            .pos = a(at(s.pos, base * 4)),
            .seg_of = a(at(s.seg_of, base * 4)),
            .segs = a(at(s.segs, si * @sizeOf(ap.Seg))),
        };
        const kv = attention.Cache{ .k = a(cache.keys(v.cache_slot)), .v = a(cache.values(v.cache_slot)) };
        try convInputForAttention(m, e, s, cache, v, base, rows, true);
        try convInputForAttention(m, e, s, cache, v, base, rows, false);
        attention.prep(&m.attention, e, rr, layer, prep, meta, .{ a(at(s.q_raw, base * state.query_width * 2)), a(at(s.k_conv, base * state.kv_width * 2)), a(at(s.v_conv, base * state.kv_width * 2)), a(at(s.rel_raw, base * state.rel_width * 2)) }, .{ a(r(m.weights.layers[v.layer].attn.q_scale)), a(r(m.weights.layers[v.layer].attn.rel_scale)) }, a(at(s.q_prep, base * state.query_width * 2)), a(at(s.rel_prep, base * state.rel_width * 2)), kv);
        attention.relTable(&m.attention, e, rr, a(at(s.rel_prep, base * state.rel_width * 2)), a(r(m.weights.layers[v.layer].attn.rel_projection)), a(at(s.rel_tab, base * 24 * ap.extent * 2)));
        try attention.attend(&m.attention, e, rr, layer, state.max_attention_stride, 0, meta, a(at(s.q_prep, base * state.query_width * 2)), kv, a(at(s.rel_tab, base * 24 * ap.extent * 2)), a(at(s.part_o, base * 24 * state.max_attention_stride * ap.head_dim * 4)), a(at(s.part_ml, base * 24 * state.max_attention_stride * 8)), a(at(s.sink, base * 24 * 2)), a(at(s.gate, base * state.query_width * 2)), a(at(s.att, base * state.query_width * 2)));
        base += rows;
    }
}

fn convInputForAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, key: bool) !void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    const raw = if (key) at(s.k_conv, base * state.kv_width * 2) else at(s.v_conv, base * state.kv_width * 2);
    const projected = if (key) at(s.k_raw, base * state.kv_width * 2) else at(s.v_raw, base * state.kv_width * 2);
    const normalized = if (key) at(s.k_norm, 0) else if (v.hasMessages() or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    const taps = cache.staged(v.cache_slot, if (key) 0 else 1, 0);
    copyBf16(m, e, cache.history(v.cache_slot, if (key) 0 else 1), taps, 3 * state.kv_width);
    copyBf16(m, e, projected, cache.staged(v.cache_slot, if (key) 0 else 1, 3), rows * state.kv_width);
    switch (if (key) w.attn.k_conv else w.attn.v_conv) {
        .static => |weight| glue.convStatic(&m.glue, e, taps, r(weight), raw, @intCast(rows), state.kv_width),
        .dynamic => |d| glue.convDynamic(&m.glue, e, taps, at(normalized.buf, normalized.off + base * state.hidden * 2), r(d.static_w), r(d.projection), raw, @intCast(rows), state.kv_width),
    }
}

fn routedExperts(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, layer: usize, rows: usize) !void {
    const w = &m.weights.layers[layer];
    const precision: experts.Format = switch (m.weight_plan.at(@intCast(layer)).precision) {
        .nvfp4 => .nvfp4,
        .mxfp8 => .mxfp8,
    };
    const gu_call = experts.Call{ .rows = @intCast(rows), .k = state.hidden, .n = state.intermediate, .format = precision };
    const down_call = experts.Call{ .rows = @intCast(rows), .k = state.intermediate, .n = state.hidden, .format = precision };
    if (precision == .nvfp4) {
        try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), .{ .amax = x(r(w.gu.amax.?)), .denominator = x(r(w.gu.denominator.?)) }, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
    } else {
        try experts.gu(&m.experts, e, x(at(s.mlp_normed, 0)), x(r(w.gu.weight)), x(r(w.gu.scale)), null, x(at(s.route_ids, 0)), x(at(s.gu, 0)), gu_call);
    }
    switch (precision) {
        .nvfp4 => try experts.nvfp4Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(r(w.down.amax.?)), x(r(w.down.denominator.?)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
        .mxfp8 => try experts.mxfp8Down(&m.experts, e, x(at(s.gu, 0)), x(r(w.down.weight)), x(r(w.down.scale)), x(at(s.route_ids, 0)), x(at(s.down, 0)), down_call),
    }
    try experts.combine(&m.experts, e, x(at(s.down, 0)), x(at(s.route_ids, 0)), x(at(s.route_weights, 0)), x(at(s.mlp_out, 0)), down_call);
}

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
    prepareAttentionInputs(m, e, s, v, rows);
    const knorm = if (gated or lp.kind == .global) at(s.k_norm, 0) else at(s.q_norm, 0);
    const vnorm = if (gated or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    dense(m, e, w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
    dense(m, e, w.attn.wk, knorm, at(s.k_raw, 0), rows);
    dense(m, e, w.attn.wv, vnorm, at(s.v_raw, 0), rows);
    dense(m, e, w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
    dense(m, e, w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
    dense(m, e, w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
    try attendSegments(m, e, s, v, segments);
    dense(m, e, w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
    var base: usize = 0;
    for (segments) |seg| {
        staticPostConv(m, e, seg.cache, v.cache_slot, 2, at(s.attn_dense, 0), w.attn_conv, at(s.attn_conv, 0), base, seg.rows);
        base += seg.rows;
    }
    runtime(m, e, .attention_partial, &.{ at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.partial, 0) }, rows * state.hidden, if (lp.boundary) lp.blocks_in + 1 else 0);
    residual(m, e, s, m.weights.mlp_queries[v.layer], lp.mlp_sources, rows);
    if (gated) {
        const c = w.controllers.?[3];
        const core = v.layer - 6;
        gates(m, e, s, at(s.mlp_ar, 0), messageRow(s, core, 3, 0), c, at(s.mlp_gated, 0), rows);
        norm(m, e, at(s.mlp_gated, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    } else norm(m, e, at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    glue.routerSplit(&m.glue, e, at(s.mlp_normed, 0), r(w.router_weight), r(w.router_bias), at(s.route_scores, 0), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
    try routedExperts(m, e, s, v.layer, rows);
    base = 0;
    for (segments) |seg| {
        staticPostConv(m, e, seg.cache, v.cache_slot, 3, at(s.mlp_out, 0), w.mlp_conv, at(s.mlp_conv, 0), base, seg.rows);
        base += seg.rows;
    }
    runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, rows * state.hidden, 0);
    if (v.cache_slot == 5) copyBf16(m, e, at(s.p, 0), at(s.prelude, 0), rows * state.hidden);
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

pub fn runShared(m: *const Model, s: *state.Scratch, segments: []const Segment, ids: []const u32, options: Options) !void {
    const rows = ids.len;
    if (rows == 0 or rows > s.rows or segments.len == 0 or segments.len > state.max_segments or (!options.all and segments.len != 1)) return error.Inkling2RowsExceeded;
    var total: usize = 0;
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        if (seg.rows == 0 or seg.rows > state.max_rows or seg.cache.len + seg.rows > seg.cache.capacity) return error.Inkling2ContextFull;
        if (seg.rows > state.window_rows and seg.cache.pending != null) return error.Inkling2CacheTransaction;
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
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    glue.embedNorm(&m.glue, e, options.ids orelse at(s.ids, 0), r(m.weights.embed), r(m.weights.embed_norm), at(s.p, 0), @intCast(rows));
    for (m.weight_plan.visits[0..m.weight_plan.n_visits]) |*v| try visit(m, e, s, v, segments, rows, options.taps);
    widen(m, e, at(s.p, 0), at(s.partial, 0), rows * state.hidden);
    const final = finalPlan(rows, options.all, options.taps != null);
    for (0..final.work_rows) |j| {
        const i = if (final.work_rows == rows) j else rows - 1;
        glue.ar(&m.glue, e, at(s.blocks, i * state.block_slots * state.hidden * 2), row(s.partial, i, state.hidden * 4), r(m.weights.output_query), row(s.mlp_ar, j, state.hidden * 4), m.weight_plan.final_sources, 1);
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
