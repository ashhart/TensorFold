//! Non-looped Inkling 2 Small EXL3 forward encoded on Metal for prompt chunks, verify windows and decode.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const exl3_proj = @import("core").exl3_projection;
const layer_plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const dense_mma = @import("dense.zig");
const small_ops = @import("small_ops.zig");
const state = @import("small_state.zig");
const model_file = @import("small_model.zig");
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

pub fn finalPlan(rows: usize, all: bool, taps: bool) FinalPlan {
    std.debug.assert(rows > 0);
    return .{
        .work_rows = if (all or taps) rows else 1,
        .logits_rows = if (all) rows else 1,
        .logits_row = if (!all and taps) rows - 1 else 0,
        .capture_rows = if (taps) rows else 0,
    };
}

/// DSpark tap channel for a non-looped Small layer index (layers 5,11,17,23,29,35,41 -> 0..6; final -> 7).
pub fn tapChannel(layer: u32) ?usize {
    return switch (layer) {
        5 => 0,
        11 => 1,
        17 => 2,
        23 => 3,
        29 => 4,
        35 => 5,
        41 => 6,
        else => null,
    };
}

const RArgs = extern struct { count: u32, boundary: u32 = 0 };
const NormArgs = extern struct { rows: u32, width: u32, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 };
pub const DenseArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
pub const DenseStep = struct { variant: usize, rows: usize };

pub fn denseStep(left: usize) DenseStep {
    std.debug.assert(left > 0);
    const take = @min(left, 32);
    const variant: usize = if (take <= 1) 0 else if (take <= 2) 1 else if (take <= 4) 2 else if (take <= 8) 3 else if (take <= 16) 4 else 5;
    return .{ .variant = variant, .rows = take };
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
fn view(x_: Ref) exl3_proj.View {
    return .{ .buffer = x_.buf, .offset = x_.off };
}
fn arRow(s: *state.Scratch, role: usize, rows: usize, i: usize) Ref {
    return at(s.ar, (role * rows + i) * state.hidden * 4);
}
fn arBatch(rows: usize) small_ops.ARRows {
    return .{ .rows = @intCast(rows), .blocks_stride = state.block_slots * state.hidden, .partial_stride = state.hidden, .out_stride = state.hidden };
}

fn runtime(m: *const Model, e: mtl.ComputeEncoder, op: model_file.RuntimeOp, refs: []const Ref, count: usize) void {
    e.setPipeline(m.pipes.runtime[@backingInt(op)]);
    for (refs, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
    e.setValue(RArgs{ .count = @intCast(count), .boundary = 0 }, refs.len);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}

fn copyBf16(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, count: usize) void {
    runtime(m, e, .copy_bf16, &.{ from, to }, count);
}
fn widen(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, count: usize) void {
    runtime(m, e, .widen, &.{ from, to }, count);
}

fn norm(m: *const Model, e: mtl.ComputeEncoder, input: Ref, weight: ck.Tensor, out: Ref, rows: usize) void {
    e.setPipeline(m.pipes.norm);
    e.setBuffer(input.buf, input.off, 0);
    e.setBuffer(weight.buffer, weight.offset, 1);
    e.setValue(NormArgs{ .rows = @intCast(rows), .width = state.hidden, .eps = 1e-6 }, 2);
    e.setBuffer(out.buf, out.off, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn dense(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    const k = weight.shape[weight.rank - 1];
    const n = weight.shape[weight.rank - 2];
    var base: usize = 0;
    while (base < rows) {
        const step = denseStep(rows - base);
        e.setPipeline(m.pipes.dense[step.variant]);
        e.setBuffer(input.buf, input.off + base * k * 2, 0);
        e.setBuffer(weight.buffer, weight.offset, 1);
        e.setValue(DenseArgs{ .rows = @intCast(step.rows), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, 2);
        e.setBuffer(out.buf, out.off + base * n * 2, 3);
        e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        base += step.rows;
    }
}

fn project(m: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, out: Ref, rows: usize) void {
    if (m.pipes.dense_mma) |mma| {
        const n: u32 = @intCast(weight.shape[weight.rank - 2]);
        const call = dense_mma.Call{ .rows = @intCast(rows), .n = n, .k = @intCast(weight.shape[weight.rank - 1]), .ldy = n };
        if (dense_mma.Mma.serves(call)) return mma.launch(e, .{ .buf = input.buf, .off = input.off }, .{ .buf = weight.buffer, .off = weight.offset }, .{ .buf = out.buf, .off = out.off }, call);
    }
    dense(m, e, weight, input, out, rows);
}

fn prepareAttentionInputs(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, layer_idx: u32, rows: usize) void {
    const lp = m.weight_plan.at(layer_idx);
    const w = &m.weights.layers[layer_idx];
    if (layer_idx == 0) {
        widen(m, e, at(s.p, 0), at(s.ar, 0), rows * state.hidden);
        norm(m, e, arRow(s, 0, rows, 0), w.attn.q_norm, at(s.q_norm, 0), rows);
        return;
    }
    const q = m.weights.attn_queries[layer_idx - 1];
    for (0..lp.attn_queries) |role| {
        const query = at(q.buffer, q.offset + role * state.hidden * 4);
        small_ops.ar(&m.pipes.small, e, at(s.blocks, 0), at(s.partial, 0), query, arRow(s, role, rows, 0), lp.attn_sources, 1, arBatch(rows));
    }
    if (lp.kind == .local) {
        norm(m, e, arRow(s, 0, rows, 0), w.attn.q_norm, at(s.q_norm, 0), rows);
    } else {
        norm(m, e, arRow(s, 0, rows, 0), w.attn.q_norm, at(s.q_norm, 0), rows);
        norm(m, e, arRow(s, 1, rows, 0), w.attn.k_norm, at(s.k_norm, 0), rows);
        norm(m, e, arRow(s, 2, rows, 0), w.attn.v_norm, at(s.v_norm, 0), rows);
    }
}

fn staticPostConv(m: *const Model, e: mtl.ComputeEncoder, cache: *state.Cache, slot: usize, conv_state: usize, raw: Ref, weight: ck.Tensor, out: Ref, base: usize, rows: usize) void {
    const channels = state.convWidth(conv_state);
    const taps = cache.staged(slot, conv_state, 0);
    copyBf16(m, e, cache.history(slot, conv_state), taps, 3 * channels);
    copyBf16(m, e, at(raw.buf, raw.off + base * channels * 2), cache.staged(slot, conv_state, 3), rows * channels);
    glue.convStatic(&m.pipes.glue, e, taps, r(weight), at(out.buf, out.off + base * channels * 2), @intCast(rows), @intCast(channels));
}

fn convInputForAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, key: bool) void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    const raw = if (key) at(s.k_conv, base * state.kv_width * 2) else at(s.v_conv, base * state.kv_width * 2);
    const projected = if (key) at(s.k_raw, base * state.kv_width * 2) else at(s.v_raw, base * state.kv_width * 2);
    const normalized = if (lp.kind == .global) (if (key) at(s.k_norm, 0) else at(s.v_norm, 0)) else at(s.q_norm, 0);
    const taps = cache.staged(v.cache_slot, if (key) 0 else 1, 0);
    copyBf16(m, e, cache.history(v.cache_slot, if (key) 0 else 1), taps, 3 * state.kv_width);
    copyBf16(m, e, projected, cache.staged(v.cache_slot, if (key) 0 else 1, 3), rows * state.kv_width);
    switch (if (key) w.attn.k_conv else w.attn.v_conv) {
        .static => |weight| glue.convStatic(&m.pipes.glue, e, taps, r(weight), raw, @intCast(rows), state.kv_width),
        .dynamic => |d| small_ops.convDynamic(&m.pipes.small, e, taps, at(normalized.buf, normalized.off + base * state.hidden * 2), r(d.static_w), r(d.projection), raw, @intCast(rows), state.kv_width),
    }
}

fn attendSegments(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment) !void {
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        const rows = seg.rows;
        const cache = seg.cache;
        const layer0 = cache.attentionLayer(&m.weight_plan, v.cache_slot, rows);
        const layer = ap.Layer{ .window = layer0.window, .ring = @intCast(cache.ring(v.cache_slot)), .chunk = layer0.chunk, .tau = layer0.tau };
        const start = cache.len;
        const rr = ap.Rows{ .rows = @intCast(rows), .heads = state.heads, .kv_heads = state.kv_heads, .last_pos = @intCast(start + rows - 1), .streams = 1 };
        const prep = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
        const meta = attention.RowMeta{
            .pos = a(at(s.pos, base * 4)),
            .seg_of = a(at(s.seg_of, base * 4)),
            .segs = a(at(s.segs, si * @sizeOf(ap.Seg))),
        };
        const kv = attention.Cache{ .k = a(cache.keys(v.cache_slot)), .v = a(cache.values(v.cache_slot)) };
        convInputForAttention(m, e, s, cache, v, base, rows, true);
        convInputForAttention(m, e, s, cache, v, base, rows, false);
        attention.prep(&m.pipes.attention, e, rr, layer, prep, meta, .{ a(at(s.q_raw, base * state.query_width * 2)), a(at(s.k_conv, base * state.kv_width * 2)), a(at(s.v_conv, base * state.kv_width * 2)), a(at(s.rel_raw, base * state.rel_width * 2)) }, .{ a(r(m.weights.layers[v.layer].attn.q_scale)), a(r(m.weights.layers[v.layer].attn.rel_scale)) }, a(at(s.q_prep, base * state.query_width * 2)), a(at(s.rel_prep, base * state.rel_width * 2)), kv);
        try attention.attend(&m.pipes.attention, e, rr, layer, state.max_attention_stride, 0, meta, a(at(s.q_prep, base * state.query_width * 2)), kv, a(at(s.rel_prep, base * state.rel_width * 2)), a(r(m.weights.layers[v.layer].attn.rel_projection)), a(at(s.part_o, base * state.heads * state.max_attention_stride * ap.head_dim * 4)), a(at(s.part_ml, base * state.heads * state.max_attention_stride * 8)), a(at(s.sink, base * state.heads * 2)), a(at(s.gate, base * state.query_width * 2)), a(at(s.att, base * state.query_width * 2)));
        base += rows;
    }
}

fn routedExperts(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, layer_idx: usize, rows: usize) !void {
    const exp = &m.weights.layers[layer_idx].experts;
    if (exp.shard_ids.len > 0) e.useResources(exp.shard_ids, mtl.ResourceUsage.read);
    const total_rows = rows * state.experts_per_token;
    const w13 = exl3_proj.Experts{
        .descriptors = .{ .buffer = exp.w13_descs, .offset = 0 },
        .count = exp.count,
        .k = state.hidden,
        .n = 2 * state.intermediate,
    };
    const w2 = exl3_proj.Experts{
        .descriptors = .{ .buffer = exp.w2_descs, .offset = 0 },
        .count = exp.count,
        .k = state.intermediate,
        .n = state.hidden,
    };
    try exl3_proj.encodeIndexed(&m.pipes.exl3, e, w13, view(at(s.mlp_normed, 0)), view(at(s.gu_raw, 0)), view(at(s.exl3_scratch, 0)), .{
        .count = total_rows,
        .x_stride = state.hidden,
        .y_stride = 2 * state.intermediate,
        .output = .bf16,
        .routes = .{ .ids = view(at(s.route_ids, 0)), .repeat = state.experts_per_token },
    });
    glue.swiglu(&m.pipes.glue, e, at(s.gu_raw, 0), at(s.gu, 0), @intCast(total_rows), state.intermediate);
    try exl3_proj.encodeIndexed(&m.pipes.exl3, e, w2, view(at(s.gu, 0)), view(at(s.down, 0)), view(at(s.exl3_scratch, 0)), .{
        .count = total_rows,
        .x_stride = state.intermediate,
        .y_stride = state.hidden,
        .output = .bf16,
        .routes = .{ .ids = view(at(s.route_ids, 0)), .repeat = 1 },
    });
    glue.expertCombine(&m.pipes.glue, e, at(s.down, 0), at(s.route_weights, 0), at(s.mlp_out, 0), @intCast(rows), state.hidden);
}

fn visit(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment, rows: usize, taps: ?state.Ref) !void {
    const lp = m.weight_plan.at(v.layer);
    if (taps) |to| {
        if (tapChannel(v.layer)) |ch| small_ops.captureTap(&m.pipes.small, e, at(s.p, 0), to, rows, ch);
    }
    const w = &m.weights.layers[v.layer];
    widen(m, e, at(s.p, 0), at(s.partial, 0), rows * state.hidden);
    prepareAttentionInputs(m, e, s, v.layer, rows);
    const knorm = if (lp.kind == .global) at(s.k_norm, 0) else at(s.q_norm, 0);
    const vnorm = if (lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    project(m, e, w.attn.wq, at(s.q_norm, 0), at(s.q_raw, 0), rows);
    project(m, e, w.attn.wk, knorm, at(s.k_raw, 0), rows);
    project(m, e, w.attn.wv, vnorm, at(s.v_raw, 0), rows);
    project(m, e, w.attn.wr, at(s.q_norm, 0), at(s.rel_raw, 0), rows);
    project(m, e, w.attn.sink, at(s.q_norm, 0), at(s.sink, 0), rows);
    project(m, e, w.attn.gate, at(s.q_norm, 0), at(s.gate, 0), rows);
    try attendSegments(m, e, s, v, segments);
    project(m, e, w.attn.wo, at(s.att, 0), at(s.attn_dense, 0), rows);
    var base: usize = 0;
    for (segments) |seg| {
        staticPostConv(m, e, seg.cache, v.cache_slot, 2, at(s.attn_dense, 0), w.attn_conv, at(s.attn_conv, 0), base, seg.rows);
        base += seg.rows;
    }
    small_ops.attentionPartial(&m.pipes.small, e, at(s.p, 0), at(s.attn_conv, 0), at(s.blocks, 0), at(s.partial, 0), rows, if (lp.boundary) lp.blocks_in + 1 else 0);
    small_ops.ar(&m.pipes.small, e, at(s.blocks, 0), at(s.partial, 0), r(m.weights.mlp_queries[v.layer]), at(s.mlp_ar, 0), lp.mlp_sources, 1, arBatch(rows));
    norm(m, e, at(s.mlp_ar, 0), w.mlp_norm, at(s.mlp_normed, 0), rows);
    small_ops.router(&m.pipes.small, e, at(s.mlp_normed, 0), r(w.router_weight), r(w.router_bias), at(s.route_ids, 0), at(s.route_weights, 0), @intCast(rows));
    try routedExperts(m, e, s, v.layer, rows);
    base = 0;
    for (segments) |seg| {
        staticPostConv(m, e, seg.cache, v.cache_slot, 3, at(s.mlp_out, 0), w.mlp_conv, at(s.mlp_conv, 0), base, seg.rows);
        base += seg.rows;
    }
    runtime(m, e, .mlp_partial, &.{ at(s.partial, 0), at(s.mlp_conv, 0), at(s.p, 0) }, rows * state.hidden);
}

fn commitFullHistories(m: *const Model, e: mtl.ComputeEncoder, segments: []const Segment) void {
    for (segments) |seg| for (0..seg.cache.slots) |slot| for (0..state.conv_states) |cs| {
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
    for (segments) |seg| {
        if (seg.rows == 0 or seg.rows > seg.cache.max_chunk_rows or seg.cache.len + seg.rows > seg.cache.capacity) return error.Inkling2ContextFull;
        if (seg.rows > state.window_rows and seg.cache.pending != null) return error.Inkling2CacheTransaction;
        if (seg.cache.pending) |pending| if (pending != seg.rows) return error.Inkling2CacheTransaction;
        total = std.math.add(usize, total, seg.rows) catch return error.Inkling2Segments;
    }
    if (total != rows) return error.Inkling2Segments;
    if (options.ids == null) {
        for (ids) |id| if (id >= m.config.text.vocab) return error.Inkling2Token;
        @memcpy(s.ids.slice(u32, rows), ids);
    }
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        for (0..seg.rows) |i| {
            s.pos.slice(i32, rows)[base + i] = @intCast(seg.cache.len + i);
            s.seg_of.slice(i32, rows)[base + i] = 0;
        }
        s.segs.slice(ap.Seg, state.max_segments)[si] = .{ .row0 = 0, .n = @intCast(seg.rows), .p0 = @intCast(seg.cache.len), .off = 0 };
        base += seg.rows;
    }

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    small_ops.embedNorm(&m.pipes.small, e, options.ids orelse at(s.ids, 0), r(m.weights.embed), r(m.weights.embed_norm), at(s.p, 0), @intCast(rows));
    for (m.weight_plan.visits[0..m.weight_plan.n_visits]) |*v| try visit(m, e, s, v, segments, rows, options.taps);
    widen(m, e, at(s.p, 0), at(s.partial, 0), rows * state.hidden);
    const final = finalPlan(rows, options.all, options.taps != null);
    const first: usize = if (final.work_rows == rows) 0 else rows - 1;
    small_ops.ar(&m.pipes.small, e, at(s.blocks, first * state.block_slots * state.hidden * 2), row(s.partial, first, state.hidden * 4), r(m.weights.output_query), at(s.mlp_ar, 0), m.weight_plan.final_sources, 1, arBatch(final.work_rows));
    norm(m, e, at(s.mlp_ar, 0), m.weights.norm, at(s.mlp_normed, 0), final.work_rows);
    if (options.taps) |to| small_ops.captureTap(&m.pipes.small, e, at(s.mlp_normed, 0), to, final.capture_rows, state.feature_taps - 1);
    const logits = if (options.all) at(s.logits_all, 0) else at(segments[0].cache.logits, 0);
    small_ops.logits(&m.pipes.small, e, at(s.mlp_normed, final.logits_row * state.hidden * 2), r(m.weights.unembed), logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab);
    commitFullHistories(m, e, segments);
    e.end();
    cb.commit();
    if (options.flight) |flight| {
        flight.* = cb.retain();
        return;
    }
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_small forward: {s}", .{msg});
        return error.GpuFailed;
    }
}
