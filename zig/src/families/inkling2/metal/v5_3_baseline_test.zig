//! Pre-V5-3 serial/backup baseline encoder and commit helper for bit-exact parity testing.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const loop = @import("loop.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const fwd = @import("forward.zig");
const ops = @import("forward_ops.zig");
const Model = model_file.Model;

pub const BaselineBuffers = struct {
    taps_3plus: mtl.Buffer,
    backup: mtl.Buffer,

    pub fn init(device: mtl.Device) !BaselineBuffers {
        const taps_bytes = state.visit_slots * (3 + state.max_rows) * state.conv_row_values * 2;
        const taps_3plus = try device.buffer(taps_bytes, mtl.ResourceOptions.private);
        errdefer taps_3plus.deinit();
        const backup = try device.buffer(state.conv_bytes, mtl.ResourceOptions.private);
        return .{ .taps_3plus = taps_3plus, .backup = backup };
    }

    pub fn deinit(b: *BaselineBuffers) void {
        b.backup.deinit();
        b.taps_3plus.deinit();
    }

    fn staged(b: *const BaselineBuffers, slot: usize, conv_state: usize, rows_before: usize) state.Ref {
        return .{ .buf = b.taps_3plus, .off = state.convByteOffset(slot, conv_state, rows_before, 3 + state.max_rows) };
    }
};

fn copyGpu(queue: mtl.Queue, from: mtl.Buffer, from_off: usize, to: mtl.Buffer, to_off: usize, bytes: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    e.copy(from, from_off, to, to_off, bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

pub fn baselineBegin(cache: *state.Cache, queue: mtl.Queue, base_bufs: *const BaselineBuffers, rows: usize) !void {
    if (cache.pending != null or rows == 0 or rows > state.window_rows or cache.len + rows > cache.capacity) return error.Inkling2CacheTransaction;
    try copyGpu(queue, cache.conv, 0, base_bufs.backup, 0, state.conv_bytes);
    cache.pending = rows;
}

pub fn baselineCommit(cache: *state.Cache, queue: mtl.Queue, base_bufs: *const BaselineBuffers, accepted: usize) !void {
    const rows = cache.pending orelse return error.Inkling2NoTransaction;
    if (accepted > rows) return error.Inkling2CacheTransaction;
    if (accepted < rows) {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.blit();
        e.copy(base_bufs.backup, 0, cache.conv, 0, state.conv_bytes);
        for (0..state.visit_slots) |slot| for (0..state.conv_states) |cs| {
            const bytes = 3 * state.convWidth(cs) * 2;
            const src_off = state.convByteOffset(slot, cs, accepted, 3 + state.max_rows);
            const dst_off = state.convByteOffset(slot, cs, 0, 3);
            e.copy(base_bufs.taps_3plus, src_off, cache.conv, dst_off, bytes);
        };
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
    }
    cache.len += accepted;
    cache.pending = null;
}

pub fn baselineRollback(cache: *state.Cache, queue: mtl.Queue, base_bufs: *const BaselineBuffers) !void {
    _ = cache.pending orelse return error.Inkling2NoTransaction;
    try copyGpu(queue, base_bufs.backup, 0, cache.conv, 0, state.conv_bytes);
    cache.pending = null;
}

fn staticPostConv(m: *const Model, e: mtl.ComputeEncoder, cache: *state.Cache, base_bufs: *const BaselineBuffers, slot: usize, conv_state: usize, raw: state.Ref, weight: ck.Tensor, out: state.Ref, base: usize, rows: usize) void {
    const channels = state.convWidth(conv_state);
    const taps = base_bufs.staged(slot, conv_state, 0);
    ops.copyBf16(m, e, cache.history(slot, conv_state), taps, 3 * channels);
    ops.copyBf16(m, e, ops.at(raw.buf, raw.off + base * channels * 2), base_bufs.staged(slot, conv_state, 3), rows * channels);
    glue.convStatic(&m.glue, e, taps, ops.r(weight), ops.at(out.buf, out.off + base * channels * 2), @intCast(rows), @intCast(channels));
}

fn convInputForAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, base_bufs: *const BaselineBuffers, v: *const layer_plan.Visit, base: usize, rows: usize, key: bool) void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    const cs: usize = if (key) 0 else 1;
    const out = if (key) ops.at(s.k_conv, base * state.kv_width * 2) else ops.at(s.v_conv, base * state.kv_width * 2);
    const projected = if (key) ops.at(s.k_raw, base * state.kv_width * 2) else ops.at(s.v_raw, base * state.kv_width * 2);
    const normalized = if (key) ops.at(s.k_norm, 0) else if (v.hasMessages() or lp.kind == .global) ops.at(s.v_norm, 0) else ops.at(s.q_norm, 0);
    const taps = base_bufs.staged(v.cache_slot, cs, 0);
    ops.copyBf16(m, e, cache.history(v.cache_slot, cs), taps, 3 * state.kv_width);
    ops.copyBf16(m, e, projected, base_bufs.staged(v.cache_slot, cs, 3), rows * state.kv_width);
    switch (if (key) w.attn.k_conv else w.attn.v_conv) {
        .static => |weight| glue.convStatic(&m.glue, e, taps, ops.r(weight), out, @intCast(rows), state.kv_width),
        .dynamic => |d| glue.convDynamic(&m.glue, e, taps, ops.at(normalized.buf, normalized.off + base * state.hidden * 2), ops.r(d.static_w), ops.r(d.projection), out, @intCast(rows), state.kv_width),
    }
}

fn attendSegments(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, base_bufs: []const BaselineBuffers, v: *const layer_plan.Visit, segments: []const fwd.Segment) !void {
    const prep_cfg = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
    const attn_w = &m.weights.layers[v.layer].attn;
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        const rows = seg.rows;
        const cache = seg.cache;
        const layer0 = cache.attentionLayer(&m.weight_plan, v.cache_slot, rows);
        const layer = ap.Layer{ .window = layer0.window, .ring = @intCast(cache.ring(v.cache_slot)), .chunk = layer0.chunk, .tau = layer0.tau };
        const rr = ap.Rows{ .rows = @intCast(rows), .heads = 24, .kv_heads = 8, .last_pos = @intCast(cache.len + rows - 1), .streams = 1 };
        const meta = attention.RowMeta{ .pos = ops.a(ops.at(s.pos, base * 4)), .seg_of = ops.a(ops.at(s.seg_of, base * 4)), .segs = ops.a(ops.at(s.segs, si * @sizeOf(ap.Seg))) };
        const kv = attention.Cache{ .k = ops.a(cache.keys(v.cache_slot)), .v = ops.a(cache.values(v.cache_slot)) };
        convInputForAttention(m, e, s, cache, &base_bufs[si], v, base, rows, true);
        convInputForAttention(m, e, s, cache, &base_bufs[si], v, base, rows, false);
        attention.prep(&m.attention, e, rr, layer, prep_cfg, meta, .{ ops.a(ops.at(s.q_raw, base * state.query_width * 2)), ops.a(ops.at(s.k_conv, base * state.kv_width * 2)), ops.a(ops.at(s.v_conv, base * state.kv_width * 2)), ops.a(ops.at(s.rel_raw, base * state.rel_width * 2)) }, .{ ops.a(ops.r(attn_w.q_scale)), ops.a(ops.r(attn_w.rel_scale)) }, ops.a(ops.at(s.q_prep, base * state.query_width * 2)), ops.a(ops.at(s.rel_prep, base * state.rel_width * 2)), kv);
        try attention.attend(&m.attention, e, rr, layer, state.max_attention_stride, 0, meta, ops.a(ops.at(s.q_prep, base * state.query_width * 2)), kv, ops.a(ops.at(s.rel_prep, base * state.rel_width * 2)), ops.a(ops.r(attn_w.rel_projection)), ops.a(ops.at(s.part_o, base * 24 * state.max_attention_stride * ap.head_dim * 4)), ops.a(ops.at(s.part_ml, base * 24 * state.max_attention_stride * 8)), ops.a(ops.at(s.sink, base * 24 * 2)), ops.a(ops.at(s.gate, base * state.query_width * 2)), ops.a(ops.at(s.att, base * state.query_width * 2)));
        base += rows;
    }
}

fn prepareInputsSerial(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, rows: usize, pre_fused: bool) !void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    if (v.layer == 0) {
        ops.widen(m, e, ops.at(s.p, 0), ops.at(s.ar, 0), rows * state.hidden);
    } else if (!pre_fused) {
        ops.widen(m, e, ops.at(s.p, 0), ops.at(s.partial, 0), rows * state.hidden);
        const q = m.weights.attn_queries[v.layer - 1];
        for (0..lp.attn_queries) |role| {
            const query = ops.at(q.buffer, q.offset + role * state.hidden * 4);
            for (0..rows) |i| glue.cachedAr(&m.glue, e, ops.at(s.blocks, i * state.block_slots * state.hidden * 2), ops.row(s.partial, i, state.hidden * 4), ops.at(s.block_inv_rms, i * state.block_slots * 4), query, ops.arRow(s, role, rows, i, false), lp.attn_sources, 1);
        }
    }
    const gated = v.hasMessages();
    if (gated) {
        const controllers = w.controllers.?;
        const core = v.layer - 6;
        for (0..3) |role| {
            const source_role = if (lp.kind == .local) 0 else role;
            try ops.gates(m, e, s, ops.arRow(s, source_role, rows, 0, false), ops.messageRow(s, core, role, 0), controllers[role], ops.arRow(s, role, rows, 0, true), rows);
        }
    }
    if (!gated and lp.kind == .local) {
        ops.norm(m, e, ops.arRow(s, 0, rows, 0, false), w.attn.q_norm, ops.at(s.q_norm, 0), rows);
    } else {
        ops.norm(m, e, ops.arRow(s, 0, rows, 0, gated), w.attn.q_norm, ops.at(s.q_norm, 0), rows);
        ops.norm(m, e, ops.arRow(s, 1, rows, 0, gated), w.attn.k_norm, ops.at(s.k_norm, 0), rows);
        ops.norm(m, e, ops.arRow(s, 2, rows, 0, gated), w.attn.v_norm, ops.at(s.v_norm, 0), rows);
    }
}

fn visitBaseline(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, base_bufs: []const BaselineBuffers, v: *const layer_plan.Visit, prev_fused: bool, next_v: ?*const layer_plan.Visit, final_rows: usize, segments: []const fwd.Segment, rows: usize, taps: ?state.Ref) !void {
    const lp = m.weight_plan.at(v.layer);
    const gated = v.hasMessages();
    if (taps) |to| {
        const channel: ?usize = switch (v.cache_slot) {
            5 => 0,
            23 => 1,
            41 => 2,
            else => null,
        };
        if (channel) |ch| ops.captureTap(m, e, ops.at(s.p, 0), to, rows, ch);
    }
    if (v.pass > 1 and v.layer == 6) ops.bankAndReset(m, e, s, rows);
    const w = &m.weights.layers[v.layer];
    try prepareInputsSerial(m, e, s, v, rows, prev_fused);
    const knorm = if (gated or lp.kind == .global) ops.at(s.k_norm, 0) else ops.at(s.q_norm, 0);
    const vnorm = if (gated or lp.kind == .global) ops.at(s.v_norm, 0) else ops.at(s.q_norm, 0);
    ops.project(m, e, w.attn.wq, ops.at(s.q_norm, 0), ops.at(s.q_raw, 0), rows);
    ops.project(m, e, w.attn.wk, knorm, ops.at(s.k_raw, 0), rows);
    ops.project(m, e, w.attn.wv, vnorm, ops.at(s.v_raw, 0), rows);
    ops.project(m, e, w.attn.wr, ops.at(s.q_norm, 0), ops.at(s.rel_raw, 0), rows);
    ops.project(m, e, w.attn.sink, ops.at(s.q_norm, 0), ops.at(s.sink, 0), rows);
    ops.project(m, e, w.attn.gate, ops.at(s.q_norm, 0), ops.at(s.gate, 0), rows);
    try attendSegments(m, e, s, base_bufs, v, segments);
    ops.project(m, e, w.attn.wo, ops.at(s.att, 0), ops.at(s.attn_dense, 0), rows);
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        staticPostConv(m, e, seg.cache, &base_bufs[si], v.cache_slot, 2, ops.at(s.attn_dense, 0), w.attn_conv, ops.at(s.attn_conv, 0), base, seg.rows);
        base += seg.rows;
    }
    glue.attnPartialAr(&m.glue, e, ops.at(s.p, 0), ops.at(s.attn_conv, 0), ops.at(s.blocks, 0), ops.at(s.block_inv_rms, 0), ops.at(s.partial, 0), ops.r(m.weights.mlp_queries[v.layer]), ops.at(s.mlp_ar, 0), @intCast(rows), lp.mlp_sources, if (lp.boundary) lp.blocks_in + 1 else 0);
    if (gated) {
        const c = w.controllers.?[3];
        try ops.gates(m, e, s, ops.at(s.mlp_ar, 0), ops.messageRow(s, v.layer - 6, 3, 0), c, ops.at(s.mlp_gated, 0), rows);
        ops.norm(m, e, ops.at(s.mlp_gated, 0), w.mlp_norm, ops.at(s.mlp_normed, 0), rows);
    } else ops.norm(m, e, ops.at(s.mlp_ar, 0), w.mlp_norm, ops.at(s.mlp_normed, 0), rows);
    if (loop.plan.splitRows(rows))
        glue.routerSplit(&m.glue, e, ops.at(s.mlp_normed, 0), ops.r(w.router_weight), ops.r(w.router_bias), ops.at(s.route_scores, 0), ops.at(s.route_ids, 0), ops.at(s.route_weights, 0), @intCast(rows))
    else
        glue.router(&m.glue, e, ops.at(s.mlp_normed, 0), ops.r(w.router_weight), ops.r(w.router_bias), ops.at(s.route_ids, 0), ops.at(s.route_weights, 0), @intCast(rows));
    try ops.routedExperts(m, e, s, v.layer, rows);
    base = 0;
    for (segments, 0..) |seg, si| {
        staticPostConv(m, e, seg.cache, &base_bufs[si], v.cache_slot, 3, ops.at(s.mlp_out, 0), w.mlp_conv, ops.at(s.mlp_conv, 0), base, seg.rows);
        base += seg.rows;
    }
    if (next_v) |nv| {
        if (fwd.canPreFuse(nv)) {
            const lp_next = m.weight_plan.at(nv.layer);
            glue.mlpPartialAr(&m.glue, e, ops.at(s.partial, 0), ops.at(s.mlp_conv, 0), ops.at(s.blocks, 0), ops.at(s.block_inv_rms, 0), ops.r(m.weights.attn_queries[nv.layer - 1]), ops.at(s.p, 0), ops.at(s.partial, 0), ops.at(s.ar, 0), @intCast(rows), lp_next.attn_sources, lp_next.attn_queries);
        } else {
            ops.runtime(m, e, .mlp_partial, &.{ ops.at(s.partial, 0), ops.at(s.mlp_conv, 0), ops.at(s.p, 0) }, rows * state.hidden, 0);
        }
        if (v.cache_slot == 5) ops.copyBf16(m, e, ops.at(s.p, 0), ops.at(s.prelude, 0), rows * state.hidden);
    } else if (final_rows == rows) {
        glue.mlpPartialAr(&m.glue, e, ops.at(s.partial, 0), ops.at(s.mlp_conv, 0), ops.at(s.blocks, 0), ops.at(s.block_inv_rms, 0), ops.r(m.weights.output_query), ops.at(s.p, 0), ops.at(s.partial, 0), ops.at(s.mlp_ar, 0), @intCast(rows), m.weight_plan.final_sources, 1);
    } else {
        const last = rows - 1;
        ops.runtime(m, e, .mlp_partial, &.{ ops.at(s.partial, 0), ops.at(s.mlp_conv, 0), ops.at(s.p, 0) }, last * state.hidden, 0);
        glue.mlpPartialAr(&m.glue, e, ops.row(s.partial, last, state.hidden * 4), ops.row(s.mlp_conv, last, state.hidden * 2), ops.at(s.blocks, last * state.block_slots * state.hidden * 2), ops.at(s.block_inv_rms, last * state.block_slots * 4), ops.r(m.weights.output_query), ops.row(s.p, last, state.hidden * 2), ops.row(s.partial, last, state.hidden * 4), ops.at(s.mlp_ar, 0), 1, m.weight_plan.final_sources, 1);
    }
}

pub fn runBaselineSerial(m: *const Model, s: *state.Scratch, base_bufs: []const BaselineBuffers, segments: []const fwd.Segment, ids: []const u32, options: fwd.Options) !void {
    std.debug.assert(segments.len == base_bufs.len);
    const rows = ids.len;
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        for (0..seg.rows) |i| {
            s.pos.slice(i32, rows)[base + i] = @intCast(seg.cache.len + i);
            s.seg_of.slice(i32, rows)[base + i] = 0;
        }
        s.segs.slice(ap.Seg, state.max_segments)[si] = .{ .row0 = 0, .n = @intCast(seg.rows), .p0 = @intCast(seg.cache.len), .off = 0 };
        base += seg.rows;
    }
    @memcpy(s.ids.slice(u32, rows), ids);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    glue.embedNorm(&m.glue, e, ops.at(s.ids, 0), ops.r(m.weights.embed), ops.r(m.weights.embed_norm), ops.at(s.p, 0), @intCast(rows));
    const visits = m.weight_plan.visits[0..m.weight_plan.n_visits];
    const final = fwd.finalPlan(rows, options.all, options.taps != null);
    for (visits, 0..) |*v, vi| {
        const prev_fused = vi > 0 and fwd.canPreFuse(v);
        const next_v: ?*const layer_plan.Visit = if (vi + 1 < visits.len) &visits[vi + 1] else null;
        try visitBaseline(m, e, s, base_bufs, v, prev_fused, next_v, final.work_rows, segments, rows, options.taps);
    }
    for (segments, 0..) |seg, si| for (0..state.visit_slots) |slot| for (0..state.conv_states) |cs| {
        ops.copyBf16(m, e, base_bufs[si].staged(slot, cs, seg.rows), seg.cache.history(slot, cs), 3 * state.convWidth(cs));
    };
    ops.norm(m, e, ops.at(s.mlp_ar, 0), m.weights.norm, ops.at(s.mlp_normed, 0), final.work_rows);
    if (options.taps) |to| ops.captureTap(m, e, ops.at(s.mlp_normed, 0), to, final.capture_rows, 3);
    const logits = if (options.all) ops.at(s.logits_all, 0) else ops.at(segments[0].cache.logits, 0);
    glue.logits(&m.glue, e, ops.at(s.mlp_normed, final.logits_row * state.hidden * 2), ops.r(m.weights.unembed), logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}
