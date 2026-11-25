//! Mini's whole 48-visit forward, encoded entirely on Metal for prompt chunks and decode.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const loop = @import("loop.zig");
const tree = @import("tree.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const ops = @import("forward_ops.zig");
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
pub const DenseArgs = ops.DenseArgs;
pub const DenseStep = ops.DenseStep;
pub const denseStep = ops.denseStep;
pub const dense = ops.dense;
pub const encodeDense = ops.encodeDense;

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

pub fn canPreFuse(v: *const layer_plan.Visit) bool {
    return v.layer != 0 and !(v.pass > 1 and v.layer == 6);
}

fn stageConvRows(m: *const Model, e: mtl.ComputeEncoder, cache: *const state.Cache, slot: usize, conv_state: usize, raw: Ref, rows: usize) void {
    const channels = state.convWidth(conv_state);
    const keep = if (cache.pending != null) blk: {
        std.debug.assert(rows <= state.window_rows);
        break :blk rows;
    } else @min(rows, 3);
    const src = ops.at(raw.buf, raw.off + (rows - keep) * channels * 2);
    ops.copyBf16(m, e, src, cache.staged(slot, conv_state, 0), keep * channels);
}

fn staticPostConv(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, slot: usize, conv_state: usize, raw: Ref, weight: ck.Tensor, out: Ref, base: usize, rows: usize, is_tree: bool) void {
    const channels = state.convWidth(conv_state);
    const raw_at = ops.at(raw.buf, raw.off + base * channels * 2);
    const out_at = ops.at(out.buf, out.off + base * channels * 2);
    const hist = cache.history(slot, conv_state);
    if (is_tree) {
        tree.convStatic(&m.tree, e, hist, raw_at, ops.r(weight), ops.at(s.conv_src, base * 4 * 4), out_at, @intCast(rows), @intCast(channels));
    } else {
        glue.convStaticInplace(&m.glue, e, hist, raw_at, ops.r(weight), out_at, @intCast(rows), @intCast(channels));
    }
    stageConvRows(m, e, cache, slot, conv_state, raw_at, rows);
}

fn convInputForAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, key: bool, is_tree: bool) !void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    const conv_state: usize = if (key) 0 else 1;
    const out = if (key) ops.at(s.k_conv, base * state.kv_width * 2) else ops.at(s.v_conv, base * state.kv_width * 2);
    const projected = if (key) ops.at(s.k_raw, base * state.kv_width * 2) else ops.at(s.v_raw, base * state.kv_width * 2);
    const normalized = if (key) ops.at(s.k_norm, 0) else if (v.hasMessages() or lp.kind == .global) ops.at(s.v_norm, 0) else ops.at(s.q_norm, 0);
    const hist = cache.history(v.cache_slot, conv_state);
    if (is_tree) {
        const src = ops.at(s.conv_src, base * 4 * 4);
        switch (if (key) w.attn.k_conv else w.attn.v_conv) {
            .static => |weight| tree.convStatic(&m.tree, e, hist, projected, ops.r(weight), src, out, @intCast(rows), state.kv_width),
            .dynamic => |d| tree.convDynamic(&m.tree, e, hist, projected, ops.at(normalized.buf, normalized.off + base * state.hidden * 2), ops.r(d.static_w), ops.r(d.projection), src, out, @intCast(rows), state.kv_width),
        }
    } else switch (if (key) w.attn.k_conv else w.attn.v_conv) {
        .static => |weight| glue.convStaticInplace(&m.glue, e, hist, projected, ops.r(weight), out, @intCast(rows), state.kv_width),
        .dynamic => |d| glue.convDynamicInplace(&m.glue, e, hist, projected, ops.at(normalized.buf, normalized.off + base * state.hidden * 2), ops.r(d.static_w), ops.r(d.projection), out, @intCast(rows), state.kv_width),
    }
    stageConvRows(m, e, cache, v.cache_slot, conv_state, projected, rows);
}

fn attendSegments(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: []const Segment, seg_meta: []const SegMeta) !void {
    var launches: [state.max_segments]ap.Launch = undefined;
    var tree_launches: [state.max_segments]ap.TreeLaunch = undefined;
    var layers: [state.max_segments]ap.Layer = undefined;
    var rrs: [state.max_segments]ap.Rows = undefined;
    var metas: [state.max_segments]attention.RowMeta = undefined;
    var kvs: [state.max_segments]attention.Cache = undefined;
    var bases: [state.max_segments]usize = undefined;
    var starts: [state.max_segments]u32 = undefined;
    var base: usize = 0;
    for (segments, seg_meta, 0..) |seg, sm, si| {
        const rows = seg.rows;
        const cache = seg.cache;
        const layer0 = cache.attentionLayer(&m.weight_plan, v.cache_slot, rows);
        bases[si] = base;
        starts[si] = @intCast(cache.len);
        layers[si] = .{ .window = layer0.window, .ring = @intCast(cache.ring(v.cache_slot)), .chunk = layer0.chunk, .tau = layer0.tau };
        rrs[si] = .{ .rows = @intCast(rows), .heads = 24, .kv_heads = 8, .last_pos = @intCast(cache.len + (if (sm.is_tree) sm.max_depth else rows - 1)), .streams = 1 };
        metas[si] = .{
            .pos = ops.a(ops.at(s.pos, base * 4)),
            .seg_of = ops.a(ops.at(s.seg_of, base * 4)),
            .segs = ops.a(ops.at(s.segs, si * @sizeOf(ap.Seg))),
        };
        kvs[si] = .{ .k = ops.a(cache.keys(v.cache_slot)), .v = ops.a(cache.values(v.cache_slot)) };
        try convInputForAttention(m, e, s, cache, v, base, rows, true, sm.is_tree);
        try convInputForAttention(m, e, s, cache, v, base, rows, false, sm.is_tree);
        base += rows;
    }
    e.barrier();
    const prep_cfg = ap.Prep{ .qk_clip = m.config.text.qk_clip, .rel_clip = m.config.text.rel_clip, .floor = @floatFromInt(m.config.text.log_floor), .alpha = m.config.text.log_alpha, .eps = m.config.text.eps };
    const attn_w = &m.weights.layers[v.layer].attn;
    for (seg_meta, 0..) |sm, si| {
        const b = bases[si];
        const raw_refs = [4]attention.Ref{ ops.a(ops.at(s.q_raw, b * state.query_width * 2)), ops.a(ops.at(s.k_conv, b * state.kv_width * 2)), ops.a(ops.at(s.v_conv, b * state.kv_width * 2)), ops.a(ops.at(s.rel_raw, b * state.rel_width * 2)) };
        const scale_refs = [2]attention.Ref{ ops.a(ops.r(attn_w.q_scale)), ops.a(ops.r(attn_w.rel_scale)) };
        const q_ref = ops.a(ops.at(s.q_prep, b * state.query_width * 2));
        const rel_ref = ops.a(ops.at(s.rel_prep, b * state.rel_width * 2));
        if (sm.is_tree) {
            attention.prepTree(&m.attention, e, rrs[si], layers[si], prep_cfg, metas[si], raw_refs, scale_refs, q_ref, rel_ref, kvs[si]);
        } else {
            attention.prep(&m.attention, e, rrs[si], layers[si], prep_cfg, metas[si], raw_refs, scale_refs, q_ref, rel_ref, kvs[si]);
        }
    }
    e.barrier();
    var ran_prefix = false;
    for (seg_meta, 0..) |sm, si| {
        if (!sm.is_tree) continue;
        const b = bases[si];
        tree_launches[si] = try ap.attendTree(rrs[si], layers[si], state.max_attention_stride, starts[si]);
        if (attention.treePrefix(&m.attention, e, tree_launches[si], rrs[si], layers[si], metas[si], ops.a(ops.at(s.q_prep, b * state.query_width * 2)), kvs[si], ops.a(ops.at(s.rel_prep, b * state.rel_width * 2)), ops.a(ops.r(attn_w.rel_projection)), ops.a(ops.at(s.part_o, b * 24 * state.max_attention_stride * ap.head_dim * 4)), ops.a(ops.at(s.part_ml, b * 24 * state.max_attention_stride * 8)), ops.a(ops.at(s.tree_tail_o, b * 2 * 24 * ap.head_dim * 4)), ops.a(ops.at(s.tree_tail_ml, b * 24 * 16)))) {
            ran_prefix = true;
        }
    }
    if (ran_prefix) e.barrier();
    for (seg_meta, 0..) |sm, si| {
        const b = bases[si];
        const q_ref = ops.a(ops.at(s.q_prep, b * state.query_width * 2));
        const rel_ref = ops.a(ops.at(s.rel_prep, b * state.rel_width * 2));
        const proj_ref = ops.a(ops.r(attn_w.rel_projection));
        const po_ref = ops.a(ops.at(s.part_o, b * 24 * state.max_attention_stride * ap.head_dim * 4));
        const pml_ref = ops.a(ops.at(s.part_ml, b * 24 * state.max_attention_stride * 8));
        if (sm.is_tree) {
            attention.treePartials(&m.attention, e, tree_launches[si], rrs[si], layers[si], metas[si], ops.a(ops.at(s.tree_paths, b * state.window_rows * 4)), ops.a(ops.at(s.tree_tail_o, b * 2 * 24 * ap.head_dim * 4)), ops.a(ops.at(s.tree_tail_ml, b * 24 * 16)), q_ref, kvs[si], rel_ref, proj_ref, po_ref, pml_ref);
        } else {
            launches[si] = try attention.partials(&m.attention, e, rrs[si], layers[si], state.max_attention_stride, 0, metas[si], q_ref, kvs[si], rel_ref, proj_ref, po_ref, pml_ref);
        }
    }
    e.barrier();
    for (seg_meta, 0..) |sm, si| {
        const b = bases[si];
        const partials_rh = if (sm.is_tree) tree_launches[si].partials_rh else launches[si].partials_rh;
        const subs = if (sm.is_tree) @as(u32, 1) else launches[si].subs;
        attention.mergeRaw(&m.attention, e, rrs[si], layers[si], partials_rh, subs, metas[si], ops.a(ops.at(s.part_o, b * 24 * state.max_attention_stride * ap.head_dim * 4)), ops.a(ops.at(s.part_ml, b * 24 * state.max_attention_stride * 8)), ops.a(ops.at(s.sink, b * 24 * 2)), ops.a(ops.at(s.gate, b * state.query_width * 2)), ops.a(ops.at(s.att, b * state.query_width * 2)));
    }
    e.barrier();
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
        if (channel) |ch| ops.captureTap(m, e, ops.at(s.p, 0), to, rows, ch);
    }
    if (v.pass > 1 and v.layer == 6) ops.bankAndReset(m, e, s, rows);
    const w = &m.weights.layers[v.layer];
    // Layer 0 boundary overwrites s.partial so widen to s.ar; non-prefused core starts widen s.p -> s.partial.
    if (v.layer == 0) {
        ops.widen(m, e, ops.at(s.p, 0), ops.at(s.ar, 0), rows * state.hidden);
        e.barrier();
    } else if (!prev_fused) {
        ops.widen(m, e, ops.at(s.p, 0), ops.at(s.partial, 0), rows * state.hidden);
        e.barrier();
    }
    try ops.prepareAttentionInputs(m, e, s, v, rows, prev_fused);
    const knorm = if (gated or lp.kind == .global) ops.at(s.k_norm, 0) else ops.at(s.q_norm, 0);
    const vnorm = if (gated or lp.kind == .global) ops.at(s.v_norm, 0) else ops.at(s.q_norm, 0);
    ops.project(m, e, w.attn.wq, ops.at(s.q_norm, 0), ops.at(s.q_raw, 0), rows);
    ops.project(m, e, w.attn.wk, knorm, ops.at(s.k_raw, 0), rows);
    ops.project(m, e, w.attn.wv, vnorm, ops.at(s.v_raw, 0), rows);
    ops.project(m, e, w.attn.wr, ops.at(s.q_norm, 0), ops.at(s.rel_raw, 0), rows);
    ops.project(m, e, w.attn.sink, ops.at(s.q_norm, 0), ops.at(s.sink, 0), rows);
    ops.project(m, e, w.attn.gate, ops.at(s.q_norm, 0), ops.at(s.gate, 0), rows);
    e.barrier();
    try attendSegments(m, e, s, v, segments, seg_meta);
    ops.project(m, e, w.attn.wo, ops.at(s.att, 0), ops.at(s.attn_dense, 0), rows);
    e.barrier();
    var base: usize = 0;
    for (segments, seg_meta) |seg, sm| {
        staticPostConv(m, e, s, seg.cache, v.cache_slot, 2, ops.at(s.attn_dense, 0), w.attn_conv, ops.at(s.attn_conv, 0), base, seg.rows, sm.is_tree);
        base += seg.rows;
    }
    e.barrier();
    glue.attnPartialAr(&m.glue, e, ops.at(s.p, 0), ops.at(s.attn_conv, 0), ops.at(s.blocks, 0), ops.at(s.block_inv_rms, 0), ops.at(s.partial, 0), ops.r(m.weights.mlp_queries[v.layer]), ops.at(s.mlp_ar, 0), @intCast(rows), lp.mlp_sources, if (lp.boundary) lp.blocks_in + 1 else 0);
    e.barrier();
    if (gated) {
        const c = w.controllers.?[3];
        const core = v.layer - 6;
        try ops.gates(m, e, s, ops.at(s.mlp_ar, 0), ops.messageRow(s, core, 3, 0), c, ops.at(s.mlp_gated, 0), rows);
        e.barrier();
        ops.norm(m, e, ops.at(s.mlp_gated, 0), w.mlp_norm, ops.at(s.mlp_normed, 0), rows);
    } else ops.norm(m, e, ops.at(s.mlp_ar, 0), w.mlp_norm, ops.at(s.mlp_normed, 0), rows);
    e.barrier();
    if (loop.plan.splitRows(rows))
        glue.routerSplit(&m.glue, e, ops.at(s.mlp_normed, 0), ops.r(w.router_weight), ops.r(w.router_bias), ops.at(s.route_scores, 0), ops.at(s.route_ids, 0), ops.at(s.route_weights, 0), @intCast(rows))
    else
        glue.router(&m.glue, e, ops.at(s.mlp_normed, 0), ops.r(w.router_weight), ops.r(w.router_bias), ops.at(s.route_ids, 0), ops.at(s.route_weights, 0), @intCast(rows));
    e.barrier();
    try ops.routedExperts(m, e, s, v.layer, rows);
    base = 0;
    for (segments, seg_meta) |seg, sm| {
        staticPostConv(m, e, s, seg.cache, v.cache_slot, 3, ops.at(s.mlp_out, 0), w.mlp_conv, ops.at(s.mlp_conv, 0), base, seg.rows, sm.is_tree);
        base += seg.rows;
    }
    e.barrier();
    if (next_v) |nv| {
        if (canPreFuse(nv)) {
            const lp_next = m.weight_plan.at(nv.layer);
            glue.mlpPartialAr(&m.glue, e, ops.at(s.partial, 0), ops.at(s.mlp_conv, 0), ops.at(s.blocks, 0), ops.at(s.block_inv_rms, 0), ops.r(m.weights.attn_queries[nv.layer - 1]), ops.at(s.p, 0), ops.at(s.partial, 0), ops.at(s.ar, 0), @intCast(rows), lp_next.attn_sources, lp_next.attn_queries);
        } else {
            ops.runtime(m, e, .mlp_partial, &.{ ops.at(s.partial, 0), ops.at(s.mlp_conv, 0), ops.at(s.p, 0) }, rows * state.hidden, 0);
        }
        e.barrier();
        if (v.cache_slot == 5) {
            ops.copyBf16(m, e, ops.at(s.p, 0), ops.at(s.prelude, 0), rows * state.hidden);
            e.barrier();
        }
    } else if (final_rows == rows) {
        glue.mlpPartialAr(&m.glue, e, ops.at(s.partial, 0), ops.at(s.mlp_conv, 0), ops.at(s.blocks, 0), ops.at(s.block_inv_rms, 0), ops.r(m.weights.output_query), ops.at(s.p, 0), ops.at(s.partial, 0), ops.at(s.mlp_ar, 0), @intCast(rows), m.weight_plan.final_sources, 1);
        e.barrier();
    } else {
        const last = rows - 1;
        ops.runtime(m, e, .mlp_partial, &.{ ops.at(s.partial, 0), ops.at(s.mlp_conv, 0), ops.at(s.p, 0) }, last * state.hidden, 0);
        glue.mlpPartialAr(&m.glue, e, ops.row(s.partial, last, state.hidden * 4), ops.row(s.mlp_conv, last, state.hidden * 2), ops.at(s.blocks, last * state.block_slots * state.hidden * 2), ops.at(s.block_inv_rms, last * state.block_slots * 4), ops.r(m.weights.output_query), ops.row(s.p, last, state.hidden * 2), ops.row(s.partial, last, state.hidden * 4), ops.at(s.mlp_ar, 0), 1, m.weight_plan.final_sources, 1);
        e.barrier();
    }
}

/// Update 48x4 convolution histories on GPU in batches of up to 8 segments using staged taps.
pub fn commitHistories(m: *const Model, e: mtl.ComputeEncoder, segments: []const Segment, accepted: []const usize) void {
    std.debug.assert(segments.len == accepted.len);
    if (segments.len == 0) return;
    e.setPipeline(m.runtime[@backingInt(model_file.RuntimeOp.commit_histories)]);
    var start: usize = 0;
    while (start < segments.len) {
        const take = @min(segments.len - start, state.max_commit_segments);
        var args: state.CommitArgs = .{
            .segments = @intCast(take),
            .taps_stride = state.window_rows,
            .accepted = @splat(0),
        };
        for (0..state.max_commit_segments) |i| {
            const seg = segments[start + (if (i < take) i else 0)];
            if (i < take) {
                std.debug.assert(accepted[start + i] <= state.window_rows);
                args.accepted[i] = @intCast(accepted[start + i]);
            }
            e.setBuffer(seg.cache.conv, 0, i);
            e.setBuffer(seg.cache.taps, 0, 8 + i);
            e.setBuffer(seg.cache.conv, 0, 16 + i);
        }
        e.setValue(args, 24);
        e.dispatchGroups(mtl.Size.of(state.visit_slots * state.commit_tiles_per_slot, take, 1), mtl.Size.of(state.commit_threads, 1, 1));
        start += take;
    }
}

fn commitPendingNullHistories(m: *const Model, e: mtl.ComputeEncoder, segments: []const Segment) void {
    var batch_segs: [state.max_segments]Segment = undefined;
    var batch_acc: [state.max_segments]usize = undefined;
    var n: usize = 0;
    for (segments) |seg| {
        if (seg.cache.pending != null) continue;
        batch_segs[n] = seg;
        batch_acc[n] = @min(seg.rows, 3);
        n += 1;
    }
    commitHistories(m, e, batch_segs[0..n], batch_acc[0..n]);
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
                s.seg_of.slice(i32, rows)[base + i] = 0;
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
                s.seg_of.slice(i32, rows)[base + i] = 0;
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
    const e = cb.compute(.concurrent);
    glue.embedNorm(&m.glue, e, options.ids orelse ops.at(s.ids, 0), ops.r(m.weights.embed), ops.r(m.weights.embed_norm), ops.at(s.p, 0), @intCast(rows));
    e.barrier();
    if (options.images.len > 0) {
        for (options.images) |im| ops.copyBf16(m, e, ops.at(im.buf, im.off), ops.at(s.p, im.row * state.hidden * 2), im.rows * state.hidden);
        e.barrier();
    }
    const visits = m.weight_plan.visits[0..m.weight_plan.n_visits];
    const final = finalPlan(rows, options.all, options.taps != null);
    for (visits, 0..) |*v, vi| {
        const prev_fused = vi > 0 and canPreFuse(v);
        const next_v: ?*const layer_plan.Visit = if (vi + 1 < visits.len) &visits[vi + 1] else null;
        try visit(m, e, s, v, prev_fused, next_v, final.work_rows, segments, seg_meta[0..segments.len], rows, options.taps);
    }
    commitPendingNullHistories(m, e, segments);
    ops.norm(m, e, ops.at(s.mlp_ar, 0), m.weights.norm, ops.at(s.mlp_normed, 0), final.work_rows);
    e.barrier();
    if (options.taps) |to| ops.captureTap(m, e, ops.at(s.mlp_normed, 0), to, final.capture_rows, 3);
    const logits = if (options.all) ops.at(s.logits_all, 0) else ops.at(segments[0].cache.logits, 0);
    glue.logits(&m.glue, e, ops.at(s.mlp_normed, final.logits_row * state.hidden * 2), ops.r(m.weights.unembed), logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab);
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
