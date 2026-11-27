//! Phased attention, in-place staging and asynchronous history helpers.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const loop = @import("loop.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const Model = model_file.Model;

const Ref = state.Ref;
const RArgs = extern struct { count: u32, boundary: u32 = 0 };
fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
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
fn gateNorm(s: *state.Scratch, role: usize, rows: usize) Ref {
    return at(s.gate_norm, role * rows * 2 * state.hidden * 2);
}
fn gateZeta(s: *state.Scratch, role: usize, rows: usize) Ref {
    return at(s.gate_zeta, role * rows * 64 * 4);
}
fn copyBf16(m: *const Model, e: mtl.ComputeEncoder, from: Ref, to: Ref, count: usize) void {
    e.setPipeline(m.runtime[@backingInt(model_file.RuntimeOp.copy_bf16)]);
    e.setBuffer(from.buf, from.off, 0);
    e.setBuffer(to.buf, to.off, 1);
    e.setValue(RArgs{ .count = @intCast(count), .boundary = 0 }, 2);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn stageConvRows(m: *const Model, e: mtl.ComputeEncoder, cache: *const state.Cache, slot: usize, conv_state: usize, raw: Ref, base: usize, rows: usize) void {
    const channels = state.convWidth(conv_state);
    const keep = if (cache.pending != null) blk: {
        std.debug.assert(rows <= state.window_rows);
        break :blk rows;
    } else @min(rows, 3);
    const src = at(raw.buf, raw.off + (base + rows - keep) * channels * 2);
    copyBf16(m, e, src, cache.staged(slot, conv_state, 0), keep * channels);
}

pub fn gateRoles(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, lp: *const layer_plan.Layer, w: *const model_file.LayerWeights, core: usize, rows: usize) !void {
    const controllers = w.controllers.?[0..3];
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

pub fn attendSerial(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: anytype, seg_meta: anytype, specs: []const ap.BatchedSegSpec, prep_cfg: ap.Prep, proj_ref: attention.Ref, scale_refs: [2]attention.Ref) !void {
    var launches: [state.max_segments]ap.Launch = undefined;
    var tree_launches: [state.max_segments]ap.TreeLaunch = undefined;
    var layers: [state.max_segments]ap.Layer = undefined;
    var rrs: [state.max_segments]ap.Rows = undefined;
    var strides: [state.max_segments]usize = undefined;
    var metas: [state.max_segments]attention.RowMeta = undefined;
    var kvs: [state.max_segments]attention.Cache = undefined;
    var bases: [state.max_segments]usize = undefined;
    var starts: [state.max_segments]u32 = undefined;
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        const rows = seg.rows;
        const cache = seg.cache;
        const layer0 = cache.attentionLayer(&m.weight_plan, v.cache_slot, rows);
        bases[si] = base;
        starts[si] = @intCast(cache.len);
        layers[si] = .{ .window = layer0.window, .ring = @intCast(cache.ring(v.cache_slot)), .chunk = layer0.chunk, .tau = layer0.tau };
        rrs[si] = .{ .rows = @intCast(rows), .heads = 24, .kv_heads = 8, .last_pos = specs[si].last_pos, .streams = 1 };
        metas[si] = .{
            .pos = a(at(s.pos, base * 4)),
            .seg_of = a(at(s.seg_of, base * 4)),
            .segs = a(at(s.segs, 0)),
            .prompt_segs = a(at(s.segs, si * @sizeOf(ap.Seg))),
        };
        kvs[si] = .{ .k = a(cache.keys(v.cache_slot)), .v = a(cache.values(v.cache_slot)) };
        base += rows;
    }
    for (seg_meta, 0..) |sm, si| {
        const b = bases[si];
        const raw_refs = [4]attention.Ref{ a(at(s.q_raw, b * state.query_width * 2)), a(at(s.k_conv, b * state.kv_width * 2)), a(at(s.v_conv, b * state.kv_width * 2)), a(at(s.rel_raw, b * state.rel_width * 2)) };
        const q_ref = a(at(s.q_prep, b * state.query_width * 2));
        const rel_ref = a(at(s.rel_prep, b * state.rel_width * 2));
        if (sm.is_tree) {
            attention.prepTree(&m.attention, e, rrs[si], layers[si], prep_cfg, metas[si], raw_refs, scale_refs, q_ref, rel_ref, kvs[si]);
        } else {
            strides[si] = ap.slotsFor(layers[si], @as(usize, specs[si].last_pos) + 1);
            if (strides[si] > state.max_attention_stride) return error.Inkling2Partials;
            attention.prep(&m.attention, e, rrs[si], layers[si], prep_cfg, metas[si], raw_refs, scale_refs, q_ref, rel_ref, kvs[si]);
        }
    }
    e.barrier();
    var ran_prefix = false;
    for (seg_meta, 0..) |sm, si| {
        if (!sm.is_tree) continue;
        const b = bases[si];
        tree_launches[si] = try ap.attendTree(rrs[si], layers[si], state.max_attention_stride, starts[si]);
        if (attention.treePrefix(&m.attention, e, tree_launches[si], rrs[si], layers[si], metas[si], a(at(s.q_prep, b * state.query_width * 2)), kvs[si], a(at(s.rel_prep, b * state.rel_width * 2)), proj_ref, a(at(s.part_o, b * 24 * state.max_attention_stride * ap.head_dim * 4)), a(at(s.part_ml, b * 24 * state.max_attention_stride * 8)), a(at(s.tree_tail_o, b * 2 * 24 * ap.head_dim * 4)), a(at(s.tree_tail_ml, b * 24 * 16)))) {
            ran_prefix = true;
        }
    }
    if (ran_prefix) e.barrier();
    for (seg_meta, 0..) |sm, si| {
        const b = bases[si];
        const q_ref = a(at(s.q_prep, b * state.query_width * 2));
        const rel_ref = a(at(s.rel_prep, b * state.rel_width * 2));
        const po_ref = a(at(s.part_o, b * 24 * state.max_attention_stride * ap.head_dim * 4));
        const pml_ref = a(at(s.part_ml, b * 24 * state.max_attention_stride * 8));
        if (sm.is_tree) {
            attention.treePartials(&m.attention, e, tree_launches[si], rrs[si], layers[si], metas[si], a(at(s.tree_paths, b * state.window_rows * 4)), a(at(s.tree_tail_o, b * 2 * 24 * ap.head_dim * 4)), a(at(s.tree_tail_ml, b * 24 * 16)), q_ref, kvs[si], rel_ref, proj_ref, po_ref, pml_ref);
        } else {
            launches[si] = try attention.partials(&m.attention, e, rrs[si], layers[si], @intCast(strides[si]), 0, metas[si], q_ref, kvs[si], rel_ref, proj_ref, po_ref, pml_ref);
        }
    }
    e.barrier();
    for (seg_meta, 0..) |sm, si| {
        const b = bases[si];
        const partials_rh = if (sm.is_tree) tree_launches[si].partials_rh else launches[si].partials_rh;
        const subs = if (sm.is_tree) @as(u32, 1) else launches[si].subs;
        attention.mergeRaw(&m.attention, e, rrs[si], layers[si], partials_rh, subs, metas[si], a(at(s.part_o, b * 24 * state.max_attention_stride * ap.head_dim * 4)), a(at(s.part_ml, b * 24 * state.max_attention_stride * 8)), a(at(s.sink, b * 24 * 2)), a(at(s.gate, b * state.query_width * 2)), a(at(s.att, b * state.query_width * 2)));
    }
    e.barrier();
}

/// Update 48x4 convolution histories on GPU in batches of up to 8 segments using staged taps.
pub fn commitHistories(m: *const Model, e: mtl.ComputeEncoder, segments: anytype, accepted: []const usize) void {
    std.debug.assert(segments.len == accepted.len);
    if (segments.len == 0) return;
    e.setPipeline(m.runtime[@backingInt(model_file.RuntimeOp.commit_histories)]);
    var start: usize = 0;
    while (start < segments.len) {
        const take = @min(segments.len - start, state.max_commit_segments);
        var args: state.CommitArgs = .{ .segments = @intCast(take), .taps_stride = state.window_rows, .accepted = @splat(0) };
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

pub fn commitPendingNullHistories(m: *const Model, e: mtl.ComputeEncoder, segments: anytype) void {
    var batch_segs: [state.max_segments]@TypeOf(segments[0]) = undefined;
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
