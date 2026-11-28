//! Inkling 2 Mini short-convolution table prefill, resource residency and fused/segmented kernels.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const glue = @import("glue.zig");
const tree = @import("tree.zig");
const ops = @import("forward_ops.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");

const Model = model_file.Model;
pub const Ref = state.Ref;
pub const Segment = struct { cache: *state.Cache, rows: usize };

const StageConvArgs = extern struct { rows: u32, width: u32, commit: u32, pairs: u32 };
const CommitHistoriesArgs = extern struct { offset: u32, slots: u32, tap_rows: u32 };

fn addr(ref: Ref) u64 {
    return ref.buf.gpuAddress() + ref.off;
}

fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}

fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

pub fn staticPostConv(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, slot: usize, conv_state: usize, raw: Ref, weight: ck.Tensor, out: Ref, base: usize, rows: usize, is_tree: bool) void {
    const channels = state.convWidth(conv_state);
    const raw_at = at(raw.buf, raw.off + base * channels * 2);
    const out_at = at(out.buf, out.off + base * channels * 2);
    const hist = cache.history(slot, conv_state);
    if (is_tree) {
        tree.convStatic(&m.tree, e, hist, raw_at, r(weight), at(s.conv_src, base * 4 * 4), out_at, @intCast(rows), @intCast(channels));
    } else {
        glue.convStaticInplace(&m.glue, e, hist, raw_at, r(weight), out_at, @intCast(rows), @intCast(channels));
    }
    ops.stageConvRows(m, e, cache, slot, conv_state, raw, base, rows);
}

pub fn fusedConv(segments: anytype, meta: anytype) bool {
    for (meta) |sm| if (sm.is_tree) return false;
    return segments.len > 1 or segments[0].rows <= state.window_rows;
}

pub fn declareConvResources(e: mtl.ComputeEncoder, s: *state.Scratch, segments: anytype) void {
    var ids: [state.max_segments * 2 + 12]mtl.objc.Id = undefined;
    var n: usize = 0;
    for (segments) |seg| {
        ids[n] = seg.cache.conv.id;
        ids[n + 1] = seg.cache.taps.id;
        n += 2;
    }
    for ([_]mtl.Buffer{ s.k_raw, s.v_raw, s.k_conv, s.v_conv, s.attn_dense, s.attn_conv, s.mlp_out, s.mlp_conv, s.q_norm, s.k_norm, s.v_norm, s.conv_stage }) |b| {
        ids[n] = b.id;
        n += 1;
    }
    e.useResources(ids[0..n], mtl.ResourceUsage.read | mtl.ResourceUsage.write);
}

pub fn convSegsRef(s: *state.Scratch, slot: usize, kind: usize) Ref {
    return at(s.conv_segs, (slot * 3 + kind) * state.max_segments * @sizeOf(state.InkConvSeg));
}

pub fn convDynSegsRef(s: *state.Scratch, slot: usize, cs: usize) Ref {
    return at(s.conv_dyn_segs, (slot * 2 + cs) * state.max_segments * @sizeOf(glue.InkConvDynSeg));
}

pub fn prepareConvTables(m: *const Model, s: *state.Scratch, segments: anytype) void {
    const conv_segs = s.conv_segs.slice(state.InkConvSeg, state.visit_slots * 3 * state.max_segments);
    const conv_dyn = s.conv_dyn_segs.slice(glue.InkConvDynSeg, state.visit_slots * 2 * state.max_segments);
    var seg_bases: [state.max_segments]usize = undefined;
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        seg_bases[si] = base;
        base += seg.rows;
    }
    for (0..state.visit_slots) |slot| {
        const is_global = m.weight_plan.at(m.weight_plan.visit(@intCast(slot)).layer).kind == .global;
        for (segments, 0..) |seg, si| {
            const cache = seg.cache;
            const b = seg_bases[si];
            const commit: u32 = @intFromBool(cache.pending == null);
            const stage_base = state.convStageByteOffset(b, si);
            const st_k = at(s.conv_stage, stage_base);
            const st_v = at(s.conv_stage, stage_base + (3 + seg.rows) * state.kv_width * 2);
            conv_segs[(slot * 3 + 0) * state.max_segments + si] = .{
                .hist0 = addr(cache.history(slot, 0)),
                .raw0 = addr(at(s.k_raw, b * state.kv_width * 2)),
                .staged0 = addr(st_k),
                .out0 = addr(at(s.k_conv, b * state.kv_width * 2)),
                .hist1 = addr(cache.history(slot, 1)),
                .raw1 = addr(at(s.v_raw, b * state.kv_width * 2)),
                .staged1 = addr(st_v),
                .out1 = addr(at(s.v_conv, b * state.kv_width * 2)),
                .rows = @intCast(seg.rows),
                .commit = commit,
            };
            if (is_global) {
                for ([_]usize{ 0, 1 }) |cs| {
                    const norm_buf = if (cs == 0) s.k_norm else s.v_norm;
                    const out_buf = if (cs == 0) s.k_conv else s.v_conv;
                    const st_kv = if (cs == 0) st_k else st_v;
                    conv_dyn[(slot * 2 + cs) * state.max_segments + si] = .{
                        .taps = addr(st_kv),
                        .input = addr(at(norm_buf, b * state.hidden * 2)),
                        .out = addr(at(out_buf, b * state.kv_width * 2)),
                        .row0 = @intCast(b),
                        .rows = @intCast(seg.rows),
                    };
                }
            }
            inline for (.{ .{ 1, 2, s.attn_dense, s.attn_conv }, .{ 2, 3, s.mlp_out, s.mlp_conv } }) |entry| {
                const st_post = at(s.conv_stage, stage_base);
                const h_post = cache.history(slot, entry[1]);
                conv_segs[(slot * 3 + entry[0]) * state.max_segments + si] = .{
                    .hist0 = addr(h_post),
                    .raw0 = addr(at(entry[2], b * state.hidden * 2)),
                    .staged0 = addr(st_post),
                    .out0 = addr(at(entry[3], b * state.hidden * 2)),
                    .hist1 = addr(h_post),
                    .raw1 = addr(at(entry[2], b * state.hidden * 2)),
                    .staged1 = addr(st_post),
                    .out1 = addr(at(entry[3], b * state.hidden * 2)),
                    .rows = @intCast(seg.rows),
                    .commit = commit,
                };
            }
        }
    }
}

pub fn stageConv(m: *const Model, e: mtl.ComputeEncoder, hist0: Ref, raw0: Ref, staged0: Ref, hist1: Ref, raw1: Ref, staged1: Ref, rows: usize, width: usize, commit: bool, pairs: usize) void {
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

pub fn stageConvSegs(m: *const Model, e: mtl.ComputeEncoder, segs: Ref, n_segs: usize, width: usize, pairs: usize) void {
    std.debug.assert(pairs == 1 or pairs == 2);
    e.setPipeline(m.runtime[@backingInt(model_file.RuntimeOp.stage_conv_segs)]);
    e.setBuffer(segs.buf, segs.off, 0);
    e.setValue(state.StageConvSegsArgs{ .width = @intCast(width), .n_segs = @intCast(n_segs), .pairs = @intCast(pairs) }, 1);
    e.dispatchThreads(mtl.Size.of(pairs * width, n_segs, 1), mtl.Size.of(256, 1, 1));
}

pub fn stageConvStatic(m: *const Model, e: mtl.ComputeEncoder, segs: Ref, weight0: Ref, weight1: Ref, n_segs: usize, width: usize, pairs: usize) void {
    std.debug.assert(pairs == 1 or pairs == 2);
    e.setPipeline(m.runtime[@backingInt(model_file.RuntimeOp.stage_conv_static)]);
    e.setBuffer(segs.buf, segs.off, 0);
    e.setBuffer(weight0.buf, weight0.off, 1);
    e.setBuffer(weight1.buf, weight1.off, 2);
    e.setValue(state.StageConvSegsArgs{ .width = @intCast(width), .n_segs = @intCast(n_segs), .pairs = @intCast(pairs) }, 3);
    e.dispatchThreads(mtl.Size.of(pairs * width, n_segs, 1), mtl.Size.of(256, 1, 1));
}

/// Encode fused K/V staging and dynamic or static convolution for all attention segments.
pub fn fusedAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, segments: anytype) void {
    const w = &m.weights.layers[v.layer];
    const segs_ref = convSegsRef(s, v.cache_slot, 0);
    if (w.attn.k_conv == .static and w.attn.v_conv == .static) {
        stageConvStatic(m, e, segs_ref, r(w.attn.k_conv.static), r(w.attn.v_conv.static), segments.len, state.kv_width, 2);
    } else {
        stageConvSegs(m, e, segs_ref, segments.len, state.kv_width, 2);
        e.barrier();
        if (w.attn.k_conv == .dynamic and w.attn.v_conv == .dynamic) {
            var total: usize = 0;
            for (segments) |seg| total += seg.rows;
            const kd = w.attn.k_conv.dynamic;
            const vd = w.attn.v_conv.dynamic;
            glue.convDynamicSegs(&m.glue, e, convDynSegsRef(s, v.cache_slot, 0), r(kd.static_w), r(kd.projection), @intCast(segments.len), @intCast(total), state.kv_width);
            glue.convDynamicSegs(&m.glue, e, convDynSegsRef(s, v.cache_slot, 1), r(vd.static_w), r(vd.projection), @intCast(segments.len), @intCast(total), state.kv_width);
        } else {
            var offset: usize = 0;
            for (segments, 0..) |seg, si| {
                const stage_base = state.convStageByteOffset(offset, si);
                convInputForAttention(m, e, s, v, offset, seg.rows, stage_base, true);
                convInputForAttention(m, e, s, v, offset, seg.rows, stage_base, false);
                offset += seg.rows;
            }
        }
    }
}

/// Every stream's static convolution of `raw` into `out`: fused stage+conv for decode windows, split for large prefill.
pub fn postConv(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, segments: anytype, slot: usize, conv_state: usize, raw: Ref, weight: ck.Tensor, out: Ref) void {
    const channels = state.convWidth(conv_state);
    if (segments.len > 1 or segments[0].rows <= state.window_rows) {
        const w = r(weight);
        stageConvStatic(m, e, convSegsRef(s, slot, conv_state - 1), w, w, segments.len, channels, 1);
        e.barrier();
        var base: usize = 0;
        var staged = false;
        for (segments) |seg| {
            if (seg.cache.pending != null) {
                ops.stageConvRows(m, e, seg.cache, slot, conv_state, raw, base, seg.rows);
                staged = true;
            }
            base += seg.rows;
        }
        if (staged) e.barrier();
        return;
    }
    var base: usize = 0;
    for (segments, 0..) |seg, si| {
        const taps = at(s.conv_stage, state.convStageByteOffset(base, si));
        const h = seg.cache.history(slot, conv_state);
        const raw_ref = at(raw.buf, raw.off + base * channels * 2);
        stageConv(m, e, h, raw_ref, taps, h, raw_ref, taps, seg.rows, channels, seg.cache.pending == null, 1);
        base += seg.rows;
    }
    e.barrier();
    base = 0;
    for (segments, 0..) |seg, si| {
        const taps = at(s.conv_stage, state.convStageByteOffset(base, si));
        glue.convStatic(&m.glue, e, taps, r(weight), at(out.buf, out.off + base * channels * 2), @intCast(seg.rows), @intCast(channels));
        base += seg.rows;
    }
    e.barrier();
}

const ConvInput = struct { raw: Ref, projected: Ref, normalized: Ref, taps: Ref, state: usize };

fn convInput(s: *state.Scratch, local_ungated: bool, base: usize, rows: usize, stage_base: usize, key: bool) ConvInput {
    const cs: usize = if (key) 0 else 1;
    const normalized = if (key) at(s.k_norm, 0) else if (local_ungated) at(s.q_norm, 0) else at(s.v_norm, 0);
    const taps = at(s.conv_stage, stage_base + if (key) @as(usize, 0) else (3 + rows) * state.kv_width * 2);
    return .{
        .raw = if (key) at(s.k_conv, base * state.kv_width * 2) else at(s.v_conv, base * state.kv_width * 2),
        .projected = if (key) at(s.k_raw, base * state.kv_width * 2) else at(s.v_raw, base * state.kv_width * 2),
        .normalized = at(normalized.buf, normalized.off + base * state.hidden * 2),
        .taps = taps,
        .state = cs,
    };
}

pub fn convInputForAttention(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, v: *const layer_plan.Visit, base: usize, rows: usize, stage_base: usize, key: bool) void {
    const w = &m.weights.layers[v.layer];
    const c = convInput(s, !v.hasMessages() and m.weight_plan.at(v.layer).kind != .global, base, rows, stage_base, key);
    switch (if (key) w.attn.k_conv else w.attn.v_conv) {
        .static => |weight| glue.convStatic(&m.glue, e, c.taps, r(weight), c.raw, @intCast(rows), state.kv_width),
        .dynamic => |d| glue.convDynamic(&m.glue, e, c.taps, c.normalized, r(d.static_w), r(d.projection), c.raw, @intCast(rows), state.kv_width),
    }
}
