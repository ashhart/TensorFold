//! Mini's whole 48-visit forward, encoded entirely on Metal for prompt chunks and decode.
const std = @import("std");
const mtl = @import("metal");
const ap = @import("attention_plan.zig");
const glue = @import("glue.zig");
const state = @import("state.zig");
const model_file = @import("model.zig");
const layers = @import("forward_layers.zig");

const Model = model_file.Model;
pub const Ref = state.Ref;
pub const Segment = layers.Segment;
pub const Meta = layers.Meta;
pub const LogitMode = enum { logits, argmax, both };
pub const dense = layers.dense;
pub const commitHistories = layers.commitHistories;

pub const Options = struct {
    all: bool = false,
    ids: ?Ref = null,
    meta: ?Ref = null,
    flight: ?*mtl.CommandBuffer = null,
    taps: ?state.Ref = null,
    logits_mode: LogitMode = .logits,
    argmax_out: ?Ref = null,
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
    if (options.meta != null and (segments.len != 1 or rows != 1)) return error.Inkling2Segments;
    var total: usize = 0;
    var base: usize = 0;
    var need_backup = false;
    for (segments, 0..) |seg, si| {
        if (seg.rows == 0 or seg.rows > state.max_rows or seg.cache.len + seg.rows > seg.cache.capacity) return error.Inkling2ContextFull;
        if (seg.rows > state.window_rows and seg.cache.pending != null) return error.Inkling2CacheTransaction;
        if (segments.len > 1 and seg.rows > state.window_rows) return error.Inkling2RowsExceeded;
        if (seg.cache.pending) |pending| {
            if (pending != seg.rows) return error.Inkling2CacheTransaction;
            if (!seg.cache.conv_backed_up) need_backup = true;
        }
        total += seg.rows;
        if (options.meta == null) {
            const pos = s.pos.slice(i32, rows);
            const seg_of = s.seg_of.slice(i32, rows);
            for (0..seg.rows) |i| pos[base + i] = @intCast(seg.cache.len + i);
            @memset(seg_of[base .. base + seg.rows], 0);
            s.segs.slice(ap.Seg, state.max_segments)[si] = .{ .row0 = 0, .n = @intCast(seg.rows), .p0 = @intCast(seg.cache.len), .off = 0 };
        }
        base += seg.rows;
    }
    if (total != rows) return error.Inkling2Segments;
    if (options.meta) |mref| {
        const meta: *Meta = @ptrCast(@alignCast(mref.buf.contents() + mref.off));
        meta.seg = .{ .row0 = 0, .n = 1, .p0 = @intCast(segments[0].cache.len), .off = 0 };
        meta.pos = @intCast(segments[0].cache.len);
        meta.seg_of = 0;
    }
    if (options.ids == null) {
        for (ids) |id| if (id >= m.config.text.vocab) return error.Inkling2Token;
        @memcpy(s.ids.slice(u32, rows), ids);
    }
    const use_fused_conv = segments.len > 1 or rows <= state.window_rows;
    if (use_fused_conv) layers.prepareConvTables(m, s, segments, options.meta);
    const split_cuts = options.flight == null and usesCuts(rows);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var cb = m.queue.commandBuffer();
    if (need_backup) {
        const blit = cb.blit();
        for (segments) |seg| if (seg.cache.pending != null and !seg.cache.conv_backed_up) seg.cache.backupBlit(blit);
        blit.end();
    }
    var e = cb.compute(.concurrent);
    if (use_fused_conv) layers.declareConvResources(e, s, segments);
    // Earlier parts of this forward, already committed: the GPU runs them while the host encodes the next.
    var parts: [cuts.len]mtl.CommandBuffer = undefined;
    var n_parts: usize = 0;
    var cb_committed = false;
    defer for (parts[0..n_parts]) |part| part.release();
    errdefer {
        if (cb_committed) cb.wait() else if (n_parts > 0) parts[n_parts - 1].wait();
    }
    glue.embedNorm(&m.glue, e, options.ids orelse layers.at(s.ids, 0), layers.r(m.weights.embed), layers.r(m.weights.embed_norm), layers.at(s.p, 0), @intCast(rows));
    e.barrier();
    for (m.weight_plan.visits[0..m.weight_plan.n_visits], 0..) |*v, vi| {
        try layers.visit(m, e, s, v, segments, rows, options.taps, options.meta);
        if (split_cuts and n_parts < cuts.len and vi + 1 == cuts[n_parts]) {
            e.end();
            cb.commit();
            if (n_parts == 0) for (segments) |seg| if (seg.cache.pending != null) {
                seg.cache.conv_backed_up = true;
            };
            parts[n_parts] = cb.retain();
            n_parts += 1;
            cb = m.queue.commandBuffer();
            e = cb.compute(.concurrent);
            if (use_fused_conv) layers.declareConvResources(e, s, segments);
        }
    }
    const final = finalPlan(rows, options.all, options.taps != null);
    if (final.work_rows == rows) {
        glue.arRowsBf16(&m.glue, e, layers.at(s.blocks, 0), layers.at(s.p, 0), layers.r(m.weights.output_query), layers.at(s.mlp_ar, 0), m.weight_plan.final_sources, 1, @intCast(rows));
    } else {
        const i = rows - 1;
        glue.arBf16(&m.glue, e, layers.at(s.blocks, i * state.block_slots * state.hidden * 2), layers.row(s.p, i, state.hidden * 2), layers.r(m.weights.output_query), layers.at(s.mlp_ar, 0), m.weight_plan.final_sources, 1);
    }
    e.barrier();
    layers.norm(m, e, layers.at(s.mlp_ar, 0), m.weights.norm, layers.at(s.mlp_normed, 0), final.work_rows);
    e.barrier();
    if (options.taps) |to| layers.captureTap(m, e, layers.at(s.mlp_normed, 0), to, final.capture_rows, 3);
    const logits = if (options.all) layers.at(s.logits_all, 0) else layers.at(segments[0].cache.logits, 0);
    const hidden_rows = layers.at(s.mlp_normed, final.logits_row * state.hidden * 2);
    switch (options.logits_mode) {
        .logits => glue.logits(&m.glue, e, hidden_rows, layers.r(m.weights.unembed), logits, @intCast(final.logits_rows), m.config.text.unpadded_vocab),
        .argmax, .both => glue.logitsArgmax(
            &m.glue,
            e,
            hidden_rows,
            layers.r(m.weights.unembed),
            logits,
            layers.at(s.argmax_vals, 0),
            layers.at(s.argmax_part_ids, 0),
            options.argmax_out orelse layers.at(s.argmax_ids, 0),
            @intCast(final.logits_rows),
            m.config.text.unpadded_vocab,
            options.logits_mode == .both,
        ),
    }
    commitFullHistories(m, e, s, segments);
    e.end();
    cb.commit();
    cb_committed = true;
    if (n_parts == 0) for (segments) |seg| if (seg.cache.pending != null) {
        seg.cache.conv_backed_up = true;
    };
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
pub const cuts = [_]usize{ 2, 12, 24, 36 };

pub fn usesCuts(rows: usize) bool {
    return rows > state.batch_rows;
}
