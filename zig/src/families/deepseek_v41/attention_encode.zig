//! The attention pass's encode entry points.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("core").exl3_projection;
const checkpoint = @import("core").checkpoint_metal;
const weights = @import("weights.zig");
const config = @import("config.zig");
const ops = @import("ops.zig");
const csa2 = @import("csa2.zig");
const kv = @import("kv.zig");
const index = @import("csa2_index.zig");
const rows_phase = @import("attention_rows.zig");
const rope = @import("rope.zig");
const common = @import("attention_common.zig");
const stages = @import("attention_stages.zig");
const window = @import("attention_window.zig");
const Stage = stages.Stage;
const View = common.View;
const State = common.State;
const Seg = common.Seg;
const Prof = common.Prof;
const profLayerOn = common.profLayerOn;
const DiagnosticStage = common.DiagnosticStage;
const DiagnosticObserver = common.DiagnosticObserver;
const Pipelines = common.Pipelines;
const regions = common.regions;
const Region = common.Region;
const Workspace = common.Workspace;
const tensor = common.tensor;
const core = common.core;
const projection = common.projection;
const norm = common.norm;
const normRepeat = common.normRepeat;
const rotate = common.rotate;
const rotateRepeat = common.rotateRepeat;
const attnCountOn = common.attnCountOn;
const attnBegin = common.attnBegin;
const attnMark = common.attnMark;
const precedingIndex = common.precedingIndex;
const partialBytes = common.partialBytes;
const denseKvBytes = common.denseKvBytes;
const splitScan = common.splitScan;

pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, state: *State, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, rows: u32, c: config.Config) !void {
    return encodeDiagnostic(p, e, w, state, work, x, y, ep, scratch, layer, rows, c, false, null, null, &.{});
}

/// The same encode with the per-block GPU split armed (TF_DSV41_ATTN_PROF).
pub fn encodeProfiled(p: *const Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, state: *State, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, rows: u32, c: config.Config, prof: *Prof) !void {
    return encodeDiagnostic(p, e, w, state, work, x, y, ep, scratch, layer, rows, c, false, null, prof, &.{});
}

/// Same encoder and arithmetic as encode, with optional copies at operator boundaries.
pub fn encodeMulti(p: *const Pipelines, e_in: mtl.ComputeEncoder, w: *const weights.Attention, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, c: config.Config, segs: []const Seg) !void {
    var e = e_in;
    const nsegs = segs.len;
    if (nsegs < 2 or nsegs > common.RowsScratch.slots) return error.InvalidAttentionChunk;
    const rows: u32 = segs[nsegs - 1].r0 + segs[nsegs - 1].n;
    if (layer >= 40 or rows == 0 or rows > work.max_rows or rows > 8) return error.InvalidAttentionChunk;
    for (segs) |sg| {
        if (sg.n != 1) return error.InvalidAttentionChunk;
        if (sg.state.position != sg.p0) return error.PositionMismatch;
        if (sg.p0 > sg.state.plan.capacity or sg.n > sg.state.plan.capacity - sg.p0) return error.InvalidAttentionChunk;
    }
    const ratio = c.compress_ratios.items()[layer];
    const q_width = c.heads * c.head_dim;
    const qr = work.view(.qr);
    const q = work.view(.query);
    const raw = work.view(.raw_kv);
    const iq_width = c.index_heads * c.index_dim;
    var kv_src = false;
    if (ratio != 0) {
        const src0 = try segs[0].state.source(layer, ratio);
        kv_src = src0.owner == layer;
    }
    const ix_src = ratio != 0 and kv.cache.isIndexSource(layer);
    if (std.c.getenv("TF_DSV41_BATCH_DEBUG") != null) std.log.info("encodeMulti: layer={d} segs={d} rows={d}", .{ layer, nsegs, rows });
    // The row-wise phase: one call over every segment's rows.
    try projection(ep, e, .{ .trellis = w.wq_a }, x, qr, scratch, rows, c.hidden, c.q_lora, .bf16);
    norm(p, e, qr, w.q_norm, rows, c.q_lora, c.eps);
    try projection(ep, e, .{ .trellis = w.wq_b }, qr, q, scratch, rows, c.q_lora, q_width, .bf16);
    try projection(ep, e, .{ .trellis = w.wkv }, x, raw, scratch, rows, c.hidden, c.head_dim, .bf16);
    if (kv_src) {
        const compressor = if (w.compressor) |*value| value else return error.MissingAttentionCompressor;
        const output: exl3.Output = if (ratio == 1) .bf16 else .f32;
        try projection(ep, e, compressor.wkv, x, work.view(.values), scratch, rows, c.hidden, c.head_dim, output);
        if (ratio > 1) {
            const gate = compressor.wgate orelse return error.MissingAttentionCompressorGate;
            try projection(ep, e, gate, x, work.view(.gates), scratch, rows, c.hidden, c.head_dim, .f32);
        }
    }
    if (ix_src) {
        const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
        try projection(ep, e, selector.wq_b, qr, work.view(.index_q), scratch, rows, c.q_lora, iq_width, .bf16);
        const iq_w_args: ops.Dense = .{ .rows = rows, .k = c.hidden, .n = c.index_heads, .x_stride = c.hidden, .w_stride = c.hidden, .y_stride = c.index_heads, .w_dtype = switch (selector.weights_proj.dtype) {
            .bf16 => .bf16,
            else => .f16,
        } };
        const dense_col_env = std.c.getenv("TF_DSV41_DENSE_COL");
        const dense_col = (dense_col_env == null or
            !std.mem.eql(u8, std.mem.span(dense_col_env.?), "0")) and
            ops.encodeDenseCol(e, &p.plain, x, tensor(selector.weights_proj), work.view(.index_weights), iq_w_args);
        if (!dense_col) ops.encodeDense(e, &p.plain, x, tensor(selector.weights_proj), work.view(.index_weights), iq_w_args);
    }
    e.barrier();
    // The slot-bound phase: the batched form issues each stage once for every slot, else one slot at a time.
    const batched = rows_phase.usable(p, w, work, c, segs);
    if (batched) try rows_phase.head(p, e, w, work, q, raw, c, ratio, segs);
    var s: usize = 0;
    while (s < nsegs) : (s += 1) {
        const sg = segs[s];
        const st = sg.state;
        const pos = sg.p0;
        const r0 = sg.r0;
        const sel = work.selectionFor(s);
        const qs = q.shifted(@as(usize, r0) * q_width * 2);
        const raws = raw.shifted(@as(usize, r0) * c.head_dim * 2);
        if (!batched) {
            try rotate(p, e, qs, 1, c.heads, c.head_dim, pos, ratio, 1, false, c);
        }
        const chunk = sel.windowChunk();
        if (!batched and w.kv_norm.dtype == .bf16 and c.head_dim == 512) {
            const factor_offset: usize = if (ratio == 0) 0 else @as(usize, p.rotary.capacity) * p.rotary.rotary * @sizeOf(f32);
            const factors: View = .{ .buffer = p.rotary.factors, .offset = factor_offset };
            try p.cache.encodeGlue(e, raws, tensor(w.kv_norm), factors, chunk, chunk, .{
                .rows = 1,
                .width = c.head_dim,
                .stride = c.head_dim,
                .packed_stride = kv.cache.ring_stride,
                .block = 32,
                .position = pos,
                .rotary = c.rope_dim,
                .write_y = 0,
                .eps = c.eps,
            });
        } else if (!batched) {
            norm(p, e, raws, w.kv_norm, 1, c.head_dim, c.eps);
            try rotate(p, e, raws, 1, 1, c.head_dim, pos, ratio, 1, false, c);
            try p.cache.encodePack(e, raws, chunk, .{ .rows = 1, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
        }
        e.barrier();
        var main: View = chunk;
        var compressed_rows: u32 = 0;
        if (ratio != 0) {
            const source = try st.source(layer, ratio);
            compressed_rows = (pos + 1) / ratio;
            const complete = compressed_rows - pos / ratio;
            if (source.owner == layer) {
                const compressor = if (w.compressor) |*value| value else return error.MissingAttentionCompressor;
                const values_row: usize = if (ratio == 1) c.head_dim * 2 else c.head_dim * 4;
                try p.cache.encodePool(e, work.view(.values).shifted(@as(usize, r0) * values_row), work.view(.gates).shifted(@as(usize, r0) * c.head_dim * 4), source.partial, tensor(compressor.norm), work.view(.latent).shifted(@as(usize, r0) * c.head_dim * 2), source.capture, .{
                    .rows = 1,
                    .ratio = ratio,
                    .position = pos,
                    .stride = c.head_dim,
                    .input_f32 = @intFromBool(ratio > 1),
                    .eps = c.eps,
                    .capture_rows = if (ratio > 1) kv.cache.capture_rows else 0,
                });
                e.barrier();
                if (complete != 0) {
                    const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
                    const wk = selector.wk orelse return error.MissingAttentionIndexKeyProjection;
                    const k_norm = selector.k_norm orelse return error.MissingAttentionIndexKeyNorm;
                    const latent = work.view(.latent).shifted(@as(usize, r0) * c.head_dim * 2);
                    const key = work.view(.index_k).shifted(@as(usize, r0) * c.index_dim * 2);
                    try projection(ep, e, wk, latent, key, scratch, 1, c.head_dim, c.index_dim, .bf16);
                    norm(p, e, key, k_norm, 1, c.index_dim, c.eps);
                    try rotate(p, e, key, 1, 1, c.index_dim, pos / ratio * ratio, ratio, ratio, false, c);
                    try p.cache.encodePack(e, key, source.index, .{ .rows = 1, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .first = pos / ratio });
                    e.barrier();
                }
            }
            if (ix_src) {
                const iq = work.view(.index_q).shifted(@as(usize, r0) * iq_width * 2);
                if (compressed_rows != 0 and compressed_rows > c.index_topk) {
                    try rotate(p, e, iq, 1, c.index_heads, c.index_dim, pos, ratio, 1, false, c);
                    try p.cache.encodePack(e, iq, sel.queryPacked(), .{ .rows = c.index_heads, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .inplace = 1 });
                    e.barrier();
                    const selp: index.Params = .{ .rows = 1, .q_stride = iq_width, .weight_stride = c.index_heads, .position = pos, .ratio = ratio, .keys = compressed_rows, .make_candidates = 0, .use_candidates = 0, .layer = layer, .skip_mask = 0 };
                    try p.select.encode(e, iq, work.view(.index_weights).shifted(@as(usize, r0) * c.index_heads * 2), source.index, sel.*, selp);
                    e.barrier();
                } else {
                    const selp: index.Params = .{ .rows = 1, .q_stride = iq_width, .weight_stride = c.index_heads, .position = pos, .ratio = ratio, .keys = compressed_rows, .layer = layer };
                    p.select.encodeIdentity(e, sel.*, selp);
                }
                sel.sel_layer = layer;
                sel.sel_position = pos;
                sel.sel_rows = 1;
            } else {
                const prev = precedingIndex(layer);
                if (sel.sel_layer == null or sel.sel_layer.? != prev or sel.sel_position != pos or sel.sel_rows != 1) return error.MissingAttentionSelection;
            }
            if (source.owner == layer and complete != 0) {
                const latent = work.view(.latent).shifted(@as(usize, r0) * c.head_dim * 2);
                try rotate(p, e, latent, 1, 1, c.head_dim, pos / ratio * ratio, ratio, ratio, false, c);
                try p.cache.encodePack(e, latent, source.main, .{ .rows = 1, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.main_stride, .block = 16, .kind = .main_fp4, .first = pos / ratio });
                e.barrier();
            }
            main = source.main;
        }
        if (!batched) {
            const ring = try st.ring(layer);
            try p.sparse.encode(e, qs, ring, chunk, main, sel.positions(), tensor(w.sink), qs, work.dense(), work.splitPartials(), .{
                .rows = 1,
                .q_stride = q_width,
                .position = pos,
                .chunk_rows = 1,
                .ratio = ratio,
                .compressed_rows = compressed_rows,
                .topk = @min(c.index_topk, compressed_rows),
                .splits = splitScan(),
            });
            e.barrier();
            try p.cache.publishWindow(e, chunk, ring, 1, pos);
            try rotate(p, e, qs, 1, c.heads, c.head_dim, pos, ratio, 1, true, c);
            e.barrier();
        }
    }
    if (batched) try rows_phase.tail(p, e, w, work, q, c, layer, ratio, segs);
    // The output phase: one call over every segment's rows.
    const grouped = work.view(.grouped);
    const group_input = q_width / c.o_groups;
    const group_output = c.o_groups * c.o_lora;
    if (w.wo_a_descriptors != null and std.c.getenv("TF_DSV41_NO_FUSE_WO_A") == null) {
        for (&w.wo_a) |*m| exl3.useMatrix(e, m.*);
        try exl3.encodeIndexed(ep, e, .{ .descriptors = .{ .buffer = w.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = group_input, .n = c.o_lora }, core(q), core(grouped), scratch, .{
            .count = rows * c.o_groups,
            .x_stride = group_input,
            .y_stride = c.o_lora,
            .output = .bf16,
            .output_rows = rows * c.o_groups,
            .routes = .{ .ids = core(.{ .buffer = work.ids, .offset = 0 }), .repeat = 1 },
        });
    } else {
        var r0: u32 = 0;
        while (r0 < rows) : (r0 += 1) {
            for (w.wo_a, 0..) |matrix, group| {
                try projection(ep, e, .{ .trellis = matrix }, q.shifted(@as(usize, r0) * q_width * 2).shifted(group * group_input * 2), grouped.shifted(@as(usize, r0) * group_output * 2).shifted(group * c.o_lora * 2), scratch, 1, q_width, group_output, .bf16);
            }
        }
    }
    e.barrier();
    try projection(ep, e, .{ .trellis = w.wo_b }, grouped, y, scratch, rows, group_output, c.hidden, .bf16);
}

/// The attention pass, in the release's own stages (`inference/model.py`, revision 314113f8).
pub fn encodeDiagnostic(p: *const Pipelines, e_in: mtl.ComputeEncoder, w: *const weights.Attention, state: *State, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, rows: u32, c: config.Config, comptime diagnostic: bool, observer: ?DiagnosticObserver, prof: ?*Prof, segs: []const Seg) !void {
    if (segs.len > 1) return encodeMulti(p, e_in, w, work, x, y, ep, scratch, layer, c, segs);
    var e = e_in;
    if (layer >= 40 or rows == 0 or rows > work.max_rows or
        state.position > state.plan.capacity or rows > state.plan.capacity - state.position)
        return error.InvalidAttentionChunk;
    attnBegin(layer);
    // Start the first block from a fresh buffer so the "pre" window is not charged to `qkv`.
    if (prof) |pf| pf.split("pre", layer, &e);
    const position = state.position;
    const ratio = c.compress_ratios.items()[layer];
    const end = position + rows;
    const q_width = c.heads * c.head_dim;
    const qr = work.view(.qr);
    const q = work.view(.query);
    const raw = work.view(.raw_kv);
    const ring = try state.ring(layer);
    const chunk = work.selection.windowChunk();
    e.barrier();
    const dense_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_DENSE") != null) 2 else 1;
    const dup_view: View = .{ .buffer = work.dup, .offset = 0 };
    var st = Stage{
        .p = p,
        .e = e,
        .w = w,
        .state = state,
        .work = work,
        .x = x,
        .y = y,
        .ep = ep,
        .scratch = scratch,
        .layer = layer,
        .rows = rows,
        .c = c,
        .observer = observer,
        .prof = prof,
        .position = position,
        .ratio = ratio,
        .end = end,
        .q_width = q_width,
        .qr = qr,
        .q = q,
        .raw = raw,
        .ring = ring,
        .chunk = chunk,
        .dense_passes = dense_passes,
        .dup_view = dup_view,
        .qkv_merged = false,
        .main = chunk,
        .compressed_rows = 0,
    };
    try st.queryStage(diagnostic);
    try window.windowStage(&st, diagnostic);
    try st.compressedStage();
    try st.readStage(diagnostic);
    try st.outputStage(diagnostic);
    e = st.e;
    if (prof) |pf| pf.enc = e;
}
