//! The model's forward pass.
const builtin = @import("builtin");
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const exl3 = @import("core").exl3_projection;
const config = @import("config.zig");
const load = @import("load.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
const attention = @import("attention.zig");
const attention_common = @import("attention_common.zig");
const kv = @import("kv.zig");
const ops = @import("ops.zig");
const mhc = @import("mhc.zig");
const csa2 = @import("csa2.zig");
const csa2_index = @import("csa2_index.zig");
const rope = @import("rope.zig");
const route = @import("route.zig");
const engram = @import("engram.zig");
const engram_stream = @import("engram_rows_safetensors.zig");
const gpu_counters = @import("profiler.zig");
const expert_groups = @import("core").expert_groups;
const scan_probe = @import("scan_probe.zig");
const model = @import("model.zig");
const Model = model.Model;
const View = model.View;
const Logits = model.Logits;
const Seg = model.Seg;
const HidHash = model.HidHash;
const tensor = model.tensor;
const splitStage = model.splitStage;
const completed = model.completed;
const nowNs = model.nowNs;
const tapsOn = model.tapsOn;
const tapsSharedBaseOn = model.tapsSharedBaseOn;
const condAlignOn = model.condAlignOn;
const concurrentOn = model.concurrentOn;
const forward_cone = @import("forward_cone.zig");

/// One forward over the segments; the cone prefill (forward_cone.zig) takes a long prompt's chunks first.
pub fn forwardSegs(m: *Model, segs: []const Seg, logits: Logits) ![]const f32 {
    if (!m.valid) return error.InvalidModelState;
    if (try forward_cone.route(m, segs, logits)) |out| return out;
    return forwardCore(m, segs, logits);
}

/// The forward itself; `m.cone.phase` makes it the cone's encoder half (stops at the decoder) or decoder half (replay).
pub fn forwardCore(m: *Model, segs: []const Seg, logits: Logits) ![]const f32 {
    if (!m.valid) return error.InvalidModelState;
    if (segs.len == 0 or segs.len > 8) return error.InvalidLaneSlot;
    const first_seg = segs[0];
    const slot = first_seg.slot;
    if (slot >= m.lanes) return error.InvalidLaneSlot;
    var total_rows: u32 = 0;
    for (segs) |sg| {
        if (sg.slot >= m.lanes) return error.InvalidLaneSlot;
        if (sg.ids.len == 0) return error.InvalidPrefillChunk;
        if (sg.p0 != m.caches[sg.slot].position) return error.PositionMismatch;
        if (m.caches[sg.slot].position > m.caches[sg.slot].plan.capacity or
            sg.ids.len > m.caches[sg.slot].plan.capacity - m.caches[sg.slot].position) return error.ContextFull;
        if (total_rows + sg.ids.len > m.workspace.plan.rows) return error.InvalidPrefillChunk;
        total_rows += @intCast(sg.ids.len);
    }
    if (segs.len > 1 and (m.profile != null or m.observer != null or m.stream == null or
        std.c.getenv("TF_DSV41_LAYER_CB") != null or
        std.c.getenv("TF_DSV41_ATTN_PROF") != null or std.c.getenv("TF_DSV41_HID_HASH") != null))
        return error.BatchUnsupported;
    const state = &m.caches[slot];
    const cone_phase = m.cone.phase;
    const cone_d = m.cone.d;
    defer m.attention_workspace.cone = .full;
    const rows: u32 = total_rows;
    if (segs.len > 1 and std.c.getenv("TF_DSV41_BATCH_DEBUG") != null) std.log.info("forwardSegs: segs={d} rows={d}", .{ segs.len, rows });
    // A forward starts from the embedding: its stream lives in `.stream` and no sublayer branch is pending.
    m.stream_alt = false;
    m.pending_expand = false;
    m.engram_pending = false;
    for (segs) |sg| for (sg.ids, 0..) |id, at| if (id >= 129280) {
        std.log.err("forwardOn: unsupported token id {d} at row {d} of {d} (position {d})", .{ id, at, sg.ids.len, sg.p0 });
        return error.UnsupportedToken;
    };
    m.logits_rows = 0;
    {
        const dst = m.workspace.tokenIds(rows);
        var at: usize = 0;
        for (segs) |sg| {
            @memcpy(dst[at..][0..sg.ids.len], sg.ids);
            at += sg.ids.len;
        }
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    // The forward-stage hash's staging buffer, one forward's life.
    var hid = HidHash{};
    var hid_stage: ?mtl.Buffer = null;
    defer if (hid_stage) |b| b.deinit();
    if (HidHash.on()) {
        hid = HidHash.init(rows);
        hid_stage = try m.device.buffer(hid.bytes, mtl.ResourceOptions.shared);
        hid.stage = hid_stage.?;
    }
    // Probe: the sparse scan's own input and output, one layer, one forward's staging.
    var sp_stage: ?mtl.Buffer = null;
    defer if (sp_stage) |b| {
        b.deinit();
        scan_probe.stage = null;
    };
    if (scan_probe.on()) {
        scan_probe.init(rows, @intCast(m.cfg.heads * m.cfg.head_dim));
        sp_stage = try m.device.buffer(scan_probe.cap, mtl.ResourceOptions.shared);
        scan_probe.stage = sp_stage.?;
        scan_probe.reset();
    }
    if (m.stream != null and cone_phase != .decoder and std.c.getenv("TF_DSV41_SKIP_ENGRAM") == null) {
        const hash_start: u64 = if (m.host_prof) nowNs() else 0;
        m.host_stage_ns = 0;
        m.host_stage_misses = 0;
        m.host_stage_faults = 0;
        if (segs.len == 1) try m.hashPass(slot, rows) else try m.hashPassSegs(segs, rows);
        if (m.host_prof) m.host_hash_ns = nowNs() - hash_start;
    }
    const host_enc_start: u64 = nowNs();
    const dispatches_before = mtl.command.dispatch_count.load(.monotonic);
    // TF_DSV41_LAYER_CB: one command buffer a layer, with the GPU's own seconds for each.
    const layer_cb = std.c.getenv("TF_DSV41_LAYER_CB") != null;
    // THE PIPELINED FORWARD: commit the first n layers before waiting, so the GPU runs while the host encodes.
    const pipe_layers: u32 = if (rows <= 8 and m.profile == null and m.observer == null and
        std.c.getenv("TF_DSV41_LAYER_CB") == null and std.c.getenv("TF_DSV41_STAGE_CB") == null and
        std.c.getenv("TF_DSV41_ATTN_PROF") == null)
        (if (std.c.getenv("TF_DSV41_PIPE_LAYERS")) |v|
            (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2)
        else
            2)
    else
        0;
    var pipe_prev: mtl.CommandBuffer = undefined;
    var pipe_prev_set = false;
    var pipe_gpu_extra: f64 = 0;
    // The command buffers committed before the second Engram layer's staged read.
    var engram_prev: [4]mtl.CommandBuffer = undefined;
    var engram_prev_n: usize = 0;
    var e = m.queue.commandBufferUnretained();
    var enc = e.compute(if (concurrentOn()) .concurrent else .serial);
    for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
    var encoding = true;
    var record_started = false;
    errdefer {
        if (m.profile) |recorder| {
            if (record_started) {
                if (encoding) recorder.cancelEncoding(&enc) catch {} else recorder.discardAfterCompletion() catch {};
            } else if (encoding) enc.end();
        } else if (encoding) enc.end();
    }
    if (m.profile) |recorder| {
        try recorder.beginCommand(e);
        record_started = true;
    }
    try m.profileSpan(&enc, .other);
    const p = &m.pipelines;
    const ws = &m.workspace;
    if (m.stream == null and cone_phase != .decoder) {
        for (engram.hash.layers, 0..) |meta, index| {
            try engram.encodeHash(enc, &p.engram, meta, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), ws.view(if (index == 0) .engram_ids_0 else .engram_ids_1), rows);
        }
        try engram.encodeCommit(enc, &p.engram, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), rows);
    }
    if (cone_phase == .decoder) forward_cone.loadStash(m, enc, rows) else {
        ops.encodeEmbedding(enc, &p.ops, ws.view(.ids), tensor(m.weights.embed), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .table_stride = 5120, .y_stride = 5120, .copies = 1 });
        enc.barrier();
        mhc.encodeExpand(enc, &p.mhc, ws.view(.collapsed), ws.view(.stream), ws.view(.previous_pre), .{ .rows = rows, .width = 5120, .x_stride = 5120, .stream_stride = 20480 });
        enc.barrier();
    }
    hid.record(enc, &p.ops, m.streamView(), rows, .embed, 0);
    var asegs: [8]attention.Seg = undefined;
    {
        var at: u32 = 0;
        for (segs, 0..) |sg, si| {
            asegs[si] = .{ .state = &m.caches[sg.slot], .p0 = sg.p0, .r0 = at, .n = @intCast(sg.ids.len) };
            at += @intCast(sg.ids.len);
        }
    }
    for (&m.weights.layers, 0..) |*layer, index| {
        if (cone_phase == .decoder and index < m.cone.lo) continue;
        if (pipe_layers != 0 and index == pipe_layers) {
            enc.end();
            e.commit();
            pipe_prev = e;
            pipe_prev_set = true;
            e = m.queue.commandBufferUnretained();
            enc = e.compute(if (concurrentOn()) .concurrent else .serial);
            for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
        }
        // THE ENGRAM GATE READS THE EXPANDED STREAM: flush the pending expand before the gate sees post-branch hidden.
        if (layer.engram != null and m.pending_expand and
            std.c.getenv("TF_DSV41_SKIP_ENGRAM") == null and
            std.c.getenv("TF_DSV41_NO_ENGRAM_FLUSH") == null)
        {
            const cur = m.streamView();
            const nxt = if (m.stream_alt) ws.view(.stream) else ws.view(.stream_next);
            mhc.encodePost(enc, &p.mhc, ws.view(.branch), cur, ws.view(.post), ws.view(.comb), nxt, .{ .rows = rows, .width = 5120, .branch_stride = 5120, .residual_stride = 20480, .y_stride = 20480 });
            m.stream_alt = !m.stream_alt;
            m.pending_expand = false;
            enc.barrier();
        }
        hid.record(enc, &p.ops, m.streamView(), rows, .stream, @intCast(index));
        // TF_DSV41_SKIP_ENGRAM: a diagnostic arm for the Engram term (its gate kernels and the host hash pass).
        if (layer.engram != null and std.c.getenv("TF_DSV41_SKIP_ENGRAM") == null) {
            const g = &layer.engram.?;
            if (m.engram_pending and @as(u32, @intCast(index)) == engram.hash.layers[1].layer_id and m.engramPipeRows(rows)) {
                // Commit the layers encoded so far, so the GPU is running them while the second layer's rows are read.
                enc.end();
                e.commit();
                if (engram_prev_n < engram_prev.len) {
                    engram_prev[engram_prev_n] = e;
                    engram_prev_n += 1;
                }
                e = m.queue.commandBufferUnretained();
                enc = e.compute(if (concurrentOn()) .concurrent else .serial);
                for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
                m.engram_pending = false;
                const stage_t0: u64 = if (m.host_prof) nowNs() else 0;
                try m.stageLayer(rows, 1);
                if (m.host_prof) m.host_stage_ns += nowNs() - stage_t0;
                m.stageHash(rows, "l1");
            }
            try m.profileSpan(&enc, .engram);
            const engram_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ENGRAM") != null) 2 else 1;
            for (0..engram_passes) |_| if (m.stream_stage) |staged| {
                const at = @as(usize, if (index == 1) 0 else 1) * @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
                try m.project(enc, g.wkv, View{ .buffer = staged, .offset = at }, ws.view(.engram_kv), rows, engram.gather_dim, 25600, .bf16);
            } else {
                try engram.encodeGather(enc, &p.engram, .{ .weight = tensor(g.table.?), .scale = tensor(g.scales.?), .rows = @intCast(g.table.?.shape[0]) }, ws.view(if (index == 1) .engram_ids_0 else .engram_ids_1), ws.view(.engram_gather), rows, 6144);
                try m.project(enc, g.wkv, ws.view(.engram_gather), ws.view(.engram_kv), rows, 6144, 25600, .bf16);
            };
            try engram.encodeGate(enc, &p.engram, .{ .q_weight = tensor(g.q_weight), .k_weight = tensor(g.k_weight), .dim = 5120, .copies = 4, .weight_dtype = switch (g.q_weight.dtype) {
                .bf16 => .bf16,
                else => .f32,
            }, .hidden_stride = 20480, .kv_stride = 25600, .output_stride = 20480, .eps = m.cfg.eps }, m.streamView(), ws.view(.engram_kv), null, m.streamView(), rows);
        }
        // The drafter reads the mean of this layer's input streams at its three target layers.
        if (segs.len == 1 and tapsOn() and index >= 37 and index <= 39) {
            // The last rows of this pass, a ring of 128: the drafter's conditioning is the window's rows.
            const n: u32 = @min(rows, 128);
            const skip = @as(usize, rows - n) * 20480 * 2;
            // THE CAPTURE'S FIRST POSITION: a merged tail is not 128-aligned, so it cannot reuse the chunk start.
            const base = (state.position + (rows - n)) % 128;
            // A MERGED TAIL CHUNK ALSO STASHES THE PREFIX -- the taps ring as it stands at the mark.
            if (m.publish_limit != 0) {
                const q = @min(m.publish_limit, rows);
                const sn: u32 = @min(q, 128);
                ops.encodeStreamMean(enc, &p.ops, m.streamView().shifted(@as(usize, q - sn) * 20480 * 2), .{ .buffer = m.taps_stash }, .{ .rows = sn, .width = 5120, .stream_stride = 20480, .out_stride = 5120, .slot = @intCast(index - 37), .row_base = (state.position + q - sn) % 128 });
            }
            // THE TARGET LAYERS SHARE ONE row_base: taps are interleaved by position, so the ring advances once.
            const shared = tapsSharedBaseOn();
            if (shared and index == 37) {
                // THE TAPS RING'S SLOT IS THE TOKEN POSITION, not the cursor: the two advance at different rates.
                m.taps_base = if (condAlignOn()) state.position % 128 else m.taps_row;
                // A fresh capture's rows are all valid; a rollback sets the prefix.
                m.taps_rows_valid = 0;
                m.taps_row = (m.taps_row + n) % 128;
                m.taps_n = @min(m.taps_n + n, 128);
            }
            ops.encodeStreamMean(enc, &p.ops, m.streamView().shifted(skip), .{ .buffer = m.taps }, .{ .rows = n, .width = 5120, .stream_stride = 20480, .out_stride = 5120, .slot = @intCast(index - 37), .row_base = if (shared) base else m.taps_row });
            // The capture's row count, so the conditioning knows how many tail rows of the taps ring are consecutive.
            m.taps_rows_last = n;
            if (!shared) {
                // A fresh capture's rows are all valid; a rollback sets the prefix.
                m.taps_rows_valid = 0;
                m.taps_row = (m.taps_row + n) % 128;
                m.taps_n = @min(m.taps_n + n, 128);
            }
            enc.barrier();
        }
        try splitStage(m, &enc, &e, "engram", index);
        try m.profileSpan(&enc, .mhc);
        if (cone_phase == .encoder and index == cone_d) forward_cone.stashRows(m, enc, rows, state.position);
        if (p.hc_boundary) |*bp| {
            m.boundary(enc, bp, layer.hc_attn, ws.view(.previous_pre), ws.view(.attention_pre), layer.attn_norm, rows, true);
            enc.barrier();
        } else {
            m.mixes(enc, layer.hc_attn, ws.view(.attention_pre), rows);
            mhc.encodeCollapse(enc, &p.mhc, m.streamView(), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            m.normalize(enc, ws.view(.collapsed), layer.attn_norm, ws.view(.normalized), rows);
        }
        hid.record(enc, &p.ops, ws.view(.normalized), rows, .attn_in, @intCast(index));
        try splitStage(m, &enc, &e, "mhc_attn", index);
        try m.profileSpan(&enc, .attention);
        m.attention_workspace.cone = if (index != cone_d) .full else switch (cone_phase) {
            .encoder => .publish,
            .decoder => .skip,
            .off => .full,
        };
        if (std.c.getenv("TF_DSV41_SKIP_ATTN") == null) {
            if (attention.profLayerOn(@intCast(index))) {
                var aprof = attention.Prof{ .queue = m.queue, .cb = e, .enc = enc, .on = true };
                try attention.encodeProfiled(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg, &aprof);
                enc = aprof.enc;
                e = aprof.cb;
            } else if (m.attention_observer) |attention_observer|
                try attention.encodeDiagnostic(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg, true, attention_observer, null, &.{})
            else if (segs.len > 1)
                try attention.encodeDiagnostic(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg, false, null, null, asegs[0..segs.len])
            else
                try attention.encode(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg);
        }
        enc.barrier();
        if (cone_phase == .encoder and index == cone_d) break;
        hid.record(enc, &p.ops, ws.view(.branch), rows, .attn, @intCast(index));
        if (hid.full_layer >= 0 and index == @as(usize, @intCast(hid.full_layer))) {
            // The attention block's own regions, in execution order, so the first one that moves names the kernel.
            const aw = &m.attention_workspace;
            const li: u32 = @intCast(index);
            hid.recordBytes(enc, &p.ops, aw.view(.qr), rows, 1280 * 2, .a_qr, li);
            hid.recordBytes(enc, &p.ops, aw.view(.query), rows, 32768 * 2, .a_q, li);
            hid.recordBytes(enc, &p.ops, aw.view(.raw_kv), rows, 512 * 2, .a_kv, li);
            hid.recordBytes(enc, &p.ops, aw.view(.values), rows, 512 * 4, .a_vals, li);
            hid.recordBytes(enc, &p.ops, aw.view(.gates), rows, 512 * 4, .a_gates, li);
            // A compressed region is written for this chunk's `complete` new rows only.
            const aratio = m.cfg.compressRatio(@intCast(index));
            const complete: u32 = if (aratio == 0) 0 else @intCast((state.position + rows) / aratio - state.position / aratio);
            if (complete != 0) {
                hid.recordBytes(enc, &p.ops, aw.view(.latent), complete, 512 * 2, .a_latent, li);
                hid.recordBytes(enc, &p.ops, aw.view(.index_k), complete, 128 * 2, .a_idxk, li);
            }
            hid.recordBytes(enc, &p.ops, aw.view(.index_q), rows, 4096 * 2, .a_idxq, li);
            hid.recordBytes(enc, &p.ops, aw.view(.index_weights), rows, 32 * 2, .a_idxw, li);
            // The picks in 64-row bands, each its own line, so the first band that moves names the row range.
            var band: u32 = 0;
            while (@as(usize, band) * 64 < rows) : (band += 1) {
                const r0: u32 = band * 64;
                const nb: u32 = @min(64, rows - r0);
                hid.recordBytes(enc, &p.ops, aw.selection.positions().shifted(@as(usize, r0) * 512 * 4), nb, 512 * 4, .a_pos, 200000 + band);
            }
            hid.recordBytes(enc, &p.ops, aw.selection.positions().shifted(@as(usize, rows - 1) * 512 * 4), 1, 512 * 4, .a_pos1, li);
            hid.recordBytes(enc, &p.ops, aw.selection.windowChunk(), rows, 528, .a_chunk, li);
            if (aratio != 0) {
                const srcv = try state.source(@intCast(index), aratio);
                const compv: u32 = @intCast((state.position + rows) / aratio);
                if (compv != 0) hid.recordBytes(enc, &p.ops, srcv.main, compv, 288, .a_main, li);
            }
            if (aratio != 0) {
                const keys_all: u32 = @intCast((state.position + rows) / aratio);
                if (keys_all != 0 and keys_all <= 4096)
                    hid.recordStrided(enc, &p.ops, aw.selection.scores(), rows, 4096 * 4, keys_all * 4, .a_sco, 999999);
            }
            if (aratio != 0 and rows > 1) {
                var k: u32 = 0;
                while (k < 16 and k < rows) : (k += 1) {
                    const row: u32 = (rows - 1) - @as(u32, @intCast((@as(usize, rows - 1) * k) / 15));
                    const vis: u32 = @intCast(@min((state.position + row + 1) / aratio, (state.position + rows) / aratio));
                    if (vis == 0) continue;
                    hid.recordBytes(enc, &p.ops, aw.selection.scores().shifted(@as(usize, row) * 4096 * 4), 1, vis * 4, .a_sco, 100000 + row);
                }
            }
            hid.recordBytes(enc, &p.ops, aw.view(.grouped), rows, 8192 * 2, .a_grp, li);
        }
        try splitStage(m, &enc, &e, "attention", index);
        try m.profileSpan(&enc, .mhc);
        if (p.hc_boundary != null) m.pending_expand = true else m.post(enc, rows);
        if (m.observer) |observer| observer.encode(observer.ctx, enc, @intCast(index), .attention, m);
        if (p.hc_boundary) |*bp| {
            m.boundary(enc, bp, layer.hc_ffn, ws.view(.attention_pre), ws.view(.previous_pre), layer.ffn_norm, rows, true);
            enc.barrier();
        } else {
            m.mixes(enc, layer.hc_ffn, ws.view(.previous_pre), rows);
            mhc.encodeCollapse(enc, &p.mhc, m.streamView(), ws.view(.attention_pre), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            m.normalize(enc, ws.view(.collapsed), layer.ffn_norm, ws.view(.normalized), rows);
        }
        hid.record(enc, &p.ops, ws.view(.normalized), rows, .ffn_in, @intCast(index));
        try splitStage(m, &enc, &e, "mhc_ffn", index);
        try m.profileSpan(&enc, .moe);
        if (std.c.getenv("TF_DSV41_SKIP_MOE") == null) try m.moe(&enc, &e, &layer.moe, rows, @intCast(index), m.cfg.topk, null);
        hid.record(enc, &p.ops, ws.view(.branch), rows, .moe, @intCast(index));
        try m.profileSpan(&enc, .mhc);
        if (p.hc_boundary != null) m.pending_expand = true else m.post(enc, rows);
        if (m.observer) |observer| observer.encode(observer.ctx, enc, @intCast(index), .moe, m);
        try splitStage(m, &enc, &e, "moe", index);
        if (cone_phase == .decoder and index + 1 == m.cone.hi and m.cone.hi < m.cfg.layers) {
            forward_cone.stashRows(m, enc, rows, state.position);
            break;
        }
        if (layer_cb) {
            enc.end();
            e.commit();
            e.wait();
            try completed(e);
            const gpu: f64 = e.gpuSeconds() * 1e3;
            const host: f64 = @as(f64, @floatFromInt(nowNs() - host_enc_start)) / 1e6;
            std.log.info("layer {d:>2}: command gpu {d:7.2} ms, wall so far {d:8.2} ms, ratio {d}, engram {}", .{ index, gpu, host, m.cfg.compressRatio(@intCast(index)), layer.engram != null });
            e = m.queue.commandBufferUnretained();
            enc = e.compute(.serial);
            for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
        }
    }
    if (logits != .none) {
        try m.profileSpan(&enc, .head);
        const output_rows: u32 = if (logits == .all) rows else 1;
        const first: usize = if (logits == .all) 0 else rows - 1;
        if (p.hc_boundary) |*bp| {
            if (m.pending_expand) {
                const last = &m.weights.layers[m.weights.layers.len - 1].hc_ffn;
                const wsx = &m.workspace;
                const cur = if (m.stream_alt) wsx.view(.stream_next) else wsx.view(.stream);
                const next = if (m.stream_alt) wsx.view(.stream) else wsx.view(.stream_next);
                enc.setPipeline(bp.expand_mix);
                inline for (.{ cur, wsx.view(.branch), wsx.view(.post), wsx.view(.comb), tensor(last.projection), wsx.view(.hc_part), next }, 0..) |r, i| enc.setBuffer(r.buffer, r.offset, i);
                enc.dispatchGroups(.{ .width = 20, .height = if (logits == .all) rows else 1 }, .{ .width = 256 });
                enc.barrier();
                m.stream_alt = !m.stream_alt;
                m.pending_expand = false;
            }
        }
        mhc.encodeCollapse(enc, &p.mhc, m.streamView().shifted(first * 20480 * 2), ws.view(.previous_pre).shifted(first * 4 * 4), ws.view(.collapsed), .{ .rows = output_rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
        enc.barrier();
        m.normalize(enc, ws.view(.collapsed), m.weights.norm, ws.view(.normalized), output_rows);
        // The trunk's own head input (the row 0 of the window), so the drafter's head input can be compared with it.
        if (std.c.getenv("TF_DSV41_MTP_DEBUG_TOP") != null) {
            if (m.mtp_trunk_hidden == null) {
                m.mtp_trunk_hidden = try m.device.buffer(5120 * 4, mtl.ResourceOptions.shared);
                m.residency.add(m.mtp_trunk_hidden.?);
            }
            ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = m.mtp_trunk_hidden.?, .offset = 0 }, 5120, false);
        }
        const head_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_HEAD") != null) 2 else 1;
        for (0..head_passes) |_| try m.project(enc, m.weights.head, ws.view(.normalized), ws.view(.logits), output_rows, 5120, 129280, .f32);
        hid.recordLogits(enc, &p.ops, ws.view(.logits).shifted(@as(usize, output_rows - 1) * 129280 * 4));
    }
    // TF_DSV41_ROUTE_DEBUG5: the head's own logits (first row), the number that decides the token.
    if (std.c.getenv("TF_DSV41_ROUTE_DEBUG5") != null and !model.gate_dump_done3 and logits != .none) {
        model.gate_dump_done3 = true;
        const n: u32 = 129280;
        const stg = try m.device.buffer(@as(usize, n) * 4, mtl.ResourceOptions.shared);
        ops.encodeDump(enc, &p.ops, ws.view(.logits), .{ .buffer = stg, .offset = 0 }, n, false);
        try splitStage(m, &enc, &e, "head", 0);
        const v: [*]const f32 = @ptrCast(@alignCast(stg.contents()));
        var best: f32 = -1e30;
        var best_id: u32 = 0;
        for (0..n) |i| if (v[i] > best) {
            best = v[i];
            best_id = @intCast(i);
        };
        std.log.info("head logits {d:.4} {d:.4} {d:.4} | argmax {d} @ {d:.4}", .{ v[0], v[1], v[2], best_id, best });
    }
    if (m.profile) |recorder| {
        if (recorder.activeSpan()) try recorder.endSpan(&enc);
        try recorder.finishEncoding(&enc);
    } else enc.end();
    encoding = false;
    m.valid = false;
    const host_commit: u64 = nowNs();
    e.commit();
    e.wait();
    const host_done: u64 = nowNs();
    try completed(e);
    hid.report();
    scan_probe.report();
    if (pipe_prev_set) {
        // The earlier chunks completed in queue order by the last wait; check them and count their GPU time.
        try completed(pipe_prev);
        pipe_gpu_extra = pipe_prev.gpuSeconds() * 1e3;
    }
    for (engram_prev[0..engram_prev_n]) |cb| {
        try completed(cb);
        pipe_gpu_extra += cb.gpuSeconds() * 1e3;
    }
    // TF_DSV41_MTP_DRAFTS: the drafter's proposal after this forward, from the token the head just chose.
    if (m.profile) |recorder| {
        _ = try recorder.readAfterCompletion();
        const agg = recorder.aggregates();
        var total: f64 = 0;
        for (agg) |a| total += a.gpu_seconds;
        for (agg, gpu_counters.stage_names) |a, name| {
            if (a.calls != 0) std.log.info("stage {s}: {d} calls, {d:.1} ms, {d:.1}% of recorded", .{ name, a.calls, a.gpu_seconds * 1e3, if (total > 0) a.gpu_seconds / total * 100 else 0 });
        }
        std.log.info("stage total: {d:.1} ms", .{total * 1e3});
        if (m.profile_open_stage) |prev_stage| {
            m.profile_stage_dispatch[@backingInt(prev_stage)] += mtl.command.dispatch_count.load(.monotonic) - m.profile_last_count;
            m.profile_open_stage = null;
        }
        for (gpu_counters.stage_names, m.profile_stage_dispatch) |name, d| {
            if (d != 0) std.log.info("stage dispatches {s}: {d}", .{ name, d });
        }
        var hist: [20]u64 = undefined;
        for (&hist, 0..) |*h, i| h.* = mtl.command.dispatch_hist[i].load(.monotonic);
        std.log.info("dispatch histogram (log2 threadgroups): {any}, threadgroups {d}, indirect {d}", .{ hist, mtl.command.dispatch_threadgroups.load(.monotonic), mtl.command.dispatch_indirect.load(.monotonic) });
    }
    // TF_DSV41_HOST_PROF without the counter recorder (which is broken on this build).
    if (m.host_prof) {
        std.log.info("host: rows {d}, engram hash {d:.2} ms (of which stage {d:.2} ms, misses {d}, faults {d}, pf {d}, thr {d})", .{
            rows,
            @as(f64, @floatFromInt(m.host_hash_ns)) / 1e6,
            @as(f64, @floatFromInt(m.host_stage_ns)) / 1e6,
            m.host_stage_misses,
            m.host_stage_faults,
            m.host_stage_pf,
            m.host_stage_thr,
        });
        std.log.info("forward: rows {d}, encode {d:.2} ms, commit+wait {d:.2} ms, command gpu {d:.2} ms, wall {d:.2} ms, dispatches {d}, pipe {d}", .{
            rows,
            @as(f64, @floatFromInt(host_commit - host_enc_start)) / 1e6,
            @as(f64, @floatFromInt(host_done - host_commit)) / 1e6,
            e.gpuSeconds() * 1e3 + pipe_gpu_extra,
            @as(f64, @floatFromInt(host_done - host_enc_start)) / 1e6,
            mtl.command.dispatch_count.load(.monotonic) - dispatches_before,
            pipe_layers,
        });
    }
    for (segs) |sg| try m.caches[sg.slot].append(@intCast(sg.ids.len));
    m.last_gpu_seconds = e.gpuSeconds() + pipe_gpu_extra / 1e3;
    m.valid = true;
    m.logits_rows = if (logits == .all) rows else if (logits == .last) 1 else 0;
    return ws.logits(m.logits_rows);
}
