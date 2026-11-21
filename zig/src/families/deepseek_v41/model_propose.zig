//! The drafter's round.

// The drafter's head loop follows the release's DSparkBlock.forward_head (inference/model.py, revision 314113f8).
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
const CondProf = Model.CondProf;
const tensor = model.tensor;
const asF32 = model.asF32;
const completed = model.completed;
const draft_round = @import("model_draft_round.zig");
const DraftRound = draft_round.DraftRound;
pub fn proposeTokens(m: *Model, slot: u32, pending: u32, count: u32) ![4]u32 {
    const dspark = m.weights.dspark orelse return error.NoDrafter;
    // Bisect arms: TF_DSV41_MTP_STUB returns before the draft forward, TF_DSV41_MTP_STUB_COND after the conditioning.
    if (std.c.getenv("TF_DSV41_MTP_STUB") != null) return .{ 0, 0, 0, 0 };
    // The pending row is this round's first verified row; start its page-in now, under the drafter's own GPU window.
    if (Model.engramPendingOn()) m.prefetchPending(slot, pending);
    // The draft round runs its own boundaries through the shared coefficient views and the stream parity.
    const alt0 = m.stream_alt;
    const pend0 = m.pending_expand;
    defer {
        m.stream_alt = alt0;
        m.pending_expand = pend0;
    }
    const ws = &m.workspace;
    const p = &m.pipelines;
    const state = try m.drafterState();
    // The conditioning's cursor is THIS slot's committed position, not lane 0's.
    const slot_pos = m.caches[slot].position;
    state.position = slot_pos;
    // THE DRAFTER RUNS `depth + 1` ROWS, NOT FIVE: extra rows displace the real history its attention reads.
    const n: u32 = if (std.c.getenv("TF_DSV41_MTP_ROWS")) |v|
        (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 5)
    else
        @min(@max(count, 1) + 1, 5);
    var out: [4]u32 = @splat(0);
    var cb = m.queue.commandBuffer();
    var enc = cb.compute(if (model.concurrentOn()) .concurrent else .serial);
    // The conditioning's per-part GPU split, and the host's own encode time against the GPU's window.
    var cond_prof = CondProf{ .queue = m.queue, .cb = cb, .enc = enc, .on = std.c.getenv("TF_DSV41_COND_PROF") != null, .blk_on = std.c.getenv("TF_DSV41_BLOCK_PROF") != null };
    const host_prof = std.c.getenv("TF_DSV41_HOST_PROF") != null;
    const t_start: f64 = if (host_prof) mtl.clock.seconds() else 0;
    // The conditioning: target-layer stream means via main_proj/main_norm, then keys into each ring.
    const pos_now = slot_pos;
    // The captured rows are the last `taps_rows_last` rows of the last forward.
    const cap_n: u32 = @min(m.taps_rows_last, 128);
    const cap_pos: u32 = pos_now -| cap_n;
    const true_pos = model.dsparkTruePosOn();
    // The capture's valid rows: the rollback's prefix (the rejected rows' taps are not the committed history).
    const cap_rows: u32 = @min(@min(m.taps_rows_last, m.taps_n), 128);
    // The conditioning publishes the capture's whole rows at their own positions.
    const dspark_capture_all: u32 = if (std.c.getenv("TF_DSV41_DSPARK_CAPTURE_ALL")) |v|
        (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1)
    else
        1;
    const tail: u32 = if (std.c.getenv("TF_DSV41_DSPARK_COND_ONE") != null)
        1
    else if (m.taps_rows_valid != 0)
        @min(m.taps_rows_valid, cap_rows)
    else
        cap_rows;
    // The published rows' positions: the capture's own (the release's form) or the earlier shifted form.
    const pub_pos: u32 = if (true_pos) cap_pos else pos_now - tail;
    // The drafter's own rows' rope position: the next token's position.
    const draft_pos: u32 = if (true_pos) cap_pos + tail else slot_pos;
    // The windowed context's row count.
    var nc: u32 = 1;
    if (model.condRowsWanted() != 0) {
        const have: u32 = if (model.condAlignOn()) @min(pos_now, 128) else @min(m.taps_n, pos_now);
        const want: u32 = @min(model.condRowsWanted(), have);
        nc = if (want >= 4) (want / 4) * 4 else want;
    }
    var round = DraftRound{
        .m = m,
        .dspark = dspark,
        .slot = slot,
        .pending = pending,
        .count = count,
        .n = n,
        .state = state,
        .pos_now = pos_now,
        .cap_n = cap_n,
        .cap_pos = cap_pos,
        .true_pos = true_pos,
        .cap_rows = cap_rows,
        .dspark_capture_all = dspark_capture_all,
        .tail = tail,
        .pub_pos = pub_pos,
        .draft_pos = draft_pos,
        .nc = nc,
        .enc = enc,
        .cb = cb,
        .cond_prof = &cond_prof,
    };
    try round.conditionStage();
    enc = round.enc;
    cb = round.cb;
    // THE DRAFTER'S STOP POINT 1 (after the conditioning): commits what was encoded so completion fires.
    if (model.mtpStopPoint() == 1 or std.c.getenv("TF_DSV41_MTP_STUB_COND") != null) {
        enc.end();
        cb.commit();
        cb.wait();
        std.log.info("mtp stop 1: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
        try completed(cb);
        return .{ 0, 0, 0, 0 };
    }
    // THE DRAFTER'S OWN EARLY COMMIT: a command-buffer boundary is a full barrier, and the queue is in-order.
    var pipe_prev: mtl.CommandBuffer = undefined;
    var pipe_prev_set = false;
    const mtp_pipe: bool = blk: {
        if (cond_prof.on or cond_prof.blk_on) break :blk false;
        const raw = std.c.getenv("TF_DSV41_MTP_PIPE") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.span(raw), "0");
    };
    if (mtp_pipe) {
        enc.end();
        cb.commit();
        pipe_prev = cb;
        pipe_prev_set = true;
        cb = m.queue.commandBuffer();
        enc = cb.compute(if (model.concurrentOn()) .concurrent else .serial);
        cond_prof.cb = cb;
        cond_prof.enc = enc;
    }
    const ids = try m.drafterRound(pending, n);
    ops.encodeEmbedding(enc, &p.ops, ids, tensor(m.weights.embed), ws.view(.branch), .{ .rows = n, .width = 5120, .table_stride = 5120, .y_stride = 5120, .copies = 1 });
    enc.barrier();
    if (cond_prof.blk_on) cond_prof.split("blk", "prologue/embed", 0, &enc);
    const dbg: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_MTP_DEBUG_ROWS") != null) try m.device.buffer(model.mtp_dbg_total3 * 4, mtl.ResourceOptions.shared) else null;
    defer if (dbg) |b| b.deinit();
    if (dbg) |b| ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 0 }, 5 * 5120, true);
    // The mHC's expanded stream and its `pre` vector carry across the drafter's blocks, as the trunk's do.
    mhc.encodeExpand(enc, &p.mhc, ws.view(.branch), ws.view(.stream), ws.view(.previous_pre), .{ .rows = n, .width = 5120, .x_stride = 5120, .stream_stride = 20480, .pre_stride = 4 });
    if (cond_prof.blk_on) cond_prof.split("blk", "prologue/expand", 0, &enc);
    round.enc = enc;
    round.cb = cb;
    try round.blockStage(dbg);
    enc = round.enc;
    cb = round.cb;
    // THE DRAFTER'S STOP POINT 2 (after the three blocks).
    if (model.mtpStopPoint() == 2 or std.c.getenv("TF_DSV41_MTP_STUB_BLOCKS") != null) {
        enc.end();
        cb.commit();
        cb.wait();
        std.log.info("mtp stop 2: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
        try completed(cb);
        return .{ 0, 0, 0, 0 };
    }
    const logits = try round.headStage();
    enc = round.enc;
    cb = round.cb;
    // THE DRAFTER'S STOP POINT 3 (after the head projection).
    if (model.mtpStopPoint() == 3 or std.c.getenv("TF_DSV41_MTP_STUB_HEAD") != null) {
        enc.end();
        cb.commit();
        cb.wait();
        std.log.info("mtp stop 3: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
        try completed(cb);
        return .{ 0, 0, 0, 0 };
    }
    const markov_run = try round.markovStage(logits);
    enc = round.enc;
    cb = round.cb;
    const price_dbg = markov_run.price_dbg;
    defer if (price_dbg) |b| b.deinit();
    const groups = markov_run.groups;
    // The drafter's logits row 0 against the trunk's own row-0 logits.
    const top_buf: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_MTP_DEBUG_TOP") != null)
        try m.device.buffer(2 * 129280 * 4 + 5120 * 4, mtl.ResourceOptions.shared)
    else
        null;
    defer if (top_buf) |b| b.deinit();
    if (top_buf) |b| {
        ops.encodeDump(enc, &p.ops, logits, .{ .buffer = b, .offset = 0 }, 129280, false);
        ops.encodeDump(enc, &p.ops, ws.view(.logits), .{ .buffer = b, .offset = 129280 * 4 }, 129280, false);
        ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 2 * 129280 * 4 }, 5120, false);
        enc.barrier();
    }
    // The host's own encode time, the wait, and the GPU's window.
    const t_encoded: f64 = if (host_prof) mtl.clock.seconds() else 0;
    enc.end();
    cb.commit();
    cb.wait();
    const t_done: f64 = if (host_prof) mtl.clock.seconds() else 0;
    try completed(cb);
    if (price_dbg) |b| {
        const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
        const g2: usize = @as(usize, groups) * 2;
        var ps: usize = 0;
        while (ps < n) : (ps += 1) {
            var v1: f32 = -std.math.inf(f32);
            var id1: u32 = 0;
            var v2: f32 = -std.math.inf(f32);
            var id2: u32 = 0;
            for (0..groups) |g| {
                const v = f[ps * g2 + g * 2];
                const id: u32 = @bitCast(f[ps * g2 + g * 2 + 1]);
                if (v > v1 or (v == v1 and id < id1)) {
                    v2 = v1;
                    id2 = id1;
                    v1 = v;
                    id1 = id;
                } else if (v > v2 or (v == v2 and id < id2)) {
                    v2 = v;
                    id2 = id;
                }
            }
            std.log.info("treeprice propose round {d} step {d} top2 {d} {d}", .{ model.price_round, ps, id1, id2 });
        }
    }
    // The early buffer completed in queue order; check it and count its GPU time.
    var gpu_extra_ms: f64 = 0;
    if (pipe_prev_set) {
        try completed(pipe_prev);
        gpu_extra_ms = pipe_prev.gpuSeconds() * 1e3;
    }
    // The drafter's own forward's GPU time and wall.
    if (host_prof) {
        const encode_ms = (t_encoded - t_start) * 1e3;
        const wait_ms = (t_done - t_encoded) * 1e3;
        if (cond_prof.on or cond_prof.blk_on) {
            cond_prof.total_ms += cb.gpuSeconds() * 1e3;
            std.log.info("mtp forward: rows {d} gpu {d:.3} ms (split sum, {d} parts) encode {d:.2} ms wait {d:.2} ms", .{ n, cond_prof.total_ms, cond_prof.parts + 1, encode_ms, wait_ms });
        } else {
            std.log.info("mtp forward: rows {d} gpu {d:.2} ms encode {d:.2} ms wait {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 + gpu_extra_ms, encode_ms, wait_ms });
        }
    }
    if (!host_prof and cond_prof.blk_on) {
        cond_prof.total_ms += cb.gpuSeconds() * 1e3;
        std.log.info("mtp forward: rows {d} gpu {d:.3} ms (split sum, {d} parts)", .{ n, cond_prof.total_ms, cond_prof.parts + 1 });
    }
    if (top_buf) |b| {
        const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
        var dout: [5]u32 = undefined;
        var dval: [5]f32 = undefined;
        var tout: [5]u32 = undefined;
        var tval: [5]f32 = undefined;
        model.logitsTopK(f, 129280, &dout, &dval);
        model.logitsTopK(f + 129280, 129280, &tout, &tval);
        std.log.info("mtp top: drafter argmax={d} vals={d:.4} {d:.4} {d:.4} ids={d} {d} {d} {d} {d}", .{ dout[0], dval[0], dval[1], dval[2], dout[0], dout[1], dout[2], dout[3], dout[4] });
        std.log.info("mtp top: trunk   argmax={d} vals={d:.4} {d:.4} {d:.4} ids={d} {d} {d} {d} {d}", .{ tout[0], tval[0], tval[1], tval[2], tout[0], tout[1], tout[2], tout[3], tout[4] });
        std.log.info("mtp top: agree={} drafter@trunk_argmax={d:.4} trunk@drafter_argmax={d:.4}", .{ dout[0] == tout[0], f[@intCast(tout[0])], f[129280 + @as(usize, @intCast(dout[0]))] });
        // The drafter's head input against the trunk's own head input (the verify's row 0).
        if (m.mtp_trunk_hidden) |tb| {
            const t: [*]const f32 = @ptrCast(@alignCast(tb.contents()));
            const dh = f + 2 * 129280;
            var dot: f64 = 0;
            var dn: f64 = 0;
            var tn: f64 = 0;
            var c: usize = 0;
            while (c < 5120) : (c += 1) {
                dot += @as(f64, dh[c]) * @as(f64, t[c]);
                dn += @as(f64, dh[c]) * @as(f64, dh[c]);
                tn += @as(f64, t[c]) * @as(f64, t[c]);
            }
            std.log.info("mtp hidden: drafter rms={d:.5} trunk rms={d:.5} cosine={d:.5}", .{ @sqrt(dn / 5120.0), @sqrt(tn / 5120.0), dot / @sqrt(dn * tn) });
        }
    }
    if (std.c.getenv("TF_DSV41_MTP_DEBUG_LOGITS") != null) {
        {
            const ev = m.weights.embed;
            std.log.info("mtp embed buffer {d} bytes offset {d}", .{ ev.buffer.length(), ev.offset });
            const tp: [*]const u16 = @ptrCast(@alignCast(ev.buffer.contents() + ev.offset));
            var kk: usize = 0;
            while (kk < 3) : (kk += 1) {
                const base = (128799 + kk) * 5120;
                std.log.info("mtp embed row {d}: {d:.5} {d:.5} {d:.5}", .{ 128799 + kk, asF32(tp[base]), asF32(tp[base + 1]), asF32(tp[base + 2]) });
            }
        }
        if (m.drafter_ids) |idsbuf| {
            const ip: [*]const u32 = @ptrCast(@alignCast(idsbuf.contents()));
            std.log.info("mtp ids: {d} {d} {d} {d} {d} (buffer {d} bytes)", .{ ip[0], ip[1], ip[2], ip[3], ip[4], idsbuf.length() });
        }
        if (dbg) |b| {
            const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
            inline for (.{ "embed", "attn_in", "attn_out", "moe_in", "moe_out" }, 0..) |name, seg| {
                var rr: usize = 0;
                while (rr < 5) : (rr += 1) {
                    var bad: usize = 0;
                    var cc: usize = 0;
                    var acc: f64 = 0;
                    while (cc < 5120) : (cc += 1) {
                        if (!std.math.isFinite(f[(seg * 5 + rr) * 5120 + cc])) bad += 1;
                        acc += @as(f64, f[(seg * 5 + rr) * 5120 + cc]) * @as(f64, f[(seg * 5 + rr) * 5120 + cc]);
                    }
                    // The row's RMS: the attention's contribution.
                    std.log.info("mtp {s} row {d}: nonfinite={d} rms={d:.5} first={d:.5} {d:.5}", .{ name, rr, bad, @sqrt(acc / 5120.0), f[(seg * 5 + rr) * 5120], f[(seg * 5 + rr) * 5120 + 1] });
                }
            }
            // The drafter's key rows (the pack's input), its query's two stages, the decoded key table the scan reads.
            {
                var rrk: usize = 0;
                while (rrk < 5) : (rrk += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 512) : (cc += 1) {
                        const v = f[model.mtp_dbg_raw + rrk * 512 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp key row {d}: nonfinite={d} maxabs={d:.5}", .{ rrk, bad, mx });
                }
                var rrq: usize = 0;
                while (rrq < 5) : (rrq += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 1280) : (cc += 1) {
                        const v = f[model.mtp_dbg_qr + rrq * 1280 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp qr row {d}: nonfinite={d} maxabs={d:.5}", .{ rrq, bad, mx });
                }
                var rrp: usize = 0;
                while (rrp < 5) : (rrp += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 32768) : (cc += 1) {
                        const v = f[model.mtp_dbg_qpre + rrp * 32768 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp qpre row {d}: nonfinite={d} maxabs={d:.5}", .{ rrp, bad, mx });
                }
                // The drafter's query's per-head RMS -- the instrument's read of the scale the softmax saw.
                {
                    const r = model.headRms(f + model.mtp_dbg_qpre, 64, 512);
                    std.log.info("mtp qpre head rms: min={d:.5} max={d:.5} mean={d:.5}", .{ r.min, r.max, r.mean });
                }
                var rrd: usize = 0;
                while (rrd < 133) : (rrd += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 512) : (cc += 1) {
                        const v = f[model.mtp_dbg_dense + rrd * 512 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    if (bad != 0 or rrd >= 126) std.log.info("mtp dense row {d}: nonfinite={d} maxabs={d:.5}", .{ rrd, bad, mx });
                }
                // THE RING'S DECODED KEYS: the RMS of a ring row and the cosine between two ADJACENT positions.
                {
                    const ring0 = model.mtp_dbg_dense;
                    const ring1 = model.mtp_dbg_dense + 127 * 512;
                    const ring2 = model.mtp_dbg_dense + 126 * 512;
                    var a0: f64 = 0;
                    var a1: f64 = 0;
                    var a2: f64 = 0;
                    var d01: f64 = 0;
                    var c01: f64 = 0;
                    var c02: f64 = 0;
                    var cc: usize = 0;
                    while (cc < 512) : (cc += 1) {
                        const v0 = f[ring0 + cc];
                        const v1 = f[ring1 + cc];
                        const v2 = f[ring2 + cc];
                        a0 += @as(f64, v0) * @as(f64, v0);
                        a1 += @as(f64, v1) * @as(f64, v1);
                        a2 += @as(f64, v2) * @as(f64, v2);
                        d01 += (@as(f64, v1) - @as(f64, v0)) * (@as(f64, v1) - @as(f64, v0));
                        c01 += @as(f64, v0) * @as(f64, v1);
                        c02 += @as(f64, v0) * @as(f64, v2);
                    }
                    std.log.info("mtp ring keys: rms(slot0)={d:.5} rms(slot127)={d:.5} rms(slot126)={d:.5} cos(0,127)={d:.5} cos(0,126)={d:.5} rmsdiff(0,127)={d:.5}", .{ @sqrt(a0 / 512.0), @sqrt(a1 / 512.0), @sqrt(a2 / 512.0), c01 / @sqrt(a0 * a1), c02 / @sqrt(a0 * a2), @sqrt(d01 / 512.0) });
                }
                var rrz: usize = 0;
                while (rrz < 5) : (rrz += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 32768) : (cc += 1) {
                        const v = f[model.mtp_dbg_qpost + rrz * 32768 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp qpost row {d}: nonfinite={d} maxabs={d:.5}", .{ rrz, bad, mx });
                }
                var rrc: usize = 0;
                while (rrc < 5) : (rrc += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 5120) : (cc += 1) {
                        const v = f[model.mtp_dbg_coll + rrc * 5120 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp collapsed row {d}: nonfinite={d} maxabs={d:.5}", .{ rrc, bad, mx });
                }
                var rre: usize = 0;
                while (rre < 5) : (rre += 1) {
                    var bad: usize = 0;
                    var cc: usize = 0;
                    while (cc < 4) : (cc += 1) {
                        if (!std.math.isFinite(f[model.mtp_dbg_pre + rre * 4 + cc])) bad += 1;
                    }
                    std.log.info("mtp pre row {d}: nonfinite={d} vals={d:.5} {d:.5} {d:.5} {d:.5}", .{ rre, bad, f[model.mtp_dbg_pre + rre * 4], f[model.mtp_dbg_pre + rre * 4 + 1], f[model.mtp_dbg_pre + rre * 4 + 2], f[model.mtp_dbg_pre + rre * 4 + 3] });
                }
                var rrn: usize = 0;
                while (rrn < 5) : (rrn += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 5120) : (cc += 1) {
                        const v = f[(1 * 5 + rrn) * 5120 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp norm row {d}: nonfinite={d} maxabs={d:.5}", .{ rrn, bad, mx });
                }
                var rri: usize = 0;
                while (rri < 5) : (rri += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 5120) : (cc += 1) {
                        const v = f[model.mtp_dbg_pin + rri * 5120 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp pin row {d}: nonfinite={d} maxabs={d:.5}", .{ rri, bad, mx });
                }
                var rro: usize = 0;
                while (rro < 5) : (rro += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 5120) : (cc += 1) {
                        const v = f[model.mtp_dbg_prot + rro * 5120 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp prot row {d}: nonfinite={d} maxabs={d:.5}", .{ rro, bad, mx });
                }
                var rrs: usize = 0;
                while (rrs < 5) : (rrs += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 512) : (cc += 1) {
                        const v = f[model.mtp_dbg_psum + rrs * 512 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp psum row {d}: nonfinite={d} maxabs={d:.5}", .{ rrs, bad, mx });
                }
                var rrt: usize = 0;
                while (rrt < 5) : (rrt += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 5120) : (cc += 1) {
                        const v = f[model.mtp_dbg_rot + rrt * 5120 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp rot row {d}: nonfinite={d} maxabs={d:.5}", .{ rrt, bad, mx });
                }
                var rrg: usize = 0;
                while (rrg < 5) : (rrg += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 8192) : (cc += 1) {
                        const v = f[model.mtp_dbg_grouped + rrg * 8192 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp grouped row {d}: nonfinite={d} maxabs={d:.5}", .{ rrg, bad, mx });
                }
                var rrw: usize = 0;
                while (rrw < 5) : (rrw += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 1024) : (cc += 1) {
                        const v = f[model.mtp_dbg_woasums + rrw * 1024 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp woasums row {d}: nonfinite={d} maxabs={d:.5}", .{ rrw, bad, mx });
                }
                var rru: usize = 0;
                while (rru < 5) : (rru += 1) {
                    var bad: usize = 0;
                    var mx: f32 = 0;
                    var cc: usize = 0;
                    while (cc < 512) : (cc += 1) {
                        const v = f[model.mtp_dbg_sums + rru * 512 + cc];
                        if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                    }
                    std.log.info("mtp sums row {d}: nonfinite={d} maxabs={d:.5}", .{ rru, bad, mx });
                }
            }
        }
        inline for (.{ .{ "branch", work.Field.branch }, .{ "normalized", work.Field.normalized }, .{ "collapsed", work.Field.collapsed } }) |spec| {
            const v = ws.view(spec[1]);
            const b: [*]const u16 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
            var rrz: usize = 0;
            while (rrz < 5) : (rrz += 1) {
                const w = b[rrz * 5120 ..];
                const a0 = asF32(w[0]);
                const a1 = asF32(w[1]);
                const a2 = asF32(w[2]);
                var bad: usize = 0;
                var cc: usize = 0;
                while (cc < 5120) : (cc += 1) {
                    if (!std.math.isFinite(asF32(w[cc]))) bad += 1;
                }
                std.log.info("mtp {s} row {d}: nonfinite={d} first={d:.5} {d:.5} {d:.5}", .{ spec[0], rrz, bad, a0, a1, a2 });
            }
        }
        const f: [*]const f32 = @ptrCast(@alignCast(m.mtp_logits.?.contents()));
        var row: usize = 0;
        while (row < 5) : (row += 1) {
            var bad: usize = 0;
            var best: f32 = -std.math.inf(f32);
            var best_col: usize = 0;
            var col: usize = 0;
            while (col < 129280) : (col += 1) {
                const v = f[row * 129280 + col];
                if (!std.math.isFinite(v)) bad += 1 else if (v > best) {
                    best = v;
                    best_col = col;
                }
            }
            std.log.info("mtp logits row {d}: nonfinite={d} max={d:.5} at {d}", .{ row, bad, best, best_col });
        }
    }
    const idbytes = m.markov_ids.?.contents();
    const rows_out: u32 = @min(count, n);
    var r: u32 = 0;
    while (r < rows_out) : (r += 1) out[r] = std.mem.readInt(u32, idbytes[(r + 1) * 4 ..][0..4], .little);
    return out;
}
