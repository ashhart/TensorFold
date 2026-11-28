//! Adaptive Hybrid DFlash2 <-> Native MTP per-stream chooser with zero-loss 1-layer MTP KV catch-up.
const std = @import("std");
const mtl = @import("metal");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const pw = @import("prompt.zig");
const df2 = @import("dflash2.zig");
const tr = @import("tree.zig");
const R = gpu.max_rows;

const CompactTapsArgs = extern struct { kept: u32, k_loc: u32, pad0: u32 = 0, pad1: u32 = 0, path: [8]u32 = @splat(0) };

/// Read exact 7-node GPU best-first speculative tree grown inside `df2_merge_walk` (`seq[1..22]`).
pub fn readDf2Tree(seq_u: []const u32, seq_f: []const f32, depth: u32, use_tree: bool) tr.Plan {
    var plan: tr.Plan = .{};
    const cap = @min(depth, 7);
    if (!use_tree or cap < 2) return plan;
    var has_branch = false;
    var branch_strong = false;
    for (0..cap) |k| {
        const tok = seq_u[1 + k];
        const rp: f64 = std.math.clamp(@as(f64, seq_f[8 + k]), 0.0, 0.99);
        if (tok == 0xFFFFFFFF or (k > 0 and rp < 0.03)) break;
        const par: i32 = @bitCast(seq_u[16 + k]);
        if (par < -1 or par >= @as(i32, @intCast(k))) break;
        plan.tokens[k] = tok;
        plan.parents[k] = par;
        plan.probs[k] = rp;
        if (par != @as(i32, @intCast(k)) - 1) {
            has_branch = true;
            if (rp >= 0.16) branch_strong = true;
        }
        plan.n += 1;
    }
    plan.use_tree = has_branch and branch_strong and plan.n >= 2;
    return plan;
}

/// Single-CB fused verify + GPU greedy accept + DFlash2 absorb + parallel 7-token block draft!
pub fn windowFused(d: *df2.Drafter, e: anytype, slot: u32, tokens: []const u32, pred_depth: u32, out: []u32) !void {
    if (slot >= e.slots.len) return error.WindowOutOfStep;
    const rows: u32 = @intCast(tokens.len);
    if (rows == 0 or rows > df2.block_size or pred_depth > df2.draft_tokens or out.len < rows) return error.WindowOutOfStep;
    const s = &e.slots[slot];
    if (s.open > 0) {
        s.len += s.open;
        s.kept = s.open;
        s.open = 0;
    }
    const base_pos = s.len;
    if (!s.used or d.slots[slot].end != base_pos or base_pos + rows + df2.block_size + 1 > e.context) return error.WindowOutOfStep;
    const top = base_pos + rows - 1;
    s.win_row = 0;
    s.fused_start = base_pos;
    s.open = 0;
    s.kept = 0;
    s.held_n = 0;
    s.held_probs_valid = false;
    s.fused_valid = false;
    s.fused_depth = 0;
    for (tokens, 0..) |t, j| {
        e.pass[0].ids.slice(u32, R)[j] = t;
        e.setRow(0, j, slot, base_pos + @as(u32, @intCast(j)));
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var rname_buf: [16]u8 = undefined;
    const rname = std.fmt.bufPrint(&rname_buf, "wdf{d}_{d}", .{ rows, pred_depth }) catch "wdf";
    e.queue.roundBegin(rname);
    defer e.queue.roundEnd();
    var cb = e.queue.commandBuffer();
    var enc = cb.compute(e.win_dispatch);
    const s0 = e.scratch(0);
    try e.m.backboneTap(&s0, enc, rows, top, .{ .buf = e.pass[0].ids }, @as(?*df2.Drafter, d));
    const sl = &d.slots[slot];
    if (e.hasMtp() and !sl.use_mtp and sl.df2_rounds < 24 and (sl.df2_rounds > 0 or base_pos == s.prompt.items.len) and sl.hist_n + rows <= 128) {
        kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.q.idx }, d.c.hidden, d.c.hidden, d.c.hidden, .{ .buf = d.mtp_hist, .off = @as(usize, sl.hist_n) * d.c.hidden * 4 }, rows);
    }
    try e.m.pick(enc, .{ .buf = s0.x }, rows, .{ .buf = e.pass[0].picks });
    enc.setPipeline(d.pipes.accept);
    enc.setBuffer(e.pass[0].ids, 0, 0);
    enc.setBuffer(e.pass[0].picks, 0, 1);
    enc.setValue(df2.AcceptArgs{ .rows = rows, .base_pos = base_pos, .mask_tok = d.c.mask }, 2);
    enc.setBuffer(d.meta, 0, 3);
    enc.setBuffer(d.ids, 0, 4);
    enc.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
    enc.barrier();
    d.absorb(e.pair, e.queue, &cb, &enc, e.win_dispatch, slot, base_pos, 0, rows, null, true);
    if (pred_depth > 0) d.draftBlock(&e.m, enc, slot, 0, base_pos, true, base_pos, rows, e.use_tree);
    enc.end();
    try e.finish(cb);
    const kept = d.meta.slice(u32, 4)[0];
    const picks = e.pass[0].picks.slice(u32, rows);
    @memcpy(out[0..rows], picks);
    @memcpy(s.fused_picks[0..rows], picks);
    s.fused_path = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    d.slots[slot].end = base_pos + kept;
    s.len = base_pos + kept;
    s.kept = kept;
    s.open = 0;
    s.held_n = pred_depth;
    s.fused_depth = pred_depth;
    s.fused_valid = true;
    s.held_probs_valid = pred_depth > 0;
    if (pred_depth > 0) d.readSeq(base_pos + kept, kept, pred_depth, e.gpu_ms, &s.held, &s.held_probs);
    s.tree = readDf2Tree(d.seq.slice(u32, 32), d.seq.slice(f32, 32), pred_depth, e.use_tree);
    try afterFused(d, e, slot, base_pos, tokens[0], picks[0..kept], pred_depth);
}

/// Single-CB fused 0-ULP tree verify + GPU tree accept + indirect KV/taps compact + DFlash2 absorb + block draft!
pub fn windowTreeFused(d: *df2.Drafter, e: anytype, slot: u32, tokens: []const u32, parents: []const i32, pred_depth: u32, out: []u32) !void {
    if (slot >= e.slots.len) return error.WindowOutOfStep;
    const rows: u32 = @intCast(tokens.len);
    if (rows == 0 or rows > df2.block_size or parents.len != rows or pred_depth > df2.draft_tokens or out.len < rows) return error.WindowOutOfStep;
    const s = &e.slots[slot];
    if (s.open > 0) {
        s.len += s.open;
        s.kept = s.open;
        s.open = 0;
    }
    const base_pos = s.len;
    if (!s.used or d.slots[slot].end != base_pos or base_pos + rows + df2.block_size + 1 > e.context) return error.WindowOutOfStep;
    var ds: [8]u32 = undefined;
    const targs = try tr.packAnc(base_pos, parents, &ds);
    var top_tree: u32 = base_pos;
    s.win_row = 0;
    s.win_tok0 = tokens[0];
    s.fused_start = base_pos;
    s.open = 0;
    s.kept = 0;
    s.held_n = 0;
    s.held_probs_valid = false;
    s.fused_valid = false;
    s.fused_depth = 0;
    s.tree = .{};
    for (tokens, 0..) |t, j| {
        e.pass[0].ids.slice(u32, R)[j] = t;
        const pos = base_pos + ds[j];
        e.setRow(0, j, slot, pos);
        top_tree = @max(top_tree, pos);
    }
    var ta_acc: df2.TreeAcceptArgs = .{ .rows = rows, .base_pos = base_pos, .mask_tok = d.c.mask, .k_loc = d.k_loc };
    for (parents, 0..) |par, j| ta_acc.parents[j] = par;
    const sl = &d.slots[slot];
    const rec_hist = e.hasMtp() and !sl.use_mtp and sl.df2_rounds < 24 and (sl.df2_rounds > 0 or base_pos == s.prompt.items.len) and sl.hist_n + rows <= 128;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var rname_buf: [16]u8 = undefined;
    const rname = std.fmt.bufPrint(&rname_buf, "wdft{d}_{d}", .{ rows, pred_depth }) catch "wdft";
    e.queue.roundBegin(rname);
    defer e.queue.roundEnd();
    var cb = e.queue.commandBuffer();
    var enc = cb.compute(e.win_dispatch);
    const s0 = e.scratch(0);
    try e.m.backboneTapTree(&s0, enc, rows, top_tree, .{ .buf = e.pass[0].ids }, @as(?*df2.Drafter, d), targs);
    if (rec_hist) kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.q.idx }, d.c.hidden, d.c.hidden, d.c.hidden, .{ .buf = e.xs }, rows);
    try e.m.pick(enc, .{ .buf = s0.x }, rows, .{ .buf = e.pass[0].picks });
    enc.setPipeline(d.pipes.tree_accept);
    enc.setBuffer(e.pass[0].ids, 0, 0);
    enc.setBuffer(e.pass[0].picks, 0, 1);
    enc.setValue(ta_acc, 2);
    enc.setBuffer(d.meta, 0, 3);
    enc.setBuffer(d.ids, 0, 4);
    enc.setBuffer(e.compact_disp, 0, 5);
    enc.setBuffer(e.pass[2].idx, 0, 6);
    enc.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
    enc.barrier();
    tr.compactKvIndirect(&e.m, enc, slot, base_pos, e.context, d.meta, e.compact_disp, 6);
    enc.setPipeline(d.pipes.compact_taps_indirect);
    enc.setBuffer(d.taps_loc, 0, 0);
    enc.setBuffer(d.meta, 0, 1);
    enc.setValue(d.k_loc, 2);
    enc.dispatchIndirect(e.compact_disp, 16, .{ .width = 256 });
    if (rec_hist) kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[2].idx }, d.c.hidden, d.c.hidden, d.c.hidden, .{ .buf = d.mtp_hist, .off = @as(usize, sl.hist_n) * d.c.hidden * 4 }, rows);
    enc.barrier();
    d.absorb(e.pair, e.queue, &cb, &enc, e.win_dispatch, slot, base_pos, 0, rows, null, true);
    if (pred_depth > 0) d.draftBlock(&e.m, enc, slot, 0, base_pos, true, base_pos, rows, e.use_tree);
    enc.end();
    try e.finish(cb);
    const meta = d.meta.slice(u32, 16);
    const kept = meta[0];
    const picks = e.pass[0].picks.slice(u32, rows);
    @memcpy(out[0..rows], picks);
    for (0..kept) |k| {
        const pr = meta[6 + k];
        s.fused_path[k] = pr;
        s.fused_picks[k] = picks[pr];
    }
    d.slots[slot].end = base_pos + kept;
    s.len = base_pos + kept;
    s.kept = kept;
    s.open = 0;
    s.held_n = pred_depth;
    s.fused_depth = pred_depth;
    s.fused_valid = true;
    s.held_probs_valid = pred_depth > 0;
    if (pred_depth > 0) d.readSeq(base_pos + kept, kept, pred_depth, e.gpu_ms, &s.held, &s.held_probs);
    s.tree = readDf2Tree(d.seq.slice(u32, 32), d.seq.slice(f32, 32), pred_depth, e.use_tree);
    try afterFused(d, e, slot, base_pos, tokens[0], s.fused_picks[0..kept], pred_depth);
}

/// DFlash2 draft after prompt, window, or branching windowTree (compacts KV + taps in-CB, updates hybrid state).
pub fn draftSlot(d: *df2.Drafter, e: anytype, q: anytype) !void {
    if (q.slot >= e.slots.len) return error.DraftOutOfStep;
    const s = &e.slots[q.slot];
    const sl = &d.slots[q.slot];
    const n: u32 = @intCast(q.follow.len);
    if (!s.used or q.depth > df2.draft_tokens or n == 0) return error.DraftOutOfStep;
    const base_before = s.len;
    const has_path = !q.prompt and s.open > 0 and q.path.len == n;
    const compact = has_path and tr.needsCompact(q.path);
    if (q.prompt and (n != 1 or s.open > 0 or s.len != s.prompt.items.len)) return error.DraftOutOfStep;
    if (!q.prompt and s.open > 0) {
        if (has_path) {
            if (!tr.validPath(q.path, s.open)) return error.DraftOutOfStep;
        } else if (n > s.open) return error.DraftOutOfStep;
        s.len += n;
        s.kept = n;
        s.open = 0;
    } else if (!q.prompt and n != s.kept) return error.DraftOutOfStep;
    if (s.len + df2.draft_tokens + 1 > e.context) return error.ContextFull;
    const to_abs = s.len -| sl.end;
    const record_hist = !q.prompt and to_abs == n and e.hasMtp() and !sl.use_mtp and sl.df2_rounds < 24 and
        (sl.df2_rounds > 0 or base_before == s.prompt.items.len) and sl.hist_n + n <= 128;
    if (compact or record_hist) {
        const hid = e.m.c.hidden;
        if (record_hist) for (0..n) |j| {
            e.pass[2].idx.slice(u32, R)[j] = if (has_path) q.path[j] else @intCast(j);
        };
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var cb = e.queue.commandBuffer();
        var enc = cb.compute(e.win_dispatch);
        if (compact) {
            tr.compactKv(&e.m, enc, q.slot, base_before, e.context, q.path);
            var ca: CompactTapsArgs = .{ .kept = n, .k_loc = d.k_loc };
            for (q.path, 0..) |r_idx, k| ca.path[k] = r_idx;
            enc.setPipeline(d.pipes.compact_taps);
            enc.setBuffer(d.taps_loc, 0, 0);
            enc.setValue(ca, 1);
            enc.dispatchThreads(.{ .width = d.k_loc }, .{ .width = 256 });
        }
        if (record_hist) {
            kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = d.mtp_hist, .off = @as(usize, sl.hist_n) * hid * 4 }, n);
        }
        enc.barrier();
        if (to_abs > 0) d.absorb(e.pair, e.queue, &cb, &enc, e.win_dispatch, q.slot, sl.end, s.win_row, to_abs, null, false);
        if (q.depth > 0) d.draftBlock(&e.m, enc, q.slot, q.follow[n - 1], s.len, false, s.len, 0, e.use_tree);
        enc.end();
        try e.finish(cb);
        if (q.depth > 0) d.readSeq(s.len, to_abs, q.depth, e.gpu_ms, &s.held, &s.held_probs);
    } else {
        e.gpu_ms = try d.absorbAndDraft(&e.m, e.queue, e.win_dispatch, q.slot, sl.end, s.win_row, to_abs, q.follow[n - 1], s.len, q.depth, e.use_tree, &s.held, &s.held_probs);
    }
    s.held_n = q.depth;
    s.held_probs_valid = q.depth > 0;
    s.tree = readDf2Tree(d.seq.slice(u32, 32), d.seq.slice(f32, 32), q.depth, e.use_tree);
    if (record_hist) {
        try afterFused(d, e, q.slot, base_before, s.win_tok0, q.follow, q.depth);
        s.fused_valid = false;
    }
}

/// Record a fused DFlash2 decode round and switch to Native MTP on prose/low-TPR streams in lockstep across all ranks.
pub fn afterFused(d: *df2.Drafter, e: anytype, slot: u32, base_pos: u32, tok0: u32, picks: []const u32, pred_depth: u32) !void {
    const s = &e.slots[slot];
    const sl = &d.slots[slot];
    if (!e.hasMtp() or sl.use_mtp or sl.df2_rounds >= 24) return;
    if (sl.df2_rounds == 0 and base_pos != s.prompt.items.len) return;
    const kept: u32 = @intCast(picks.len);
    if (sl.df2_rounds == 0) {
        sl.hist_pos0 = base_pos - 1;
        sl.hist_toks[0] = tok0;
        sl.hist_n = 0;
    }
    if (sl.hist_n + kept + 1 > sl.hist_toks.len) {
        sl.df2_rounds = 24;
        return;
    }
    for (picks, 0..) |pk, j| sl.hist_toks[sl.hist_n + 1 + j] = pk;
    sl.hist_n += kept;
    sl.last6[sl.df2_rounds % 6] = kept;
    sl.df2_rounds += 1;
    sl.df2_toks += kept;
    var sum6: u32 = 0;
    for (sl.last6) |v| sum6 += v;
    const long_ctx = s.prompt.items.len > d.c.window;
    const should_switch = if (long_ctx)
        (sl.df2_rounds == 1 and kept <= 1) or
            (sl.df2_rounds == 2 and sl.df2_toks <= 5) or
            (sl.df2_rounds == 3 and sl.df2_toks <= 9) or
            (sl.df2_rounds == 4 and sl.df2_toks <= 12) or
            (sl.df2_rounds >= 6 and sum6 < 20)
    else
        (sl.df2_rounds == 1 and kept <= 1) or
            (sl.df2_rounds == 2 and sl.df2_toks <= 5) or
            (sl.df2_rounds == 3 and sl.df2_toks <= 9) or
            (sl.df2_rounds == 4 and sl.df2_toks <= 13) or
            (sl.df2_rounds == 5 and sl.df2_toks <= 16) or
            (sl.df2_rounds >= 6 and sum6 < 19);
    if (should_switch and sl.hist_n > 0 and s.len + pred_depth + 1 <= e.context) {
        if (d.tp_rank == 0) std.log.info("glm_dsa hybrid: slot {d} DFlash2 {d} tok/{d} rounds (last6={d}) -> switching to Native MTP", .{
            slot, sl.df2_toks, sl.df2_rounds, sum6,
        });
        try switchToMtp(d, e, slot, pred_depth);
    } else if (sl.df2_rounds == 24 and d.tp_rank == 0) {
        std.log.info("glm_dsa hybrid: slot {d} locked DFlash2 ({d} tok/24 rounds, tpr={d:.2})", .{
            slot, sl.df2_toks, @as(f64, @floatFromInt(sl.df2_toks)) / 24.0,
        });
    }
}

/// Catch up Native MTP's 1-layer KV cache over K accepted tokens (~0.8 ms) and draft d=1..pred_depth.
fn switchToMtp(d: *df2.Drafter, e: anytype, slot: u32, pred_depth: u32) !void {
    const s = &e.slots[slot];
    const sl = &d.slots[slot];
    const K = sl.hist_n;
    const hid = e.m.c.hidden;
    const use_prompt = K + 1 > 8;
    const s0 = e.scratch(0);
    const s1 = e.scratch(1);
    e.pass[1].idx.slice(u32, R)[0] = 2 * R + slot;
    if (use_prompt) {
        for (0..K) |j| {
            e.q.ids_mtp.slice(u32, gpu.prompt_rows)[j] = sl.hist_toks[j];
            e.q.rows_mtp.slice(kn.Row, gpu.prompt_rows)[j] = .{ .slot = slot, .pos = sl.hist_pos0 + @as(u32, @intCast(j)), .list = @intCast(j * e.m.c.i_topk) };
        }
        e.pass[1].ids.slice(u32, R)[0] = sl.hist_toks[K];
        e.setRow(1, 0, slot, sl.hist_pos0 + K);
        e.pass[3].idx.slice(u32, R)[0] = K - 1;
        e.pass[2].idx.slice(u32, R)[0] = 0;
    } else {
        for (0..K + 1) |j| {
            e.pass[1].ids.slice(u32, R)[j] = sl.hist_toks[j];
            e.setRow(1, j, slot, sl.hist_pos0 + @as(u32, @intCast(j)));
        }
        e.pass[2].idx.slice(u32, R)[0] = K;
    }
    const main_n = @min(pred_depth, 6);
    tr.prepRows(e, slot, s.len, main_n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.queue.roundBegin("df2_to_mtp");
    defer e.queue.roundEnd();
    const cb = e.queue.commandBuffer();
    const enc = cb.compute(e.win_dispatch);
    if (use_prompt) {
        kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[1].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, 1);
        if (K > 1) kn.gather(&e.m.K.k, enc, .{ .buf = d.mtp_hist }, .{ .buf = e.q.idx }, hid, hid, hid, .{ .buf = e.m.head.prev, .off = @as(usize, hid) * 4 }, K - 1);
        enc.barrier();
        try pw.draft(&e.m, &s0, &e.q, enc, K, sl.hist_pos0);
        enc.barrier();
        kn.gather(&e.m.K.k, enc, .{ .buf = d.mtp_hist }, .{ .buf = e.pass[3].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, 1);
        enc.barrier();
        try e.m.draft(&s1, enc, 1, sl.hist_pos0 + K, .{ .buf = e.pass[1].ids }, .{ .buf = e.m.head.prev });
    } else {
        kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[1].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, 1);
        kn.gather(&e.m.K.k, enc, .{ .buf = d.mtp_hist }, .{ .buf = e.q.idx }, hid, hid, hid, .{ .buf = e.m.head.prev, .off = @as(usize, hid) * 4 }, K);
        enc.barrier();
        try e.m.draft(&s1, enc, K + 1, sl.hist_pos0 + K, .{ .buf = e.pass[1].ids }, .{ .buf = e.m.head.prev });
    }
    var all_prob = false;
    var has_b1 = false;
    if (main_n > 0) {
        kn.gather(&e.m.K.k, enc, .{ .buf = s1.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.draft_x }, 1);
        enc.barrier();
        all_prob = try e.m.pickDraft(enc, .{ .buf = e.draft_x }, 1, .{ .buf = e.pass[1].picks });
        const res = try tr.runBatchedMtp(e, enc, main_n, s.len, false, all_prob, all_prob and e.use_tree and main_n >= 2 and s.len < 2048);
        all_prob = all_prob and res.all_prob;
        has_b1 = res.has_b1;
    }
    enc.end();
    try e.finish(cb);
    sl.use_mtp = true;
    s.fused_depth = main_n;
    s.fused_valid = true;
    s.fused_b1 = has_b1;
    tr.finishMtpPicks(e, s, main_n, all_prob, has_b1);
}
