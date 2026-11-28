//! 100% bit-exact speculative tree helpers: ancestor-nibble packing, KV compaction, and best-first MTP tree builder.
const std = @import("std");
const mtl = @import("metal");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const R = gpu.max_rows;

/// Inline constant-buffer tree descriptor appended to MlaArgs and IndexArgs (48 bytes, 16-byte aligned).
pub const TreeArgs = extern struct {
    base: u32 = 0xFFFFFFFF,
    p0: u32 = 0,
    p1: u32 = 0,
    p2: u32 = 0,
    anc: [8]u32 = @splat(0),
};

/// Pack each row's ancestor chain [0, ..., r] into 4-bit nibbles (depth <= 7) for 0-ULP tree attention.
pub fn packAnc(base_pos: u32, parents: []const i32, depths_out: []u32) !TreeArgs {
    if (parents.len == 0 or parents.len > 8 or depths_out.len < parents.len) return error.WindowOutOfStep;
    var t: TreeArgs = .{ .base = base_pos };
    for (parents, 0..) |par, r| {
        if (r == 0) {
            if (par != -1) return error.WindowOutOfStep;
            depths_out[0] = 0;
            t.anc[0] = 0;
            continue;
        }
        if (par < 0 or par >= @as(i32, @intCast(r))) return error.WindowOutOfStep;
        const d = depths_out[@intCast(par)] + 1;
        if (d > 7) return error.WindowOutOfStep;
        depths_out[r] = d;
        var chain: [8]u32 = undefined;
        var cur: i32 = @intCast(r);
        var idx: usize = d;
        while (true) {
            chain[idx] = @intCast(cur);
            if (idx == 0) break;
            idx -= 1;
            cur = parents[@intCast(cur)];
        }
        var packed_anc: u32 = 0;
        for (0..d + 1) |i| packed_anc |= (chain[i] & 0xF) << @intCast(4 * i);
        t.anc[r] = packed_anc;
    }
    return t;
}

/// True when `path` selects a non-contiguous tree branch needing KV/index cache compaction.
pub fn needsCompact(path: []const u32) bool {
    for (path, 0..) |r, k| if (r != k) return true;
    return false;
}

/// Validate strictly increasing tree accept path starting at row 0 within `max_r`.
pub fn validPath(path: []const u32, max_r: u32) bool {
    if (path.len == 0 or path.len > 8 or path[0] != 0) return false;
    var prev: u32 = 0;
    for (path[1..]) |r| {
        if (r <= prev or r >= max_r) return false;
        prev = r;
    }
    return true;
}

/// Compact accepted tree branch KV + index cache slots across all backbone layers (<0.04 ms total).
pub fn compactKv(m: anytype, enc: mtl.ComputeEncoder, slot: u32, base_pos: u32, slot_keys: u32, path: []const u32) void {
    if (!needsCompact(path) or path.len <= 1) return;
    var a: kn.fast.CompactKvArgs = .{
        .slot = slot,
        .base_pos = base_pos,
        .slot_keys = slot_keys,
        .kept = @intCast(path.len),
    };
    for (path, 0..) |r, k| a.path[k] = r;
    for (m.layers) |*L| {
        a.has_idx = @intFromBool(L.index_cache != null);
        kn.fast.compactKv(&m.K.k.fast_p, enc, .{ .buf = L.cache }, .{ .buf = L.index_cache orelse L.cache }, a);
    }
}

/// Indirect-dispatched KV + index compaction reading `kept`/`path` from `meta` (0 TGs when branch 0 wins!).
pub fn compactKvIndirect(m: anytype, enc: mtl.ComputeEncoder, slot: u32, base_pos: u32, slot_keys: u32, meta: mtl.Buffer, disp: mtl.Buffer, path_off: u32) void {
    var a: kn.fast.CompactKvArgs = .{ .slot = slot, .base_pos = base_pos, .slot_keys = slot_keys, .kept = 0, .pad0 = path_off };
    for (m.layers) |*L| {
        a.has_idx = @intFromBool(L.index_cache != null);
        kn.fast.compactKvIndirect(&m.K.k.fast_p, enc, .{ .buf = L.cache }, .{ .buf = L.index_cache orelse L.cache }, .{ .buf = meta }, a, disp);
    }
}

/// Calibrated d=0 (depth 1) rank-0 probability matching empirical Native MTP Q4 prose acceptance.
pub fn calD0(p: f64) f64 {
    const x = std.math.clamp(p, 0.0, 1.0);
    return if (x <= 1e-6) 0.0 else std.math.clamp(std.math.pow(f64, x, 1.65), 0.0, 0.99);
}

/// Calibrated d>=1 continuation rank-0 probability matching empirical Native MTP Q4 prose acceptance.
pub fn calNext(p: f64, long_c: bool) f64 {
    const x = std.math.clamp(p, 0.0, 1.0);
    if (x <= 1e-6) return 0.0;
    return std.math.clamp(if (long_c) 0.96 * x * x * x else 0.94 * std.math.pow(f64, x, 1.85), 0.0, 0.99);
}

/// Calibrated alternate-rank (rank >= 1) probability matching empirical top-4 oracle hit rates on prose.
pub fn calAlt(p: f64) f64 {
    const x = std.math.clamp(p, 0.0, 1.0);
    return if (x <= 1e-6) 0.0 else std.math.clamp(std.math.pow(f64, x, 0.80), 0.0, 0.99);
}

/// Best-first speculative draft tree (up to 7 draft nodes = R<=8 total rows with root).
pub const Plan = struct {
    n: u32 = 0,
    use_tree: bool = false,
    tokens: [7]u32 = @splat(0),
    parents: [7]i32 = @splat(-1),
    probs: [7]f64 = @splat(0),
};

const Cand = struct {
    tok: u32,
    par_cand: i8, // -1: root (row 0), else index in `cands`
    q: f64,
    picked: bool = false,
    out_idx: i8 = -1,
};

fn addCand(cands: *[18]Cand, len: *usize, tok: u32, par_cand: i8, q: f64) i8 {
    if (tok == 0xFFFFFFFF or q < 0.04 or len.* >= cands.len) return -1;
    // Deduplicate siblings under the same parent.
    for (cands[0..len.*]) |c| if (c.par_cand == par_cand and c.tok == tok) return -1;
    const idx: i8 = @intCast(len.*);
    cands[len.*] = .{ .tok = tok, .par_cand = par_cand, .q = q };
    len.* += 1;
    return idx;
}

/// Build the optimal monotone best-first tree from main chain (d=1..6 top-4) + Branch-1 (t1 -> d=2..3 top-4).
pub fn build(
    main_ids: *const [6][4]u32,
    main_p: *const [6][4]f32,
    main_n: u32,
    b1_ids: *const [2][4]u32,
    b1_p: *const [2][4]f32,
    has_b1: bool,
    max_nodes: u32,
    scale0: f64,
    scale: f64,
    long_c: bool,
) Plan {
    var plan: Plan = .{};
    const cap = @min(max_nodes, 7);
    if (cap == 0 or main_n == 0) return plan;

    var cands: [18]Cand = undefined;
    var nc: usize = 0;

    // Depth 1 (children of root): Top-4 from d=1; enforce q0 >= q1 monotonicity while preserving true q1..q3.
    const q0 = std.math.clamp(calD0(main_p[0][0]) * scale0, 0.0, 0.99);
    const c_t0 = addCand(&cands, &nc, main_ids[0][0], -1, q0);
    var rem: f64 = @max(0.0, 0.99 - q0);
    const q1 = @min(rem, calAlt(main_p[0][1]));
    const c_t1 = addCand(&cands, &nc, main_ids[0][1], -1, q1);
    rem = @max(0.0, rem - q1);
    const q2 = @min(rem, 0.95 * calAlt(main_p[0][2]));
    _ = addCand(&cands, &nc, main_ids[0][2], -1, q2);
    rem = @max(0.0, rem - q2);
    const q3 = @min(rem, 0.90 * calAlt(main_p[0][3]));
    _ = addCand(&cands, &nc, main_ids[0][3], -1, q3);

    // Branch 0 continuations (d=2..6): scaled by online `scale` + zero-cost rank-2 sibling!
    var par_c: i8 = c_t0;
    var par_q: f64 = q0;
    const lim = @min(main_n, 6);
    for (1..lim) |d| {
        if (par_c < 0 or main_p[d][0] <= 1e-6) break;
        const pc0 = std.math.clamp(calNext(main_p[d][0], long_c) * scale, 0.0, 0.99);
        const qc0 = par_q * pc0;
        const next_c = addCand(&cands, &nc, main_ids[d][0], par_c, qc0);
        if (d <= 3 and main_p[d][1] > 0.12) {
            const pc1 = @min(@max(0.0, 0.99 - pc0), calAlt(main_p[d][1]) * scale);
            _ = addCand(&cands, &nc, main_ids[d][1], par_c, par_q * pc1);
        }
        par_c = next_c;
        par_q = qc0;
    }

    // Branch 1 continuations (t1 -> t1_1, t1_1b -> t1_2) from batched r=2 MTP at d=2,3!
    if (has_b1 and c_t1 >= 0 and b1_p[0][0] > 1e-6) {
        const pb0 = std.math.clamp(calNext(b1_p[0][0], long_c) * scale, 0.0, 0.99);
        const qb0 = q1 * pb0;
        const c_b1 = addCand(&cands, &nc, b1_ids[0][0], c_t1, qb0);
        if (b1_p[0][1] > 0.14) {
            const pb0b = @min(@max(0.0, 0.99 - pb0), calAlt(b1_p[0][1]) * scale);
            _ = addCand(&cands, &nc, b1_ids[0][1], c_t1, q1 * pb0b);
        }
        if (c_b1 >= 0 and b1_p[1][0] > 1e-6) {
            const pb1 = std.math.clamp(calNext(b1_p[1][0], long_c) * scale, 0.0, 0.99);
            _ = addCand(&cands, &nc, b1_ids[1][0], c_b1, qb0 * pb1);
        }
    }

    // Best-first greedy selection (parents always have q >= child -> strictly monotone & topological!).
    var has_branch = false;
    var branch_strong = false;
    while (plan.n < cap) {
        var best_i: ?usize = null;
        if (plan.n == 0 and c_t0 >= 0) {
            best_i = @intCast(c_t0);
        } else {
            var best_q: f64 = 0.04;
            for (cands[0..nc], 0..) |c, i| {
                if (c.picked) continue;
                if (c.par_cand >= 0 and cands[@intCast(c.par_cand)].out_idx < 0) continue;
                if (c.q > best_q) {
                    best_q = c.q;
                    best_i = i;
                }
            }
        }
        const bi = best_i orelse break;
        const out_i = plan.n;
        const par_out: i32 = if (cands[bi].par_cand < 0) -1 else @intCast(cands[@intCast(cands[bi].par_cand)].out_idx);
        cands[bi].picked = true;
        cands[bi].out_idx = @intCast(out_i);
        plan.tokens[out_i] = cands[bi].tok;
        plan.parents[out_i] = par_out;
        plan.probs[out_i] = cands[bi].q;
        if (par_out != @as(i32, @intCast(out_i)) - 1) {
            has_branch = true;
            if (cands[bi].q >= 0.095) branch_strong = true;
        }
        plan.n += 1;
    }
    plan.use_tree = has_branch and branch_strong;
    return plan;
}

/// Tree attention masks for batched r=2 MTP at d=2 (tb+0/1) and gated r=1 main chain d=3..6 (tb+2..5).
pub fn b1TreeArgs(d: u32) TreeArgs {
    return switch (d) {
        2 => .{ .base = 0xFFFFFFFE, .p0 = 0, .p1 = 0, .anc = .{ 0x0, 0x1, 0, 0, 0, 0, 0, 0 } },
        3 => .{ .base = 0xFFFFFFFE, .p0 = 2, .p1 = 1, .anc = .{ 0x20, 0, 0, 0, 0, 0, 0, 0 } },
        4 => .{ .base = 0xFFFFFFFE, .p0 = 3, .p1 = 2, .anc = .{ 0x320, 0, 0, 0, 0, 0, 0, 0 } },
        5 => .{ .base = 0xFFFFFFFE, .p0 = 4, .p1 = 3, .anc = .{ 0x4320, 0, 0, 0, 0, 0, 0, 0 } },
        else => .{ .base = 0xFFFFFFFE, .p0 = 5, .p1 = 4, .anc = .{ 0x54320, 0, 0, 0, 0, 0, 0, 0 } },
    };
}

/// Populate `pass[2..main_n].rows` (2 rows for d=2 batched tree MTP; 1 row for d>=3).
pub fn prepRows(e: anytype, slot: u32, base_pos: u32, main_n: u32) void {
    var d: u32 = 2;
    while (d <= main_n) : (d += 1) {
        e.setRow(d, 0, slot, base_pos + d - 2);
        if (d == 2 and main_n >= 3) e.setRow(d, 1, slot, base_pos);
    }
}

/// True when cumulative calibrated probability across `probs` still exceeds the GPU early-stop threshold.
pub fn shouldExtend(probs: []const f32, scale0: f64, scale: f64, long_c: bool) bool {
    var cum: f64 = 1.0;
    for (probs, 0..) |p, j| cum *= std.math.clamp(if (j == 0) calD0(p) * scale0 else calNext(p, long_c) * scale, 0.0, 0.99);
    return cum >= (if (probs.len <= 2) @as(f64, 0.42) else @as(f64, 0.26));
}

/// Batched r=2 tree-masked MTP at d=2 (t0,t1) for main_n>=3; gated r=1 MTP for main_n==2 or long ctx!
pub fn runBatchedMtp(e: anytype, enc: mtl.ComputeEncoder, main_n: u32, pos0: u32, use_draft_rows: bool, use_gate: bool, use_tree: bool) !struct { all_prob: bool, has_b1: bool } {
    if (main_n <= 1) return .{ .all_prob = use_gate, .has_b1 = false };
    const hid = e.m.c.hidden;
    var ok = use_gate;
    if (use_gate and use_tree and main_n >= 3 and pos0 < 2048) {
        const sd2 = if (use_draft_rows) e.scratchDraftAt(0) else e.scratch(2);
        kn.gather(&e.m.K.k, enc, .{ .buf = e.pass[1].picks }, .{ .buf = e.pass[8].idx }, 1, 1, 1, .{ .buf = e.pass[2].ids }, 2);
        kn.gather(&e.m.K.k, enc, .{ .buf = e.draft_x }, .{ .buf = e.pass[7].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, 2);
        e.gateMtp(enc, 200, pos0, false, e.pass[1].picks);
        try e.m.draftGated(&sd2, enc, 2, pos0, .{ .buf = e.pass[2].ids }, .{ .buf = e.m.head.prev }, null, null, e.gate_meta, b1TreeArgs(2));
        ok = (try e.m.pickDraft(enc, .{ .buf = sd2.x }, 2, .{ .buf = e.pass[2].picks })) and ok;
        if (main_n >= 3) {
            e.gateMtp(enc, 11, pos0, false, e.pass[1].picks);
            e.gateMtp(enc, 12, pos0 + 1, true, e.pass[2].picks);
            var d: u32 = 3;
            while (d <= main_n) : (d += 1) {
                const sd = if (use_draft_rows) e.scratchDraftAt(d - 1) else e.scratch(d);
                ok = (try e.runDraftStep(enc, d, 1, pos0 + d - 2, e.q.idx, e.s.x, e.q.idx, &sd, true, d < main_n, b1TreeArgs(d))) and ok;
            }
        }
        return .{ .all_prob = ok, .has_b1 = ok };
    }
    if (use_gate) e.gateMtp(enc, 1, pos0, true, e.pass[1].picks);
    var d: u32 = 2;
    while (d <= main_n) : (d += 1) {
        const sd = if (use_draft_rows) e.scratchDraftAt(d - 2) else e.scratch(d);
        const h_src = if (d == 2) e.draft_x else e.s.x;
        ok = (try e.runDraftStep(enc, d, 1, pos0 + d - 2, e.q.idx, h_src, e.q.idx, &sd, use_gate, d < main_n, .{})) and ok;
    }
    return .{ .all_prob = ok, .has_b1 = false };
}

/// Populate `s.held`, `s.held_probs`, `s.held_top4_*`, and best-first `s.tree` after MTP passes complete.
pub fn finishMtpPicks(e: anytype, s: anytype, main_n: u32, all_prob: bool, has_b1: bool) void {
    const lim = @min(main_n, 7);
    s.held_n = lim;
    s.held_probs_valid = all_prob and lim > 0;
    s.tree = .{};
    for (0..lim) |j| {
        s.held[j] = e.pass[j + 1].picks.slice(u32, 2 * R)[0];
        s.held_probs[j] = if (all_prob) e.pass[j + 1].picks.slice(f32, 2 * R)[1] else 0;
    }
    if (!all_prob or lim == 0) return;
    for (0..4) |k| {
        s.held_top4_ids[k] = e.pass[1].picks.slice(u32, 2 * R)[2 * k];
        s.held_top4_p[k] = e.pass[1].picks.slice(f32, 2 * R)[2 * k + 1];
    }
    var main_ids: [6][4]u32 = @splat(@splat(0xFFFFFFFF));
    var main_p: [6][4]f32 = @splat(@splat(0));
    const mn = @min(lim, 6);
    for (0..mn) |j| {
        const pu = e.pass[j + 1].picks.slice(u32, 2 * R);
        const pf = e.pass[j + 1].picks.slice(f32, 2 * R);
        for (0..4) |k| {
            main_ids[j][k] = pu[2 * k];
            main_p[j][k] = pf[2 * k + 1];
        }
    }
    var b1_ids: [2][4]u32 = @splat(@splat(0xFFFFFFFF));
    var b1_p: [2][4]f32 = @splat(@splat(0));
    const use_b1 = has_b1 and mn >= 2 and main_p[0][1] >= 0.08;
    if (use_b1) {
        const pu0 = e.pass[2].picks.slice(u32, 2 * R);
        const pf0 = e.pass[2].picks.slice(f32, 2 * R);
        for (0..4) |k| {
            b1_ids[0][k] = pu0[8 + 2 * k];
            b1_p[0][k] = pf0[8 + 2 * k + 1];
        }
    }
    s.tree = build(&main_ids, &main_p, mn, &b1_ids, &b1_p, use_b1, 7, s.cal0_scale, s.cal_scale, s.len >= 2048);
}

/// Verify one 0-ULP bit-exact speculative tree window (`tokens.len <= 8`) for `slot`.
pub fn windowTree(e: anytype, slot: u32, tokens: []const u32, parents: []const i32, out: []u32) !void {
    if (slot >= e.slots.len or tokens.len == 0 or tokens.len > 8 or parents.len != tokens.len or out.len < tokens.len)
        return error.WindowOutOfStep;
    const s = &e.slots[slot];
    if (s.open > 0) e.settleSlot(slot, s.open);
    if (!s.used or s.len + tokens.len + 8 > e.context) return error.WindowOutOfStep;
    var ds: [8]u32 = undefined;
    const targs = try packAnc(s.len, parents, &ds);
    const rows: u32 = @intCast(tokens.len);
    var top: u32 = s.len;
    s.win_row = 0;
    s.win_tok0 = tokens[0];
    s.open = rows;
    s.kept = 0;
    s.held_n = 0;
    s.held_probs_valid = false;
    s.fused_valid = false;
    s.fused_depth = 0;
    s.tree = .{};
    for (tokens, 0..) |tok, j| {
        e.pass[0].ids.slice(u32, R)[j] = tok;
        const pos = s.len + ds[j];
        e.setRow(0, j, slot, pos);
        top = @max(top, pos);
    }
    const hid = e.m.c.hidden;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.queue.roundBegin("win_tree");
    defer e.queue.roundEnd();
    const cb = e.queue.commandBuffer();
    const enc = cb.compute(e.win_dispatch);
    const s0 = e.scratch(0);
    const tap_d = if (e.dflash2) |d| (if (!d.slots[slot].use_mtp) d else null) else null;
    try e.m.backboneTapTree(&s0, enc, rows, top, .{ .buf = e.pass[0].ids }, tap_d, targs);
    if (e.hasMtp()) {
        for (e.pass[2].idx.slice(u32, rows), 0..) |*x, r| x.* = @intCast(r);
        kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.xs }, rows);
    }
    try e.m.pick(enc, .{ .buf = s0.x }, rows, .{ .buf = e.pass[0].picks });
    enc.end();
    try e.finish(cb);
    @memcpy(out[0..rows], e.pass[0].picks.slice(u32, rows));
}

/// Single-CB fused 0-ULP tree verify + GPU tree accept + indirect KV compact + DFlash2/MTP draft (0 CB2!).
pub fn windowTreeDraft(e: anytype, slot: u32, tokens: []const u32, parents: []const i32, pred_depth: u32, out: []u32) !void {
    if (e.dflash2) |d| if (!d.slots[slot].use_mtp) return @import("dflash2_hybrid.zig").windowTreeFused(d, e, slot, tokens, parents, pred_depth, out);
    if (!e.hasMtp() or slot >= e.slots.len) return error.WindowOutOfStep;
    const rows: u32 = @intCast(tokens.len);
    if (rows == 0 or rows > 8 or parents.len != rows or pred_depth > 7 or out.len < rows) return error.WindowOutOfStep;
    const s = &e.slots[slot];
    if (s.open > 0) e.settleSlot(slot, s.open);
    const base_pos = s.len;
    if (!s.used or base_pos + rows + pred_depth + 3 > e.context) return error.WindowOutOfStep;
    var ds: [8]u32 = undefined;
    const targs = try packAnc(base_pos, parents, &ds);
    var top_tree: u32 = base_pos;
    const top_lin = base_pos + rows - 1;
    s.win_row = 0;
    s.win_tok0 = tokens[0];
    s.fused_start = base_pos;
    s.open = 0;
    s.kept = 0;
    s.held_n = 0;
    s.held_probs_valid = false;
    s.fused_valid = false;
    s.fused_b1 = false;
    s.fused_depth = 0;
    s.tree = .{};
    for (tokens, 0..) |tok, j| {
        e.pass[0].ids.slice(u32, R)[j] = tok;
        const pos = base_pos + ds[j];
        e.setRow(0, j, slot, pos);
        e.setRow(1, j, slot, base_pos + @as(u32, @intCast(j)));
        top_tree = @max(top_tree, pos);
    }
    const main_n = @min(pred_depth, 6);
    const hid = e.m.c.hidden;
    const has_q4 = e.m.vocab.lm_q4_w != null and e.m.vocab.lm_q4_s != null and (hid % 1024 == 0);
    const batch_r2 = has_q4 and e.use_tree and main_n >= 3 and (top_lin + 1) < 2048;
    var ta_acc: kn.fast.MtpTreeAcceptArgs = .{ .rows = rows, .slot = slot, .base_pos = base_pos, .depth = main_n, .i_topk = e.m.c.i_topk, .pad0 = @intFromBool(batch_r2) };
    for (parents, 0..) |par, j| ta_acc.parents[j] = par;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var rname_buf: [16]u8 = undefined;
    const rname = std.fmt.bufPrint(&rname_buf, "wtd{d}_{d}", .{ rows, main_n }) catch "wtd";
    e.queue.roundBegin(rname);
    defer e.queue.roundEnd();
    const cb = e.queue.commandBuffer();
    const enc = cb.compute(e.win_dispatch);
    const s0 = e.scratch(0);
    try e.m.backboneTapTree(&s0, enc, rows, top_tree, .{ .buf = e.pass[0].ids }, null, targs);
    kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.q.idx }, hid, hid, hid, .{ .buf = e.xs }, rows);
    try e.m.pick(enc, .{ .buf = s0.x }, rows, .{ .buf = e.pass[0].picks });
    kn.fast.mtpTreeAccept(&e.m.K.k.fast_p, enc, .{ .buf = e.pass[0].ids }, .{ .buf = e.pass[0].picks }, ta_acc, .{ .buf = e.accept_meta }, .{ .buf = e.compact_disp }, .{ .buf = e.pass[1].ids }, .{ .buf = e.pass[1].idx }, .{ .buf = e.pass[2].idx }, .{ .buf = e.draft_rows });
    enc.barrier();
    compactKvIndirect(&e.m, enc, slot, base_pos, e.context, e.accept_meta, e.compact_disp, 2);
    kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[1].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, rows);
    enc.barrier();
    const s1 = e.scratch(1);
    try e.m.draftGated(&s1, enc, rows, top_lin, .{ .buf = e.pass[1].ids }, .{ .buf = e.m.head.prev }, null, null, e.accept_meta, .{});
    var all_prob = false;
    var has_b1 = false;
    if (main_n > 0) {
        kn.gather(&e.m.K.k, enc, .{ .buf = s1.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.draft_x }, 1);
        enc.barrier();
        all_prob = try e.m.pickDraft(enc, .{ .buf = e.draft_x }, 1, .{ .buf = e.pass[1].picks });
        const res = try runBatchedMtp(e, enc, main_n, top_lin + 1, true, all_prob, batch_r2);
        all_prob = all_prob and res.all_prob;
        has_b1 = res.has_b1;
    }
    enc.end();
    try e.finish(cb);
    const meta = e.accept_meta.slice(u32, 12);
    const kept = meta[0];
    const picks = e.pass[0].picks.slice(u32, rows);
    @memcpy(out[0..rows], picks);
    for (0..kept) |k| {
        const pr = meta[2 + k];
        s.fused_path[k] = pr;
        s.fused_picks[k] = picks[pr];
    }
    s.len = base_pos + kept;
    s.kept = kept;
    s.open = 0;
    s.fused_depth = main_n;
    s.fused_valid = true;
    s.fused_b1 = has_b1;
    finishMtpPicks(e, s, main_n, all_prob, has_b1);
}

/// Settle a tree window's winning `path` (compacting backbone KV/index caches if non-contiguous).
pub fn keepPath(e: anytype, slot: u32, path: []const u32) !void {
    if (slot >= e.slots.len) return error.KeepOutOfStep;
    const s = &e.slots[slot];
    if (!s.used or s.open == 0 or !validPath(path, s.open)) return error.KeepOutOfStep;
    const base_pos = s.len;
    if (needsCompact(path)) {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(e.win_dispatch);
        compactKv(&e.m, enc, slot, base_pos, e.context, path);
        enc.end();
        try e.finish(cb);
    }
    e.settleSlot(slot, @intCast(path.len));
}

/// Extend a still-valid fused windowDraft chain from `s.fused_depth` (0..6) to `want` with batched r=2 tree MTP.
pub fn extendDraft(e: anytype, slot: u32, want: u32) !void {
    if (!e.hasMtp() or slot >= e.slots.len) return error.DraftOutOfStep;
    const s = &e.slots[slot];
    const main_n = @min(want, 6);
    if (!s.fused_valid or main_n <= s.fused_depth) return error.DraftOutOfStep;
    if (s.len + main_n + 1 > e.context) return error.ContextFull;
    const hid = e.m.c.hidden;
    const from0 = s.fused_depth == 0;
    const start_d: u32 = if (from0) 2 else s.fused_depth + 1;
    const tree_cont = !from0 and s.fused_b1 and s.fused_depth >= 2;
    if (!from0) e.pass[start_d].ids.slice(u32, 1)[0] = s.held[s.fused_depth - 1];
    e.pass[1].idx.slice(u32, 1)[0] = s.kept - 1;
    if (!from0) e.pass[start_d].idx.slice(u32, 1)[0] = if (s.fused_depth == 1) s.kept - 1 else 0;
    if (from0) prepRows(e, slot, s.len, main_n) else {
        var d: u32 = start_d;
        while (d <= main_n) : (d += 1) e.setRow(d, 0, slot, s.len + d - 2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.queue.roundBegin("ext_draft");
    defer e.queue.roundEnd();
    const cb = e.queue.commandBuffer();
    const enc = cb.compute(e.win_dispatch);
    var all_prob = if (from0) true else s.held_probs_valid;
    var has_b1 = tree_cont;
    if (from0) {
        kn.gather(&e.m.K.k, enc, .{ .buf = e.s.x }, .{ .buf = e.pass[1].idx }, hid, hid, hid, .{ .buf = e.draft_x }, 1);
        enc.barrier();
        all_prob = try e.m.pickDraft(enc, .{ .buf = e.draft_x }, 1, .{ .buf = e.pass[1].picks });
        const res = try runBatchedMtp(e, enc, main_n, s.len, false, all_prob, all_prob and e.use_tree and main_n >= 2 and s.len < 2048);
        all_prob = all_prob and res.all_prob;
        has_b1 = res.has_b1;
    } else {
        var d: u32 = start_d;
        while (d <= main_n) : (d += 1) {
            const p = &e.pass[d];
            const prev_idx: ?mtl.Buffer = if (d > start_d) e.q.idx else null;
            const h_idx = if (d == start_d) p.idx else e.q.idx;
            const sd = e.scratch(d);
            all_prob = (try e.runDraftStep(enc, d, 1, s.len + d - 2, prev_idx, e.s.x, h_idx, &sd, false, false, if (tree_cont) b1TreeArgs(d) else .{})) and all_prob;
        }
    }
    enc.end();
    try e.finish(cb);
    s.fused_depth = 0;
    s.fused_valid = false;
    s.fused_b1 = false;
    finishMtpPicks(e, s, main_n, all_prob, has_b1);
}

test "packAnc packs 0-ULP ancestor nibbles for branching trees" {
    const parents = [_]i32{ -1, 0, 0, 1, 2, 4 };
    var ds: [6]u32 = undefined;
    const t = try packAnc(100, &parents, &ds);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 2, 2, 3 }, &ds);
    try std.testing.expectEqual(@as(u32, 0x0), t.anc[0]);
    try std.testing.expectEqual(@as(u32, 0x10), t.anc[1]);
    try std.testing.expectEqual(@as(u32, 0x20), t.anc[2]);
    try std.testing.expectEqual(@as(u32, 0x310), t.anc[3]);
    try std.testing.expectEqual(@as(u32, 0x420), t.anc[4]);
    try std.testing.expectEqual(@as(u32, 0x5420), t.anc[5]);
}

test "build produces monotone topological best-first tree" {
    var mids: [6][4]u32 = @splat(@splat(0xFFFFFFFF));
    var mp: [6][4]f32 = @splat(@splat(0));
    mids[0] = .{ 10, 20, 30, 40 };
    mp[0] = .{ 0.55, 0.30, 0.08, 0.02 };
    mids[1] = .{ 11, 12, 0xFFFFFFFF, 0xFFFFFFFF };
    mp[1] = .{ 0.90, 0.05, 0, 0 };
    var b1ids: [2][4]u32 = @splat(@splat(0xFFFFFFFF));
    var b1p: [2][4]f32 = @splat(@splat(0));
    b1ids[0] = .{ 21, 22, 0xFFFFFFFF, 0xFFFFFFFF };
    b1p[0] = .{ 0.90, 0.05, 0, 0 };
    const p = build(&mids, &mp, 2, &b1ids, &b1p, true, 7, 1.0, 1.0, false);
    try std.testing.expect(p.use_tree);
    try std.testing.expect(p.n >= 4);
    for (1..p.n) |i| try std.testing.expect(p.probs[i - 1] >= p.probs[i]);
    for (0..p.n) |i| try std.testing.expect(p.parents[i] < @as(i32, @intCast(i)));
}
