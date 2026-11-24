//! Living-memory bank calibration (whiteners, shared prefix) and teachChunk compilation of one chunk into a unit.
const std = @import("std");
const whitener = @import("whitener.zig");
const gate = @import("gate.zig");
const types = @import("types.zig");
const Bank = @import("bank.zig").Bank;
const RawState = types.RawState;
const KvLayerData = types.KvLayerData;
const RecLayerData = types.RecLayerData;
const TeachChunkInput = types.TeachChunkInput;
const Unit = types.Unit;
const compressRecurrentLayers = @import("compress.zig").compressRecurrentLayers;

/// Calibrates and compiles the shared prefix once; if loaded calibrated, keeps units and rebuilds `cal_prompts`.
pub fn ensureCalibrated(
    self: *Bank,
    cal_ids: []const []const u32,
    prefix_ids: []const u32,
    suffix_ids: []const u32,
) !void {
    const arch = self.backend.arch;
    if (self.calibrated and self.w_lex != null and self.w_sem != null and self.prefix_state != null) {
        if (self.cal_prompts.items.len == 0 and cal_ids.len > 0) {
            try populateCalPrompts(self, cal_ids);
        }
        return;
    }
    self.clearMemory();

    // Extract dual-layer hidden states on all calibration texts
    var lex_all: std.ArrayList(f32) = .empty;
    defer lex_all.deinit(self.gpa);
    var sem_all: std.ArrayList(f32) = .empty;
    defer sem_all.deinit(self.gpa);
    var lengths: std.ArrayList(usize) = .empty;
    defer lengths.deinit(self.gpa);

    for (cal_ids) |ids| {
        if (ids.len == 0) continue;
        const n = ids.len;
        const hl = try self.gpa.alloc(f32, n * arch.hidden);
        defer self.gpa.free(hl);
        const hs = try self.gpa.alloc(f32, n * arch.hidden);
        defer self.gpa.free(hs);
        try self.backend.extractHidden(self.backend.ctx, ids, arch.layer_lex, arch.layer_sem, hl, hs);
        try lex_all.appendSlice(self.gpa, hl);
        try sem_all.appendSlice(self.gpa, hs);
        try lengths.append(self.gpa, n);
    }
    const total_toks = lex_all.items.len / arch.hidden;
    if (total_toks == 0) return error.EmptyCalibration;

    self.w_lex = try whitener.LayerWhitener.fit(self.gpa, lex_all.items, total_toks, arch.hidden, 16, 560);
    self.w_sem = try whitener.LayerWhitener.fit(self.gpa, sem_all.items, total_toks, arch.hidden, 32, 480);

    var off: usize = 0;
    for (lengths.items) |n| {
        const hl = lex_all.items[off * arch.hidden ..][0 .. n * arch.hidden];
        const hs = sem_all.items[off * arch.hidden ..][0 .. n * arch.hidden];
        const tl = try self.w_lex.?.transform(self.gpa, hl, n);
        const ts = try self.w_sem.?.transform(self.gpa, hs, n);
        try self.cal_prompts.append(self.gpa, .{ .lex = tl, .sem = ts });
        off += n;
    }

    // Compile shared prefix state (KV + recurrent state)
    self.prefix_ids = try self.gpa.dupe(u32, prefix_ids);
    self.suffix_ids = try self.gpa.dupe(u32, suffix_ids);
    var pref = try RawState.init(self.gpa, arch, @intCast(prefix_ids.len));
    errdefer pref.deinit(self.gpa);
    try self.backend.compileState(self.backend.ctx, null, prefix_ids, &pref);
    self.prefix_state = pref;
    self.calibrated = true;
}

fn populateCalPrompts(self: *Bank, cal_ids: []const []const u32) !void {
    const arch = self.backend.arch;
    for (cal_ids) |ids| {
        if (ids.len == 0) continue;
        const n = ids.len;
        const hl = try self.gpa.alloc(f32, n * arch.hidden);
        defer self.gpa.free(hl);
        const hs = try self.gpa.alloc(f32, n * arch.hidden);
        defer self.gpa.free(hs);
        try self.backend.extractHidden(self.backend.ctx, ids, arch.layer_lex, arch.layer_sem, hl, hs);
        const tl = try self.w_lex.?.transform(self.gpa, hl, n);
        const ts = try self.w_sem.?.transform(self.gpa, hs, n);
        try self.cal_prompts.append(self.gpa, .{ .lex = tl, .sem = ts });
    }
}

/// Compiles one chunk (fact, doc section, digest or policy) into KV, recurrent deltas, gate prototypes and bridges.
pub fn teachChunk(self: *Bank, inp: TeachChunkInput) !u32 {
    if (!self.calibrated or self.prefix_state == null or self.w_lex == null or self.w_sem == null) {
        return error.BankNotCalibrated;
    }
    if (inp.body_ids.len == 0) return error.EmptyChunk;
    if (!inp.is_standing and !inp.is_digest and inp.gate_ids.len == 0) return error.EmptyChunk;
    const arch = self.backend.arch;
    const pref = &self.prefix_state.?;
    const b_len: usize = inp.body_ids.len;

    // 1. Compile KV + recurrent state by running body_ids on top of shared prefix
    var body_raw = try RawState.init(self.gpa, arch, @intCast(b_len));
    defer body_raw.deinit(self.gpa);
    try self.backend.compileState(self.backend.ctx, pref, inp.body_ids, &body_raw);

    // 2. K and V kept exactly as computed (BF16, K at rows prefix_len + t): compose shifts K, never rounds twice
    const kv = try self.gpa.alloc(KvLayerData, arch.num_attn);
    var kv_made: usize = 0;
    errdefer {
        for (kv[0..kv_made]) |*k| k.deinit(self.gpa);
        self.gpa.free(kv);
    }
    for (0..arch.num_attn) |a| {
        const k = try self.gpa.dupe(u16, body_raw.kv_k[a]);
        errdefer self.gpa.free(k);
        kv[a] = .{ .k = k, .v = try self.gpa.dupe(u16, body_raw.kv_v[a]) };
        kv_made += 1;
    }

    // 3. Recurrent delta rank: 0 for pure attention, svd_rank_short for small facts/policies, else svd_rank.
    const target_rank: usize = if (arch.num_rec == 0) 0 else if (inp.is_small_fact or inp.is_standing) arch.svd_rank_short else arch.svd_rank;
    const rank: usize = if (target_rank == 0) 0 else @min(b_len + 2, @min(target_rank, @min(arch.rec_rows, arch.rec_cols)));
    const rec: []RecLayerData = if (rank == 0)
        try self.gpa.alloc(RecLayerData, 0)
    else
        try compressRecurrentLayers(self.gpa, arch, pref, &body_raw, rank);
    errdefer {
        for (rec) |*r| r.deinit(self.gpa);
        self.gpa.free(rec);
    }

    // 4. Build dual-layer gate prototypes (standing & digest units skip Hop-1 gating -> 0 prototype bytes!)
    var ug: gate.UnitGate = undefined;
    if (inp.is_standing or inp.is_digest) {
        ug = .{
            .dim = arch.hidden,
            .lex_protos = try self.gpa.alloc(f32, 0),
            .lex_weights = try self.gpa.alloc(f32, 0),
            .n_lex = 0,
            .n_body_lex = 0,
            .sem_protos = try self.gpa.alloc(f32, 0),
            .sem_weights = try self.gpa.alloc(f32, 0),
            .n_sem = 0,
            .neg_lex_max = 0.0,
            .neg_sem_max = 0.0,
            .is_small_fact = false,
            .is_standing = inp.is_standing,
            .is_digest = inp.is_digest,
        };
    } else {
        var lex_vecs: std.ArrayList(f32) = .empty;
        errdefer lex_vecs.deinit(self.gpa);
        var lex_wts: std.ArrayList(f32) = .empty;
        errdefer lex_wts.deinit(self.gpa);
        var sem_vecs: std.ArrayList(f32) = .empty;
        errdefer sem_vecs.deinit(self.gpa);
        var sem_wts: std.ArrayList(f32) = .empty;
        errdefer sem_wts.deinit(self.gpa);

        var n_body_lex: usize = 0;
        {
            const n = inp.gate_ids.len;
            const hl = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hl);
            const hs = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hs);
            try self.backend.extractHidden(self.backend.ctx, inp.gate_ids, arch.layer_lex, arch.layer_sem, hl, hs);
            var tl = try self.w_lex.?.transform(self.gpa, hl, n);
            defer tl.deinit(self.gpa);
            var ts = try self.w_sem.?.transform(self.gpa, hs, n);
            defer ts.deinit(self.gpa);

            const max_lex = @min(@as(usize, 180), @max(@as(usize, 24), n / 2));
            const max_sem = @min(@as(usize, 96), @max(@as(usize, 16), n / 3));
            var lp = try whitener.selectPrototypes(self.gpa, &tl, arch.hidden, self.w_lex.?.norm_floor, max_lex, 0.35, 0.93);
            defer lp.deinit(self.gpa);
            var sp = try whitener.selectPrototypes(self.gpa, &ts, arch.hidden, self.w_sem.?.norm_floor, max_sem, 0.30, 0.87);
            defer sp.deinit(self.gpa);

            try lex_vecs.appendSlice(self.gpa, lp.vecs);
            try lex_wts.appendSlice(self.gpa, lp.weights);
            try sem_vecs.appendSlice(self.gpa, sp.vecs);
            try sem_wts.appendSlice(self.gpa, sp.weights);
            n_body_lex = lp.count;

            // Small-fact pooled semantic centroid, used to match paraphrases without cue words.
            if (inp.is_small_fact and sp.count >= 2) {
                const mean_s = try self.gpa.alloc(f32, arch.hidden);
                defer self.gpa.free(mean_s);
                @memset(mean_s, 0.0);
                var wsum: f32 = 0.0;
                for (0..sp.count) |pi| {
                    const w = @max(sp.weights[pi], 0.15);
                    const v = sp.vecs[pi * arch.hidden ..][0..arch.hidden];
                    for (mean_s, v) |*dst, x| dst.* += w * x;
                    wsum += w;
                }
                var sq: f32 = 0.0;
                for (mean_s) |x| sq += x * x;
                if (sq > 1e-12) {
                    const inv = 1.0 / @sqrt(sq);
                    for (mean_s) |*x| x.* *= inv;
                    try sem_vecs.appendSlice(self.gpa, mean_s);
                    try sem_wts.append(self.gpa, @min(0.95, wsum / @as(f32, @floatFromInt(sp.count)) + 0.08));
                }
            }
        }

        if (inp.title_ids.len > 0) {
            const n = inp.title_ids.len;
            const hl = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hl);
            const hs = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hs);
            try self.backend.extractHidden(self.backend.ctx, inp.title_ids, arch.layer_lex, arch.layer_sem, hl, hs);
            var tl = try self.w_lex.?.transform(self.gpa, hl, n);
            defer tl.deinit(self.gpa);
            var lp = try whitener.selectPrototypes(self.gpa, &tl, arch.hidden, self.w_lex.?.norm_floor, 12, 0.40, 0.92);
            defer lp.deinit(self.gpa);
            try lex_vecs.appendSlice(self.gpa, lp.vecs);
            for (lp.weights) |w| try lex_wts.append(self.gpa, w * 0.72);
        }

        if (inp.cue_ids.len > 0) {
            const n = inp.cue_ids.len;
            const hl = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hl);
            const hs = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hs);
            try self.backend.extractHidden(self.backend.ctx, inp.cue_ids, arch.layer_lex, arch.layer_sem, hl, hs);
            var tl = try self.w_lex.?.transform(self.gpa, hl, n);
            defer tl.deinit(self.gpa);
            var ts = try self.w_sem.?.transform(self.gpa, hs, n);
            defer ts.deinit(self.gpa);
            var lp = try whitener.selectPrototypes(self.gpa, &tl, arch.hidden, self.w_lex.?.norm_floor, 18, 0.35, 0.92);
            defer lp.deinit(self.gpa);
            var sp = try whitener.selectPrototypes(self.gpa, &ts, arch.hidden, self.w_sem.?.norm_floor, 20, 0.25, 0.87);
            defer sp.deinit(self.gpa);
            try lex_vecs.appendSlice(self.gpa, lp.vecs);
            for (lp.weights) |w| try lex_wts.append(self.gpa, w * 0.90);
            try sem_vecs.appendSlice(self.gpa, sp.vecs);
            for (sp.weights) |w| try sem_wts.append(self.gpa, w * 0.94);
        }

        const n_lex = lex_wts.items.len;
        const n_sem = sem_wts.items.len;
        // each slice freed by this block until the gate holds all four, then by the gate's cleanup alone
        ug = gate: {
            const lex_p = try lex_vecs.toOwnedSlice(self.gpa);
            errdefer self.gpa.free(lex_p);
            const lex_w = try lex_wts.toOwnedSlice(self.gpa);
            errdefer self.gpa.free(lex_w);
            const sem_p = try sem_vecs.toOwnedSlice(self.gpa);
            errdefer self.gpa.free(sem_p);
            break :gate .{
                .dim = arch.hidden,
                .lex_protos = lex_p,
                .lex_weights = lex_w,
                .n_lex = n_lex,
                .n_body_lex = n_body_lex,
                .sem_protos = sem_p,
                .sem_weights = try sem_wts.toOwnedSlice(self.gpa),
                .n_sem = n_sem,
                .neg_lex_max = 0.0,
                .neg_sem_max = 0.0,
                .is_small_fact = inp.is_small_fact,
                .is_standing = false,
                .is_digest = inp.is_digest,
            };
        };
        errdefer ug.deinit(self.gpa);

        // Calibrate negative semantic ceiling against the background calibration prompts
        var neg_sem_max: f32 = 0.0;
        for (self.cal_prompts.items) |*cp| {
            const sc = try gate.scoreUnitDual(
                self.gpa,
                &ug,
                &cp.lex,
                &cp.sem,
                self.w_lex.?.norm_floor,
                self.w_sem.?.norm_floor,
            );
            if (sc.sem_score > neg_sem_max) neg_sem_max = sc.sem_score;
        }
        ug.neg_sem_max = neg_sem_max;
    }
    errdefer ug.deinit(self.gpa);

    const uid = self.next_unit_id;
    self.next_unit_id += 1;
    ug.epoch = self.next_epoch;
    self.next_epoch += 1;
    ug.superseded_by = 0;
    ug.reinforcements = 1;
    try ug.computeSummaries(self.gpa);
    if (ug.is_standing) self.freeSkillRec();
    if (self.trace) {
        std.debug.print(
            "lw6 teach: uid={d} pid={d} ep={d} tok={d} kv=q{d} rk={d} n_lex={d}(body={d}) n_sem={d} small={} standing={} digest={} neg_sem={d:.3}\n",
            .{ uid, inp.parent_id, ug.epoch, b_len, @as(u8, 16), rank, ug.n_lex, ug.n_body_lex, ug.n_sem, ug.is_small_fact, ug.is_standing, ug.is_digest, ug.neg_sem_max },
        );
    }
    const unit: Unit = .{ .unit_id = uid, .parent_id = inp.parent_id, .n_tokens = @intCast(b_len), .gate = ug, .kv = kv, .rec = rec };
    if (inp.staged) { // unseen until its lesson is published: no index change either
        try self.staged.append(self.gpa, unit);
        return uid;
    }
    try self.units.append(self.gpa, unit);
    // the unit's parts stay this function's until the index takes it: a failed rebuild gives it back, freed once
    self.rebuildBridges() catch |e| {
        _ = self.units.pop();
        self.rebuildBridges() catch {}; // the index of the units before this one, as it was
        return e;
    };
    self.version += 1;
    return uid;
}
