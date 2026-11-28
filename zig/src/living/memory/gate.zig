//! Dual-layer neural gate: lexical + semantic scoring, ClusterIndex, coverage, bridges, conflicts, standing policies.
const std = @import("std");
const quant = @import("quant.zig");
const whitener = @import("whitener.zig");
const Allocator = std.mem.Allocator;
const cluster_index = @import("cluster_index.zig");
const bridges_mod = @import("bridges.zig");
const normalizeInPlace = cluster_index.normalizeInPlace;
pub const ClusterIndex = cluster_index.ClusterIndex;
pub const BridgeEdge = bridges_mod.BridgeEdge;
pub const computeBridgeEdges = bridges_mod.computeBridgeEdges;

pub const UnitGate = struct {
    dim: usize,
    lex_protos: []f32, // [n_lex, dim]
    lex_weights: []f32, // [n_lex] (specificity, 0.72x on repeated doc title, 0.90x on cues)
    n_lex: usize,
    /// Number of leading `lex_protos` coming from the chunk's unique body (before title/cue protos).
    n_body_lex: usize = 0,
    sem_protos: []f32, // [n_sem, dim]
    sem_weights: []f32, // [n_sem]
    n_sem: usize,
    neg_lex_max: f32 = 0.0,
    neg_sem_max: f32 = 0.0,
    is_small_fact: bool = false,
    is_standing: bool = false,
    is_digest: bool = false,
    /// Temporal belief lifecycle: creation epoch, superseding parent_id (0 = active), reinforcement count.
    epoch: u64 = 0,
    superseded_by: u32 = 0,
    reinforcements: u16 = 1,
    /// L2-normalized pooled unit signatures [dim] for ClusterIndex coarse routing (computed lazily or on teach).
    lex_summary: []f32 = &.{},
    sem_summary: []f32 = &.{},

    pub fn bodyLexCount(self: *const UnitGate) usize {
        return if (self.n_body_lex > 0 and self.n_body_lex <= self.n_lex) self.n_body_lex else self.n_lex;
    }

    pub fn isActive(self: *const UnitGate) bool {
        return self.superseded_by == 0;
    }

    /// Computes L2-normalized specificity-weighted pooled lexical & semantic summary vectors for ClusterIndex.
    pub fn computeSummaries(self: *UnitGate, gpa: Allocator) !void {
        if (self.lex_summary.len > 0) {
            gpa.free(self.lex_summary);
            self.lex_summary = &.{};
        }
        if (self.sem_summary.len > 0) {
            gpa.free(self.sem_summary);
            self.sem_summary = &.{};
        }
        if (self.n_lex > 0) {
            const s = try gpa.alloc(f32, self.dim);
            @memset(s, 0.0);
            for (0..self.n_lex) |p| {
                const w = @max(self.lex_weights[p], 0.1);
                const v = self.lex_protos[p * self.dim ..][0..self.dim];
                for (s, v) |*dst, x| dst.* += w * x;
            }
            normalizeInPlace(s);
            self.lex_summary = s;
        }
        if (self.n_sem > 0) {
            const s = try gpa.alloc(f32, self.dim);
            @memset(s, 0.0);
            for (0..self.n_sem) |p| {
                const w = @max(self.sem_weights[p], 0.1);
                const v = self.sem_protos[p * self.dim ..][0..self.dim];
                for (s, v) |*dst, x| dst.* += w * x;
            }
            normalizeInPlace(s);
            self.sem_summary = s;
        }
    }

    pub fn deinit(self: *UnitGate, gpa: Allocator) void {
        gpa.free(self.lex_protos);
        gpa.free(self.lex_weights);
        gpa.free(self.sem_protos);
        gpa.free(self.sem_weights);
        if (self.lex_summary.len > 0) gpa.free(self.lex_summary);
        if (self.sem_summary.len > 0) gpa.free(self.sem_summary);
        self.* = undefined;
    }
};

pub const DualScore = struct {
    lex_gate: f32,
    lex_rank: f32,
    sem_score: f32,
    n_strong_lex: u32 = 0,
    best_lex: f32 = 0.0,
};

pub const RouteConfig = struct {
    lex_floor: f32 = 0.44,
    sem_margin: f32 = 0.095,
    sem_floor: f32 = 0.31,
    top_k: usize = 4,
    max_per_parent: usize = 3,
    min_ratio: f32 = 0.40,
    /// Minimum IDF-weighted pure-vector bridge strength for Hop-2 associative spreading activation.
    same_parent_bridge_min: f32 = 0.52,
    cross_parent_bridge_min: f32 = 0.85,
};

pub const UnitView = struct {
    index: usize,
    parent_id: u32,
    n_tokens: u32,
    gate: *const UnitGate,
};

/// Dual-layer score of a query against one unit; optional `out_tok_scores` gets per-query-token lexical scores.
pub fn scoreUnitDualDetailed(
    gpa: Allocator,
    unit: *const UnitGate,
    ql: *const whitener.Transformed,
    qs: *const whitener.Transformed,
    lex_norm_floor: f32,
    sem_norm_floor: f32,
    out_tok_scores: ?[]f32,
) !DualScore {
    const dim = unit.dim;
    const q_len = ql.norms.len;
    std.debug.assert(q_len == qs.norms.len and q_len > 0);
    if (out_tok_scores) |buf| {
        std.debug.assert(buf.len == q_len);
        @memset(buf, 0.0);
    }
    if (unit.n_lex == 0 or unit.n_sem == 0) {
        return .{ .lex_gate = 0.0, .lex_rank = 0.0, .sem_score = 0.0 };
    }

    // 1. Lexical layer (two-regime gate + additive multi-token evidence rank)
    var tok_lex: std.ArrayList(f32) = .empty;
    defer tok_lex.deinit(gpa);
    for (0..q_len) |i| {
        if (ql.spec[i] < 0.35 or ql.norms[i] < lex_norm_floor * 0.65) continue;
        const qv = ql.unit[i * dim ..][0..dim];
        var best_cos: f32 = -1.0;
        var best_idx: usize = 0;
        for (0..unit.n_lex) |p| {
            const c = quant.dotF32(qv, unit.lex_protos[p * dim ..][0..dim]);
            if (c > best_cos) {
                best_cos = c;
                best_idx = p;
            }
        }
        const matched_w = unit.lex_weights[best_idx];
        const joint_spec = @sqrt(@max(ql.spec[i] * matched_w, 0.0));
        const val = best_cos * joint_spec;
        if (out_tok_scores) |buf| buf[i] = val;
        try tok_lex.append(gpa, val);
    }

    var lex_gate: f32 = 0.0;
    var lex_rank: f32 = 0.0;
    var n_strong_lex: u32 = 0;
    var best_lex: f32 = 0.0;
    if (tok_lex.items.len > 0) {
        std.mem.sort(f32, tok_lex.items, {}, std.sort.desc(f32));
        const best1 = tok_lex.items[0];
        best_lex = best1;
        const top2: f32 = if (tok_lex.items.len >= 2)
            0.5 * (tok_lex.items[0] + tok_lex.items[1])
        else
            best1 * 0.75;
        // Two-regime trigger: >=2 corroborating domain tokens OR 1 ultra-specific rare entity
        const single_cut: f32 = if (tok_lex.items.len <= 2) 0.65 else 0.74;
        lex_gate = @max(top2, best1 * (0.45 / single_cut));
        var strong_sum: f32 = 0.0;
        var n_strong: usize = 0;
        for (tok_lex.items) |v| {
            if (v >= 0.42) {
                strong_sum += v;
                n_strong += 1;
            }
        }
        n_strong_lex = @intCast(n_strong);
        lex_rank = if (n_strong > 0) strong_sum else top2;
    }

    // 2. Deep semantic layer (catches paraphrases like "shade" <-> "colour", "pet" <-> "dog")
    var tok_sem: std.ArrayList(f32) = .empty;
    defer tok_sem.deinit(gpa);
    for (0..q_len) |i| {
        if (qs.spec[i] < 0.25 or qs.norms[i] < sem_norm_floor * 0.60) continue;
        try tok_sem.append(gpa, scoreSemToken(unit, qs.unit[i * dim ..][0..dim], qs.spec[i]));
    }
    if (tok_sem.items.len == 0) {
        // Fallback to top-2 tokens by spec * norm
        const idx = try gpa.alloc(usize, q_len);
        defer gpa.free(idx);
        for (idx, 0..) |*o, i| o.* = i;
        const Ctx = struct { s: []const f32, n: []const f32 };
        std.mem.sort(usize, idx, Ctx{ .s = qs.spec, .n = qs.norms }, struct {
            fn less(c: Ctx, a: usize, b: usize) bool {
                return c.s[a] * c.n[a] > c.s[b] * c.n[b];
            }
        }.less);
        const take = @min(2, q_len);
        for (idx[0..take]) |i| {
            try tok_sem.append(gpa, scoreSemToken(unit, qs.unit[i * dim ..][0..dim], qs.spec[i]));
        }
    }
    std.mem.sort(f32, tok_sem.items, {}, std.sort.desc(f32));
    const s0 = tok_sem.items[0];
    const s_top2: f32 = if (tok_sem.items.len >= 2) 0.5 * (tok_sem.items[0] + tok_sem.items[1]) else s0;
    const sem_score = 0.60 * s_top2 + 0.40 * s0;

    return .{
        .lex_gate = lex_gate,
        .lex_rank = lex_rank,
        .sem_score = sem_score,
        .n_strong_lex = n_strong_lex,
        .best_lex = best_lex,
    };
}

pub fn scoreUnitDual(
    gpa: Allocator,
    unit: *const UnitGate,
    ql: *const whitener.Transformed,
    qs: *const whitener.Transformed,
    lex_norm_floor: f32,
    sem_norm_floor: f32,
) !DualScore {
    return scoreUnitDualDetailed(gpa, unit, ql, qs, lex_norm_floor, sem_norm_floor, null);
}

fn scoreSemToken(unit: *const UnitGate, qv: []const f32, q_spec: f32) f32 {
    const dim = unit.dim;
    var best: f32 = -1.0;
    for (0..unit.n_sem) |p| {
        const c = quant.dotF32(qv, unit.sem_protos[p * dim ..][0..dim]);
        const pw = std.math.clamp(0.5 + 0.5 * unit.sem_weights[p], 0.5, 1.0);
        const v = c * pw;
        if (v > best) best = v;
    }
    const qw = std.math.clamp(0.4 + 0.6 * q_spec, 0.4, 1.0);
    return best * qw;
}

pub const ConflictMatch = struct {
    parent_id: u32,
    lex_gate: f32,
    sem_score: f32,
    combined: f32,
};

/// Entity conflict detector: scores a new small fact against each active small fact's body prototypes only.
pub fn findConflictParents(
    gpa: Allocator,
    units: []const UnitView,
    ql: *const whitener.Transformed,
    qs: *const whitener.Transformed,
    lex_norm_floor: f32,
    sem_norm_floor: f32,
    max_out: usize,
) ![]ConflictMatch {
    var matches: std.ArrayList(ConflictMatch) = .empty;
    errdefer matches.deinit(gpa);

    for (units) |u| {
        if (!u.gate.isActive() or !u.gate.is_small_fact or u.gate.is_standing or u.gate.is_digest) continue;
        // Restrict lexical matching strictly to the unit's own body prototypes (excluding title/cue anchors)
        var body_only = u.gate.*;
        body_only.n_lex = u.gate.bodyLexCount();
        const sc = try scoreUnitDual(gpa, &body_only, ql, qs, lex_norm_floor, sem_norm_floor);
        // Require entity + semantic overlap above the unit's negative floor so shared generic verbs do not collide
        const sem_cut = @max(@as(f32, 0.44), u.gate.neg_sem_max + 0.14);
        const overlap = (sc.lex_gate >= 0.56 and sc.sem_score >= sem_cut) or
            (sc.n_strong_lex >= 2 and sc.lex_gate >= 0.50 and sc.sem_score >= @max(@as(f32, 0.40), u.gate.neg_sem_max + 0.12));
        if (!overlap) continue;
        const comb = sc.lex_gate + 0.8 * sc.sem_score;
        var existing: ?usize = null;
        for (matches.items, 0..) |m, mi| {
            if (m.parent_id == u.parent_id) {
                existing = mi;
                break;
            }
        }
        if (existing) |mi| {
            if (comb > matches.items[mi].combined) {
                matches.items[mi] = .{ .parent_id = u.parent_id, .lex_gate = sc.lex_gate, .sem_score = sc.sem_score, .combined = comb };
            }
        } else {
            try matches.append(gpa, .{ .parent_id = u.parent_id, .lex_gate = sc.lex_gate, .sem_score = sc.sem_score, .combined = comb });
        }
    }
    std.mem.sort(ConflictMatch, matches.items, {}, struct {
        fn less(_: void, a: ConflictMatch, b: ConflictMatch) bool {
            return a.combined > b.combined;
        }
    }.less);
    if (matches.items.len > max_out) matches.shrinkRetainingCapacity(max_out);
    return matches.toOwnedSlice(gpa);
}

/// Routes a query against all units (backward-compatible wrapper around `routeUnitsIndexed`).
pub fn routeUnits(
    gpa: Allocator,
    units: []const UnitView,
    bridges: []const BridgeEdge,
    ql: *const whitener.Transformed,
    qs: *const whitener.Transformed,
    lex_norm_floor: f32,
    sem_norm_floor: f32,
    cfg: RouteConfig,
) ![]usize {
    return routeUnitsIndexed(gpa, units, bridges, null, ql, qs, lex_norm_floor, sem_norm_floor, cfg);
}

/// Routes a query: standing units, optional ClusterIndex shortlist, Hop-1 gate + coverage, Hop-2 digest/bridge.
pub fn routeUnitsIndexed(
    gpa: Allocator,
    units: []const UnitView,
    bridges: []const BridgeEdge,
    cluster_idx: ?*const ClusterIndex,
    ql: *const whitener.Transformed,
    qs: *const whitener.Transformed,
    lex_norm_floor: f32,
    sem_norm_floor: f32,
    cfg: RouteConfig,
) ![]usize {
    if (units.len == 0) return try gpa.alloc(usize, 0);
    const q_len = ql.norms.len;

    const Candidate = struct {
        rank_key: f32,
        excess: f32,
        sem_score: f32,
        view_pos: usize,
        index: usize,
        parent_id: u32,
        n_tokens: u32,
        is_digest: bool,
        tok_scores: []f32,
        picked: bool = false,
    };

    var standing: std.ArrayList(usize) = .empty;
    defer standing.deinit(gpa);
    for (units) |u| {
        if (u.gate.isActive() and u.gate.is_standing) {
            try standing.append(gpa, u.index);
        }
    }

    var shortlist_owned: ?[]usize = null;
    defer if (shortlist_owned) |sl| gpa.free(sl);

    if (cluster_idx) |ci| {
        if (ci.isReady() and units.len >= 64) {
            const min_sl = @max(@as(usize, 48), ci.members.len / 6);
            shortlist_owned = try ci.shortlist(gpa, ql, qs, min_sl);
        }
    }

    var passed: std.ArrayList(Candidate) = .empty;
    defer {
        for (passed.items) |c| gpa.free(c.tok_scores);
        passed.deinit(gpa);
    }

    const n_eval = if (shortlist_owned) |sl| sl.len else units.len;
    for (0..n_eval) |ei| {
        const pos = if (shortlist_owned) |sl| sl[ei] else ei;
        const u = units[pos];
        if (!u.gate.isActive() or u.gate.is_standing or u.gate.is_digest) continue;
        const tbuf = try gpa.alloc(f32, q_len);
        errdefer gpa.free(tbuf);
        const sc = try scoreUnitDualDetailed(gpa, u.gate, ql, qs, lex_norm_floor, sem_norm_floor, tbuf);
        const t_sem = @max(cfg.sem_floor, u.gate.neg_sem_max + cfg.sem_margin);
        // Single-token lexical hits on a document chunk must clear semantic corroboration (subword collisions)
        const single_sem_floor = @max(@as(f32, 0.26), u.gate.neg_sem_max + 0.03);
        const lex_ok = if (!u.gate.is_small_fact and sc.n_strong_lex <= 1)
            (sc.lex_gate >= cfg.lex_floor) and (sc.sem_score >= single_sem_floor)
        else
            (sc.lex_gate >= cfg.lex_floor) and (sc.sem_score >= 0.22 or sc.lex_gate >= 0.68);
        const sem_ok = (sc.sem_score >= t_sem) and (sc.lex_gate >= 0.28 or u.gate.is_small_fact);
        if (lex_ok or sem_ok) {
            const ex_lex: f32 = if (lex_ok) sc.lex_gate - cfg.lex_floor else -1.0;
            const ex_sem: f32 = if (sem_ok) sc.sem_score - t_sem else -1.0;
            try passed.append(gpa, .{
                .rank_key = sc.lex_rank + 1.2 * sc.sem_score,
                .excess = @max(ex_lex, ex_sem),
                .sem_score = sc.sem_score,
                .view_pos = pos,
                .index = u.index,
                .parent_id = u.parent_id,
                .n_tokens = u.n_tokens,
                .is_digest = u.gate.is_digest,
                .tok_scores = tbuf,
            });
        } else {
            gpa.free(tbuf);
        }
    }

    if (passed.items.len == 0) {
        return standing.toOwnedSlice(gpa);
    }

    // Find top_rank for min_ratio filtering
    var top_rank: f32 = -1.0;
    for (passed.items) |c| {
        if (c.rank_key > top_rank) top_rank = c.rank_key;
    }

    const SelItem = struct {
        view_pos: usize,
        index: usize,
        parent_id: u32,
        n_tokens: u32,
    };
    var selected: std.ArrayList(SelItem) = .empty;
    defer selected.deinit(gpa);

    // Greedy submodular coverage: first pick by `rank_key`, then reward uncovered query tokens
    const covered = try gpa.alloc(f32, q_len);
    defer gpa.free(covered);
    @memset(covered, 0.0);

    var step: usize = 0;
    while (step < cfg.top_k) : (step += 1) {
        var best_i: ?usize = null;
        var best_gain: f32 = -1.0;
        var best_excess: f32 = -1.0;

        for (passed.items, 0..) |cand, ci| {
            if (cand.picked) continue;
            if (cand.rank_key < top_rank * cfg.min_ratio) continue;
            var parent_count: usize = 0;
            for (selected.items) |s| if (s.parent_id == cand.parent_id) {
                parent_count += 1;
            };
            if (parent_count >= cfg.max_per_parent) continue;

            const gain: f32 = if (step == 0)
                cand.rank_key
            else blk: {
                var marg_lex: f32 = 0.0;
                for (0..q_len) |t| {
                    const m = cand.tok_scores[t];
                    if (m >= 0.42) {
                        marg_lex += @max(0.0, m - 0.65 * covered[t]);
                    }
                }
                // Blend marginal new coverage with small baseline rank tie-breaker
                break :blk marg_lex + 0.15 * cand.rank_key + 0.55 * cand.sem_score;
            };

            if (best_i == null or gain > best_gain or (gain == best_gain and cand.excess > best_excess)) {
                best_i = ci;
                best_gain = gain;
                best_excess = cand.excess;
            }
        }

        const chosen_ci = best_i orelse break;
        passed.items[chosen_ci].picked = true;
        const c = passed.items[chosen_ci];
        try selected.append(gpa, .{
            .view_pos = c.view_pos,
            .index = c.index,
            .parent_id = c.parent_id,
            .n_tokens = c.n_tokens,
        });
        for (0..q_len) |t| {
            if (c.tok_scores[t] > covered[t]) covered[t] = c.tok_scores[t];
        }
    }

    // Hop-2A: collect the winning document's digest unit so it sits before its detail chunks
    var digests: std.ArrayList(usize) = .empty;
    defer digests.deinit(gpa);
    if (selected.items.len > 0) {
        const seed_parent = selected.items[0].parent_id;
        for (units) |u| {
            if (!u.gate.isActive() or !u.gate.is_digest or u.parent_id != seed_parent) continue;
            try digests.append(gpa, u.index);
            break;
        }
    }

    // Hop-2B: only after a Hop-1 hit with room left, follow the top seed's strongest bridge
    if (selected.items.len > 0 and selected.items.len < cfg.top_k and bridges.len > 0) {
        const seed_pos = selected.items[0].view_pos;
        var best_edge: ?BridgeEdge = null;
        for (bridges) |e| {
            if (e.from != seed_pos) continue;
            const tgt = units[e.to];
            if (!tgt.gate.isActive()) continue;
            const min_w = if (e.same_parent) cfg.same_parent_bridge_min else cfg.cross_parent_bridge_min;
            if (e.weight < min_w) continue;
            // Skip if target already selected or parent cap reached
            var already = false;
            var pcount: usize = 0;
            for (selected.items) |s| {
                if (s.view_pos == e.to) {
                    already = true;
                    break;
                }
                if (s.parent_id == tgt.parent_id) pcount += 1;
            }
            if (already or pcount >= cfg.max_per_parent) continue;
            best_edge = e;
            break; // edges are sorted descending by weight per `from`
        }
        if (best_edge) |e| {
            const tgt = units[e.to];
            try selected.append(gpa, .{
                .view_pos = e.to,
                .index = tgt.index,
                .parent_id = tgt.parent_id,
                .n_tokens = tgt.n_tokens,
            });
        }
    }

    // Place longer detail units earlier and shorter detail units closer to the user turn!
    std.mem.sort(SelItem, selected.items, {}, struct {
        fn less(_: void, a: SelItem, b: SelItem) bool {
            return a.n_tokens > b.n_tokens;
        }
    }.less);

    // Output layout: standing policy units first, then cross-section digest index, then authoritative detail chunks.
    const out = try gpa.alloc(usize, standing.items.len + digests.items.len + selected.items.len);
    var k: usize = 0;
    for (standing.items) |idx| {
        out[k] = idx;
        k += 1;
    }
    for (digests.items) |idx| {
        out[k] = idx;
        k += 1;
    }
    for (selected.items) |s| {
        out[k] = s.index;
        k += 1;
    }
    return out;
}

test {
    _ = @import("gate_test.zig");
}
