//! Dual-layer neural gate (v6): two-regime lexical + deep semantic scoring with single-token
//! semantic corroboration guard, temporal supersession filtering, 2-level spherical ClusterIndex
//! for 10,000+ unit scaling, greedy submodular query-token coverage, pure-weight 2-hop associative
//! bridge spreading activation, entity conflict detection, and always-on standing policy units.
const std = @import("std");
const quant = @import("quant.zig");
const whitener = @import("whitener.zig");
const Allocator = std.mem.Allocator;

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
    /// Temporal belief lifecycle (v6): monotonic creation epoch, superseding parent_id (0 = active), reinforcement count.
    epoch: u64 = 0,
    superseded_by: u32 = 0,
    reinforcements: u16 = 1,
    /// L2-normalized pooled unit signatures [dim] for O(K·D) 2-level ClusterIndex coarse routing (computed lazily/on teach).
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

fn normalizeInPlace(v: []f32) void {
    var sq: f32 = 0.0;
    for (v) |x| sq += x * x;
    const inv: f32 = if (sq > 1e-12) 1.0 / @sqrt(sq) else 0.0;
    for (v) |*x| x.* *= inv;
}

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

/// Directed associative link between two memory units discovered purely from shared rare
/// unique-body hidden-state prototypes (zero plaintext!).
pub const BridgeEdge = struct {
    from: usize,
    to: usize,
    weight: f32,
    same_parent: bool,
};

/// Computes the IDF-filtered pure-vector associative bridge graph across all non-standing units.
/// Rare entities shared between 2-3 units (e.g. "Rotherham" linking Section 5 <-> Section 1, or
/// a dog's name linking two separate facts) produce strong edges; document-wide terms are suppressed by IDF.
pub fn computeBridgeEdges(gpa: Allocator, units: []const UnitView) ![]BridgeEdge {
    const n = units.len;
    if (n < 2) return try gpa.alloc(BridgeEdge, 0);

    // Pairwise accumulated bridge weights [n * n] (n is typically <= 256; symmetric accumulation).
    const pair_w = try gpa.alloc(f32, n * n);
    defer gpa.free(pair_w);
    @memset(pair_w, 0.0);

    const matched = try gpa.alloc(f32, n);
    defer gpa.free(matched);

    // For each unit i, check each rare unique-body lexical prototype against all other units j != i.
    for (0..n) |i| {
        const ui = units[i];
        if (!ui.gate.isActive() or ui.gate.is_standing or ui.gate.is_digest) continue;
        const dim = ui.gate.dim;
        const nb_i = ui.gate.bodyLexCount();

        for (0..nb_i) |pi| {
            const wi = ui.gate.lex_weights[pi];
            if (wi < 0.56) continue; // only high-specificity rare entity tokens form bridges
            const vi = ui.gate.lex_protos[pi * dim ..][0..dim];

            @memset(matched, 0.0);
            var df: usize = 1; // unit i itself
            var same_p_df: usize = 1;

            for (0..n) |j| {
                if (j == i) continue;
                const uj = units[j];
                if (!uj.gate.isActive() or uj.gate.is_standing or uj.gate.is_digest) continue;
                const nb_j = uj.gate.bodyLexCount();
                var best_cos: f32 = 0.0;
                var best_wj: f32 = 0.0;
                for (0..nb_j) |pj| {
                    const wj = uj.gate.lex_weights[pj];
                    if (wj < 0.56) continue;
                    const c = quant.dotF32(vi, uj.gate.lex_protos[pj * dim ..][0..dim]);
                    if (c > best_cos) {
                        best_cos = c;
                        best_wj = wj;
                    }
                }
                if (best_cos >= 0.87) {
                    matched[j] = best_cos * @sqrt(wi * best_wj);
                    df += 1;
                    if (uj.parent_id == ui.parent_id) same_p_df += 1;
                }
            }

            if (df <= 1) continue;
            // Suppress boilerplate / document-wide terms appearing in too many chunks
            const max_df: usize = if (n >= 6) 3 else if (n >= 4) 2 else 2;
            if (df > max_df) continue;
            const idf: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(df)));

            for (0..n) |j| {
                if (matched[j] > 0.0) {
                    pair_w[i * n + j] += matched[j] * idf;
                }
            }
        }
    }

    // Sort only clean associative edges:
    // - Same-parent chunks (`same_p == true`): links sections of a multi-chunk doc via rare shared entities (e.g. Rotherham!).
    // - Cross-parent units (`same_p == false`): ONLY between two small facts sharing an ultra-rare entity
    //   (NEVER bridges one whole document onto a different whole document!).
    var edges: std.ArrayList(BridgeEdge) = .empty;
    errdefer edges.deinit(gpa);
    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) continue;
            const same_p = units[i].parent_id == units[j].parent_id;
            if (!same_p and !(units[i].gate.is_small_fact and units[j].gate.is_small_fact)) continue;
            const sym = 0.5 * (pair_w[i * n + j] + pair_w[j * n + i]);
            const min_w: f32 = if (same_p) 0.45 else 0.95;
            if (sym >= min_w) {
                try edges.append(gpa, .{
                    .from = i,
                    .to = j,
                    .weight = sym,
                    .same_parent = same_p,
                });
            }
        }
    }

    // Sort by (from asc, weight desc) so looking up top bridge neighbors of a seed is O(deg).
    std.mem.sort(BridgeEdge, edges.items, {}, struct {
        fn less(_: void, a: BridgeEdge, b: BridgeEdge) bool {
            if (a.from != b.from) return a.from < b.from;
            return a.weight > b.weight;
        }
    }.less);
    return edges.toOwnedSlice(gpa);
}

/// Dual-layer score of a query (transformed at L_lex and L_sem) against one memory unit.
/// If `out_tok_scores` is provided (`len == ql.norms.len`), writes per-query-token lexical scores
/// (0 for masked tokens) for submodular coverage ranking.
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

/// 2-Level Spherical Cluster Index for sub-millisecond routing across 1,000–50,000+ memory units.
/// Stage 1 scores the query against `K = clamp(isqrt(N), 8, 128)` cluster centroids (O(K·D));
/// Stage 2 runs exact dual-layer scoring + submodular coverage ONLY on the shortlisted top clusters.
pub const ClusterIndex = struct {
    dim: usize = 0,
    n_clusters: usize = 0,
    /// Flat centroid arrays `[n_clusters * dim]` (L2-normalized pooled lexical & semantic cluster means).
    lex_centroids: []f32 = &.{},
    sem_centroids: []f32 = &.{},
    /// CSR member list storing `UnitView` positions belonging to each cluster.
    offsets: []usize = &.{}, // len = n_clusters + 1
    members: []usize = &.{},

    pub fn deinit(self: *ClusterIndex, gpa: Allocator) void {
        if (self.lex_centroids.len > 0) gpa.free(self.lex_centroids);
        if (self.sem_centroids.len > 0) gpa.free(self.sem_centroids);
        if (self.offsets.len > 0) gpa.free(self.offsets);
        if (self.members.len > 0) gpa.free(self.members);
        self.* = .{};
    }

    pub fn isReady(self: *const ClusterIndex) bool {
        return self.n_clusters > 0 and self.members.len > 0;
    }

    /// Rebuilds the spherical cluster index over all active non-standing, non-digest units.
    /// Automatically stays empty for small banks (`active < 32`) where linear scan is already < 0.2 ms.
    pub fn rebuild(self: *ClusterIndex, gpa: Allocator, units: []const UnitView, dim: usize) !void {
        self.deinit(gpa);
        if (dim == 0) return;

        var active_pos: std.ArrayList(usize) = .empty;
        defer active_pos.deinit(gpa);
        for (units, 0..) |u, pos| {
            if (!u.gate.isActive() or u.gate.is_standing or u.gate.is_digest) continue;
            if (u.gate.lex_summary.len != dim or u.gate.sem_summary.len != dim) continue;
            try active_pos.append(gpa, pos);
        }
        const n = active_pos.items.len;
        if (n < 32) return;

        const k_raw = std.math.sqrt(n);
        const k: usize = std.math.clamp(k_raw, 8, @min(@as(usize, 128), n));

        const lex_c = try gpa.alloc(f32, k * dim);
        errdefer gpa.free(lex_c);
        const sem_c = try gpa.alloc(f32, k * dim);
        errdefer gpa.free(sem_c);

        // Deterministic farthest-first / strided initialization for well-spread centroids
        const stride = @max(@as(usize, 1), n / k);
        for (0..k) |ci| {
            const u = units[active_pos.items[(ci * stride) % n]];
            @memcpy(lex_c[ci * dim ..][0..dim], u.gate.lex_summary);
            @memcpy(sem_c[ci * dim ..][0..dim], u.gate.sem_summary);
        }

        const assign = try gpa.alloc(usize, n);
        defer gpa.free(assign);
        const counts = try gpa.alloc(usize, k);
        defer gpa.free(counts);

        // 2 spherical k-means iterations (fast & well-converged on unit-norm hidden-state summaries)
        var iter: usize = 0;
        while (iter < 2) : (iter += 1) {
            for (active_pos.items, 0..) |pos, idx| {
                const u = units[pos];
                var best_c: usize = 0;
                var best_s: f32 = -1e9;
                for (0..k) |ci| {
                    const dl = quant.dotF32(u.gate.lex_summary, lex_c[ci * dim ..][0..dim]);
                    const ds = quant.dotF32(u.gate.sem_summary, sem_c[ci * dim ..][0..dim]);
                    const s = 0.55 * dl + 0.45 * ds;
                    if (s > best_s) {
                        best_s = s;
                        best_c = ci;
                    }
                }
                assign[idx] = best_c;
            }
            @memset(lex_c, 0.0);
            @memset(sem_c, 0.0);
            @memset(counts, 0);
            for (active_pos.items, 0..) |pos, idx| {
                const ci = assign[idx];
                const u = units[pos];
                const lc = lex_c[ci * dim ..][0..dim];
                const sc = sem_c[ci * dim ..][0..dim];
                for (lc, u.gate.lex_summary) |*dst, x| dst.* += x;
                for (sc, u.gate.sem_summary) |*dst, x| dst.* += x;
                counts[ci] += 1;
            }
            for (0..k) |ci| {
                if (counts[ci] > 0) {
                    normalizeInPlace(lex_c[ci * dim ..][0..dim]);
                    normalizeInPlace(sem_c[ci * dim ..][0..dim]);
                }
            }
        }

        // Pack CSR member list
        const offsets = try gpa.alloc(usize, k + 1);
        errdefer gpa.free(offsets);
        offsets[0] = 0;
        for (0..k) |ci| offsets[ci + 1] = offsets[ci] + counts[ci];
        const members = try gpa.alloc(usize, n);
        errdefer gpa.free(members);
        const cursor = try gpa.alloc(usize, k);
        defer gpa.free(cursor);
        @memcpy(cursor, offsets[0..k]);
        for (active_pos.items, 0..) |pos, idx| {
            const ci = assign[idx];
            members[cursor[ci]] = pos;
            cursor[ci] += 1;
        }

        self.dim = dim;
        self.n_clusters = k;
        self.lex_centroids = lex_c;
        self.sem_centroids = sem_c;
        self.offsets = offsets;
        self.members = members;
    }

    /// Stage-1 coarse shortlist: scores query tokens + query mean against all K centroids and
    /// collects the top clusters covering at least `min_shortlist` units.
    pub fn shortlist(
        self: *const ClusterIndex,
        gpa: Allocator,
        ql: *const whitener.Transformed,
        qs: *const whitener.Transformed,
        min_shortlist: usize,
    ) ![]usize {
        const k = self.n_clusters;
        const dim = self.dim;
        const q_len = ql.norms.len;

        // Build pooled query summary vectors [dim]
        const q_lex_mean = try gpa.alloc(f32, dim);
        defer gpa.free(q_lex_mean);
        @memset(q_lex_mean, 0.0);
        const q_sem_mean = try gpa.alloc(f32, dim);
        defer gpa.free(q_sem_mean);
        @memset(q_sem_mean, 0.0);

        for (0..q_len) |i| {
            const wl = @max(ql.spec[i], 0.05);
            const ws = @max(qs.spec[i], 0.05);
            const vl = ql.unit[i * dim ..][0..dim];
            const vs = qs.unit[i * dim ..][0..dim];
            for (q_lex_mean, vl) |*dst, x| dst.* += wl * x;
            for (q_sem_mean, vs) |*dst, x| dst.* += ws * x;
        }
        normalizeInPlace(q_lex_mean);
        normalizeInPlace(q_sem_mean);

        // Also grab top-3 highest-specificity lexical query tokens so rare entities hit their cluster directly!
        var top_toks: [3]usize = .{ 0, 0, 0 };
        var top_specs: [3]f32 = .{ -1.0, -1.0, -1.0 };
        var n_top: usize = 0;
        for (0..q_len) |i| {
            const sp = ql.spec[i];
            if (sp < 0.35) continue;
            var slot: ?usize = null;
            if (n_top < 3) {
                slot = n_top;
                n_top += 1;
            } else {
                var min_i: usize = 0;
                if (top_specs[1] < top_specs[min_i]) min_i = 1;
                if (top_specs[2] < top_specs[min_i]) min_i = 2;
                if (sp > top_specs[min_i]) slot = min_i;
            }
            if (slot) |s| {
                top_toks[s] = i;
                top_specs[s] = sp;
            }
        }

        const CScore = struct { ci: usize, score: f32 };
        const cscores = try gpa.alloc(CScore, k);
        defer gpa.free(cscores);
        for (0..k) |ci| {
            const lc = self.lex_centroids[ci * dim ..][0..dim];
            const sc = self.sem_centroids[ci * dim ..][0..dim];
            var max_tok: f32 = 0.0;
            for (0..n_top) |ti| {
                const d = quant.dotF32(ql.unit[top_toks[ti] * dim ..][0..dim], lc);
                if (d > max_tok) max_tok = d;
            }
            const ml = quant.dotF32(q_lex_mean, lc);
            const ms = quant.dotF32(q_sem_mean, sc);
            cscores[ci] = .{ .ci = ci, .score = 0.45 * ml + 0.35 * ms + 0.45 * max_tok };
        }
        std.mem.sort(CScore, cscores, {}, struct {
            fn less(_: void, a: CScore, b: CScore) bool {
                return a.score > b.score;
            }
        }.less);

        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(gpa);
        for (cscores) |cs| {
            const start = self.offsets[cs.ci];
            const end = self.offsets[cs.ci + 1];
            try out.appendSlice(gpa, self.members[start..end]);
            if (out.items.len >= min_shortlist) break;
        }
        return out.toOwnedSlice(gpa);
    }
};

pub const ConflictMatch = struct {
    parent_id: u32,
    lex_gate: f32,
    sem_score: f32,
    combined: f32,
};

/// Pure-vector (< 1 ms) entity conflict detector for Temporal Belief Revision (v6):
/// Given a newly arriving small fact's dual-layer transformed tokens, scores STRICTLY against
/// each active small fact's unique-body prototypes (`n_body_lex`, NEVER cue prototypes!) so
/// unrelated facts sharing generic cues ("called/named") NEVER collide.
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
        // Require strong entity + deep semantic overlap well above this unit's own negative calibration floor
        // so unrelated facts sharing a generic verb ("called", "is", "located") NEVER collide!
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

/// Routes a query against all active units:
///   1. Always-on active standing policy units are included first.
///   2. Optional Stage-1 `ClusterIndex` shortlist activates for large banks (`N >= 64`).
///   3. Hop-1 dual-layer gate with single-token semantic corroboration guard (eliminates subword
///      collisions like "loom" <-> "Loombook") + greedy submodular query-token coverage.
///   4. Hop-2A (consolidated verbatim index) + Hop-2B (pure-vector 2-hop associative bridge).
/// Automatically ignores any unit with `superseded_by != 0`.
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
        // Zero-edge single-token semantic guard (v6):
        // - Single-token lexical match on a document chunk (n_strong_lex <= 1): MUST clear model-adaptive
        //   semantic corroboration (`max(0.26, neg_sem_max + 0.03)`) to block subword/homonym collisions!
        // - Multi-token lexical match (n_strong_lex >= 2) or small fact: standard semantic sanity floor.
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

    // Greedy submodular query-token coverage selection!
    // Step 0 picks the highest total `rank_key`; subsequent steps reward uncovered query tokens
    // so multi-section queries (e.g. "Halcyon-7 vs Halcyon-8 payload") pull BOTH relevant sections.
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

    // Hop-2A: If the winning multi-chunk document has a consolidated digest unit, collect it separately
    // so it sits BEFORE the full detail chunks (bird's-eye index first, verbatim detail chunks closest to the user turn!).
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

    // Hop-2B: Pure-weight 2-hop associative bridge spreading activation!
    // ONLY runs when Hop-1 already fired (zero drift on unrelated queries!) and we have room,
    // following the strongest IDF-filtered rare-entity bridge from the top seed unit.
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

test "gate: hits matching unit, rejects orthogonal query, enforces parent diversity" {
    const gpa = std.testing.allocator;
    const dim = 16;

    var lp0: [dim]f32 = @splat(0);
    lp0[0] = 1.0;
    var lp2: [dim]f32 = @splat(0);
    lp2[1] = 1.0;

    var g0 = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &lp0),
        .lex_weights = try gpa.dupe(f32, &.{0.9}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &lp0),
        .sem_weights = try gpa.dupe(f32, &.{0.9}),
        .n_sem = 1,
        .neg_sem_max = 0.1,
        .is_small_fact = true,
    };
    defer g0.deinit(gpa);
    var g2 = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &lp2),
        .lex_weights = try gpa.dupe(f32, &.{0.9}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &lp2),
        .sem_weights = try gpa.dupe(f32, &.{0.9}),
        .n_sem = 1,
        .neg_sem_max = 0.1,
        .is_small_fact = false,
    };
    defer g2.deinit(gpa);

    const views = [_]UnitView{
        .{ .index = 0, .parent_id = 10, .n_tokens = 20, .gate = &g0 },
        .{ .index = 1, .parent_id = 20, .n_tokens = 100, .gate = &g2 },
    };

    var q_hit_u: [2 * dim]f32 = @splat(0);
    q_hit_u[0] = 1.0;
    q_hit_u[dim + 0] = 1.0;
    var tr_hit = whitener.Transformed{
        .unit = &q_hit_u,
        .norms = @constCast(&[_]f32{ 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9 }),
    };
    const active = try routeUnits(gpa, &views, &.{}, &tr_hit, &tr_hit, 1.0, 1.0, .{});
    defer gpa.free(active);
    try std.testing.expectEqualSlices(usize, &.{0}, active);

    var q_miss_u: [2 * dim]f32 = @splat(0);
    q_miss_u[5] = 1.0;
    q_miss_u[dim + 6] = 1.0;
    var tr_miss = whitener.Transformed{
        .unit = &q_miss_u,
        .norms = @constCast(&[_]f32{ 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9 }),
    };
    const none = try routeUnits(gpa, &views, &.{}, &tr_miss, &tr_miss, 1.0, 1.0, .{});
    defer gpa.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

test "gate: submodular coverage picks complementary section over redundant duplicate" {
    const gpa = std.testing.allocator;
    const dim = 16;

    // Query has 3 tokens: dim0 (Halcyon-8), dim1 (Halcyon-7), dim2 (payload)
    // Unit 0 covers dim0 + dim2 (Halcyon-8 payload -> highest single score!)
    // Unit 1 covers dim0 + dim2 slightly lower (redundant with Unit 0)
    // Unit 2 covers dim1 + dim2 (Halcyon-7 payload -> covers UNCOVERED dim1!)
    var p0: [2 * dim]f32 = @splat(0);
    p0[0] = 1.0;
    p0[dim + 2] = 1.0;
    var p1: [2 * dim]f32 = @splat(0);
    p1[0] = 0.98;
    p1[3] = @sqrt(1.0 - 0.98 * 0.98);
    p1[dim + 2] = 0.98;
    p1[dim + 4] = @sqrt(1.0 - 0.98 * 0.98);
    var p2: [2 * dim]f32 = @splat(0);
    p2[1] = 0.95;
    p2[5] = @sqrt(1.0 - 0.95 * 0.95);
    p2[dim + 2] = 0.90;
    p2[dim + 6] = @sqrt(1.0 - 0.90 * 0.90);

    var g0 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p0), .lex_weights = try gpa.dupe(f32, &.{ 0.9, 0.9 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p0), .sem_weights = try gpa.dupe(f32, &.{ 0.9, 0.9 }), .n_sem = 2 };
    defer g0.deinit(gpa);
    var g1 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p1), .lex_weights = try gpa.dupe(f32, &.{ 0.9, 0.9 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p1), .sem_weights = try gpa.dupe(f32, &.{ 0.9, 0.9 }), .n_sem = 2 };
    defer g1.deinit(gpa);
    var g2 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p2), .lex_weights = try gpa.dupe(f32, &.{ 0.85, 0.85 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p2), .sem_weights = try gpa.dupe(f32, &.{ 0.85, 0.85 }), .n_sem = 2 };
    defer g2.deinit(gpa);

    const views = [_]UnitView{
        .{ .index = 0, .parent_id = 1, .n_tokens = 120, .gate = &g0 },
        .{ .index = 1, .parent_id = 1, .n_tokens = 110, .gate = &g1 },
        .{ .index = 2, .parent_id = 1, .n_tokens = 100, .gate = &g2 },
    };

    var qu: [3 * dim]f32 = @splat(0);
    qu[0] = 1.0;
    qu[dim + 1] = 1.0;
    qu[2 * dim + 2] = 1.0;
    var tr = whitener.Transformed{
        .unit = &qu,
        .norms = @constCast(&[_]f32{ 2.0, 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9, 0.9 }),
    };

    // With top_k=2, submodular coverage MUST pick Unit 0 AND Unit 2 (skipping redundant Unit 1)!
    const picked = try routeUnits(gpa, &views, &.{}, &tr, &tr, 1.0, 1.0, .{ .top_k = 2, .max_per_parent = 2 });
    defer gpa.free(picked);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, picked);
}

test "gate: 2-hop associative bridge links rare shared entity without causing drift on unrelated queries" {
    const gpa = std.testing.allocator;
    const dim = 16;

    // Unit 0 (Sec 5): dim0 (Windtunnel-9), dim1 (Rotherham bridge!)
    // Unit 1 (Sec 1): dim1 (Rotherham bridge!), dim2 (wing spars)
    // Unit 2 (Sec 2): dim3 (unrelated section)
    var p0: [2 * dim]f32 = @splat(0);
    p0[0] = 1.0;
    p0[dim + 1] = 1.0;
    var p1: [2 * dim]f32 = @splat(0);
    p1[1] = 1.0;
    p1[dim + 2] = 1.0;
    var p2: [dim]f32 = @splat(0);
    p2[3] = 1.0;

    var g0 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p0), .lex_weights = try gpa.dupe(f32, &.{ 0.9, 0.85 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p0), .sem_weights = try gpa.dupe(f32, &.{ 0.9, 0.85 }), .n_sem = 2 };
    defer g0.deinit(gpa);
    var g1 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p1), .lex_weights = try gpa.dupe(f32, &.{ 0.85, 0.9 }), .n_lex = 2, .n_body_lex = 2, .sem_protos = try gpa.dupe(f32, &p1), .sem_weights = try gpa.dupe(f32, &.{ 0.85, 0.9 }), .n_sem = 2 };
    defer g1.deinit(gpa);
    var g2 = UnitGate{ .dim = dim, .lex_protos = try gpa.dupe(f32, &p2), .lex_weights = try gpa.dupe(f32, &.{0.9}), .n_lex = 1, .n_body_lex = 1, .sem_protos = try gpa.dupe(f32, &p2), .sem_weights = try gpa.dupe(f32, &.{0.9}), .n_sem = 1 };
    defer g2.deinit(gpa);

    const views = [_]UnitView{
        .{ .index = 0, .parent_id = 6, .n_tokens = 200, .gate = &g0 },
        .{ .index = 1, .parent_id = 6, .n_tokens = 180, .gate = &g1 },
        .{ .index = 2, .parent_id = 6, .n_tokens = 150, .gate = &g2 },
    };

    const bridges = try computeBridgeEdges(gpa, &views);
    defer gpa.free(bridges);
    try std.testing.expect(bridges.len >= 2);

    // Query mentions ONLY dim0 (Windtunnel-9) — never mentions Rotherham (dim1) or wing spars (dim2)!
    var q_wt: [2 * dim]f32 = @splat(0);
    q_wt[0] = 1.0;
    q_wt[dim + 0] = 1.0;
    var tr_wt = whitener.Transformed{
        .unit = &q_wt,
        .norms = @constCast(&[_]f32{ 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9 }),
    };
    const picked = try routeUnits(gpa, &views, bridges, &tr_wt, &tr_wt, 1.0, 1.0, .{});
    defer gpa.free(picked);
    // Hop-1 picks Unit 0 (Windtunnel-9); Hop-2 bridge follows Rotherham -> activates Unit 1 too!
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, picked);

    // Unrelated query (dim 9) -> STILL zero hits (zero drift!)
    var q_unrel: [2 * dim]f32 = @splat(0);
    q_unrel[9] = 1.0;
    q_unrel[dim + 10] = 1.0;
    var tr_unrel = whitener.Transformed{
        .unit = &q_unrel,
        .norms = @constCast(&[_]f32{ 2.0, 2.0 }),
        .spec = @constCast(&[_]f32{ 0.9, 0.9 }),
    };
    const none = try routeUnits(gpa, &views, bridges, &tr_unrel, &tr_unrel, 1.0, 1.0, .{});
    defer gpa.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

test "gate(v6): single-token subword veto blocks low-semantic collision and skips superseded units" {
    const gpa = std.testing.allocator;
    const dim = 16;

    // Document chunk with lexical vector on dim 0 ("Loombook"), semantic vector on dim 1 (aerospace software)
    var lp: [dim]f32 = @splat(0);
    lp[0] = 1.0;
    var sp: [dim]f32 = @splat(0);
    sp[1] = 1.0;

    var g_doc = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &lp),
        .lex_weights = try gpa.dupe(f32, &.{0.95}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &sp),
        .sem_weights = try gpa.dupe(f32, &.{0.90}),
        .n_sem = 1,
        .neg_sem_max = 0.15,
        .is_small_fact = false,
    };
    defer g_doc.deinit(gpa);

    // Superseded small fact on dim 2 (old belief: "Barnaby") vs active new belief on dim 2 ("Biscuit")
    var p_dog: [dim]f32 = @splat(0);
    p_dog[2] = 1.0;
    var g_old = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &p_dog),
        .lex_weights = try gpa.dupe(f32, &.{0.9}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &p_dog),
        .sem_weights = try gpa.dupe(f32, &.{0.9}),
        .n_sem = 1,
        .is_small_fact = true,
        .superseded_by = 99,
    };
    defer g_old.deinit(gpa);
    var g_new = UnitGate{
        .dim = dim,
        .lex_protos = try gpa.dupe(f32, &p_dog),
        .lex_weights = try gpa.dupe(f32, &.{0.9}),
        .n_lex = 1,
        .n_body_lex = 1,
        .sem_protos = try gpa.dupe(f32, &p_dog),
        .sem_weights = try gpa.dupe(f32, &.{0.9}),
        .n_sem = 1,
        .is_small_fact = true,
        .superseded_by = 0,
    };
    defer g_new.deinit(gpa);

    const views = [_]UnitView{
        .{ .index = 0, .parent_id = 6, .n_tokens = 250, .gate = &g_doc },
        .{ .index = 1, .parent_id = 2, .n_tokens = 15, .gate = &g_old },
        .{ .index = 2, .parent_id = 99, .n_tokens = 15, .gate = &g_new },
    };

    // 1. Single-token lexical hit on dim 0 ("loom") with orthogonal semantics (dim 8: weaving) -> MUST NOT FIRE!
    var ql_loom: [2 * dim]f32 = @splat(0);
    ql_loom[0] = 0.96; // "loom" matches "Loombook"
    ql_loom[dim + 7] = 1.0; // "warp" orthogonal
    var qs_loom: [2 * dim]f32 = @splat(0);
    qs_loom[8] = 1.0;
    qs_loom[dim + 9] = 1.0;
    var tr_ql = whitener.Transformed{ .unit = &ql_loom, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };
    var tr_qs = whitener.Transformed{ .unit = &qs_loom, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };
    const hit_loom = try routeUnits(gpa, &views, &.{}, &tr_ql, &tr_qs, 1.0, 1.0, .{});
    defer gpa.free(hit_loom);
    try std.testing.expectEqual(@as(usize, 0), hit_loom.len);

    // 2. Query on dim 2 ("dog") -> MUST pick ONLY active unit 2 (pid=99), NEVER superseded unit 1 (pid=2)!
    var q_dog: [2 * dim]f32 = @splat(0);
    q_dog[2] = 1.0;
    q_dog[dim + 2] = 1.0;
    var tr_dog = whitener.Transformed{ .unit = &q_dog, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };
    const hit_dog = try routeUnits(gpa, &views, &.{}, &tr_dog, &tr_dog, 1.0, 1.0, .{});
    defer gpa.free(hit_dog);
    try std.testing.expectEqualSlices(usize, &.{2}, hit_dog);

    // 3. Entity conflict detector finds active pid=99 (and NOT superseded pid=2)!
    const conflicts = try findConflictParents(gpa, &views, &tr_dog, &tr_dog, 1.0, 1.0, 4);
    defer gpa.free(conflicts);
    try std.testing.expectEqual(@as(usize, 1), conflicts.len);
    try std.testing.expectEqual(@as(u32, 99), conflicts[0].parent_id);
}

test "gate(v6): 2-level ClusterIndex scales to 2,048 units with exact target recall" {
    const gpa = std.testing.allocator;
    const dim = 32;
    const n_units: usize = 2048;

    const gates = try gpa.alloc(UnitGate, n_units);
    defer {
        for (gates) |*g| g.deinit(gpa);
        gpa.free(gates);
    }
    const views = try gpa.alloc(UnitView, n_units);
    defer gpa.free(views);

    var prng = std.Random.DefaultPrng.init(0x5eed1234);
    const rng = prng.random();

    for (0..n_units) |i| {
        const lp = try gpa.alloc(f32, dim);
        const sp = try gpa.alloc(f32, dim);
        for (0..dim) |d| {
            lp[d] = rng.floatNorm(f32);
            sp[d] = rng.floatNorm(f32);
        }
        // Plant a crisp rare needle at unit #1337 on dimensions 0 and 1
        if (i == 1337) {
            @memset(lp, 0.0);
            @memset(sp, 0.0);
            lp[0] = 1.0;
            sp[1] = 1.0;
        } else {
            // Keep distractors away from the exact planted needle axis
            lp[0] *= 0.05;
            sp[1] *= 0.05;
            normalizeInPlace(lp);
            normalizeInPlace(sp);
        }
        gates[i] = .{
            .dim = dim,
            .lex_protos = lp,
            .lex_weights = try gpa.dupe(f32, &.{0.9}),
            .n_lex = 1,
            .n_body_lex = 1,
            .sem_protos = sp,
            .sem_weights = try gpa.dupe(f32, &.{0.9}),
            .n_sem = 1,
            .neg_sem_max = 0.25,
            .is_small_fact = true,
        };
        try gates[i].computeSummaries(gpa);
        views[i] = .{ .index = i, .parent_id = @intCast(i + 1), .n_tokens = 20, .gate = &gates[i] };
    }

    var idx = ClusterIndex{};
    defer idx.deinit(gpa);
    try idx.rebuild(gpa, views, dim);
    try std.testing.expect(idx.isReady());
    try std.testing.expect(idx.n_clusters >= 16);

    var ql_u: [2 * dim]f32 = @splat(0);
    ql_u[0] = 1.0;
    ql_u[dim + 0] = 1.0;
    var qs_u: [2 * dim]f32 = @splat(0);
    qs_u[1] = 1.0;
    qs_u[dim + 1] = 1.0;
    var tr_l = whitener.Transformed{ .unit = &ql_u, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };
    var tr_s = whitener.Transformed{ .unit = &qs_u, .norms = @constCast(&[_]f32{ 2.0, 2.0 }), .spec = @constCast(&[_]f32{ 0.9, 0.9 }) };

    const active = try routeUnitsIndexed(gpa, views, &.{}, &idx, &tr_l, &tr_s, 1.0, 1.0, .{});
    defer gpa.free(active);
    var found = false;
    for (active) |u_idx| if (u_idx == 1337) {
        found = true;
        break;
    };
    try std.testing.expect(found);
}
