//! Two-level spherical ClusterIndex for coarse unit routing, plus the shared L2 normalization helper.
const std = @import("std");
const quant = @import("quant.zig");
const whitener = @import("whitener.zig");
const UnitView = @import("gate.zig").UnitView;
const Allocator = std.mem.Allocator;

pub fn normalizeInPlace(v: []f32) void {
    var sq: f32 = 0.0;
    for (v) |x| sq += x * x;
    const inv: f32 = if (sq > 1e-12) 1.0 / @sqrt(sq) else 0.0;
    for (v) |*x| x.* *= inv;
}

/// Two-level spherical cluster index: score `K = clamp(isqrt(N), 8, 128)` centroids, then exact-score top clusters.
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

    /// Rebuilds the cluster index over active non-standing, non-digest units; stays empty below 32 units.
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

    /// Stage-1 shortlist: scores the query against all centroids, keeps top clusters covering `min_shortlist` units.
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
