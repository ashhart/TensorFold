//! IDF-filtered associative bridge graph between memory units (Hop-2 spreading activation edges).
const std = @import("std");
const quant = @import("quant.zig");
const UnitView = @import("gate.zig").UnitView;
const Allocator = std.mem.Allocator;

/// Directed associative link between two memory units that share rare unique-body hidden-state prototypes.
pub const BridgeEdge = struct {
    from: usize,
    to: usize,
    weight: f32,
    same_parent: bool,
};

/// IDF-filtered associative bridge graph over non-standing units: rare shared entities link, common terms do not.
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

    // Keep same-parent edges (sections of one doc) and cross-parent edges only between two small facts.
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
