//! Best-first candidate verify tree builder for Inkling 2 Mini DFlash 2 (up to 15 draft nodes under window_rows = 16).
const std = @import("std");
const sys = @import("sys");
const lanes = @import("lanes");
const w = @import("dflash2_weights.zig");
const tp = @import("inkling2_tree_plan");

pub const max_window: usize = tp.window_rows;
pub const max_tree_nodes: usize = max_window - 1;
pub const max_top_k: usize = lanes.draft_tree.lp_top;
pub const branch: usize = 4;
pub const edge_weight: f64 = 0.6;
pub const scale: f64 = 1.5;
pub const noise_weight: f64 = 0.7;
/// Alternate branches below this cumulative probability collapse away, keeping confident rounds on the chain fast path.
pub const min_branch_prob: f64 = 0.01;
const max_branch_cost: f64 = -@log(min_branch_prob);

pub const Landing = lanes.draft_tree.Landing;

pub const Candidates = struct {
    tids: []const u32, // [rows * topk]
    tvals: []const f32, // [rows * topk] raw unary logits (before softcap)
    hid: []const u16, // [rows * rank] bf16 selector projections
    topk: usize,
    rank: usize,
    valid_vocab: usize,
    pred_cb: []const u16, // [vocab * rank] bf16
    succ_cb: []const u16, // [vocab * rank] bf16
    softcap: f32 = 30.0,
    sampling: ?lanes.Sampling = null,
    position: u64 = 0,
};

/// Tree node budget under Inkling 2 Mini's 16-row verify window: 0 under TF_NO_TREES=1 or TF_DRAFT_NODES=0, default 15.
pub fn nodes() !usize {
    const no_trees = try sys.environment.read(bool, "TF_NO_TREES", false);
    return lanes.draft_tree.nodes(max_window, no_trees);
}

fn bf32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn roundBf16(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x)) @truncate((bits >> 16) | 0x40) else @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    return @bitCast(@as(u32, rounded) << 16);
}

fn activeSampling(s: ?lanes.Sampling) ?lanes.Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

fn scoreCandidate(c: Candidates, row: usize, cand: u32, raw_unary: f32, edge: f64) f64 {
    const combined = @as(f64, raw_unary) + edge_weight * edge;
    const capped = if (c.softcap > 0.0) @as(f64, c.softcap) * std.math.tanh(combined / @as(f64, c.softcap)) else combined;
    const s = activeSampling(c.sampling) orelse return capped;
    const u = @min(lanes.gpu_rule.uniform24(s.seed, @intCast(c.position + row), cand), @as(f32, @bitCast(@as(u32, 0x3F7FFFFF))));
    return capped / @max(s.temperature, 1e-6) + noise_weight * -@log(-@log(@as(f64, u)));
}

/// Computes per-candidate log-probabilities at `row` after `pred`, deduplicating any repeated token IDs.
fn chancesAfter(c: Candidates, row: usize, pred: u32, out: *[max_top_k]f64, valid_mask: *[max_top_k]bool) usize {
    const topk = @min(c.topk, max_top_k);
    var ah: [w.rank]f32 = undefined;
    const use_edge = c.rank > 0 and pred < c.valid_vocab;
    if (use_edge) {
        std.debug.assert(c.rank <= w.rank);
        const p_row = c.pred_cb[@as(usize, pred) * c.rank ..][0..c.rank];
        const h_row = c.hid[row * c.rank ..][0..c.rank];
        for (0..c.rank) |d| ah[d] = roundBf16(bf32(p_row[d]) * bf32(h_row[d]));
    }
    const tids = c.tids[row * c.topk ..][0..topk];
    const tvals = c.tvals[row * c.topk ..][0..topk];
    var most: f64 = -std.math.inf(f64);
    var valid_count: usize = 0;
    for (0..topk) |j| {
        const cand = tids[j];
        const unary = tvals[j];
        var ok = cand < c.valid_vocab and std.math.isFinite(unary);
        if (ok) {
            for (0..j) |prev| if (valid_mask[prev] and tids[prev] == cand) {
                ok = false;
                break;
            };
        }
        valid_mask[j] = ok;
        if (!ok) {
            out[j] = -std.math.inf(f64);
            continue;
        }
        var edge: f64 = 0.0;
        if (use_edge) {
            const s_row = c.succ_cb[@as(usize, cand) * c.rank ..][0..c.rank];
            for (0..c.rank) |d| edge += @as(f64, ah[d]) * @as(f64, bf32(s_row[d]));
        }
        const sc = scoreCandidate(c, row, cand, unary, edge) / scale;
        if (!std.math.isFinite(sc)) {
            valid_mask[j] = false;
            out[j] = -std.math.inf(f64);
            continue;
        }
        out[j] = sc;
        most = @max(most, sc);
        valid_count += 1;
    }
    if (valid_count == 0) return 0;
    var total: f64 = 0.0;
    for (0..topk) |j| if (valid_mask[j]) {
        total += @exp(out[j] - most);
    };
    const log_total = most + @log(total);
    for (0..topk) |j| if (valid_mask[j]) {
        out[j] -= log_total;
    };
    return valid_count;
}

/// Best-first candidate tree of up to `most` (<= 15) draft nodes in topological order; returns node count grown.
pub fn grow(c: Candidates, pending: u32, rows: usize, most: usize, tokens: []u32, parents: []i32, own: []f64) usize {
    return growWith(c, pending, rows, most, true, tokens, parents, own);
}

/// Same as `grow`, collapsing to the primary selector-scored chain when `allow_branch` is false.
pub fn growWith(c: Candidates, pending: u32, rows: usize, most: usize, allow_branch: bool, tokens: []u32, parents: []i32, own: []f64) usize {
    const cap = @min(most, @min(max_tree_nodes, @min(tokens.len, @min(parents.len, own.len))));
    if (cap == 0 or rows == 0 or c.topk == 0) return 0;
    const Open = struct { cost: f64, parent: i32, token: u32, row: usize, own: f64, on_chain: bool };
    var open: [branch * (max_window + 1)]Open = undefined;
    var n_open: usize = 0;
    var count: usize = 0;
    var pred = pending;
    var parent: i32 = -1;
    var row: usize = 0;
    var path: f64 = 0.0;
    var on_chain = true;
    while (true) {
        if (row < rows) {
            var lc: [max_top_k]f64 = undefined;
            var valid: [max_top_k]bool = @splat(false);
            const valid_n = chancesAfter(c, row, pred, &lc, &valid);
            if (valid_n > 0) {
                const topk = @min(c.topk, max_top_k);
                var order: [max_top_k]usize = undefined;
                var ord_n: usize = 0;
                for (0..topk) |j| if (valid[j]) {
                    order[ord_n] = j;
                    ord_n += 1;
                };
                std.mem.sort(usize, order[0..ord_n], &lc, struct {
                    fn more(v: *const [max_top_k]f64, a: usize, b: usize) bool {
                        return v[a] > v[b] or (v[a] == v[b] and a < b);
                    }
                }.more);
                const max_rank = if (allow_branch) @min(branch, ord_n) else @min(@as(usize, 1), ord_n);
                for (order[0..max_rank], 0..) |j, rank| {
                    if (!std.math.isFinite(lc[j]) or n_open == open.len) break;
                    const child_cost = path - lc[j];
                    const child_chain = on_chain and rank == 0;
                    if (!child_chain and child_cost > max_branch_cost) continue;
                    open[n_open] = .{
                        .cost = child_cost,
                        .parent = parent,
                        .token = c.tids[row * c.topk + j],
                        .row = row,
                        .own = lc[j],
                        .on_chain = child_chain,
                    };
                    n_open += 1;
                }
            }
        }
        if (count == cap or n_open == 0) break;
        var best: usize = 0;
        for (open[1..n_open], 1..) |o, i| {
            const b = open[best];
            if (o.cost < b.cost or (o.cost == b.cost and (o.parent < b.parent or (o.parent == b.parent and (o.token < b.token or (o.token == b.token and o.row < b.row)))))) best = i;
        }
        const node = open[best];
        open[best] = open[n_open - 1];
        n_open -= 1;
        tokens[count] = node.token;
        parents[count] = node.parent;
        own[count] = node.own;
        pred = node.token;
        parent = @intCast(count);
        row = node.row + 1;
        path = node.cost;
        on_chain = node.on_chain;
        count += 1;
    }
    return count;
}

fn bf(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

test "high-confidence distributions collapse cleanly to the primary chain" {
    const rows = 7;
    const topk = 4;
    var tids: [rows * topk]u32 = undefined;
    var tvals: [rows * topk]f32 = undefined;
    for (0..rows) |r| {
        for (0..topk) |j| {
            tids[r * topk + j] = @intCast(100 + r * 10 + j);
            tvals[r * topk + j] = if (j == 0) 20.0 else 0.0;
        }
    }
    const zeros: [16]u16 = @splat(0);
    const c: Candidates = .{
        .tids = &tids,
        .tvals = &tvals,
        .hid = &zeros,
        .topk = topk,
        .rank = 0,
        .valid_vocab = 1000,
        .pred_cb = &zeros,
        .succ_cb = &zeros,
        .softcap = 30.0,
    };
    var tokens: [max_tree_nodes]u32 = undefined;
    var parents: [max_tree_nodes]i32 = undefined;
    var own: [max_tree_nodes]f64 = undefined;
    const n = grow(c, 7, rows, max_tree_nodes, &tokens, &parents, &own);
    try std.testing.expectEqual(@as(usize, rows), n);
    try std.testing.expectEqualSlices(u32, &.{ 100, 110, 120, 130, 140, 150, 160 }, tokens[0..n]);
    const rp = try lanes.accept.rowParents(std.testing.allocator, 1 + n, parents[0..n]);
    defer std.testing.allocator.free(rp);
    try std.testing.expect(tp.isChain(rp));
    var depths: [tp.window_rows]u32 = undefined;
    try tp.validateParents(rp, depths[0..rp.len]);
}

test "uncertain positions branch up to 15 nodes with valid topology, unique siblings, and state plan parity" {
    const rows = 7;
    const topk = 4;
    var tids: [rows * topk]u32 = undefined;
    var tvals: [rows * topk]f32 = undefined;
    for (0..rows) |r| {
        for (0..topk) |j| {
            tids[r * topk + j] = @intCast(200 + r * 10 + j);
            tvals[r * topk + j] = switch (j) {
                0 => if (r == 0) 5.0 else 6.0,
                1 => if (r == 0) 4.8 else 4.5,
                2 => 3.5,
                else => 1.0,
            };
        }
    }
    // Inject a duplicate candidate ID at row 0 to verify sibling deduplication.
    tids[2] = tids[0];
    const zeros: [16]u16 = @splat(0);
    const c: Candidates = .{
        .tids = &tids,
        .tvals = &tvals,
        .hid = &zeros,
        .topk = topk,
        .rank = 0,
        .valid_vocab = 1000,
        .pred_cb = &zeros,
        .succ_cb = &zeros,
        .softcap = 30.0,
    };
    var tokens: [max_tree_nodes]u32 = undefined;
    var parents: [max_tree_nodes]i32 = undefined;
    var own: [max_tree_nodes]f64 = undefined;
    const n = grow(c, 5, rows, max_tree_nodes, &tokens, &parents, &own);
    try std.testing.expectEqual(@as(usize, max_tree_nodes), n);
    // Topological ordering and zero duplicate siblings under the same parent.
    for (0..n) |i| {
        try std.testing.expect(parents[i] >= -1 and parents[i] < @as(i32, @intCast(i)));
        try std.testing.expect(own[i] <= 0.0 and std.math.isFinite(own[i]));
        for (0..i) |prev| {
            if (parents[prev] == parents[i]) try std.testing.expect(tokens[prev] != tokens[i]);
        }
    }
    // Root must branch to both 200 and 201 (the close top-1 / top-2 at row 0).
    var saw_200 = false;
    var saw_201 = false;
    for (tokens[0..n], parents[0..n]) |tok, par| {
        if (par == -1 and tok == 200) saw_200 = true;
        if (par == -1 and tok == 201) saw_201 = true;
    }
    try std.testing.expect(saw_200 and saw_201);
    // Full Inkling 2 Metal tree plan validation: validateParents, buildConvSrc, buildTreePaths.
    const rp = try lanes.accept.rowParents(std.testing.allocator, 1 + n, parents[0..n]);
    defer std.testing.allocator.free(rp);
    try std.testing.expect(!tp.isChain(rp));
    var depths: [tp.window_rows]u32 = undefined;
    try tp.validateParents(rp, depths[0..rp.len]);
    var conv_src: [tp.window_rows * 4]u32 = undefined;
    tp.buildConvSrc(rp, conv_src[0 .. rp.len * 4]);
    var tree_paths: [tp.window_rows * tp.window_rows]u32 = undefined;
    tp.buildTreePaths(rp, depths[0..rp.len], tree_paths[0 .. rp.len * tp.window_rows]);
    for (0..rp.len) |row| {
        try std.testing.expectEqual(@as(u32, 0), tree_paths[row * tp.window_rows]);
        try std.testing.expectEqual(@as(u32, @intCast(row)), tree_paths[row * tp.window_rows + depths[row]]);
    }
}

test "selector edge, softcap, and sampled Gumbel noise steer tree ranking deterministically" {
    const rows = 2;
    const topk = 2;
    const rank = 2;
    const tids = [_]u32{ 1, 2, 3, 4 };
    const tvals = [_]f32{ 5.0, 4.9, 5.0, 5.0 };
    const hid = [_]u16{ bf(1.0), bf(1.0), bf(1.0), bf(1.0) };
    // pred=0 has [+2, +2], succ[2] has [+2, +2] -> strong positive edge for token 2 over token 1.
    var pred_cb: [8 * rank]u16 = @splat(bf(0.0));
    var succ_cb: [8 * rank]u16 = @splat(bf(0.0));
    pred_cb[0] = bf(2.0);
    pred_cb[1] = bf(2.0);
    succ_cb[2 * rank + 0] = bf(2.0);
    succ_cb[2 * rank + 1] = bf(2.0);
    var c: Candidates = .{
        .tids = &tids,
        .tvals = &tvals,
        .hid = &hid,
        .topk = topk,
        .rank = rank,
        .valid_vocab = 8,
        .pred_cb = &pred_cb,
        .succ_cb = &succ_cb,
        .softcap = 30.0,
    };
    var tokens: [6]u32 = undefined;
    var parents: [6]i32 = undefined;
    var own: [6]f64 = undefined;
    const n_greedy = grow(c, 0, rows, 6, &tokens, &parents, &own);
    try std.testing.expectEqual(@as(usize, 6), n_greedy);
    try std.testing.expectEqual(@as(u32, 2), tokens[0]); // selector flips 1 -> 2

    // Sampled mode (T=1.0, seed) is deterministic across repeated calls.
    c.sampling = .{ .seed = 0x5eed_cafe, .temperature = 1.0, .top_k = 16, .top_p = 0.95 };
    c.position = 42;
    var t_a: [6]u32 = undefined;
    var p_a: [6]i32 = undefined;
    var o_a: [6]f64 = undefined;
    var t_b: [6]u32 = undefined;
    var p_b: [6]i32 = undefined;
    var o_b: [6]f64 = undefined;
    const na = grow(c, 0, rows, 6, &t_a, &p_a, &o_a);
    const nb = grow(c, 0, rows, 6, &t_b, &p_b, &o_b);
    try std.testing.expectEqual(na, nb);
    try std.testing.expectEqualSlices(u32, t_a[0..na], t_b[0..nb]);
    try std.testing.expectEqualSlices(i32, p_a[0..na], p_b[0..nb]);
    try std.testing.expectEqualSlices(f64, o_a[0..na], o_b[0..nb]);
}

test "growWith allow_branch=false collapses uncertain distributions to the primary chain with valid own chances" {
    const rows = 7;
    const topk = 4;
    var tids: [rows * topk]u32 = undefined;
    var tvals: [rows * topk]f32 = undefined;
    for (0..rows) |r| {
        for (0..topk) |j| {
            tids[r * topk + j] = @intCast(300 + r * 10 + j);
            tvals[r * topk + j] = switch (j) {
                0 => 5.0,
                1 => 4.9,
                2 => 4.5,
                else => 1.0,
            };
        }
    }
    const zeros: [16]u16 = @splat(0);
    const c: Candidates = .{
        .tids = &tids,
        .tvals = &tvals,
        .hid = &zeros,
        .topk = topk,
        .rank = 0,
        .valid_vocab = 1000,
        .pred_cb = &zeros,
        .succ_cb = &zeros,
        .softcap = 30.0,
    };
    var tokens: [max_tree_nodes]u32 = undefined;
    var parents: [max_tree_nodes]i32 = undefined;
    var own: [max_tree_nodes]f64 = undefined;
    const n_tree = growWith(c, 5, rows, max_tree_nodes, true, &tokens, &parents, &own);
    try std.testing.expectEqual(@as(usize, max_tree_nodes), n_tree);
    try std.testing.expect(!lanes.accept.isChain(parents[0..n_tree]));
    const n_chain = growWith(c, 5, rows, max_tree_nodes, false, &tokens, &parents, &own);
    try std.testing.expectEqual(@as(usize, rows), n_chain);
    try std.testing.expect(lanes.accept.isChain(parents[0..n_chain]));
    try std.testing.expectEqualSlices(u32, &.{ 300, 310, 320, 330, 340, 350, 360 }, tokens[0..n_chain]);
    for (own[0..n_chain]) |lp| try std.testing.expect(lp <= 0.0 and std.math.isFinite(lp));
}
