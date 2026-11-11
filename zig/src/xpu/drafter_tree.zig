//! DFlash2 draft trees: the best-first policy over the drafter's lattice, and one verified tree round.
const std = @import("std");
const dmod = @import("drafter.zig");
const Drafter = dmod.Drafter;
const MAXL = dmod.MAXL;
const TOPK = dmod.TOPK;
const RANK = dmod.RANK;
// the tree policy: edge weight, weight of the target's noise, temperature of the pops, children a node pushes
pub const EDGE = 0.6;
pub const NOISE = 0.7;
pub const SCALE = 1.5;
pub const BRANCH = 4;

/// The tree search's frontier: a few dozen items, popped smallest (score, parent, token, depth) first like heapq.
pub const Heap = struct {
    const Item = struct { score: f64, parent: i32, token: u32, depth: u32, own: f64 = 0 };
    items: [MAXL * BRANCH + BRANCH]Item = undefined,
    n: usize = 0,
    fn push(h: *Heap, it: Item) void {
        if (h.n < h.items.len) {
            h.items[h.n] = it;
            h.n += 1;
        }
    }
    fn before(a: Item, b: Item) bool {
        if (a.score != b.score) return a.score < b.score;
        if (a.parent != b.parent) return a.parent < b.parent;
        if (a.token != b.token) return a.token < b.token;
        return a.depth < b.depth;
    }
    fn pop(h: *Heap) ?Item {
        if (h.n == 0) return null;
        var best: usize = 0;
        for (1..h.n) |i| if (before(h.items[i], h.items[best])) {
            best = i;
        };
        const it = h.items[best];
        h.n -= 1;
        h.items[best] = h.items[h.n];
        return it;
    }
};

/// The chain over the current lattice (propose's choice).
pub fn chain(dr: *const Drafter, pending: u32, count: u32, out: []u32) void {
    var parent = pending;
    for (0..count) |depth| {
        var best: usize = 0;
        var best_score: f64 = -std.math.inf(f64);
        for (0..TOPK) |j| {
            const tok = dr.cand[depth][j];
            const score = dr.unary[depth][j] + EDGE * edge(dr, parent, depth, tok);
            if (score > best_score or (score == best_score and tok < dr.cand[depth][best])) {
                best_score = score;
                best = j;
            }
        }
        out[depth] = dr.cand[depth][best];
        parent = out[depth];
    }
}

pub fn edge(dr: *const Drafter, parent: u32, depth: usize, tok: u32) f64 {
    var e: f64 = 0;
    for (0..RANK) |rr| e += code(dr.pred, parent, rr) * @as(f64, dr.sp[depth * RANK + rr]) * code(dr.succ, tok, rr);
    return e;
}

/// A best-first tree of ≤ `max_nodes` drafts over the last lattice, `parents[i]` −1 at the root; returns the count.
pub fn tree(dr: *Drafter, pending: u32, max_nodes: u32, noise: ?*const [MAXL][TOPK]f64, temp: f64, tokens: []u32, parents: []i32) u32 {
    return treeOwn(dr, pending, max_nodes, noise, temp, tokens, parents, null);
}

/// `tree`, also giving each draft's own log-chance under its parent (`own`).
pub fn treeOwn(dr: *Drafter, pending: u32, max_nodes: u32, noise: ?*const [MAXL][TOPK]f64, temp: f64, tokens: []u32, parents: []i32, own: ?[]f64) u32 {
    var heap: Heap = .{};
    expand(dr, &heap, noise, @max(temp, 1e-6), pending, 0, -1, 0);
    var n: u32 = 0;
    while (n < max_nodes) {
        const it = heap.pop() orelse break;
        tokens[n] = it.token;
        parents[n] = it.parent;
        if (own) |o| o[n] = it.own;
        const me: i32 = @intCast(n);
        n += 1;
        if (it.depth + 1 < dr.depths) expand(dr, &heap, noise, @max(temp, 1e-6), it.token, it.depth + 1, me, it.score);
    }
    return n;
}

/// Pushes the BRANCH best children of `token` at `depth`, scored by path − log p under the policy.
pub fn expand(dr: *const Drafter, heap: *Heap, noise: ?*const [MAXL][TOPK]f64, temp: f64, token: u32, depth: u32, parent: i32, path: f64) void {
    var v: [TOPK]f64 = undefined;
    for (0..TOPK) |j| {
        v[j] = (dr.unary[depth][j] + EDGE * edge(dr, token, depth, dr.cand[depth][j])) / temp;
        if (noise) |nz| v[j] += NOISE * nz[depth][j];
        v[j] /= SCALE;
    }
    var mx = v[0];
    for (v) |x| mx = @max(mx, x);
    var sum: f64 = 0;
    for (&v) |*x| {
        x.* -= mx;
        sum += @exp(x.*);
    }
    const lse = @log(sum);
    var taken: [TOPK]bool = @splat(false);
    for (0..BRANCH) |_| { // the BRANCH largest, ties to the lower index (numpy's argsort of −logp)
        var j: usize = TOPK;
        for (0..TOPK) |i| if (!taken[i] and (j == TOPK or v[i] > v[j])) {
            j = i;
        };
        taken[j] = true;
        const lp = v[j] - lse;
        if (!std.math.isFinite(lp)) break;
        heap.push(.{ .score = path - lp, .parent = parent, .token = dr.cand[depth][j], .depth = depth, .own = lp });
    }
}

pub fn code(t: []const u8, row: u32, col: usize) f64 {
    const at = (@as(usize, row) * RANK + col) * 2;
    const bits = std.mem.readInt(u16, t[at..][0..2], .little);
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

/// How a round draws the target's tokens: greedy without one; else row `row` at `position` with keyed Gumbel noise.
pub const Draw = struct {
    ctx: *anyopaque,
    row: *const fn (*anyopaque, row: u32, position: u64) anyerror!u32,
    gumbel: *const fn (*anyopaque, position: u64, id: u64) f64,
    temperature: f64,
};

/// `path`: the rows of the accepted path, row 0 first, `rows` of them.
pub const Round = struct { tokens: []const u32, terminal: u32, drafted: u32, accepted: u32, path: [MAXL]u32 = undefined, rows: u32 = 0 };

pub const Walk = struct { len: u32, terminal: u32 };

/// The target's walk down a verified tree: `path` the rows it takes from row 0, then its pick past the last.
pub fn walk(ids: []const u32, rows: []const i32, greedy: []const u32, draw: ?Draw, p0: u64, path: *[MAXL]u32) !Walk {
    path[0] = 0;
    var len: u32 = 1;
    var terminal = if (draw) |dw| try dw.row(dw.ctx, 0, p0 + 1) else greedy[0];
    descend: while (true) {
        for (1..ids.len) |c| if (rows[c] == path[len - 1] and ids[c] == terminal) {
            path[len] = @intCast(c);
            len += 1;
            terminal = if (draw) |dw| try dw.row(dw.ctx, @intCast(c), p0 + len) else greedy[c];
            continue :descend;
        };
        break;
    }
    return .{ .len = len, .terminal = terminal };
}

/// A grown tree: `n` drafts (ids[1..], their parents, their own log-chances) over the last lattice, from position `p0`.
pub const Grown = struct { ids: [MAXL]u32 = undefined, par: [MAXL]i32 = undefined, own: [MAXL]f64 = undefined, n: u32 = 0, p0: u64 = 0 };

/// The lattice after `pending`, then its best-first tree of at most `nodes` drafts (see `grown`).
pub fn grow(dr: *Drafter, pending: u32, depths: u32, nodes: u32, draw: ?Draw) !Grown {
    try dr.lattice(pending, depths);
    return grown(dr, pending, depths, nodes, draw, dr.r.pos);
}

/// A best-first tree of at most `nodes` drafts over the last lattice, on the host; nothing is verified.
pub fn grown(dr: *Drafter, pending: u32, depths: u32, nodes: u32, draw: ?Draw, p0: u64) Grown {
    var g: Grown = .{ .p0 = p0 };
    var noise: [MAXL][TOPK]f64 = undefined;
    if (draw) |dw| for (0..depths) |dd| for (0..TOPK) |j| {
        noise[dd][j] = dw.gumbel(dw.ctx, p0 + 1 + dd, dr.cand[dd][j]);
    };
    g.ids[0] = pending;
    g.n = treeOwn(dr, pending, @min(nodes, MAXL - 1), if (draw != null) &noise else null, if (draw) |dw| dw.temperature else 1.0, g.ids[1..], g.par[1..], g.own[1..]);
    return g;
}

/// The first `n` drafts in verified rows: row 0 the pending token, a draft's row its index + 1.
pub fn treeRows(g: *const Grown, n: u32, rows: *[MAXL]i32) void {
    rows[0] = 0;
    for (1..n + 1) |i| rows[i] = if (g.par[i] < 0) 0 else g.par[i] + 1;
}

/// The target's walk down the first `nodes` drafts of a grown tree, given its pick at each verified row.
pub fn follow(g: *const Grown, nodes: u32, greedy: []const u32, draw: ?Draw, out: *[MAXL]u32) !Round {
    const n = @min(nodes, g.n);
    var rows: [MAXL]i32 = undefined;
    treeRows(g, n, &rows);
    var round: Round = .{ .tokens = &.{}, .terminal = 0, .drafted = n, .accepted = 0 };
    const w = try walk(g.ids[0 .. n + 1], rows[0 .. n + 1], greedy, draw, g.p0, &round.path);
    for (0..w.len) |i| out[i] = g.ids[round.path[i]];
    round.tokens = out[0..w.len];
    round.terminal = w.terminal;
    round.accepted = w.len - 1;
    round.rows = w.len;
    return round;
}

/// The first `nodes` drafts of a grown tree (a prefix in pop order, a child after its parent), verified and kept.
pub fn settle(dr: *Drafter, g: *const Grown, nodes: u32, draw: ?Draw, out: *[MAXL]u32) !Round {
    const r = dr.r;
    const n = @min(nodes, g.n);
    var rows: [MAXL]i32 = undefined;
    treeRows(g, n, &rows);
    const round = try follow(g, nodes, try r.verifyTree(g.ids[0 .. n + 1], rows[0 .. n + 1]), draw, out);
    try r.commitPath(round.path[0..round.rows]);
    try dr.addTaps(r.acts.taps, round.rows);
    return round;
}

/// One tree round from `pending`: drafts a tree of ≤ `nodes`, verifies it, keeps the path; returns the accepted drafts.
pub fn treeRound(dr: *Drafter, pending: u32, depths: u32, nodes: u32, draw: ?Draw, out: *[MAXL]u32) !Round {
    const g = try grow(dr, pending, depths, nodes, draw);
    return settle(dr, &g, nodes, draw, out);
}
