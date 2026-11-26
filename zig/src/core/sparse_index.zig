//! A sparse-attention indexer's key selection in float64, for any model's policy: the reference its kernels follow.
const std = @import("std");
const nvfp4_store = @import("nvfp4_store.zig");

/// The activation on each head's dot product before heads are combined.
pub const Phi = enum { relu, identity };
/// How heads combine into one score.
pub const Reduce = enum { sum, max };
/// A block's summary key from its keys (each key first scaled to unit RMS when `norm_first`).
pub const Summary = enum { mean, max, first, last };
/// What a later layer takes from the layer that selected.
pub const Reuse = enum { none, final, pool };
/// The keys block summaries average: the original keys, or the token keys as stored (keys_nvfp4 read back).
pub const SummaryInput = enum { original, stored };

pub const Policy = struct {
    topk: usize, // tokens selected a query
    block: usize = 1, // keys a block summary covers; 1: one stage, no summaries
    pool_blocks: usize = 0, // blocks the first stage keeps; 0: all
    sink: usize = 0, // the first keys, always attended
    local: usize = 0, // the most recent keys (the query's own included), always attended
    mandatory_in_topk: bool = false, // sink and local keys count inside `topk` (else they come on top)
    dense: bool = false, // a model rule: a query whose keys all fit the budget attends every key, unscored
    phi: Phi = .relu,
    reduce: Reduce = .sum,
    summary: Summary = .mean,
    norm_first: bool = false, // summaries average unit-RMS keys
    open_block: bool = true, // the last, incomplete block is summarised from the keys it has (else left out)
    key_heads: usize = 1, // index key heads a key row holds; query head h reads key head h / (heads / key_heads)
    scale: f64 = 1, // times every score
    keys_nvfp4: ?nvfp4_store.Format = null, // token keys scored as stored in NVFP4 and read back (null: as given)
    queries_nvfp4: ?nvfp4_store.Format = null, // the same for the queries
    summary_nvfp4: ?nvfp4_store.Format = null, // each block summary stored in NVFP4 and read back before scoring
    summary_input: SummaryInput = .stored, // summaries from the original keys or from the stored token keys
};

/// The first stage's blocks for keys 0..t: the open (incomplete) last one only when `open_block`.
pub fn blocks(p: Policy, t: usize) usize {
    return if (p.open_block) (t + p.block) / p.block else (t + 1) / p.block;
}

/// One query position's index heads [heads, dim] and optional per-head weights.
pub const Query = struct { q: []const f64, heads: usize, weights: ?[]const f64 = null, layer: u32 = 0, seq: u64 = 0 };

/// Index keys [n, dim], position i at row i.
pub const Keys = struct { k: []const f64, dim: usize };

/// The larger of a and b, NaN kept (as the kernels' (b > a || b != b) ? b : a).
fn larger(a: f64, b: f64) f64 {
    return if (b > a or std.math.isNan(b)) b else a;
}

/// scale * sum (or max) over heads of w_h * phi(q_h . its key head); NaN propagates through phi and max.
pub fn score(p: Policy, q: Query, key: []const f64) f64 {
    const dim = key.len / p.key_heads;
    const group = q.heads / p.key_heads;
    var acc: f64 = if (p.reduce == .max) -std.math.inf(f64) else 0;
    for (0..q.heads) |h| {
        var dot: f64 = 0;
        for (q.q[h * dim ..][0..dim], key[h / group * dim ..][0..dim]) |a, b| dot += a * b;
        if (p.phi == .relu and dot < 0) dot = 0;
        const w = if (q.weights) |ws| ws[h] else 1;
        acc = if (p.reduce == .max) larger(acc, w * dot) else acc + w * dot;
    }
    return acc * p.scale;
}

/// The scores' one order, core's and the kernels': any NaN above +inf, -0 equal to +0.
pub fn above(a: f64, b: f64) bool {
    if (std.math.isNan(a)) return !std.math.isNan(b);
    return !std.math.isNan(b) and a > b;
}

fn same(a: f64, b: f64) bool {
    return a == b or (std.math.isNan(a) and std.math.isNan(b));
}

const Scored = struct { pos: u32, s: f64 };

/// Higher score first (in `above`'s order); on equal scores the lower position.
fn before(_: void, a: Scored, b: Scored) bool {
    return above(a.s, b.s) or (same(a.s, b.s) and a.pos < b.pos);
}

/// The positions query t of layer `owner` attends and its first stage's kept candidates (for reuse), both ascending.
pub const Selection = struct {
    keys: []u32,
    pool: []u32,
    t: usize = 0,
    owner: u32 = 0,
    seq: u64 = 0, // the sequence (stream) whose keys it picked: a pool never crosses into another

    pub fn deinit(s: Selection, gpa: std.mem.Allocator) void {
        gpa.free(s.keys);
        gpa.free(s.pool);
    }
};

fn mandatory(p: Policy, t: usize, i: usize) bool {
    return i < p.sink or i + p.local > t;
}

/// Block b's summary over positions [b * block, min((b + 1) * block, t + 1)) into `out`; NaN propagates.
pub fn summarise(p: Policy, keys: Keys, b: usize, t: usize, out: []f64) void {
    const lo = b * p.block;
    const hi = @min(lo + p.block, t + 1);
    @memset(out, if (p.summary == .max) -std.math.inf(f64) else 0);
    for (lo..hi) |i| {
        const k = keys.k[i * keys.dim ..][0..keys.dim];
        var r: f64 = 1;
        if (p.norm_first) {
            var ss: f64 = 0;
            for (k) |v| ss += v * v;
            r = if (ss > 0) 1 / @sqrt(ss / @as(f64, @floatFromInt(keys.dim))) else if (ss == 0) 0 else ss;
        }
        const take = switch (p.summary) {
            .first => i == lo,
            .last => i + 1 == hi,
            else => true,
        };
        if (take) for (out, k) |*o, v| {
            o.* = if (p.summary == .max) larger(o.*, v * r) else o.* + v * r;
        };
    }
    if (p.summary == .mean) for (out) |*o| {
        o.* /= @floatFromInt(hi - lo);
    };
}

/// The most keys a selection at t may hold: topk, with the mandatory keys inside it or on top.
pub fn budget(p: Policy, t: usize) usize {
    var m: usize = 0;
    for (0..t + 1) |i| m += @intFromBool(mandatory(p, t, i));
    return if (p.mandatory_in_topk) @max(p.topk, m) else p.topk + m;
}

/// Every block fits the first stage's pool, the open one included (or there is none): the stage keeps every key.
fn fitsPool(p: Policy, t: usize) bool {
    return (p.pool_blocks == 0 or blocks(p, t) <= p.pool_blocks) and (p.open_block or (t + 1) % p.block == 0);
}

/// Rows of x (`width` wide) stored as NVFP4 under f and read back.
fn rounded(gpa: std.mem.Allocator, f: nvfp4_store.Format, x: []const f64, width: usize) ![]f64 {
    if (width % nvfp4_store.group != 0) return error.BadPolicy;
    const out = try gpa.alloc(f64, x.len);
    errdefer gpa.free(out);
    const row = try gpa.alloc(f32, width);
    defer gpa.free(row);
    const codes = try gpa.alloc(u8, width / 2);
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, width / nvfp4_store.group);
    defer gpa.free(scales);
    for (0..x.len / width) |i| {
        for (row, x[i * width ..][0..width]) |*y, v| y.* = @floatCast(v);
        try nvfp4_store.quantize(f, row, codes, scales);
        nvfp4_store.dequantize(f, codes, scales, out[i * width ..][0..width]);
    }
    return out;
}

/// Block b's summary as scored: from the original or the stored keys (`summary_input`), then stored itself if asked.
fn summaryInto(gpa: std.mem.Allocator, p: Policy, original: Keys, stored: Keys, b: usize, t: usize, out: []f64) !void {
    summarise(p, if (p.summary_input == .original) original else stored, b, t, out);
    const f = p.summary_nvfp4 orelse return;
    const r = try rounded(gpa, f, out, out.len / p.key_heads);
    defer gpa.free(r);
    @memcpy(out, r);
}

/// Block b's summary at t under p's three precisions (token keys, summary, the summary's input), as select scores it.
pub fn blockSummary(gpa: std.mem.Allocator, p: Policy, keys: Keys, b: usize, t: usize, out: []f64) !void {
    const n = (t + 1) * keys.dim;
    const kq = if (p.keys_nvfp4) |f| try rounded(gpa, f, keys.k[0..n], keys.dim / p.key_heads) else null;
    defer if (kq) |x| gpa.free(x);
    try summaryInto(gpa, p, keys, .{ .k = kq orelse keys.k, .dim = keys.dim }, b, t, out);
}

/// Another layer's selection a follower rescores: its pool, for the same sequence and t, and from `owner`.
pub const From = struct { sel: *const Selection, owner: u32 };

/// The query at position t (keys 0..t) under `p`; `pool_from`: rescore that pool instead of searching.
pub fn select(gpa: std.mem.Allocator, p: Policy, q: Query, keys: Keys, t: usize, pool_from: ?From) !Selection {
    return run(gpa, p, q, keys, t, pool_from, true);
}

/// select without its pool shortcut: every stage scored, the path the shortcut must equal (dense is a rule, kept).
pub fn selectStaged(gpa: std.mem.Allocator, p: Policy, q: Query, keys: Keys, t: usize, pool_from: ?From) !Selection {
    return run(gpa, p, q, keys, t, pool_from, false);
}

/// Keys of `a` (ascending) not in `b`: how many picks a change of scores moved across the top-k boundary.
pub fn flips(a: []const u32, b: []const u32) usize {
    var n: usize = 0;
    var j: usize = 0;
    for (a) |x| {
        while (j < b.len and b[j] < x) j += 1;
        if (j == b.len or b[j] != x) n += 1;
    }
    return n;
}

fn run(gpa: std.mem.Allocator, p: Policy, q_in: Query, keys_in: Keys, t: usize, pool_from: ?From, shortcuts: bool) !Selection {
    // a foreign pool (another position's, or not the owner's this layer follows) is refused, not read
    if (pool_from) |f| if (f.sel.t != t or f.sel.owner != f.owner or f.sel.seq != q_in.seq) return error.ForeignPool;
    if (p.topk == 0 or p.block == 0 or p.key_heads == 0 or keys_in.k.len < (t + 1) * keys_in.dim) return error.BadPolicy;
    if (q_in.heads % p.key_heads != 0 or keys_in.dim % p.key_heads != 0) return error.BadPolicy;
    // dense (a policy rule, not a shortcut): when every causal key fits the final budget, all are attended
    if (p.dense and pool_from == null and t + 1 <= budget(p, t)) {
        const all = try gpa.alloc(u32, t + 1);
        errdefer gpa.free(all);
        for (all, 0..) |*x, i| x.* = @intCast(i);
        return .{ .keys = all, .pool = try gpa.dupe(u32, all), .t = t, .owner = q_in.layer, .seq = q_in.seq };
    }
    // scores from NVFP4-stored keys or queries when the policy asks (the rounding's effect on the boundary)
    const kq = if (p.keys_nvfp4) |f| try rounded(gpa, f, keys_in.k[0 .. (t + 1) * keys_in.dim], keys_in.dim / p.key_heads) else null;
    defer if (kq) |x| gpa.free(x);
    const qq = if (p.queries_nvfp4) |f| try rounded(gpa, f, q_in.q, q_in.q.len / q_in.heads) else null;
    defer if (qq) |x| gpa.free(x);
    const keys: Keys = .{ .k = kq orelse keys_in.k, .dim = keys_in.dim };
    var q = q_in;
    if (qq) |x| q.q = x;
    // the candidates: every causal key, the first stage's blocks, or the pool handed down
    var cands: std.ArrayList(u32) = .empty;
    defer cands.deinit(gpa);
    if (pool_from) |f| {
        for (f.sel.pool) |i| if (i <= t) try cands.append(gpa, i);
    } else if (p.block == 1 or (shortcuts and fitsPool(p, t))) {
        // one stage, or every block would be kept: tokens scored directly
        for (0..t + 1) |i| try cands.append(gpa, @intCast(i));
    } else {
        const n_blocks = blocks(p, t);
        const sum = try gpa.alloc(f64, keys.dim);
        defer gpa.free(sum);
        var scored = try gpa.alloc(Scored, n_blocks);
        defer gpa.free(scored);
        for (0..n_blocks) |b| {
            try summaryInto(gpa, p, keys_in, keys, b, t, sum);
            scored[b] = .{ .pos = @intCast(b), .s = score(p, q, sum) };
        }
        std.mem.sort(Scored, scored, {}, before);
        const keep = if (p.pool_blocks == 0) n_blocks else @min(n_blocks, p.pool_blocks);
        for (scored[0..keep]) |s| for (s.pos * p.block..@min((s.pos + 1) * p.block, t + 1)) |i| try cands.append(gpa, @intCast(i));
        // keys of a block left out (the open one) are only reachable as mandatory keys
    }
    std.mem.sort(u32, cands.items, {}, std.sort.asc(u32));
    const pool = try gpa.dupe(u32, cands.items);
    errdefer gpa.free(pool);
    // the second stage: tokens by their own score; mandatory keys apart unless they count inside topk
    var scored: std.ArrayList(Scored) = .empty;
    defer scored.deinit(gpa);
    for (cands.items) |i| if (p.mandatory_in_topk or !mandatory(p, t, i)) {
        try scored.append(gpa, .{ .pos = i, .s = score(p, q, keys.k[i * keys.dim ..][0..keys.dim]) });
    };
    std.mem.sort(Scored, scored.items, {}, before);
    const in = try gpa.alloc(bool, t + 1);
    defer gpa.free(in);
    var left = p.topk;
    var count: usize = 0;
    for (in, 0..) |*x, i| {
        x.* = mandatory(p, t, i);
        if (x.*) count += 1;
        if (x.* and p.mandatory_in_topk) left -|= 1;
    }
    var taken: usize = 0;
    for (scored.items) |s| {
        if (taken == left) break;
        if (in[s.pos]) continue;
        in[s.pos] = true;
        taken += 1;
    }
    const out = try gpa.alloc(u32, count + taken);
    var j: usize = 0;
    for (in, 0..) |x, i| if (x) {
        out[j] = @intCast(i);
        j += 1;
    };
    return .{ .keys = out, .pool = pool, .t = t, .owner = q_in.layer, .seq = q_in.seq };
}

/// What a selection must be: ascending, no repeats, never past t, every mandatory key in, at most the budget.
pub fn check(p: Policy, t: usize, keys: []const u32) error{ Unsorted, Future, MissingMandatory, OverBudget }!void {
    for (keys, 0..) |k, j| {
        if (j > 0 and keys[j - 1] >= k) return error.Unsorted;
        if (k > t) return error.Future;
    }
    for (0..t + 1) |i| if (mandatory(p, t, i)) {
        if (std.sort.binarySearch(u32, keys, @as(u32, @intCast(i)), struct {
            fn cmp(a: u32, b: u32) std.math.Order {
                return std.math.order(a, b);
            }
        }.cmp) == null) return error.MissingMandatory;
    };
    if (keys.len > budget(p, t)) return error.OverBudget;
}

test {
    _ = @import("sparse_index_test.zig");
}
