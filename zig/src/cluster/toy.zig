//! A small model split by heads (tensor parallel) and expert groups (expert parallel) for the cluster's exactness tests.
const std = @import("std");
const canon = @import("canon.zig");
const exchange = @import("exchange.zig");
const round = @import("round.zig");

const Allocator = std.mem.Allocator;

pub const Dims = struct {
    d: u32 = 16,
    heads: u32 = 8,
    hd: u32 = 4,
    layers: u32 = 2,
    experts: u32 = 16,
    top_k: u32 = 2,
    inter: u32 = 8,
    vocab: u32 = 97,
    groups: u32 = 8,
};

/// Weights every rank reads from (each uses only its heads, experts and vocabulary rows).
pub const Weights = struct {
    dims: Dims,
    embed: []f32,
    wq: []f32,
    wk: []f32,
    wv: []f32,
    wo: []f32,
    router: []f32,
    up: []f32,
    down: []f32,
    head: []f32,

    /// Magnitudes vary by head and expert, so a different summation order shows up in the bits.
    pub fn init(a: Allocator, dims: Dims, seed: u64) !Weights {
        var prng: std.Random.DefaultPrng = .init(seed);
        const r = prng.random();
        const D = dims;
        const fill = struct {
            fn f(al: Allocator, rr: std.Random, n: usize, block: usize) ![]f32 {
                const out = try al.alloc(f32, n);
                var scale: f32 = 1;
                for (out, 0..) |*v, i| {
                    if (i % block == 0) scale = std.math.pow(f32, 10, (rr.float(f32) - 0.5) * 2.5);
                    v.* = (rr.float(f32) - 0.5) * scale;
                }
                return out;
            }
        }.f;
        const hdd = D.hd * D.d;
        return .{
            .dims = D,
            .embed = try fill(a, r, D.vocab * D.d, D.d),
            .wq = try fill(a, r, D.layers * D.heads * hdd, hdd),
            .wk = try fill(a, r, D.layers * D.heads * hdd, hdd),
            .wv = try fill(a, r, D.layers * D.heads * hdd, hdd),
            .wo = try fill(a, r, D.layers * D.heads * hdd, hdd),
            .router = try fill(a, r, D.layers * D.experts * D.d, D.experts * D.d),
            .up = try fill(a, r, D.layers * D.experts * D.inter * D.d, D.inter * D.d),
            .down = try fill(a, r, D.layers * D.experts * D.d * D.inter, D.d * D.inter),
            .head = try fill(a, r, D.vocab * D.d, D.vocab * D.d),
        };
    }
};

fn dot(x: []const f32, y: []const f32) f32 {
    var s: f32 = 0;
    for (x, y) |p, q| s += p * q;
    return s;
}

fn rmsnorm(x: []const f32, out: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + 1e-5);
    for (x, out) |v, *o| o.* = v * inv;
}

/// One rank's shard: its heads' caches per slot, its expert groups, its vocabulary rows.
pub const Rank = struct {
    gpa: Allocator,
    w: *const Weights,
    me: u32,
    n: u32,
    heads: canon.Range,
    groups: canon.Range,
    vocab: canon.Range,
    vocabs: []canon.Range,
    /// The control: sum this rank's slices left to right, then ranks in order (depends on the rank count).
    naive: bool = false,
    /// Sums as reduce-scatter then all-gather instead of one-shot (the same canonical tree).
    split_sums: bool = false,
    /// Attention by streams: a slot's owner runs every head of its rows and keeps its caches; rows are then all-gathered.
    by_streams: bool = false,
    /// Every expert split by its intermediate columns (8 slices) instead of whole experts by groups.
    split_experts: bool = false,
    slices: canon.Range = .{},
    caches: std.AutoHashMapUnmanaged(u32, []std.ArrayList(f32)) = .empty,

    pub fn init(gpa: Allocator, w: *const Weights, me: u32, n: u32) !Rank {
        const ones = try gpa.alloc(u64, n);
        defer gpa.free(ones);
        @memset(ones, 1);
        var heads: [64]canon.Range = undefined;
        var groups: [64]canon.Range = undefined;
        const vocabs = try gpa.alloc(canon.Range, n);
        canon.partition(w.dims.heads, ones, heads[0..n]);
        canon.partition(w.dims.groups, ones, groups[0..n]);
        canon.partition(w.dims.vocab, ones, vocabs);
        var slices: [64]canon.Range = undefined;
        canon.partition(w.dims.inter, ones, slices[0..n]);
        return .{ .gpa = gpa, .w = w, .me = me, .n = n, .heads = heads[me], .groups = groups[me], .vocab = vocabs[me], .vocabs = vocabs, .slices = slices[me] };
    }

    pub fn deinit(r: *Rank) void {
        var it = r.caches.valueIterator();
        while (it.next()) |lists| {
            for (lists.*) |*l| l.deinit(r.gpa);
            r.gpa.free(lists.*);
        }
        r.caches.deinit(r.gpa);
        r.gpa.free(r.vocabs);
    }

    fn cache(r: *Rank, slot: u32, layer: u32, head: u32) !*std.ArrayList(f32) {
        const D = r.w.dims;
        const got = try r.caches.getOrPut(r.gpa, slot);
        if (!got.found_existing) {
            got.value_ptr.* = try r.gpa.alloc(std.ArrayList(f32), D.layers * D.heads);
            for (got.value_ptr.*) |*l| l.* = .empty;
        }
        return &got.value_ptr.*[layer * D.heads + head];
    }

    pub fn keep(r: *Rank, slot: u32, len: u32) void {
        const lists = r.caches.get(slot) orelse return;
        for (lists) |*l| if (l.items.len > len * 2 * r.w.dims.hd) l.shrinkRetainingCapacity(len * 2 * r.w.dims.hd);
    }

    pub fn release(r: *Rank, slot: u32) void {
        const kv = r.caches.fetchRemove(slot) orelse return;
        for (kv.value) |*l| l.deinit(r.gpa);
        r.gpa.free(kv.value);
    }

    /// This rank's share of one round; the leader (rank 0) receives every sampled row's full logits in `logits`.
    pub fn forward(r: *Rank, a: Allocator, x: exchange.Exchange, rows: []const round.Row, logits: ?[]f32) !void {
        const D = r.w.dims;
        const R = rows.len;
        const X = try a.alloc(f32, R * D.d);
        const XN = try a.alloc(f32, R * D.d);
        for (rows, 0..) |row, i| @memcpy(X[i * D.d ..][0..D.d], r.w.embed[row.token * D.d ..][0..D.d]);
        for (0..D.layers) |l| {
            const layer: u32 = @intCast(l);
            for (0..R) |i| rmsnorm(X[i * D.d ..][0..D.d], XN[i * D.d ..][0..D.d]);
            if (r.by_streams) {
                try r.attendOwned(a, x, layer, rows, XN, X);
            } else {
                const att = try a.alloc([]f32, r.heads.len());
                for (att, r.heads.begin..) |*part, h| part.* = try r.attend(a, layer, @intCast(h), rows, XN);
                try r.sum(a, x, D.heads, r.heads, att, R, X);
            }
            for (0..R) |i| rmsnorm(X[i * D.d ..][0..D.d], XN[i * D.d ..][0..D.d]);
            const units: u32 = if (r.split_experts) D.inter else D.groups;
            const owned = if (r.split_experts) r.slices else r.groups;
            const moe = try a.alloc([]f32, owned.len());
            for (moe) |*part| part.* = try a.alloc(f32, R * D.d);
            for (moe) |part| @memset(part, 0);
            try r.experts(a, layer, R, XN, moe);
            try r.sum(a, x, units, owned, moe, R, X);
        }
        for (0..R) |i| rmsnorm(X[i * D.d ..][0..D.d], XN[i * D.d ..][0..D.d]);
        var sampled: usize = 0;
        for (rows) |row| sampled += @intFromBool(row.sample);
        const mine = try a.alloc(f32, sampled * r.vocab.len());
        var k: usize = 0;
        for (rows, 0..) |row, i| {
            if (!row.sample) continue;
            for (r.vocab.begin..r.vocab.end, 0..) |v, j| mine[k * r.vocab.len() + j] = dot(r.w.head[v * D.d ..][0..D.d], XN[i * D.d ..][0..D.d]);
            k += 1;
        }
        var got: [exchange.max_ranks][]const u8 = undefined;
        try exchange.gather(x, 0, std.mem.sliceAsBytes(mine), got[0..r.n]);
        const out = logits orelse return;
        for (r.vocabs, got[0..r.n]) |vr, bytes| {
            const part: []align(1) const f32 = @ptrCast(bytes);
            for (0..sampled) |s| for (0..vr.len()) |j| {
                out[s * D.vocab + vr.begin + j] = part[s * vr.len() + j];
            };
        }
    }

    /// Attention by streams: every head for the rows of slots this rank owns, the heads' canonical tree here, rows gathered.
    fn attendOwned(r: *Rank, a: Allocator, x: exchange.Exchange, layer: u32, rows: []const round.Row, XN: []const f32, X: []f32) !void {
        const D = r.w.dims;
        var mine: std.ArrayList(round.Row) = .empty;
        var mine_xn: std.ArrayList(f32) = .empty;
        const owner = try a.alloc(u32, rows.len);
        for (rows, owner, 0..) |row, *o, i| {
            o.* = row.slot % r.n;
            if (o.* != r.me) continue;
            try mine.append(a, row);
            try mine_xn.appendSlice(a, XN[i * D.d ..][0..D.d]);
        }
        const parts = try a.alloc(canon.Part, D.heads);
        for (parts, 0..) |*p, h| p.* = .{ .range = .{ .begin = @intCast(h), .end = @intCast(h + 1) }, .data = try r.attend(a, layer, @intCast(h), mine.items, mine_xn.items) };
        const own = try a.alloc(f32, mine.items.len * D.d);
        canon.reduce(D.heads, parts, own);
        const all = try a.alloc(f32, rows.len * D.d);
        try exchange.gatherRows(x, own, owner, D.d, all);
        for (X, all) |*o, t| o.* += t;
    }

    /// Head `h`'s output projection for every row: cache append at the row's index, causal attention over its slot.
    fn attend(r: *Rank, a: Allocator, layer: u32, h: u32, rows: []const round.Row, XN: []const f32) ![]f32 {
        const D = r.w.dims;
        const hdd = D.hd * D.d;
        const base = (layer * D.heads + h) * hdd;
        const out = try a.alloc(f32, rows.len * D.d);
        var q: [64]f32 = undefined;
        var o: [64]f32 = undefined;
        for (rows, 0..) |row, i| {
            const xn = XN[i * D.d ..][0..D.d];
            const c = try r.cache(row.slot, layer, h);
            if (c.items.len != row.index * 2 * D.hd) return error.PositionMismatch;
            for (0..D.hd) |j| q[j] = dot(r.w.wq[base + j * D.d ..][0..D.d], xn);
            for (0..D.hd) |j| try c.append(r.gpa, dot(r.w.wk[base + j * D.d ..][0..D.d], xn));
            for (0..D.hd) |j| try c.append(r.gpa, dot(r.w.wv[base + j * D.d ..][0..D.d], xn));
            const tokens = row.index + 1;
            const scores = try a.alloc(f32, tokens);
            var m: f32 = -std.math.inf(f32);
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D.hd)));
            for (scores, 0..) |*s, t| {
                s.* = dot(q[0..D.hd], c.items[t * 2 * D.hd ..][0..D.hd]) * scale;
                m = @max(m, s.*);
            }
            var z: f32 = 0;
            for (scores) |*s| {
                s.* = @exp(s.* - m);
                z += s.*;
            }
            @memset(o[0..D.hd], 0);
            for (scores, 0..) |s, t| for (0..D.hd) |j| {
                o[j] += s / z * c.items[t * 2 * D.hd + D.hd + j];
            };
            for (0..D.d) |e| out[i * D.d + e] = dot(r.w.wo[base + e * D.hd ..][0..D.hd], o[0..D.hd]);
        }
        return out;
    }

    /// Each owned group's partial: its picked experts in id order, weighted; the router itself runs on every rank.
    fn experts(r: *Rank, a: Allocator, layer: u32, R: usize, XN: []const f32, moe: [][]f32) !void {
        const D = r.w.dims;
        const per = D.experts / D.groups;
        const scores = try a.alloc(f32, D.experts);
        var hid: [64]f32 = undefined;
        for (0..R) |i| {
            const xn = XN[i * D.d ..][0..D.d];
            for (scores, 0..) |*s, e| s.* = dot(r.w.router[(layer * D.experts + e) * D.d ..][0..D.d], xn);
            var pick: [16]u32 = undefined;
            for (0..D.top_k) |k| {
                var best: ?u32 = null;
                for (0..D.experts) |e| {
                    const id: u32 = @intCast(e);
                    if (std.mem.indexOfScalar(u32, pick[0..k], id) != null) continue;
                    if (best == null or scores[e] > scores[best.?]) best = id;
                }
                pick[k] = best.?;
            }
            var wsum: f32 = 0;
            var wts: [16]f32 = undefined;
            for (pick[0..D.top_k], 0..) |e, k| {
                wts[k] = @exp(scores[e] - scores[pick[0]]);
                wsum += wts[k];
            }
            const order = pick;
            if (r.split_experts) {
                std.mem.sort(u32, pick[0..D.top_k], {}, std.sort.asc(u32));
                for (pick[0..D.top_k]) |e| {
                    const k = std.mem.indexOfScalar(u32, order[0..D.top_k], e).?;
                    const ub = (layer * D.experts + e) * D.inter * D.d;
                    const db = (layer * D.experts + e) * D.d * D.inter;
                    for (r.slices.begin..r.slices.end, 0..) |c, ci| {
                        const h = @max(0, dot(r.w.up[ub + c * D.d ..][0..D.d], xn));
                        for (0..D.d) |j| moe[ci][i * D.d + j] += wts[k] / wsum * (r.w.down[db + j * D.inter + c] * h);
                    }
                }
                continue;
            }
            for (r.groups.begin..r.groups.end, 0..) |g, gi| {
                for (g * per..(g + 1) * per) |e| {
                    const k = std.mem.indexOfScalar(u32, pick[0..D.top_k], @intCast(e)) orelse continue;
                    const ub = (layer * D.experts + e) * D.inter * D.d;
                    for (0..D.inter) |j| hid[j] = @max(0, dot(r.w.up[ub + j * D.d ..][0..D.d], xn));
                    const db = (layer * D.experts + e) * D.d * D.inter;
                    for (0..D.d) |j| moe[gi][i * D.d + j] += wts[k] / wsum * dot(r.w.down[db + j * D.inter ..][0..D.inter], hid[0..D.inter]);
                }
            }
        }
    }

    /// X += the canonical sum of the parts (or, as the control, a rank-order sum).
    fn sum(r: *Rank, a: Allocator, x: exchange.Exchange, units: u32, mine: canon.Range, parts: []const []f32, R: usize, X: []f32) !void {
        const D = r.w.dims;
        const total = try a.alloc(f32, R * D.d);
        if (!r.naive) {
            const views = try a.alloc([]const f32, parts.len);
            for (views, parts) |*v, p| v.* = p;
            if (r.split_sums) {
                const block = try a.alloc(f32, R * D.d);
                const cols = try exchange.reduceScatter(a, x, units, mine, views, R, D.d, block);
                try exchange.allGather(x, block[0 .. R * cols.len()], R, D.d, total);
            } else try exchange.allReduce(a, x, units, mine, views, R, D.d, total);
        } else {
            const own = try a.alloc(f32, R * D.d);
            @memset(own, 0);
            for (parts) |p| for (own, p) |*o, v| {
                o.* += v;
            };
            var got: [exchange.max_ranks][]const u8 = undefined;
            var chunks: [exchange.max_ranks][]const u8 = undefined;
            for (chunks[0..r.n]) |*c| c.* = std.mem.sliceAsBytes(own);
            try x.exchange(chunks[0..r.n], got[0..r.n]);
            @memset(total, 0);
            for (got[0..r.n]) |g| {
                const v: []align(1) const f32 = @ptrCast(g);
                for (total, v) |*t, y| t.* += y;
            }
        }
        for (X, total) |*o, t| o.* += t;
    }
};
