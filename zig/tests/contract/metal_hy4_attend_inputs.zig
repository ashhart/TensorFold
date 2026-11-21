//! Hy4-flash's MLA attention inputs: slot lists over a shuffled page table, the gates' fills and float64's ranges.
const std = @import("std");
const contract = @import("contract");
const h = @import("metal_hy4.zig");
const fl = h.fl;
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const Cache = h.hy.attention_slots.Cache;

/// A slot's 576 words: the 512-word latent (both key and value) and the 64 rope words (key only).
pub const lat = h.hy.mla_weights.latent;
pub const key = h.hy.mla_weights.absorbed_stride;
pub const chunk = h.hy.mla_attention.chunk_keys;
pub const window = h.hy.serve_metal.window;
pub const recalled = h.hy.metal_index.keys_a_row;
pub const draft_stride = h.hy.serve_metal.draft_stride;
pub const draft_rows = h.hy.serve_metal.max_window;
const scale: f64 = h.hy.mla_attention.score_scale;
/// A key's columns and one more: the sink is a key of value 0 whose score is q's extra 1 against sink / scale.
const E = key + 1;
const page_tokens = 64;
const nan16: u16 = 0x7fc0;

/// Whose slot lists: a sliding layer's window, a sparse layer's (every key to 2,048, then recalled), DFly2's rows.
pub const Lists = enum { window, sparse, draft };

/// The key stride a pass binds: the window, a sparse pass's keys to a whole window or its recall, the draft's.
pub fn strideOf(l: Lists, end: usize) usize {
    return switch (l) {
        .window => window,
        .sparse => if (end > recalled) recalled else (end + window - 1) / window * window,
        .draft => draft_stride,
    };
}

/// fp32 steps: 23 a score, precise::exp a weight, a rescale a key and the merge's; o 2 a key, l 1, then the merge.
fn form(keys: usize, chunks: usize) ar.Form {
    const k: f64 = @floatFromInt(keys);
    const n: f64 = @floatFromInt(chunks);
    return .{ .score = 23, .fast_exp = false, .chunk_exp = false, .p_rel = 0, .num = 2 * k + n + 2, .den = k + n + 2, .exp_ulps = .{ 4 * (k + 2), 2 } };
}

/// Gate 4's values by residue: 127 and 16,256 in turn, the second half negated, four nudged; a window sums to 4.
fn stressValue(rho: usize) f32 {
    const at = rho % 256;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    const one: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
    return if (rho < 256) big + one else -big;
}

pub const Inputs = struct {
    a: std.mem.Allocator,
    rows: usize = 0,
    heads: usize = 0,
    stride: usize = 0,
    fill: g.Fill = .random,
    /// [rows][heads][576] absorbed queries, [slots + 1][576] the cache (the last slot all NaN), [rows][stride] lists.
    q: []u16 = &.{},
    cache: []u16 = &.{},
    lists: []u32 = &.{},
    counts: []u32 = &.{},
    sinks: []f32 = &.{},
    /// Float64's outputs and how far the kernel's fp32 steps may move each, made when a gate first asks.
    point: []f64 = &.{},
    half: []f64 = &.{},

    pub fn deinit(in: *Inputs) void {
        inline for (.{ "q", "cache", "lists", "counts", "sinks", "point", "half" }) |f| in.a.free(@field(in, f));
        in.* = .{ .a = in.a };
    }

    pub fn outputs(in: *const Inputs) usize {
        return in.rows * in.heads * lat;
    }

    fn list(in: *const Inputs, r: usize) []u32 {
        return in.lists[r * in.stride ..][0..in.stride];
    }

    /// Whether slot `s` is a 512-key chunk's first key in any row's list.
    fn leads(in: *const Inputs, s: u32) bool {
        for (0..in.rows) |r| {
            var i: usize = 0;
            while (i < in.counts[r]) : (i += chunk) if (in.list(r)[i] == s) return true;
        }
        return false;
    }

    /// A slot of row `r`'s list `back` keys before its last that leads no chunk in any list (null: none).
    fn safe(in: *const Inputs, r: usize, back: usize) ?u32 {
        var i = in.counts[r] -| (1 + back);
        while (true) : (i -= 1) {
            if (!in.leads(in.list(r)[i])) return in.list(r)[i];
            if (i == 0) return null;
        }
    }

    /// 2,048 distinct positions below `end`, a few dropped (-1) as DFly2's reuse does; gate 4's the last, in order.
    fn recall(in: *const Inputs, pick: []i32, all: []i32, end: usize, rng: std.Random) void {
        if (in.fill == .stress) {
            for (pick, 0..) |*x, i| x.* = @intCast(end - recalled + i);
            return;
        }
        for (all[0..end], 0..) |*x, i| x.* = @intCast(i);
        for (pick, 0..) |*x, i| {
            const j = i + rng.uintLessThan(usize, end - i);
            std.mem.swap(i32, &all[i], &all[j]);
            x.* = if (rng.uintLessThan(u32, 64) == 0) -1 else all[i];
        }
    }

    /// Each row's list through the page table, as attention_slots.zig gives the engine's.
    fn place(in: *Inputs, l: Lists, t: Cache, first: usize, recall_all: bool, rng: std.Random) !void {
        const all = try in.a.alloc(i32, first + in.rows + recalled);
        defer in.a.free(all);
        const pick = all[first + in.rows ..][0..recalled];
        const draft_recall = l == .draft and first >= recalled;
        if (draft_recall) in.recall(pick, all, first, rng);
        const own = try in.a.alloc(i32, recalled + draft_rows);
        defer in.a.free(own);
        for (0..in.rows) |r| {
            const end: u32 = @intCast(first + r + 1);
            const out = in.list(r);
            in.counts[r] = switch (l) {
                .window => try t.sliding(end, window, out),
                .sparse => if (recall_all) blk: {
                    in.recall(pick, all, end, rng);
                    break :blk try t.retained(end, pick, out);
                } else try t.sliding(end, end, out),
                // past 2,048 the anchor's recall below the window, then the window's own rows to this one
                .draft => if (draft_recall) blk: {
                    @memcpy(own[0..recalled], pick);
                    for (own[recalled..][0 .. r + 1], 0..) |*x, i| x.* = @intCast(first + i);
                    break :blk try t.retained(end, own[0 .. recalled + r + 1], out);
                } else try t.sliding(end, end, out),
            };
        }
    }

    /// NaN, +Inf and -Inf in a query, keys rows 1 to 3 see and two sinks; `first_inf` adds -Inf at row 4's first key.
    fn plant(in: *Inputs, first_inf: bool) void {
        const inf = fl.inf32;
        if (in.heads > 1) in.q[(0 * in.heads + 1) * key + 3] = c.toBf16(fl.nan32);
        const at = [_]struct { row: usize, back: usize, col: usize, v: f32 }{
            .{ .row = 1, .back = 1, .col = 5, .v = inf },
            .{ .row = 2, .back = 5, .col = lat + 8, .v = -inf },
            .{ .row = 3, .back = 9, .col = 7, .v = inf },
            .{ .row = 3, .back = 13, .col = 9, .v = -inf },
        };
        for (at) |p| if (p.row < in.rows) {
            const s = in.safe(p.row, p.back) orelse continue;
            in.cache[s * key + p.col] = c.toBf16(p.v);
            // other rows meet it at a score of -Inf, so there it drops out but for its own latent column's 0 x Inf
            for (0..in.rows) |r| if (r != p.row) for (0..in.heads) |hd| {
                const w = &in.q[(r * in.heads + hd) * key + p.col];
                w.* = if (p.v > 0) w.* | 0x8000 else w.* & 0x7fff;
            };
        };
        if (in.heads > 3) {
            in.sinks[2] = fl.nan32;
            in.sinks[3] = -inf;
        }
        if (!first_inf or in.rows < 5) return;
        in.cache[in.list(4)[0] * key + lat + 8] = c.toBf16(-inf);
        for (0..in.rows * in.heads) |rh| in.q[rh * key + lat + 8] = c.toBf16(1);
    }

    /// Float64's outputs and half widths (attention_range.head per row and head), rows split over threads.
    pub fn ranges(in: *Inputs) !void {
        if (in.point.len != 0) return;
        in.point = try in.a.alloc(f64, in.outputs());
        in.half = try in.a.alloc(f64, in.outputs());
        const threads = contract.reference_threads;
        var jobs: [threads.maximum]Job = undefined;
        const k = threads.count(threads.maximum, in.rows);
        for (jobs[0..k], 0..) |*j, i| j.* = .{ .in = in, .first = i, .step = k };
        threads.run(Job, jobs[0..k], .normal);
        for (jobs[0..k]) |j| if (j.err) |e| return e;
    }

    const Job = struct {
        in: *Inputs,
        first: usize,
        step: usize,
        err: ?anyerror = null,

        pub fn run(j: *Job) void {
            j.work() catch |e| {
                j.err = e;
            };
        }

        fn work(j: *Job) !void {
            const in = j.in;
            var most: usize = 0;
            for (in.counts) |n| most = @max(most, n);
            most += 1;
            const buf = try in.a.alloc(f64, 2 * most * E + 2 * most + 3 * E);
            defer in.a.free(buf);
            const kv = 2 * most * E;
            var r = j.first;
            while (r < in.rows) : (r += j.step) in.row(r, buf[0 .. most * E], buf[most * E .. kv], buf[kv..][0..most], buf[kv + most ..][0..most], buf[kv + 2 * most ..][0 .. 3 * E]);
        }
    };

    /// One row's keys gathered once, then each head through attention_range.head with its sink as one more key.
    fn row(in: *Inputs, r: usize, kf: []f64, vf: []f64, w: []f64, tau: []f64, rest: []f64) void {
        const qf = rest[0..E];
        const pt = rest[E .. 2 * E];
        const hf = rest[2 * E ..];
        const n = in.counts[r];
        for (in.list(r)[0..n], 0..) |s, j| {
            const k = kf[j * E ..][0..E];
            const v = vf[j * E ..][0..E];
            for (k[0..key], in.cache[s * key ..][0..key]) |*x, word| x.* = c.fromBf16(word);
            k[key] = 0;
            @memcpy(v[0..lat], k[0..lat]);
            @memset(v[lat..], 0);
        }
        const f = form(@min(n, chunk), (in.stride - 1) / chunk + 1);
        for (0..in.heads) |hd| {
            const sink: f64 = in.sinks[hd];
            const m = if (std.math.isNegativeInf(sink)) n else n + 1;
            @memset(kf[n * E ..][0..E], 0);
            @memset(vf[n * E ..][0..E], 0);
            kf[n * E + key] = sink / scale;
            for (qf[0..key], in.q[(r * in.heads + hd) * key ..][0..key]) |*x, word| x.* = c.fromBf16(word);
            qf[key] = 1;
            ar.head(qf, kf[0 .. m * E], vf[0 .. m * E], m, E, scale, f, w[0..m], tau[0..m], pt, hf);
            const at = (r * in.heads + hd) * lat;
            @memcpy(in.point[at..][0..lat], pt[0..lat]);
            @memcpy(in.half[at..][0..lat], hf[0..lat]);
        }
    }

    /// Gate 4's sums: every score 0 and the sinks -Inf, so each output is its row's listed values' mean, rounded.
    pub fn simulate(in: *const Inputs, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (in.fill != .stress) return error.StressRequiresEqualScores;
        const terms = try in.a.alloc(f64, in.stride);
        defer in.a.free(terms);
        for (0..in.rows) |r| {
            const n = in.counts[r];
            for (0..lat) |e| {
                for (terms[0..n], in.list(r)[0..n]) |*t, s| t.* = c.fromBf16(in.cache[s * key + e]);
                const o = fl.bf(contract.rows.sumIn(acc, order, terms[0..n]) / @as(f64, @floatFromInt(n)));
                for (0..in.heads) |hd| out[(r * in.heads + hd) * lat + e] = o;
            }
        }
    }
};

/// `shape.rows` rows of `shape.n` heads, the last at position `shape.k` - 1; null where the lists cannot serve it.
pub fn build(a: std.mem.Allocator, l: Lists, shape: c.Shape, fill: g.Fill, seed: u64, first_inf: bool) !?Inputs {
    const rows = shape.rows;
    const end = shape.k;
    if (rows == 0 or shape.n == 0 or shape.n > 64 or end < rows) return null;
    const first = end - rows;
    const recall_all = l == .sparse and end > recalled;
    if (recall_all and first + 1 < recalled) return null;
    if (l == .draft and rows > draft_rows) return null;
    var in: Inputs = .{ .a = a, .rows = rows, .heads = shape.n, .stride = strideOf(l, end), .fill = fill };
    errdefer in.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const pages = (end + page_tokens - 1) / page_tokens;
    const bases = try a.alloc(u32, pages);
    defer a.free(bases);
    for (bases, 0..) |*b, i| b.* = @intCast(i * page_tokens);
    rng.shuffle(u32, bases);
    const table: Cache = .{ .page_bases = bases, .page_tokens = page_tokens, .capacity = @intCast(pages * page_tokens) };
    // every list entry past a row's count names the last slot, and every slot no list names stays NaN
    const poison = pages * page_tokens;
    in.cache = try a.alloc(u16, (poison + 1) * key);
    @memset(in.cache, nan16);
    in.lists = try a.alloc(u32, rows * in.stride);
    @memset(in.lists, @intCast(poison));
    in.counts = try a.alloc(u32, rows);
    in.q = try a.alloc(u16, rows * shape.n * key);
    in.sinks = try a.alloc(f32, shape.n);
    try in.place(l, table, first, recall_all, rng);
    const seen = try a.alloc(bool, poison);
    defer a.free(seen);
    @memset(seen, false);
    for (0..rows) |r| for (in.list(r)[0..in.counts[r]]) |s| {
        seen[s] = true;
    };
    for (0..end) |x| {
        const s = try table.slot(@intCast(x));
        if (!seen[s]) continue;
        for (in.cache[s * key ..][0..key], 0..) |*w, e| w.* = c.toBf16(if (fill != .stress) rng.float(f32) * 2 - 1 else if (e < lat) stressValue((x + e) % window) else 0);
    }
    for (in.q) |*x| x.* = if (fill == .stress) 0 else c.toBf16(rng.float(f32) * 4 - 2);
    for (in.sinks) |*x| x.* = if (fill == .stress) -fl.inf32 else rng.float(f32) * 4 - 2;
    if (fill == .nonfinite) in.plant(first_inf);
    return in;
}
