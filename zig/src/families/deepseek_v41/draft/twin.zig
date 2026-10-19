//! A CPU twin of the lanes' model path: a one-layer attention `Target` and a DSpark-shaped `Pass` on the same vtables.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const iface = @import("iface.zig");
const dspark = @import("dspark.zig");

extern "c" fn exp(x: f64) f64;

pub const Dims = struct { vocab: u32 = 48, dim: u32 = 16, seed: u64 = 1, bigram: f32 = 6.0, head: f32 = 0.12, conf: f32 = 0.03, window: u32 = 16, block: u32 = 5 };

const Slot = struct {
    k: std.ArrayList(f32) = .empty, // committed rows' keys [len][dim]
    v: std.ArrayList(f32) = .empty,
    base: usize = 0, // rows before the last window
    rows: std.ArrayList(f32) = .empty, // the last window's keys, values and taps [rows][3 dim]
    parents: std.ArrayList(i32) = .empty,
    taps: std.ArrayList(f32) = .empty,
    live: bool = false,

    fn deinit(s: *Slot, gpa: Allocator) void {
        s.k.deinit(gpa);
        s.v.deinit(gpa);
        s.rows.deinit(gpa);
        s.parents.deinit(gpa);
        s.taps.deinit(gpa);
    }

    fn len(s: Slot, d: usize) usize {
        return s.k.items.len / d;
    }
};

pub const Twin = struct {
    gpa: Allocator,
    dims: Dims,
    emb: []f32,
    wq: []f32,
    wk: []f32,
    wv: []f32,
    wo: []f32,
    bigram: []f32,
    slots: []Slot,
    windows: u64 = 0, // rows' passes run (tests)
    widest: usize = 0, // the most rows one forward held

    pub fn init(gpa: Allocator, dims: Dims, slots: usize) !*Twin {
        var prng = std.Random.DefaultPrng.init(dims.seed);
        const r = prng.random();
        const v: usize = dims.vocab;
        const d: usize = dims.dim;
        const t = try gpa.create(Twin);
        t.* = .{ .gpa = gpa, .dims = dims, .emb = try normal(gpa, r, v * d, 1.0), .wq = try normal(gpa, r, d * d, 0.5), .wk = try normal(gpa, r, d * d, 0.5), .wv = try normal(gpa, r, d * d, 0.5), .wo = try normal(gpa, r, d * v, dims.head), .bigram = try normal(gpa, r, v * v, dims.bigram), .slots = try gpa.alloc(Slot, slots) };
        for (t.slots) |*s| s.* = .{};
        return t;
    }

    pub fn deinit(t: *Twin) void {
        for (t.slots) |*s| s.deinit(t.gpa);
        for ([_][]f32{ t.emb, t.wq, t.wk, t.wv, t.wo, t.bigram }) |x| t.gpa.free(x);
        t.gpa.free(t.slots);
        t.gpa.destroy(t);
    }

    fn normal(gpa: Allocator, r: std.Random, n: usize, scale: f32) ![]f32 {
        const out = try gpa.alloc(f32, n);
        for (out) |*o| o.* = r.floatNorm(f32) * scale;
        return out;
    }

    /// x = emb[token] + a position signal.
    pub fn input(t: *const Twin, token: u32, pos: u64, x: []f32) void {
        const d = t.dims.dim;
        for (x, t.emb[@as(usize, token) * d ..][0..d], 0..) |*o, e, j| o.* = e + 0.5 * @sin(@as(f32, @floatFromInt(pos % 4096)) / @as(f32, @floatFromInt(j + 1)));
    }

    pub fn project(t: *const Twin, w: []const f32, x: []const f32, out: []f32) void {
        const d = t.dims.dim;
        @memset(out, 0.0);
        for (x, 0..) |xi, i| {
            for (out, w[i * d ..][0..d]) |*o, wij| o.* += xi * wij;
        }
    }

    /// Attention of query `q` over keys/values (row-major [n][dim]) in their order, added to `h`.
    pub fn attend(t: *const Twin, q: []const f32, keys: []const []const f32, values: []const []const f32, h: []f32) void {
        const d = t.dims.dim;
        var m: f32 = -std.math.inf(f32);
        for (keys) |k| m = @max(m, dot(q, k) / @sqrt(@as(f32, @floatFromInt(d))));
        var sum: f32 = 0.0;
        for (keys) |k| sum += @floatCast(exp(@as(f64, dot(q, k) / @sqrt(@as(f32, @floatFromInt(d))) - m)));
        for (keys, values) |k, v| {
            const a: f32 = @as(f32, @floatCast(exp(@as(f64, dot(q, k) / @sqrt(@as(f32, @floatFromInt(d))) - m)))) / sum;
            for (h, v) |*o, vi| o.* += a * vi;
        }
    }

    pub fn logits(t: *const Twin, h: []const f32, prev: u32, out: []f32) void {
        const v = t.dims.vocab;
        @memset(out, 0.0);
        for (h, 0..) |hi, i| {
            for (out, t.wo[i * v ..][0..v]) |*o, w| o.* += hi * w;
        }
        for (out, t.bigram[@as(usize, prev) * v ..][0..v]) |*o, b| o.* += b;
    }

    /// The keyed choice over the whole vocabulary (greedy: value desc, id asc).
    pub fn choose(t: *const Twin, scratch: Allocator, row: []const f32, draw: u64, s: ?lanes.Sampling) !u32 {
        const smp = s orelse {
            var best: usize = 0;
            for (row, 0..) |x, i| if (x > row[best]) {
                best = i;
            };
            return @intCast(best);
        };
        const vals = try scratch.alloc(f64, row.len);
        defer scratch.free(vals);
        const ids = try scratch.alloc(u64, row.len);
        defer scratch.free(ids);
        for (vals, ids, row, 0..) |*o, *id, x, i| {
            o.* = x;
            id.* = i;
        }
        _ = t;
        return @intCast(try lanes.sampling.choose(scratch, vals, ids, draw, smp));
    }

    fn rowsOf(t: *Twin, s: *Slot, tokens: []const u32, parents: ?[]const i32, start: u64, draws: []const u64, smp: ?lanes.Sampling, choices: []u32) !void {
        const d = t.dims.dim;
        const a = t.gpa;
        s.rows.clearRetainingCapacity();
        s.taps.clearRetainingCapacity();
        s.parents.clearRetainingCapacity();
        const n = tokens.len;
        try s.rows.resize(a, n * 2 * d);
        try s.taps.resize(a, n * d);
        for (0..n) |r| try s.parents.append(a, if (parents) |p| p[r] else @as(i32, @intCast(r)) - 1);
        const committed = s.len(d);
        const keys = try a.alloc([]const f32, committed + n);
        defer a.free(keys);
        const vals = try a.alloc([]const f32, committed + n);
        defer a.free(vals);
        for (0..committed) |i| {
            keys[i] = s.k.items[i * d ..][0..d];
            vals[i] = s.v.items[i * d ..][0..d];
        }
        var x: [64]f32 = undefined;
        var q: [64]f32 = undefined;
        const row = try a.alloc(f32, t.dims.vocab);
        defer a.free(row);
        var chain: [64]usize = undefined;
        for (0..n) |r| {
            var depth: usize = 0;
            var at: i32 = @intCast(r);
            while (at >= 0) : (at = s.parents.items[@intCast(at)]) {
                chain[depth] = @intCast(at);
                depth += 1;
            }
            const kr = s.rows.items[r * 2 * d ..][0..d];
            const vr = s.rows.items[r * 2 * d + d ..][0..d];
            const h = s.taps.items[r * d ..][0..d];
            t.input(tokens[r], start + depth - 1, x[0..d]);
            t.project(t.wq, x[0..d], q[0..d]);
            t.project(t.wk, x[0..d], kr);
            t.project(t.wv, x[0..d], vr);
            for (0..depth) |i| { // ancestors root first, then the row itself
                const anc = chain[depth - 1 - i];
                keys[committed + i] = s.rows.items[anc * 2 * d ..][0..d];
                vals[committed + i] = s.rows.items[anc * 2 * d + d ..][0..d];
            }
            @memcpy(h, x[0..d]);
            t.attend(q[0..d], keys[0 .. committed + depth], vals[0 .. committed + depth], h);
            t.logits(h, tokens[r], row);
            choices[r] = try t.choose(a, row, draws[r], smp);
        }
        s.base = committed;
        for (0..n) |r| { // every row stays until keep (a chain that is never kept keeps them all)
            try s.k.appendSlice(a, s.rows.items[r * 2 * d ..][0..d]);
            try s.v.appendSlice(a, s.rows.items[r * 2 * d + d ..][0..d]);
        }
        t.windows += 1;
    }

    pub fn target(t: *Twin) iface.Target {
        return .{ .ptr = t, .vtable = &.{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release } };
    }

    fn self(ptr: *anyopaque) *Twin {
        return @ptrCast(@alignCast(ptr));
    }

    fn prefill(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self(ptr);
        const s = &t.slots[slot];
        if (s.live) return error.SlotLive;
        s.k.clearRetainingCapacity();
        s.v.clearRetainingCapacity();
        s.live = true;
        const draws = try t.gpa.alloc(u64, ids.len);
        defer t.gpa.free(draws);
        for (draws, 0..) |*w, i| w.* = i + 1;
        draws[ids.len - 1] = draw;
        const choices = try t.gpa.alloc(u32, ids.len);
        defer t.gpa.free(choices);
        try t.rowsOf(s, ids, null, 0, draws, smp, choices);
        return choices[ids.len - 1];
    }

    fn window(ptr: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self(ptr);
        var rows: usize = 0;
        for (segments) |g| rows += g.tokens.len;
        t.widest = @max(t.widest, rows);
        for (segments, choices) |g, c| {
            const s = &t.slots[g.slot];
            if (!s.live or g.start != s.len(t.dims.dim)) return error.PositionMismatch;
            try t.rowsOf(s, g.tokens, g.parents, g.start, g.draws, g.sampling, c);
        }
    }

    fn keep(ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self(ptr);
        const s = &t.slots[slot];
        const d = t.dims.dim;
        for (path[1..], path[0 .. path.len - 1]) |r, parent| if (s.parents.items[r] != @as(i32, @intCast(parent))) return error.NotAPath;
        s.k.shrinkRetainingCapacity(s.base * d);
        s.v.shrinkRetainingCapacity(s.base * d);
        for (path) |r| {
            try s.k.appendSlice(t.gpa, s.rows.items[r * 2 * d ..][0..d]);
            try s.v.appendSlice(t.gpa, s.rows.items[r * 2 * d + d ..][0..d]);
        }
    }

    /// Drop the slot's last window without committing a row (branches.zig: a chain whose branch lost).
    pub fn drop(t: *Twin, slot: u32) void {
        const s = &t.slots[slot];
        s.k.shrinkRetainingCapacity(s.base * t.dims.dim);
        s.v.shrinkRetainingCapacity(s.base * t.dims.dim);
    }

    fn taps(ptr: *anyopaque, slot: u32) iface.Taps {
        const t = self(ptr);
        const s = &t.slots[slot];
        return .{ .address = @intFromPtr(s.taps.items.ptr), .rows = @intCast(s.taps.items.len / t.dims.dim), .stride = t.dims.dim };
    }

    fn release(ptr: *anyopaque, slot: u32) void {
        const t = self(ptr);
        t.slots[slot].live = false;
    }

    /// The serial reference without lanes: the prompt, then one row a step, each drawn at its own position.
    pub fn serial(t: *Twin, slot: u32, prompt: []const u32, n: usize, smp: ?lanes.Sampling, eos: ?u32) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(t.gpa);
        var tok = try prefill(t, slot, prompt, smp, prompt.len);
        while (out.items.len < n) {
            try out.append(t.gpa, tok);
            if (eos != null and tok == eos.?) break;
            const at: u64 = prompt.len + out.items.len - 1;
            var c: [1]u32 = undefined;
            var cs = [_][]u32{&c};
            try window(t, &.{.{ .slot = slot, .start = at, .tokens = &.{tok}, .parents = null, .draws = &.{at + 1}, .sampling = smp }}, &cs);
            tok = c[0];
        }
        release(t, slot);
        return out.toOwnedSlice(t.gpa);
    }
};

fn dot(a: []const f32, b: []const f32) f32 {
    var s: f32 = 0.0;
    for (a, b) |x, y| s += x * y;
    return s;
}

/// The twin's DSpark: rings of target taps, a non-causal block over them, the target head, the bigram chain rule.
pub const Drafter = struct {
    gpa: Allocator,
    t: *Twin,
    shape: dspark.Shape,
    rings: [][]f32, // slot -> [ring][dim]
    tags: [][]i64, // slot -> the position each ring row holds (-1: none)
    w1: []u16,
    w2: []u16,
    cw: []f32,
    passes: u64 = 0,
    ingests: u64 = 0, // ingest calls (tests)
    /// `begin` / `collect` (passAsync): the pass reads rings at `begin`, the host takes results at `collect`; all slots
    stash_drafts: []u32,
    stash_conf: []f32,
    stash_cand: []i32,
    stash_base: []f32,
    stash_n: usize = 0,
    begun: bool = false,

    pub fn init(gpa: Allocator, t: *Twin, slots: usize) !*Drafter {
        const v = t.dims.vocab;
        const rank = std.math.ceilPowerOfTwo(u32, v) catch unreachable;
        const shape: dspark.Shape = .{ .block = t.dims.block, .noise = v - 1, .window = t.dims.window, .rank = rank, .hidden = t.dims.dim, .candidates = v };
        const x = try gpa.create(Drafter);
        x.* = .{ .gpa = gpa, .t = t, .shape = shape, .rings = try gpa.alloc([]f32, slots), .tags = try gpa.alloc([]i64, slots), .w1 = try gpa.alloc(u16, @as(usize, v) * rank), .w2 = try gpa.alloc(u16, @as(usize, v) * rank), .cw = try gpa.alloc(f32, t.dims.dim + rank), .stash_drafts = try gpa.alloc(u32, slots * shape.block), .stash_conf = try gpa.alloc(f32, slots * shape.block), .stash_cand = try gpa.alloc(i32, slots * shape.block * v), .stash_base = try gpa.alloc(f32, slots * shape.block * v) };
        const ring = shape.ring();
        for (x.rings, x.tags) |*r, *g| {
            r.* = try gpa.alloc(f32, @as(usize, ring) * t.dims.dim);
            g.* = try gpa.alloc(i64, ring);
            @memset(g.*, -1);
        }
        @memset(x.w1, 0);
        for (0..v) |p| x.w1[p * rank + p] = dspark.toBf16(1.0);
        for (0..v) |id| for (0..rank) |r| {
            x.w2[id * rank + r] = if (r < v) dspark.toBf16(t.bigram[r * v + id]) else 0;
        };
        var prng = std.Random.DefaultPrng.init(t.dims.seed ^ 0x5eed);
        for (x.cw) |*c| c.* = prng.random().floatNorm(f32) * t.dims.conf;
        return x;
    }

    pub fn deinit(x: *Drafter) void {
        for (x.rings, x.tags) |r, g| {
            x.gpa.free(r);
            x.gpa.free(g);
        }
        x.gpa.free(x.rings);
        x.gpa.free(x.tags);
        x.gpa.free(x.w1);
        x.gpa.free(x.w2);
        x.gpa.free(x.cw);
        x.gpa.free(x.stash_drafts);
        x.gpa.free(x.stash_conf);
        x.gpa.free(x.stash_cand);
        x.gpa.free(x.stash_base);
        x.gpa.destroy(x);
    }

    pub fn markovHeads(x: *const Drafter) dspark.Markov {
        return .{ .w1 = x.w1, .w2 = x.w2, .rank = x.shape.rank, .conf = x.cw };
    }

    pub fn pass(x: *Drafter) iface.Pass {
        return .{ .ptr = x, .vtable = &.{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markovFn } };
    }

    /// The same pass with `begin` / `collect` (iface.Pass): the speculative pass's device path on the CPU.
    pub fn passAsync(x: *Drafter) iface.Pass {
        return .{ .ptr = x, .vtable = &.{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markovFn, .begin = begin, .collect = collect } };
    }

    fn stashed(x: *Drafter, i: usize) iface.Proposal {
        const b = x.shape.block;
        const v = x.t.dims.vocab;
        return .{ .drafts = x.stash_drafts[i * b ..][0..b], .conf = x.stash_conf[i * b ..][0..b], .cand = x.stash_cand[i * b * v ..][0 .. b * v], .base = x.stash_base[i * b * v ..][0 .. b * v] };
    }

    fn begin(ptr: *anyopaque, asks: []const iface.Ask) anyerror!void {
        const x = self(ptr);
        if (x.begun) return error.PassBegun;
        var props: [16]iface.Proposal = undefined;
        if (asks.len > props.len or asks.len * x.shape.block > x.stash_drafts.len) return error.TooManySlots;
        for (props[0..asks.len], 0..) |*p, i| p.* = x.stashed(i);
        try propose(ptr, asks, props[0..asks.len]);
        x.stash_n = asks.len;
        x.begun = true;
    }

    fn collect(ptr: *anyopaque, out: []iface.Proposal) anyerror!void {
        const x = self(ptr);
        if (!x.begun or out.len != x.stash_n) return error.NoPassBegun;
        x.begun = false;
        for (out, 0..) |o, i| {
            const p = x.stashed(i);
            @memcpy(o.drafts, p.drafts[0..o.drafts.len]);
            @memcpy(o.conf, p.conf[0..o.conf.len]);
            if (o.cand) |c| @memcpy(c, p.cand.?[0..c.len]);
            if (o.base) |b| @memcpy(b, p.base.?[0..b.len]);
        }
    }

    fn self(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn ingest(ptr: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) anyerror!void {
        const x = self(ptr);
        const d = x.t.dims.dim;
        const ring = x.shape.ring();
        if (x.begun) return error.PassBegun; // the lanes collect a begun pass before any other pass work
        x.ingests += 1;
        const src: [*]const f32 = @ptrFromInt(taps.address);
        for (rows, 0..) |r, i| {
            if (r >= taps.rows) return error.TapOutOfRange;
            const pos = start + i;
            @memcpy(x.rings[slot][(pos % ring) * d ..][0..d], src[@as(usize, r) * taps.stride ..][0..d]);
            x.tags[slot][pos % ring] = @intCast(pos);
        }
    }

    fn reset(ptr: *anyopaque, slot: u32) void {
        @memset(self(ptr).tags[slot], -1);
    }

    fn markovFn(ptr: *anyopaque) ?dspark.Markov {
        return self(ptr).markovHeads();
    }

    fn propose(ptr: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(ptr);
        if (x.begun) return error.PassBegun;
        const t = x.t;
        const d = t.dims.dim;
        const n = x.shape.block;
        const v = t.dims.vocab;
        const a = x.gpa;
        const ring = x.shape.ring();
        x.passes += 1;
        var ids: [16]u32 = undefined;
        var xs: [16][64]f32 = undefined;
        var ks: [16][64]f32 = undefined;
        var vs: [16][64]f32 = undefined;
        const hid = try a.alloc(f32, n * d);
        defer a.free(hid);
        const cand = try a.alloc(i32, n * v);
        defer a.free(cand);
        const base = try a.alloc(f32, n * v);
        defer a.free(base);
        const keys = try a.alloc([]const f32, x.shape.window + n);
        defer a.free(keys);
        const vals = try a.alloc([]const f32, x.shape.window + n);
        defer a.free(vals);
        var kbuf = try a.alloc(f32, (x.shape.window) * 2 * d);
        defer a.free(kbuf);
        for (asks, out) |ask, o| {
            const ctx = dspark.Context.of(x.shape, ask.start, 0);
            for (0..ctx.count) |i| { // the ring must hold exactly the committed rows before P
                const pos = ctx.first + i;
                if (x.tags[ask.slot][pos % ring] != @as(i64, @intCast(pos))) return error.RingStale;
                const row = x.rings[ask.slot][(pos % ring) * d ..][0..d];
                t.project(t.wk, row, kbuf[i * 2 * d ..][0..d]);
                t.project(t.wv, row, kbuf[i * 2 * d + d ..][0..d]);
                keys[i] = kbuf[i * 2 * d ..][0..d];
                vals[i] = kbuf[i * 2 * d + d ..][0..d];
            }
            dspark.blockIds(x.shape, ask.anchor, ids[0..n]);
            for (0..n) |i| {
                t.input(ids[i], ask.start + i, xs[i][0..d]);
                t.project(t.wk, xs[i][0..d], ks[i][0..d]);
                t.project(t.wv, xs[i][0..d], vs[i][0..d]);
                keys[ctx.count + i] = ks[i][0..d];
                vals[ctx.count + i] = vs[i][0..d];
            }
            for (0..n) |i| {
                var q: [64]f32 = undefined;
                t.project(t.wq, xs[i][0..d], q[0..d]);
                const h = hid[i * d ..][0..d];
                @memcpy(h, xs[i][0..d]);
                t.attend(q[0..d], keys[0 .. ctx.count + n], vals[0 .. ctx.count + n], h);
                const row = base[i * v ..][0..v];
                @memset(row, 0.0);
                for (h, 0..) |hi, j| {
                    for (row, t.wo[j * v ..][0..v]) |*r, w| r.* += hi * w;
                }
                for (cand[i * v ..][0..v], 0..) |*c, id| c.* = @intCast(id);
            }
            if (o.cand) |c| @memcpy(c[0 .. n * v], cand);
            if (o.base) |b| @memcpy(b[0 .. n * v], base);
            try dspark.chain(a, x.markovHeads(), cand, base, v, ask.anchor, ask.params, hid, o.drafts[0..n], o.conf[0..n]);
        }
    }
};
