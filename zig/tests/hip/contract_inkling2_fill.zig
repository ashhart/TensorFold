//! Inkling 2's HIP attention inputs: each checkpoint's geometry, the rows' streams, each gate's fills, the FP64 pool.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_hip");

const contract = core.contract;
const ref = contract.inkling2_attn_ref;
const c = contract.spec;
const g = contract.gates;
const D = ref.D;

/// A checkpoint's attention: query and key heads, the local window (a local layer's extent), rel_extent, n_floor.
pub const Model = struct {
    heads: usize,
    kv: usize,
    window: u32,
    rel_extent: usize,
    floor: f64,

    pub fn group(m: Model) usize {
        return m.heads / m.kv;
    }

    /// A layer's bias extent (config.zig's extent(i)): the window on a local layer, rel_extent on a global one.
    pub fn extent(m: Model, layer: K.Layer) usize {
        return if (layer.window > 0) layer.window else m.rel_extent;
    }
};

/// Inkling 2 Mini: 24 query heads over 8 key heads (GQA 3:1), window and rel_extent 1,024, n_floor 8,192 (config.json).
pub const mini: Model = .{ .heads = 24, .kv = 8, .window = 1024, .rel_extent = 1024, .floor = 8192 };
/// tiny-inkling2-mini@ee242a48: 4 query heads over 2 key heads (GQA 2:1), window 128, rel_extent 256, n_floor 8,192.
pub const tiny: Model = .{ .heads = 4, .kv = 2, .window = 128, .rel_extent = 256, .floor = 8192 };
/// Global keys to 1M wrap a 4,096-slot ring (key p in slot p % ring): no 2 GiB cache needed.
pub const global_ring = 4096;

/// Whether `m` is tiny-inkling2-mini's geometry: its registrations take their own names and shapes.
pub fn isTiny(m: Model) bool {
    return std.meta.eql(m, tiny);
}

/// A call's streams: rows of stream i sit at its own positions and its caches `segs[i].off` elements in.
pub const Layout = struct {
    model: Model,
    layer: K.Layer,
    rows: usize,
    streams: usize,
    segs: [3]K.Seg,

    /// Stream 0's rows end at the context (k - 1), stream 1's near k / 2, stream 2's near k / 3: far from its end.
    pub fn of(model: Model, layer: K.Layer, rows: usize, k: usize, streams: usize) Layout {
        var x: Layout = .{ .model = model, .layer = layer, .rows = rows, .streams = streams, .segs = undefined };
        const cache: i64 = @intCast(@as(usize, layer.ring) * model.kv * D);
        var row0: usize = 0;
        for (0..3) |i| {
            const n = if (i >= streams) 0 else if (i > 0) rows / streams else rows - (streams - 1) * (rows / streams);
            const last = if (i == 0) k else @max(n, k / (i + 1));
            x.segs[i] = .{ .row0 = @intCast(row0), .n = @intCast(n), .p0 = @intCast(last - n), .off = cache * @as(i64, @intCast(i)) };
            row0 += n;
        }
        return x;
    }

    pub fn streamOf(x: *const Layout, row: usize) usize {
        var st: usize = 0;
        while (st + 1 < x.streams and row >= @as(usize, @intCast(x.segs[st + 1].row0))) st += 1;
        return st;
    }

    pub fn posOf(x: *const Layout, row: usize) usize {
        const s = x.segs[x.streamOf(row)];
        return @as(usize, @intCast(s.p0)) + row - @as(usize, @intCast(s.row0));
    }

    /// The layer's bias extent: a table row's distances in the reference's layout.
    pub fn extent(x: *const Layout) usize {
        return x.model.extent(x.layer);
    }
};

/// Gate 4's value at key p of column col: its second half the first's negation a key along, + 1 on 8.
fn periodic(p: usize, col: usize, period: usize) f64 {
    const half = period / 2;
    const i = p % period;
    if (i < half) return @floatFromInt(scale(i, period) * (129 + (i * 73 + col * 31) % 127));
    const j = i - half;
    const k = (j + 1) % half;
    return -@as(f64, @floatFromInt(scale(k, period) * (129 + (k * 73 + col * 31) % 127))) + @as(f64, if (j < 8) 1 else 0);
}

/// A window shorter than 1,024 keys: the second 32 of every 64 are 32 times larger, so 32-lane fp16 sums lose bits too.
fn scale(i: usize, period: usize) usize {
    return if (period < 1024 and i % 64 >= 32) 32 else 1;
}

/// One call's host inputs, as prep leaves them for the attention launches.
pub const Data = struct {
    q: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    tab: []u16 = &.{},
    sink: []u16 = &.{},
    gate: []u16 = &.{},
    pos: []usize = &.{},
    stream: []usize = &.{},
    ext: usize = ref.extent, // tab's distances a (row, head): the layer's extent

    pub const makeExactP = fillExactP;

    pub fn free(d: *Data, a: std.mem.Allocator) void {
        inline for (.{ "q", "kc", "vc", "tab", "sink", "gate", "pos", "stream" }) |f| a.free(@field(d, f));
        d.* = .{};
    }

    /// The device's table: each (row, head)'s `ext` distances, then zeros to the kernels' row of K.table_row.
    pub fn padded(d: *const Data, a: std.mem.Allocator) ![]u16 {
        const n = d.tab.len / d.ext;
        const t = try a.alloc(u16, n * K.table_row);
        for (0..n) |rh| {
            @memcpy(t[rh * K.table_row ..][0..d.ext], d.tab[rh * d.ext ..][0..d.ext]);
            @memset(t[rh * K.table_row + d.ext ..][0 .. K.table_row - d.ext], 0);
        }
        return t;
    }

    /// q, k clipped as prep leaves them, v rms-normed, the table rel . proj rounded once; stress: gate 4's rows.
    pub fn make(d: *Data, a: std.mem.Allocator, lay: Layout, fill: g.Fill, seed: u64) !void {
        const rows = lay.rows;
        const ring: usize = lay.layer.ring;
        const heads = lay.model.heads;
        const kv = lay.model.kv;
        const ext = lay.extent();
        // gate 4's values cancel over a period: a local layer's window, else 1,024 keys
        const period: usize = if (lay.layer.window > 0) lay.layer.window else 1024;
        d.ext = ext;
        d.pos = try a.alloc(usize, rows);
        d.stream = try a.alloc(usize, rows);
        for (0..rows) |r| {
            d.stream[r] = lay.streamOf(r);
            d.pos[r] = lay.posOf(r);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const stress = fill == .stress;
        d.q = try a.alloc(u16, rows * heads * D);
        d.kc = try a.alloc(u16, lay.streams * kv * ring * D);
        d.vc = try a.alloc(u16, d.kc.len);
        d.tab = try a.alloc(u16, rows * heads * ext);
        d.sink = try a.alloc(u16, rows * heads);
        d.gate = try a.alloc(u16, rows * heads * D);
        const amp: f64 = if (fill == .nonfinite) 0.3 else 3; // non-finite rows: no p underflows, so 0 x inf never meets
        var x: [D]f64 = undefined;
        for (0..rows * heads) |rh| {
            for (&x) |*v| v.* = if (stress) 0 else rng.floatNorm(f64) * amp;
            const f = ref.clipFactor(&x, ref.qk_clip);
            for (d.q[rh * D ..][0..D], x) |*o, v| o.* = ref.bits(v * f); // q/k L2 clip, then one bf16 rounding
        }
        for (0..lay.streams * kv * ring) |slot| {
            for (&x) |*v| v.* = rng.floatNorm(f64) * 2.5;
            const f = ref.clipFactor(&x, ref.qk_clip);
            for (d.kc[slot * D ..][0..D], x) |*o, v| o.* = ref.bits(v * f);
            var ss: f64 = 0;
            for (&x) |*v| {
                v.* = rng.floatNorm(f64);
                ss += v.* * v.*;
            }
            const inv = 1 / @sqrt(ss / D + ref.eps);
            const col0 = (slot / ring % kv) * D;
            for (d.vc[slot * D ..][0..D], x, 0..) |*o, v, dd| o.* = ref.bits(if (stress) periodic(slot % ring, col0 + dd, period) else v * inv);
        }
        var proj_all: [ref.d_rel * ref.extent]f64 = undefined;
        const proj = proj_all[0 .. ref.d_rel * ext]; // rel_logits_proj [16][extent]
        for (proj) |*p| p.* = ref.rn(rng.floatNorm(f64) * 0.5);
        var rel: [ref.d_rel]f64 = undefined;
        for (0..rows * heads) |rh| {
            for (&rel) |*v| v.* = rng.floatNorm(f64);
            const f = ref.clipFactor(&rel, ref.rel_clip);
            for (&rel) |*v| v.* = ref.rn(v.* * f);
            for (d.tab[rh * ext ..][0..ext], 0..) |*o, dd| {
                var acc: f64 = 0;
                for (rel, 0..) |rv, j| acc += rv * proj[j * ext + dd];
                o.* = if (stress) 0 else ref.bits(acc); // the bf16 table: rel . proj rounded once (transformers' bf16 matmul)
            }
            d.sink[rh] = if (stress) 0xff80 else ref.bits(8 + 16 * rng.float(f64)); // a raw logit
            for (d.gate[rh * D ..][0..D]) |*o| o.* = ref.bits(if (stress) 32 else rng.floatNorm(f64) * 2); // raw, pre-sigmoid
        }
        if (fill == .nonfinite) d.plantNonfinite(ring, heads, kv);
    }

    /// NaN q, a +inf sink, both gate infinities, +inf in a key and +inf, -inf in values every row of stream 0 sees.
    fn plantNonfinite(d: *Data, ring: usize, heads: usize, kv: usize) void {
        const nan = c.toBf16(std.math.nan(f32));
        if (d.pos.len > 3) {
            d.sink[3 * heads + 4 % heads] = nan;
            d.gate[(3 * heads + 5 % heads) * D + 7] = nan;
        }
        const inf = c.toBf16(std.math.inf(f32));
        const rows = d.pos.len;
        const p0 = d.pos[0];
        const at = struct {
            fn f(kvh: usize, slot: usize, dd: usize, r: usize) usize {
                return (kvh * r + slot) * D + dd;
            }
        }.f;
        d.q[(0 * heads + 1) * D + 5] = nan;
        if (rows > 1) d.sink[1 * heads + 2] = inf;
        if (rows > 2) {
            d.gate[(2 * heads + 3) * D + 0] = inf;
            d.gate[(2 * heads + 3) * D + 1] = inf | 0x8000;
        }
        d.kc[at(2 % kv, p0 % ring, 7, ring)] = inf;
        d.vc[at(1, p0 % ring, 9, ring)] = inf;
        d.vc[at(1, p0 % ring, 10, ring)] = inf;
        if (p0 > 0) d.vc[at(1, (p0 - 1) % ring, 10, ring)] = inf | 0x8000;
    }
};

/// The device's score of a bias with q = 0: (0 scale + bias) log2e, rounded once in fp32 (tile.hpp's score).
fn score(b: u16) f32 {
    const log2e: f32 = 1.4426950408889634;
    return @as(f32, @floatCast(ref.bf(b))) * log2e;
}

/// bf16(exp2f(d)) when certain: the exact power sits over 2^-21 (4 fp32 ulp, past exp2f's error) from a tie.
fn roundedP(d: f32) ?f64 {
    const e = std.math.exp2(@as(f64, d));
    const lo = ref.rn(e * (1 - 0x1p-21));
    return if (lo == ref.rn(e * (1 + 0x1p-21))) lo else null;
}

/// Exact-P rows (one 64-key tile, pos < 64): bias 2 at distance 0 (the max), the rest drawn where bf16 P is certain.
fn fillExactP(d: *Data, a: std.mem.Allocator, lay: Layout, seed: u64) !void {
    try d.make(a, lay, .random, seed);
    var prng = std.Random.DefaultPrng.init(seed ^ 0x9e3779b97f4a7c15);
    const rng = prng.random();
    @memset(d.q, 0); // every score is its bias: q.k = 0
    @memset(d.sink, 0xff80); // no sink, so the merge's rescale is exp2(0) = 1
    @memset(d.gate, ref.bits(32)); // sigmoid(32) is 1 in fp32
    for (d.vc) |*o| o.* = ref.bits(@floatFromInt(rng.intRangeAtMost(i32, -16, 16)));
    const top = ref.bits(2);
    for (0..d.sink.len) |rh| {
        const t = d.tab[rh * d.ext ..][0..d.ext];
        t[0] = top;
        for (t[1..]) |*o| { // biases in [-1, 1.75]: every P in [2^-5, 1), its bf16 lsb at least 2^-12
            o.* = ref.bits(-1 + 2.75 * rng.float(f64));
            while (roundedP(score(o.*) - score(top)) == null) o.* = ref.bits(-1 + 2.75 * rng.float(f64));
        }
    }
}

/// An exact-P row's outputs: sum bf16(P) v and sum bf16(P) are exact in fp32 (23 bits), so out = bf16(fp32(o / l)).
pub fn exactP(call: *const ref.Call, r: usize, h: usize, out: []f64) void {
    const pos = call.pos[r];
    const lo = if (call.window > 0) (pos + 1) -| call.window else 0;
    const t = call.tab[(r * call.heads + h) * call.extent ..][0..call.extent];
    const base = (call.stream[r] * call.kv + h / (call.heads / call.kv)) * call.ring;
    var o: [D]f64 = @splat(0);
    var l: f64 = 0;
    for (lo..pos + 1) |p| {
        const w = roundedP(score(t[pos - p]) - score(t[0])).?;
        l += w;
        for (&o, call.v[(base + p % call.ring) * D ..][0..D]) |*x, v| x.* += w * ref.bf(v);
    }
    for (out, o) |*y, x| y.* = c.fromBf16(c.toBf16(@as(f32, @floatCast(x)) / @as(f32, @floatCast(l))));
}

/// The reference's (row, head) pairs (gate 4: (row, key head) pairs) over up to six threads, each its own buffers.
pub const Pool = struct {
    call: *const ref.Call,
    want: []f64,
    tol: []f64,
    sim: ?struct { acc: c.Accum, order: g.Order } = null,

    const Job = struct {
        pool: *const Pool,
        first: usize,
        step: usize,
        items: usize,
        ks: *ref.Keys,
        p: []f64,
        col: []f64,

        pub fn run(j: *Job) void {
            const pool = j.pool;
            const heads = pool.call.heads;
            const kv = pool.call.kv;
            const group = heads / kv;
            var i = j.first;
            while (i < j.items) : (i += j.step) {
                if (pool.sim) |s| {
                    const r = i / kv;
                    const h0 = (i % kv) * group;
                    const o = pool.want[(r * heads + h0) * D ..][0..D];
                    ref.simulate(pool.call, r, h0, s.acc, s.order, j.p, j.col, o);
                    for (1..group) |e| @memcpy(pool.want[(r * heads + h0 + e) * D ..][0..D], o);
                } else {
                    const at = i * D;
                    ref.row(pool.call, i / heads, i % heads, j.ks, pool.want[at..][0..D], pool.tol[at..][0..D]);
                }
            }
        }
    };

    pub fn run(a: std.mem.Allocator, pool: Pool, items: usize) !void {
        const n = contract.reference_threads.count(6, items);
        var jobs: [contract.reference_threads.maximum]Job = undefined;
        var keys: usize = 0;
        for (pool.call.pos) |p| keys = @max(keys, p + 1);
        var made: usize = 0;
        defer for (jobs[0..made]) |j| {
            a.destroy(j.ks);
            a.free(j.p);
            a.free(j.col);
        };
        for (jobs[0..n], 0..) |*j, w| {
            const big = if (pool.sim != null) keys else 0;
            j.* = .{ .pool = &pool, .first = w, .step = n, .items = items, .ks = try a.create(ref.Keys), .p = try a.alloc(f64, big), .col = try a.alloc(f64, big) };
            made += 1;
        }
        contract.reference_threads.run(Job, jobs[0..n], .normal);
    }
};

test "HIP inkling2 layout: three streams' rows, positions and cache offsets" {
    const l = Layout.of(mini, K.Layer.sliding(1024, 2048), 10, 20000, 3);
    try std.testing.expectEqual([3]i32{ 0, 4, 7 }, [3]i32{ l.segs[0].row0, l.segs[1].row0, l.segs[2].row0 });
    try std.testing.expectEqual(@as(usize, 19999), l.posOf(3));
    try std.testing.expectEqual(@as(usize, 9997), l.posOf(4));
    try std.testing.expectEqual(@as(usize, 2), l.streamOf(9));
    try std.testing.expectEqual(@as(i64, 2 * 3072 * 8 * 128), l.segs[2].off);
    const one = Layout.of(mini, K.Layer.sliding(1024, 2048), 5, 300, 1);
    try std.testing.expectEqual(@as(i32, 5), one.segs[0].n);
    try std.testing.expectEqual(@as(usize, 0), one.streamOf(4));
    const small = Layout.of(tiny, K.Layer.sliding(128, 2048), 10, 20000, 3); // tiny-mini: 2 key heads, a 2,176-slot ring
    try std.testing.expectEqual(@as(u32, 2176), small.layer.ring);
    try std.testing.expectEqual(@as(i64, 2 * 2176 * 2 * 128), small.segs[2].off);
}

test "HIP inkling2 tiny-mini tables: 128 local and 256 global distances of bias, then zeros on the device's 1,024" {
    const a = std.testing.allocator;
    const global: K.Layer = .{ .window = 0, .ring = global_ring, .chunk = K.global_chunk, .tau = true };
    for ([_]K.Layer{ K.Layer.sliding(128, 2048), global }, [_]usize{ 128, 256 }) |layer, ext| {
        const lay = Layout.of(tiny, layer, 3, 300, 1);
        try std.testing.expectEqual(ext, lay.extent());
        var d: Data = .{};
        defer d.free(a);
        try d.make(a, lay, .random, 5);
        try std.testing.expectEqual(@as(usize, 3 * 4 * ext), d.tab.len);
        const t = try d.padded(a);
        defer a.free(t);
        try std.testing.expectEqual(@as(usize, 3 * 4 * K.table_row), t.len);
        for (0..3 * 4) |rh| {
            try std.testing.expectEqualSlices(u16, d.tab[rh * ext ..][0..ext], t[rh * K.table_row ..][0..ext]);
            for (t[rh * K.table_row + ext ..][0 .. K.table_row - ext]) |v| try std.testing.expectEqual(@as(u16, 0), v);
            try std.testing.expect(std.mem.indexOfNone(u16, d.tab[rh * ext ..][0..ext], &.{0}) != null);
        }
    }
    const lay = Layout.of(mini, global, 3, 300, 1); // Mini: the device's table is the reference's
    var d: Data = .{};
    defer d.free(a);
    try d.make(a, lay, .random, 5);
    const t = try d.padded(a);
    defer a.free(t);
    try std.testing.expectEqualSlices(u16, d.tab, t);
}

test "HIP inkling2 exact P: the outputs follow bf16 P's bits, and an unrounded P moves over a quarter of them" {
    const a = std.testing.allocator;
    for ([_]Model{ mini, tiny }, [_]u32{ 1024, 128 }) |m, window| {
        const heads = m.heads;
        const lay = Layout.of(m, K.Layer.sliding(window, 2048), 16, 64, 2);
        var d: Data = .{};
        defer d.free(a);
        try d.makeExactP(a, lay, 7);
        const call: ref.Call = .{ .q = d.q, .k = d.kc, .v = d.vc, .tab = d.tab, .sink = d.sink, .gate = d.gate, .pos = d.pos, .stream = d.stream, .ring = lay.layer.ring, .window = lay.layer.window, .extent = d.ext, .heads = heads, .kv = m.kv, .chain = .hip };
        var want: [D]f64 = undefined;
        var moved: usize = 0;
        for (0..16 * heads) |rh| {
            const r = rh / heads;
            const h = rh % heads;
            exactP(&call, r, h, &want);
            const pos = d.pos[r];
            const t = d.tab[rh * d.ext ..][0..d.ext];
            const base = (d.stream[r] * m.kv + h / m.group()) * lay.layer.ring;
            var o: [D]f64 = @splat(0);
            var l: f64 = 0;
            for (0..pos + 1) |p| { // the same rows with P left unrounded
                const w = std.math.exp2(@as(f64, score(t[pos - p]) - score(t[0])));
                l += w;
                for (&o, d.vc[(base + p % lay.layer.ring) * D ..][0..D]) |*x, v| x.* += w * ref.bf(v);
            }
            for (want, o) |x, y| moved += @intFromBool(x != c.fromBf16(c.toBf16(@floatCast(y / l))));
        }
        try std.testing.expect(moved * 4 > 16 * heads * D); // Mini: 19,232 of 49,152 at this seed
    }
}

test "HIP inkling2 tiny-mini stress: a 128-key window's fp32 sums are exact in every order, fp16 and bf16 miss" {
    const a = std.testing.allocator;
    const lay = Layout.of(tiny, K.Layer.sliding(128, 2048), 1, 256, 1); // keys 128..255: one whole period
    var d: Data = .{};
    defer d.free(a);
    try d.make(a, lay, .stress, 3);
    const call: ref.Call = .{ .q = d.q, .k = d.kc, .v = d.vc, .tab = d.tab, .sink = d.sink, .gate = d.gate, .pos = d.pos, .stream = d.stream, .ring = lay.layer.ring, .window = lay.layer.window, .extent = d.ext, .exact = true, .heads = tiny.heads, .kv = tiny.kv, .chain = .hip };
    var want: [4 * D]f64 = undefined;
    var tol: [4 * D]f64 = undefined;
    var got: [4 * D]f64 = undefined;
    try Pool.run(a, .{ .call = &call, .want = &want, .tol = &tol }, 4);
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |o| {
        try Pool.run(a, .{ .call = &call, .want = &got, .tol = &.{}, .sim = .{ .acc = .f32, .order = o } }, 2);
        for (got, want, tol) |x, w, t| try std.testing.expect(@abs(x - w) <= t);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try Pool.run(a, .{ .call = &call, .want = &got, .tol = &.{}, .sim = .{ .acc = acc, .order = o } }, 2);
            var worst: f64 = 0;
            for (got, want, tol) |x, w, t| worst = @max(worst, if (std.math.isFinite(x)) @abs(x - w) / t else std.math.inf(f64));
            try std.testing.expect(worst >= 100);
        }
    }
}
