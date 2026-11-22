//! Inkling 2's HIP attention inputs: Mini's shapes, the rows' streams, the fills of each gate and the FP64 pool.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_hip");

const contract = core.contract;
const ref = contract.inkling2_attn_ref;
const c = contract.spec;
const g = contract.gates;
const D = ref.D;

/// Mini's attention: 24 query heads over 8 key heads (GQA 3:1), n_floor 8,192 (config.json).
pub const heads = 24;
pub const kv = 8;
pub const group = heads / kv;
pub const floor_pos: f64 = 8192;
/// Global keys to 1M wrap a 4,096-slot ring (key p in slot p % ring): no 2 GiB cache needed.
pub const global_ring = 4096;

/// A call's streams: rows of stream i sit at its own positions and its caches `segs[i].off` elements in.
pub const Layout = struct {
    layer: K.Layer,
    rows: usize,
    streams: usize,
    segs: [3]K.Seg,

    /// Stream 0's rows end at the context (k - 1), stream 1's near k / 2, stream 2's near k / 3: far from its end.
    pub fn of(layer: K.Layer, rows: usize, k: usize, streams: usize) Layout {
        var x: Layout = .{ .layer = layer, .rows = rows, .streams = streams, .segs = undefined };
        const cache: i64 = @intCast(@as(usize, layer.ring) * kv * D);
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
};

/// Gate 4's value at key p of column col: period 1,024, its second half the first's negation a key along, + 1 on 8.
fn periodic(p: usize, col: usize) f64 {
    const i = p % 1024;
    if (i < 512) return @floatFromInt(129 + (i * 73 + col * 31) % 127);
    const j = i - 512;
    return -@as(f64, @floatFromInt(129 + ((j + 1) % 512 * 73 + col * 31) % 127)) + @as(f64, if (j < 8) 1 else 0);
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

    pub fn free(d: *Data, a: std.mem.Allocator) void {
        inline for (.{ "q", "kc", "vc", "tab", "sink", "gate", "pos", "stream" }) |f| a.free(@field(d, f));
        d.* = .{};
    }

    /// q, k clipped as prep leaves them, v rms-normed, the table rel . proj rounded once; stress: gate 4's rows.
    pub fn make(d: *Data, a: std.mem.Allocator, lay: Layout, fill: g.Fill, seed: u64) !void {
        const rows = lay.rows;
        const ring: usize = lay.layer.ring;
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
        d.tab = try a.alloc(u16, rows * heads * ref.extent);
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
            for (d.vc[slot * D ..][0..D], x, 0..) |*o, v, dd| o.* = ref.bits(if (stress) periodic(slot % ring, col0 + dd) else v * inv);
        }
        var proj: [ref.d_rel * ref.extent]f64 = undefined;
        for (&proj) |*p| p.* = ref.rn(rng.floatNorm(f64) * 0.5);
        var rel: [ref.d_rel]f64 = undefined;
        for (0..rows * heads) |rh| {
            for (&rel) |*v| v.* = rng.floatNorm(f64);
            const f = ref.clipFactor(&rel, ref.rel_clip);
            for (&rel) |*v| v.* = ref.rn(v.* * f);
            for (d.tab[rh * ref.extent ..][0..ref.extent], 0..) |*o, dd| {
                var acc: f64 = 0;
                for (rel, 0..) |rv, j| acc += rv * proj[j * ref.extent + dd];
                o.* = if (stress) 0 else ref.bits(acc); // the bf16 table: rel . proj rounded once (transformers' bf16 matmul)
            }
            d.sink[rh] = if (stress) 0xff80 else ref.bits(8 + 16 * rng.float(f64)); // a raw logit
            for (d.gate[rh * D ..][0..D]) |*o| o.* = ref.bits(if (stress) 32 else rng.floatNorm(f64) * 2); // raw, pre-sigmoid
        }
        if (fill == .nonfinite) d.plantNonfinite(ring);
    }

    /// NaN q, a +inf sink, both gate infinities, +inf in a key and +inf, -inf in values every row of stream 0 sees.
    fn plantNonfinite(d: *Data, ring: usize) void {
        const nan = c.toBf16(std.math.nan(f32));
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
        d.kc[at(2, p0 % ring, 7, ring)] = inf;
        d.vc[at(1, p0 % ring, 9, ring)] = inf;
        d.vc[at(1, p0 % ring, 10, ring)] = inf;
        if (p0 > 0) d.vc[at(1, (p0 - 1) % ring, 10, ring)] = inf | 0x8000;
    }
};

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
    const l = Layout.of(K.Layer.sliding(1024, 2048), 10, 20000, 3);
    try std.testing.expectEqual([3]i32{ 0, 4, 7 }, [3]i32{ l.segs[0].row0, l.segs[1].row0, l.segs[2].row0 });
    try std.testing.expectEqual(@as(usize, 19999), l.posOf(3));
    try std.testing.expectEqual(@as(usize, 9997), l.posOf(4));
    try std.testing.expectEqual(@as(usize, 2), l.streamOf(9));
    try std.testing.expectEqual(@as(i64, 2 * 3072 * 8 * 128), l.segs[2].off);
    const one = Layout.of(K.Layer.sliding(1024, 2048), 5, 300, 1);
    try std.testing.expectEqual(@as(i32, 5), one.segs[0].n);
    try std.testing.expectEqual(@as(usize, 0), one.streamOf(4));
}
