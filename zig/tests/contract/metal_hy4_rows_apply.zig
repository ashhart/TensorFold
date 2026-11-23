//! engram.metal's tf_hy4_engram_apply (the Engram branch added to the four streams) against float64 at its bf16 points.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const text = @import("metal_hy4_rows_text.zig");

const c = contract.spec;
const g = contract.gates;
const bf = h.fl.bf;
const Range = h.fl.Range;
const u = h.fl.u32_ulp;

const streams = h.hy.ihc.streams;
const threads = 128;
const eps = 0x1p-23;

/// A thread's strided terms (fma for the squares), simd_sum over 32 lanes (31 adds at worst), the 4 simdgroups' pairs.
fn depth(s: c.Shape) usize {
    return (s.n + threads - 1) / threads + 31 + 2;
}

/// Rows: tokens; n: the hidden width, each row's four streams written. DFly2's window first, a decode row, a chunk.
const shapes = [_]c.Shape{ .{ .rows = h.window_rows, .n = h.hidden, .k = 1 }, .{ .rows = 1, .n = h.hidden, .k = 1 }, .{ .rows = h.prompt_rows, .n = h.hidden, .k = 1 } };

/// metal::max keeps the number over a NaN; float64's formula keeps the NaN.
fn maxp(a: f64, b: f64) f64 {
    return if (std.math.isNan(a) or std.math.isNan(b)) std.math.nan(f64) else @max(a, b);
}

fn inv(ss: f64, n: usize) f64 {
    return 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + eps);
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// The gate's input sign(v) bf16(sqrt(max(|v|, bf16(1e-6)))), 0 at 0; `e` moves sqrt toward x's low or high end.
fn root(v: f64, e: f64) f64 {
    if (v == 0) return 0;
    const r = @sqrt(maxp(@abs(v), bf(1e-6)));
    return if (v > 0) bf(r * (1 + e)) else -bf(r * (1 - e));
}

/// The gate's NaN finding: a NaN dot (from a non-finite query or key) leaves the gate finite.
const nan_dot = c.Known{ .gate = .nonfinite, .finding = "a NaN dot (a stream's non-finite state or key) gives a finite gate where float64 gives NaN: max(fabs(scaled), bf(1e-6)) is metal::max, which drops the NaN, so the stream's other outputs stay finite" };

/// An accumulator and order the four sums are replayed in (gate 4).
const Sim = struct { acc: c.Accum, order: g.Order };

fn total(sim: ?Sim, terms: []const f64) f64 {
    if (sim) |m| return contract.rows.sumIn(m.acc, m.order, terms);
    var s: f64 = 0;
    for (terms) |x| s += x;
    return s;
}

pub const Apply = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    /// The finding's own entry: its plants go in the queries and keys.
    dots: bool,
    fault: ?text.Fault,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// state [rows][streams][n], keys [streams][rows][n], value [rows][n]; the gains [streams][n] (value's [n]).
    state: []u16 = &.{},
    keys: []u16 = &.{},
    key_bias: []u16 = &.{},
    key_gain: []u16 = &.{},
    query_gain: []u16 = &.{},
    delta_gain: []u16 = &.{},
    value: []u16 = &.{},
    value_bias: []u16 = &.{},
    value_gain: []u16 = &.{},
    media: []u32 = &.{},
    work: []f64 = &.{},
    ins: [10]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    const fields = .{ "state", "keys", "key_bias", "key_gain", "query_gain", "delta_gain", "value", "value_bias", "value_gain", "media", "work" };

    pub fn base(dots: bool) []const u8 {
        return if (dots) "metal hy4 engram tf_hy4_engram_apply nan dot" else "metal hy4 engram tf_hy4_engram_apply";
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, dots: bool, fault: ?text.Fault) !Apply {
        const src = try text.engram(a, fault);
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try text.pipeline(device, src, "tf_hy4_engram_apply"), .dots = dots, .fault = fault };
    }
    pub fn deinit(t: *Apply) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Apply) void {
        inline for (fields) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }

    pub fn kernel(t: *Apply) g.Kernel {
        t.name = h.fl.named(base(t.dots), if (t.fault) |f| f.drops else null, &t.label);
        const s: c.Spec = .{
            .name = t.name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .known = if (t.dots) nan_dot else null,
            // the hidden width against the group's 128 threads
            .tiles = .{ .{}, .{ .size = threads, .step = 32, .min = 32 }, .{} },
            .served = &shapes,
        };
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.fault != null) .shapes else null };
    }
    const vt = g.hooks(Apply);

    pub fn prepare(t: *Apply, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != 1) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.state = try t.a.alloc(u16, rows * streams * n);
        t.keys = try t.a.alloc(u16, streams * rows * n);
        inline for (.{ "key_bias", "key_gain", "query_gain", "delta_gain" }) |f| @field(t, f) = try t.a.alloc(u16, streams * n);
        t.value = try t.a.alloc(u16, rows * n);
        t.value_bias = try t.a.alloc(u16, n);
        t.value_gain = try t.a.alloc(u16, n);
        t.media = try t.a.alloc(u32, rows);
        t.work = try t.a.alloc(f64, 6 * n);
        const R = struct {
            fn put(r: std.Random, xs: []u16, lo: f32, hi: f32) void {
                for (xs) |*v| v.* = c.toBf16(lo + (hi - lo) * r.float(f32));
            }
            /// Integers lo..hi of either sign: squares sum exactly in fp32 and fp16 sums of them overflow.
            fn ints(r: std.Random, xs: []u16, lo: u32, hi: u32) void {
                for (xs) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(lo + r.uintLessThan(u32, hi - lo + 1))) * @as(f32, if (r.boolean()) 1 else -1));
            }
        };
        if (fill == .stress) {
            // keys and values 20-40, queries 1-3, unit gains: each square sum is under 2^23 and exact in fp32
            R.ints(rng, t.keys, 20, 40);
            R.ints(rng, t.value, 20, 40);
            // normed keys and queries lie in [0.4, 2), so the dot's bf16 terms are multiples of 2^-10 under 2^13 in all
            R.ints(rng, t.state, 1, 3);
            inline for (.{ "key_bias", "value_bias" }) |f| @memset(@field(t, f), 0);
            inline for (.{ "key_gain", "query_gain", "value_gain" }) |f| @memset(@field(t, f), c.toBf16(1));
            @memset(t.delta_gain, c.toBf16(0.5));
            @memset(t.media, 0);
        } else {
            R.put(rng, t.keys, -2, 2);
            R.put(rng, t.state, -2, 2);
            R.put(rng, t.value, -2, 2);
            R.put(rng, t.key_bias, -0.1, 0.1);
            R.put(rng, t.value_bias, -0.1, 0.1);
            inline for (.{ "key_gain", "query_gain", "value_gain" }) |f| R.put(rng, @field(t, f), 0.5, 1.5);
            R.put(rng, t.delta_gain, -0.5, 0.5);
            // a media row leaves its streams as they were (every fifth row)
            for (t.media, 0..) |*m, r| m.* = @intFromBool(r % 5 == 4);
        }
        if (fill == .nonfinite and rows >= 5) t.plant();
        const bufs = [_][]const u8{
            std.mem.sliceAsBytes(t.state),    std.mem.sliceAsBytes(t.keys),       std.mem.sliceAsBytes(t.key_bias),
            std.mem.sliceAsBytes(t.key_gain), std.mem.sliceAsBytes(t.query_gain), std.mem.sliceAsBytes(t.delta_gain),
            std.mem.sliceAsBytes(t.value),    std.mem.sliceAsBytes(t.value_bias), std.mem.sliceAsBytes(t.value_gain),
            std.mem.sliceAsBytes(t.media),
        };
        for (bufs, &t.ins) |b, *r| r.* = try t.rig.input(b);
        t.out = try t.rig.output(rows * streams * n * 2);
        return true;
    }

    /// Main entry: values (a row each, row 3 both infinities) and a media row's state; the finding's: queries and keys.
    fn plant(t: *Apply) void {
        const n = t.shape.n;
        const rows = t.shape.rows;
        const nan = c.toBf16(h.fl.planted[0]);
        const inf = c.toBf16(h.fl.planted[1]);
        const ninf = c.toBf16(h.fl.planted[2]);
        if (!t.dots) {
            t.value[0 * n + 7] = nan;
            t.value[1 * n + n / 2] = inf;
            t.value[2 * n + n - 1] = ninf;
            t.value[3 * n] = inf;
            t.value[3 * n + n - 1] = ninf;
            t.state[(4 * streams + 0) * n + 1] = nan;
            t.state[(4 * streams + 1) * n + 2] = inf;
            t.state[(4 * streams + 2) * n + 3] = ninf;
            return;
        }
        t.state[(0 * streams + 0) * n + 7] = nan;
        t.keys[(2 * rows + 1) * n + n / 2] = inf;
        t.state[(2 * streams + 1) * n + n - 1] = ninf;
        t.keys[(3 * rows + 3) * n] = inf;
        t.keys[(3 * rows + 3) * n + n - 1] = ninf;
    }

    /// metal_engram.zig's forward: a group of 128 threads for each stream and row, (rows, hidden) as its uint2.
    pub fn run(t: *Apply) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 10);
        p.e.setValue([2]u32{ @intCast(t.shape.rows), @intCast(t.shape.n) }, 11);
        p.e.dispatchGroups(mtl.Size.of(streams, t.shape.rows, 1), mtl.Size.of(threads, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Apply) usize {
        return t.shape.rows * streams * t.shape.n;
    }

    fn at(xs: []const u16, i: usize) f64 {
        return c.fromBf16(xs[i]);
    }

    /// The (row, stream)'s key, query and value inputs as the kernel forms them (bf16 sums with their biases).
    fn inputs(t: *const Apply, r: usize, s: usize) struct { k: []f64, q: []f64, v: []f64 } {
        const n = t.shape.n;
        const k = t.work[0..n];
        const q = t.work[n .. 2 * n];
        const v = t.work[2 * n .. 3 * n];
        for (k, q, v, 0..) |*kj, *qj, *vj, j| {
            kj.* = bf(at(t.keys, (s * t.shape.rows + r) * n + j) + at(t.key_bias, s * n + j));
            qj.* = at(t.state, (r * streams + s) * n + j);
            vj.* = bf(at(t.value, r * n + j) + at(t.value_bias, j));
        }
        return .{ .k = k, .q = q, .v = v };
    }

    fn squares(t: *const Apply, xs: []const f64) []f64 {
        const sq = t.work[3 * t.shape.n .. 4 * t.shape.n];
        for (sq, xs) |*o, x| o.* = x * x;
        return sq;
    }

    /// The (row, stream)'s outputs at the kernel's bf16 points, in float64 or with its sums replayed (`sim`).
    fn point(t: *const Apply, r: usize, s: usize, sim: ?Sim, out: []f64) void {
        const n = t.shape.n;
        if (t.media[r] != 0) {
            for (out, 0..) |*o, j| o.* = at(t.state, (r * streams + s) * n + j);
            return;
        }
        const x = t.inputs(r, s);
        const n0 = inv(total(sim, t.squares(x.k)), n);
        const n1 = inv(total(sim, t.squares(x.q)), n);
        const n2 = inv(total(sim, t.squares(x.v)), n);
        const terms = t.work[4 * n .. 5 * n];
        for (terms, x.k, x.q, 0..) |*d, k, q, j| {
            const key = bf(k * n0 * at(t.key_gain, s * n + j));
            const query = bf(q * n1 * at(t.query_gain, s * n + j));
            d.* = bf(key * query);
        }
        const scaled = bf(bf(total(sim, terms)) / @sqrt(@as(f64, @floatFromInt(n))));
        const gate = bf(sigmoid(root(scaled, 0)));
        for (out, x.q, x.v, 0..) |*o, q, v, j| {
            const signal = bf(bf(v * n2) * at(t.value_gain, j));
            const delta = bf(bf(gate * signal) + bf(at(t.delta_gain, s * n + j) * signal));
            o.* = bf(q + delta);
        }
    }

    pub fn reference(t: *Apply, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const n = t.shape.n;
        for (0..t.shape.rows) |r| for (0..streams) |s| t.point(r, s, null, want[(r * streams + s) * n ..][0..n]);
    }

    pub fn simulate(t: *Apply, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = t.shape.n;
        for (0..t.shape.rows) |r| for (0..streams) |s| t.point(r, s, .{ .acc = acc, .order = order }, out[(r * streams + s) * n ..][0..n]);
    }

    /// A norm's range: its squares within gamma, the mean, epsilon and rsqrt within 12 fp32 ulps.
    fn norm(ss: f64, n: usize, gam: f64) Range {
        return .{ .lo = inv(ss * (1 + gam), n) * (1 - 12 * u), .hi = inv(ss * (1 - gam), n) * (1 + 12 * u) };
    }

    /// Each output's range through the norms, bf16 keys and queries, the dot within gamma, the gate, signal and delta.
    pub fn intervals(t: *Apply, low: []f64, high: []f64) !void {
        const n = t.shape.n;
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(depth(t.shape)), u);
        for (0..t.shape.rows) |r| for (0..streams) |s| {
            const o0 = (r * streams + s) * n;
            if (t.media[r] != 0) {
                for (0..n) |j| {
                    low[o0 + j] = at(t.state, o0 + j);
                    high[o0 + j] = low[o0 + j];
                }
                continue;
            }
            const x = t.inputs(r, s);
            const n0 = norm(total(null, t.squares(x.k)), n, gam);
            const n1 = norm(total(null, t.squares(x.q)), n, gam);
            const n2 = norm(total(null, t.squares(x.v)), n, gam);
            var dot = Range.point(0);
            var size: f64 = 0;
            for (x.k, x.q, 0..) |k, q, j| {
                const key = Range.point(k).mul(n0).fp32().times(at(t.key_gain, s * n + j)).fp32().bf16();
                const query = Range.point(q).mul(n1).fp32().times(at(t.query_gain, s * n + j)).fp32().bf16();
                const d = key.mul(query).fp32().bf16();
                dot = dot.plus(d);
                size += @max(@abs(d.lo), @abs(d.hi));
            }
            const rs = 1 / @sqrt(@as(f64, @floatFromInt(n)));
            const scaled = dot.widen(gam * size).bf16().mul(.{ .lo = rs * (1 - 8 * u), .hi = rs * (1 + 8 * u) }).fp32().bf16();
            const xg = Range{ .lo = root(scaled.lo, -2 * u), .hi = root(scaled.hi, 2 * u) };
            const gate = (Range{ .lo = sigmoid(xg.lo) * (1 - 16 * u), .hi = sigmoid(xg.hi) * (1 + 16 * u) }).bf16();
            for (x.q, x.v, 0..) |q, v, j| {
                const signal = Range.point(v).mul(n2).fp32().bf16().times(at(t.value_gain, j)).fp32().bf16();
                const t1 = gate.mul(signal).fp32().bf16();
                const t2 = signal.times(at(t.delta_gain, s * n + j)).fp32().bf16();
                const o = t1.plus(t2).fp32().bf16().plus(Range.point(q)).fp32().bf16();
                low[o0 + j] = o.lo;
                high[o0 + j] = o.hi;
            }
        };
    }

    pub fn bounds(t: *Apply, want: []const f64, _: []const f64, allow: []f64) !void {
        try h.fl.allowFrom(t.a, t, want, allow);
    }
};
