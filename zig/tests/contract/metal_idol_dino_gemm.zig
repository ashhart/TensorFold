//! dn_linear under the contract: bf16 projections, fp32 sums on the tensor units, the bias in them, one bf16 rounding.
const std = @import("std");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const dn = @import("metal_idol_dino_common.zig");

const c = dn.c;
const g = dn.g;
const W = dn.W;
const Range = dn.Range;
const Ends = dn.Ends;
const bf = dn.bf;
const kref = dn.kref;

/// Rows of x and of w (with the bias) repeat at these periods, prime to every tile: 97 x 131 dots a shape.
const row_period = 97;
const col_period = 131;

/// The M5 tensor unit's fp32 sums: its 16-deep steps (attention_range's), as for the denoiser's fp16 projections.
fn naxDepth(s: c.Shape) usize {
    return s.k / 16 * contract.attention_range.mma_step + 8;
}

/// float64's dot of a bf16 row pair, the tensor unit's allowance on its fp32 sums step by step, and the terms' size.
fn naxDot(x: []const u16, w: []const u16) [3]f64 {
    var s: f64 = 0;
    var spread: f64 = 0;
    var mag: f64 = 0;
    var i: usize = 0;
    while (i < x.len) : (i += 16) {
        var t: f64 = 0;
        var abs: f64 = 0;
        for (x[i..@min(i + 16, x.len)], w[i..@min(i + 16, x.len)]) |a, b| {
            const p = @as(f64, dn.value(a)) * @as(f64, dn.value(b));
            t += p;
            abs += @abs(p);
        }
        spread += @abs(s) + abs;
        s += t;
        mag += abs;
    }
    return .{ s, spread * contract.attention_range.mma_step * 0x1p-24, mag };
}

/// The two forms by M % 64 == 0: the 4,101 rows (a last tile's rows), the 4,096 patches (whole 64-row tiles).
pub const forms = [2][]const u8{ "dn_linear_0", "dn_linear_1" };

fn envelope(form: u1) struct { served: []const c.Shape, tiles: [3]c.Tile } {
    const col: c.Tile = .{ .size = 256, .step = 128, .min = 128 };
    const k: c.Tile = .{ .size = 256, .step = 32, .min = 32 };
    if (form == 1) return .{ .served = &.{.{ .rows = 4096 / 64, .n = W, .k = dn.cfg.patch_k }}, .tiles = .{ .{ .size = 2, .step = 1, .min = 1 }, col, k } };
    // first the 4,101 rows' last tile alone (5 rows): the single-shape gates take the first served shape's rows
    return .{
        .served = &.{ .{ .rows = 5, .n = W, .k = W }, .{ .rows = 4101, .n = 3 * W, .k = W }, .{ .rows = 4101, .n = W, .k = W }, .{ .rows = 4101, .n = 2 * dn.cfg.ffn, .k = W }, .{ .rows = 4101, .n = W, .k = dn.cfg.ffn } },
        .tiles = .{ .{ .size = 64 }, col, k },
    };
}

pub const Linear = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *dn.Lib,
    form: u1,
    name: []const u8,
    m: usize = 0,
    n: usize = 0,
    k: usize = 0,
    stress: bool = false,
    x: []u16 = &.{},
    w: []u16 = &.{},
    b: []u16 = &.{},
    out: rig_.Ref = undefined,
    ins: [3]rig_.Ref = undefined,
    ends: Ends,

    /// Rows: M (form 1: 64-row tiles); n: N, whole 128-column tiles; k: K.
    pub fn spec(name: []const u8, form: u1) c.Spec {
        const e = envelope(form);
        return .{ .name = name, .accum = .f32, .depth = naxDepth, .ops = 1, .out = .bf16, .tiles = e.tiles, .served = e.served };
    }

    pub fn deinit(t: *Linear) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Linear) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.a.free(t.b);
        t.x = &.{};
        t.w = &.{};
        t.b = &.{};
    }
    pub fn kernel(t: *Linear, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Linear);

    pub fn prepare(t: *Linear, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const m = if (t.form == 1) shape.rows * 64 else shape.rows;
        if (m == 0 or shape.n == 0 or shape.n % 128 != 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        if (@intFromBool(m % 64 == 0) != t.form) return false;
        t.rig.reset();
        t.free();
        t.m = m;
        t.n = shape.n;
        t.k = shape.k;
        t.stress = fill == .stress;
        const K = shape.k;
        const px = @min(m, row_period);
        const pw = @min(shape.n, col_period);
        t.x = try t.a.alloc(u16, px * K);
        t.w = try t.a.alloc(u16, pw * K);
        t.b = try t.a.alloc(u16, pw);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // integers in [hi / 2, hi) (below 256, exact in bf16), hi^2 K <= 2^24: fp32 sums exact in any order
            const hi: i32 = @min(256, @as(i32, @intFromFloat(@sqrt(@as(f64, 0x1p24) / @as(f64, @floatFromInt(K))))));
            for (0..px) |r| for (0..K / 2) |i| {
                const v = dn.int(rng, @divTrunc(hi, 2), hi - 1);
                t.x[r * K + i] = dn.bits(v);
                // x's halves differ by 1 in their first 8: the sums cancel to a few hundred
                t.x[r * K + K / 2 + i] = dn.bits(v - @as(f32, if (i < 8) 1 else 0));
            };
            for (0..pw) |j| for (0..K / 2) |i| {
                const v = dn.int(rng, @divTrunc(hi, 2), hi - 1);
                t.w[j * K + i] = dn.bits(v);
                t.w[j * K + K / 2 + i] = dn.bits(-v);
            };
            for (t.b) |*v| v.* = dn.bits(dn.int(rng, -4, 4));
        } else {
            dn.randomBf(rng, t.x, -2, 2);
            dn.randomBf(rng, t.w, -0.25, 0.25);
            dn.randomBf(rng, t.b, -1, 1);
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, K / 3 }, .{ 1, K / 2 }, .{ 2, K - 1 }, .{ 3, 0 }, .{ 3, K - 1 } };
            for (at, [_]u16{ dn.nan, dn.inf, dn.ninf, dn.inf, dn.ninf }) |p, v| if (p[0] < px) {
                t.x[p[0] * K + p[1]] = v;
            };
            if (pw > 2) t.b[2] = dn.inf;
        }
        const xs = try dn.periodic(u16, t.a, t.x, K, m);
        defer t.a.free(xs);
        const ws = try dn.periodic(u16, t.a, t.w, K, shape.n);
        defer t.a.free(ws);
        const bs = try dn.periodic(u16, t.a, t.b, 1, shape.n);
        defer t.a.free(bs);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(xs));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(ws));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(bs));
        t.out = try t.rig.output(m * shape.n * 2);
        return true;
    }

    pub fn run(t: *Linear) !void {
        try t.lib.need(forms[t.form]);
        const p = t.rig.begin();
        t.lib.k.linear(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out), t.m, t.n, t.k, t.n);
        try t.rig.finish(p);
    }

    pub fn count(t: *Linear) usize {
        return t.m * t.n;
    }

    /// The distinct outputs' values and ranges from (sum, its range) by row and column, then every output's.
    fn table(t: *Linear, ends: *Ends, want: []f64, mag: []f64, ctx: anytype, comptime sum: anytype) !void {
        const px = t.x.len / t.k;
        const pw = t.w.len / t.k;
        const tw = try t.a.alloc(f64, 3 * px * pw);
        defer t.a.free(tw);
        for (0..px) |r| for (0..pw) |j| {
            const s = sum(ctx, t, r, j);
            const b: f64 = dn.value(t.b[j]);
            const vr = dn.bfr(s[1].add(Range.of(b)).fp32(0.5));
            tw[3 * (r * pw + j) ..][0..3].* = .{ bf(s[0] + b), vr.lo, vr.hi };
        };
        for (0..t.m) |r| for (0..t.n) |j| {
            const e = tw[3 * ((r % px) * pw + j % pw) ..][0..3];
            ends.put(want, mag, r * t.n + j, e[0], .{ .lo = e[1], .hi = e[2] });
        };
    }

    pub fn reference(t: *Linear, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const f = struct {
            fn sum(_: void, s: *const Linear, r: usize, j: usize) struct { f64, Range } {
                const d = naxDot(s.x[r * s.k ..][0..s.k], s.w[j * s.k ..][0..s.k]);
                return .{ d[0], if (s.stress) Range.of(d[0]) else Range.of(d[0]).widen(d[1]) };
            }
        };
        try t.table(&t.ends, want, mag, {}, f.sum);
    }
    pub fn bounds(t: *Linear, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Linear, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Linear, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        const terms = try t.a.alloc(f64, t.k);
        defer t.a.free(terms);
        const Ctx = struct { acc: c.Accum, order: g.Order, terms: []f64 };
        const f = struct {
            fn sum(x: Ctx, s: *const Linear, r: usize, j: usize) struct { f64, Range } {
                for (x.terms, s.x[r * s.k ..][0..s.k], s.w[j * s.k ..][0..s.k]) |*v, a, b| v.* = @as(f64, dn.value(a)) * @as(f64, dn.value(b));
                const v = dn.rows.sumIn(x.acc, x.order, x.terms);
                return .{ v, Range.of(v) };
            }
        };
        try t.table(&scratch, out, mag, Ctx{ .acc = acc, .order = order, .terms = terms }, f.sum);
    }
};
