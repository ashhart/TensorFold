//! mi_norm and mi_qk under the contract: RMS norms at the publisher's bf16 points, their fp32 sums proven in gate 4.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const mi = @import("metal_moss_image_common.zig");

const c = mi.c;
const g = mi.g;
const ks = mi.ks;
const cfg = mi.cfg;
const D = mi.D;
const Range = mi.Range;
const Ends = mi.Ends;
const bf = mi.bf;
const kref = mi.kref;
const kat = mi.kat;

// --- mi_norm: bf16(bf16(x rsqrt(mean x^2 + eps)) w), then bf16(bf16(y s) + b) when modulated --

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

/// A thread's share of the row in turn, then the simdgroup tree and the tree of the threadgroup's 8 simdgroups.
fn normDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 10;
}

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    modulated: bool,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    /// w, then s and b
    v: []u16 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    pub fn spec(name: []const u8, modulated: bool) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = normDepth,
            .ops = 2 * (mi.div_ulps + mi.rsqrt_ulps) + 1,
            .mids = if (modulated) 3 else 1,
            .out = .bf16,
            .tiles = .{ .{}, .{}, .{ .size = 256 } },
            .served = if (modulated)
                &.{ .{ .rows = 4188, .n = 3072, .k = 3072 }, .{ .rows = 4130, .n = 3072, .k = 3072 }, .{ .rows = 4125, .n = 3072, .k = 3072 } }
            else
                &.{ .{ .rows = 63, .n = 3072, .k = 3072 }, .{ .rows = 34, .n = 3072, .k = 3072 }, .{ .rows = 10, .n = 3072, .k = 3072 } },
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }

    pub fn deinit(t: *Norm) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Norm) void {
        t.a.free(t.x);
        t.a.free(t.v);
        t.x = &.{};
        t.v = &.{};
    }
    pub fn kernel(t: *Norm, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.modulated), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Norm);

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const k = shape.k;
        t.x = try t.a.alloc(u16, shape.rows * k);
        t.v = try t.a.alloc(u16, 3 * k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) mi.integers(rng, t.x, 32) else mi.randomBf16(rng, t.x, -3, 3);
        mi.randomBf16(rng, t.v[0..k], 0.25, 2);
        // the modulation's 1 + scale, and its shift
        mi.randomBf16(rng, t.v[k .. 2 * k], 0.5, 1.5);
        mi.randomBf16(rng, t.v[2 * k ..], -1, 1);
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
            for (at, [_]u16{ mi.nan16, mi.inf16, mi.ninf16, mi.inf16, mi.ninf16 }) |p, v| if (p[0] < shape.rows) {
                t.x[p[0] * k + p[1]] = v;
            };
            if (k > 7) {
                t.v[5] = mi.nan16;
                t.v[k + 6] = mi.inf16;
                t.v[2 * k + 7] = mi.ninf16;
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        for (0..3) |i| t.ins[1 + i] = try t.rig.input(std.mem.sliceAsBytes(t.v[i * k ..][0..k]));
        t.out = try t.rig.output(shape.rows * k * 2);
        return true;
    }

    pub fn run(t: *Norm) !void {
        const p = t.rig.begin();
        ks.norm(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.ins[3]), kref(t.out), t.shape.rows, t.shape.k, cfg.eps, t.modulated);
        try t.rig.finish(p);
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * t.shape.k;
    }

    fn row(t: *Norm, ends: *Ends, r: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const k = t.shape.k;
        const n = mi.invNorm(ss, ssr, @floatFromInt(k));
        for (0..k) |j| {
            const x: f64 = c.fromBf16(t.x[r * k + j]);
            const w: f64 = c.fromBf16(t.v[j]);
            var y = bf(bf(x * n.inv) * w);
            var yr = Range.of(x).mul(n.r).fp32(0.5).bf16().mul(Range.of(w)).fp32(0.5).bf16();
            if (t.modulated) {
                const s: f64 = c.fromBf16(t.v[k + j]);
                const b: f64 = c.fromBf16(t.v[2 * k + j]);
                y = bf(bf(y * s) + b);
                yr = yr.mul(Range.of(s)).fp32(0.5).bf16().add(Range.of(b)).fp32(0.5).bf16();
            }
            ends.put(want, mag, r * k + j, y, yr);
        }
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const k = t.shape.k;
        for (0..t.shape.rows) |r| {
            const s = mi.squares(t.x[r * k ..][0..k], normDepth(t.shape), t.fill == .stress);
            t.row(&t.ends, r, s.ss, s.r, want, mag);
        }
    }
    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const k = t.shape.k;
        const terms = try t.a.alloc(f64, k);
        defer t.a.free(terms);
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        for (0..t.shape.rows) |r| {
            for (terms, t.x[r * k ..][0..k]) |*v, x| {
                const f: f64 = c.fromBf16(x);
                v.* = f * f;
            }
            const ss = contract.rows.sumIn(acc, order, terms);
            t.row(&scratch, r, ss, Range.of(ss), out, mag);
        }
    }
};

// --- mi_qk: q and k heads normed, (c, c + 64) rotated by cos and sin; v copied --------------------------------------

/// Four channels a lane in turn, then the simdgroup tree.
fn qkDepth(_: c.Shape) usize {
    return 4 + 5;
}

pub const Qk = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    /// The refiner's form: q, k and v side by side in one row of 3 x 3072; else three buffers of 3072-wide rows.
    packed_rows: bool,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// q, k, v as [rows, heads * 128] each
    qkv: [3][]u16 = .{ &.{}, &.{}, &.{} },
    /// the q and k head norms' weights
    nw: []u16 = &.{},
    /// cos then sin [rows, 64], bf16 values
    rot: []f32 = &.{},
    ins: [6]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: heads (24); k: the head width (128).
    pub fn spec(name: []const u8, packed_rows: bool) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = qkDepth,
            .ops = 2 * (mi.div_ulps + mi.rsqrt_ulps) + 1,
            .mids = 3,
            .out = .bf16,
            .served = if (packed_rows)
                &.{ .{ .rows = 63, .n = 24, .k = D }, .{ .rows = 34, .n = 24, .k = D }, .{ .rows = 10, .n = 24, .k = D } }
            else
                &.{ .{ .rows = 4188, .n = 24, .k = D }, .{ .rows = 4130, .n = 24, .k = D }, .{ .rows = 4160, .n = 24, .k = D } },
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }

    pub fn deinit(t: *Qk) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Qk) void {
        for (&t.qkv) |*x| {
            t.a.free(x.*);
            x.* = &.{};
        }
        t.a.free(t.nw);
        t.a.free(t.rot);
        t.nw = &.{};
        t.rot = &.{};
    }
    pub fn kernel(t: *Qk, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.packed_rows), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Qk);

    fn width(t: *const Qk) usize {
        return t.shape.n * D;
    }

    pub fn prepare(t: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != D) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const w = t.width();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (&t.qkv, 0..) |*x, i| {
            x.* = try t.a.alloc(u16, shape.rows * w);
            if (fill == .stress and i < 2) mi.integers(rng, x.*, 128) else mi.randomBf16(rng, x.*, -4, 4);
        }
        t.nw = try t.a.alloc(u16, 2 * D);
        mi.randomBf16(rng, t.nw, 0.25, 2);
        t.rot = try t.a.alloc(f32, 2 * shape.rows * D / 2);
        const half = shape.rows * D / 2;
        for (0..half) |i| {
            const angle = rng.float(f32) * 2 * std.math.pi;
            t.rot[i] = c.fromBf16(c.toBf16(@cos(angle)));
            t.rot[half + i] = c.fromBf16(c.toBf16(@sin(angle)));
        }
        if (fill == .nonfinite) {
            const at = [_][4]usize{ .{ 0, 0, 0, 5 }, .{ 1, 1, 2, 70 }, .{ 0, 2, 3, 0 }, .{ 2, 3, 1, 9 }, .{ 1, 4, 0, 127 } };
            for (at, [_]u16{ mi.nan16, mi.inf16, mi.ninf16, mi.nan16, mi.ninf16 }) |p, v| if (p[1] < shape.rows and p[2] < shape.n) {
                t.qkv[p[0]][p[1] * w + p[2] * D + p[3]] = v;
            };
        }
        if (t.packed_rows) {
            const both = try t.a.alloc(u16, shape.rows * 3 * w);
            defer t.a.free(both);
            for (0..shape.rows) |r| for (t.qkv, 0..) |x, i| @memcpy(both[(r * 3 + i) * w ..][0..w], x[r * w ..][0..w]);
            t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(both));
        } else for (t.qkv, 0..) |x, i| {
            t.ins[i] = try t.rig.input(std.mem.sliceAsBytes(x));
        }
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.nw));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.rot[0..half]));
        t.ins[5] = try t.rig.input(std.mem.sliceAsBytes(t.rot[half..]));
        for (&t.outs) |*o| o.* = try t.rig.output(shape.rows * w * 2);
        return true;
    }

    pub fn run(t: *Qk) !void {
        const p = t.rig.begin();
        const w = t.width();
        const qkv: [3]ks.Ref = if (t.packed_rows) .{ kref(t.ins[0]), kat(t.ins[0], w * 2), kat(t.ins[0], 2 * w * 2) } else .{ kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]) };
        ks.heads(p.e, t.pipe, qkv[0], qkv[1], qkv[2], kref(t.ins[3]), kat(t.ins[3], D * 2), kref(t.ins[4]), kref(t.ins[5]), kref(t.outs[0]), kref(t.outs[1]), kref(t.outs[2]), t.shape.rows, t.shape.n, if (t.packed_rows) 3 * w else w, cfg.eps);
        try t.rig.finish(p);
    }

    pub fn count(t: *Qk) usize {
        return 3 * t.shape.rows * t.width();
    }

    /// Head h of row r of q (0) or k (1) with its sum of squares `ss` (range `ssr`), into its outputs.
    fn head(t: *Qk, ends: *Ends, which: usize, r: usize, h: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const rows = t.shape.rows;
        const x = t.qkv[which][r * t.width() + h * D ..][0..D];
        const w = t.nw[which * D ..][0..D];
        const n = mi.invNorm(ss, ssr, D);
        var normed: [D]f64 = undefined;
        var nr: [D]Range = undefined;
        for (x, w, &normed, &nr) |xv, wv, *o, *orr| {
            const f: f64 = c.fromBf16(xv);
            const wf: f64 = c.fromBf16(wv);
            o.* = bf(bf(f * n.inv) * wf);
            orr.* = Range.of(f).mul(n.r).fp32(0.5).bf16().mul(Range.of(wf)).fp32(0.5).bf16();
        }
        const half = rows * D / 2;
        const base = which * rows * t.width() + (h * rows + r) * D;
        for (0..D) |ch| {
            const j = ch % (D / 2);
            const partner = if (ch < D / 2) ch + D / 2 else ch - D / 2;
            const sign: f64 = if (ch < D / 2) -1 else 1;
            const cs: f64 = t.rot[r * D / 2 + j];
            const sn: f64 = t.rot[half + r * D / 2 + j];
            const v = bf(bf(normed[ch] * cs) + bf(sign * normed[partner] * sn));
            const vr = nr[ch].mul(Range.of(cs)).fp32(0.5).bf16().add(nr[partner].mul(Range.of(sign * sn)).fp32(0.5).bf16()).fp32(0.5).bf16();
            ends.put(want, mag, base + ch, v, vr);
        }
    }

    /// v's heads, copied bit for bit.
    fn values(t: *Qk, ends: *Ends, want: []f64, mag: []f64) void {
        const rows = t.shape.rows;
        for (0..rows) |r| for (0..t.shape.n) |h| for (0..D) |ch| {
            const f: f64 = c.fromBf16(t.qkv[2][r * t.width() + h * D + ch]);
            ends.put(want, mag, 2 * rows * t.width() + (h * rows + r) * D + ch, f, Range.of(f));
        };
    }

    pub fn reference(t: *Qk, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..2) |which| for (0..t.shape.rows) |r| for (0..t.shape.n) |h| {
            const s = mi.squares(t.qkv[which][r * t.width() + h * D ..][0..D], qkDepth(t.shape), t.fill == .stress);
            t.head(&t.ends, which, r, h, s.ss, s.r, want, mag);
        };
        t.values(&t.ends, want, mag);
    }
    pub fn bounds(t: *Qk, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Qk, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var terms: [D]f64 = undefined;
        for (0..2) |which| for (0..t.shape.rows) |r| for (0..t.shape.n) |h| {
            for (&terms, t.qkv[which][r * t.width() + h * D ..][0..D]) |*v, x| {
                const f: f64 = c.fromBf16(x);
                v.* = f * f;
            }
            const ss = contract.rows.sumIn(acc, order, &terms);
            t.head(&scratch, which, r, h, ss, Range.of(ss), out, mag);
        };
        t.values(&scratch, out, mag);
    }
};
