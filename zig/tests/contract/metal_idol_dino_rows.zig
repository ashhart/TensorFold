//! dn_heads, dn_scale_add and dn_swiglu under the contract: elementwise steps at transformers' bf16 points.
const std = @import("std");
const rig_ = @import("rig.zig");
const dn = @import("metal_idol_dino_common.zig");

const c = dn.c;
const g = dn.g;
const cfg = dn.cfg;
const W = dn.W;
const D = dn.D;
const Range = dn.Range;
const Ends = dn.Ends;
const bf = dn.bf;
const kref = dn.kref;

/// The steps' common parts: their buffers' lifetime, and the bounds and intervals from the outputs' ends.
fn Common(comptime T: type) type {
    return struct {
        pub fn deinit(t: *T) void {
            for (&t.v) |*v| {
                t.a.free(v.*);
                v.* = &.{};
            }
            t.ends.deinit();
            t.rig.deinit();
        }
        pub fn kernel(t: *T, control: ?g.Gate) g.Kernel {
            return .{ .spec = T.spec(t.name), .ctx = t, .vt = &T.vt, .control = control };
        }
        pub fn bounds(t: *T, want: []const f64, _: []const f64, allow: []f64) !void {
            t.ends.bounds(want, allow);
        }
        pub fn intervals(t: *T, low: []f64, high: []f64) !void {
            t.ends.intervals(low, high);
        }
        fn fresh(t: *T) void {
            t.rig.reset();
            for (&t.v) |*v| {
                t.a.free(v.*);
                v.* = &.{};
            }
        }
    };
}

/// bf16(a * b) for bf16 a and b: the product is exact in fp32, so one rounding.
fn prod(a: f64, b: f64) Range {
    return dn.bfr(Range.of(a * b));
}

// --- dn_heads: q, k, v [20, rows, 64] from qkv rows; q and k turned past the prefix by bf16 cos, sin ---------------

pub const Heads = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *dn.Lib,
    name: []const u8,
    rows: usize = 0,
    /// qkv [rows, 3840], then cos and sin [rows - 5, 64]
    v: [3][]u16 = .{ &.{}, &.{}, &.{} },
    ins: [3]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    ends: Ends,

    /// Rows: the stream's 4,101 (the prefix's 5 unturned); n: qkv's width.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = dn.noSums, .ops = 1, .mids = 2, .out = .bf16, .served = &.{.{ .rows = cfg.rows, .n = 3 * W, .k = 1 }} };
    }
    const C = Common(Heads);
    pub const deinit = C.deinit;
    pub const kernel = C.kernel;
    pub const bounds = C.bounds;
    pub const intervals = C.intervals;
    const vt = g.hooks(Heads);

    pub fn prepare(t: *Heads, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows <= cfg.prefix or shape.n != 3 * W) return false;
        C.fresh(t);
        t.rows = shape.rows;
        const patches = shape.rows - cfg.prefix;
        t.v[0] = try t.a.alloc(u16, shape.rows * 3 * W);
        t.v[1] = try t.a.alloc(u16, patches * D);
        t.v[2] = try t.a.alloc(u16, patches * D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        dn.randomBf(rng, t.v[0], -4, 4);
        dn.randomBf(rng, t.v[1], -1, 1);
        dn.randomBf(rng, t.v[2], -1, 1);
        if (fill == .nonfinite) {
            // a q in the prefix and two past it, a v; a cos and a sin of the patches
            for ([_]u16{ dn.nan, dn.inf, dn.ninf }, 0..) |v, i| {
                t.v[0][((i * 2) % shape.rows) * 3 * W + i * 97] = v;
                t.v[0][i * 3 * W + 2 * W + 11 * i] = v;
            }
            t.v[1][(1 % patches) * D + 5] = dn.inf;
            t.v[2][(2 % patches) * D + 40] = dn.nan;
        }
        for (&t.ins, t.v) |*r, v| r.* = try t.rig.input(std.mem.sliceAsBytes(v));
        for (&t.outs) |*r| r.* = try t.rig.output(cfg.heads * shape.rows * D * 2);
        return true;
    }

    pub fn run(t: *Heads) !void {
        try t.lib.need("dn_heads");
        const p = t.rig.begin();
        t.lib.k.heads(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.outs[0]), kref(t.outs[1]), kref(t.outs[2]), t.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *Heads) usize {
        return 3 * cfg.heads * t.rows * D;
    }

    pub fn reference(t: *Heads, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const per = cfg.heads * t.rows * D;
        for (0..t.rows) |r| for (0..cfg.heads) |h| for (0..D) |j| {
            const to = (h * t.rows + r) * D + j;
            const x = t.v[0][r * 3 * W ..];
            const vv: f64 = dn.value(x[2 * W + h * D + j]);
            t.ends.put(want, mag, 2 * per + to, vv, Range.of(vv));
            for (0..2) |m| {
                const at = m * W + h * D;
                const own: f64 = dn.value(x[at + j]);
                if (r < cfg.prefix) {
                    t.ends.put(want, mag, m * per + to, own, Range.of(own));
                    continue;
                }
                // rotate_half: -x[j + 32] for j < 32, x[j - 32] past it
                const other: f64 = if (j < 32) -@as(f64, dn.value(x[at + j + 32])) else dn.value(x[at + j - 32]);
                const tb = (r - cfg.prefix) * D + j;
                const cs: f64 = dn.value(t.v[1][tb]);
                const sn: f64 = dn.value(t.v[2][tb]);
                const a = prod(own, cs);
                const b = prod(other, sn);
                t.ends.put(want, mag, m * per + to, bf(bf(own * cs) + bf(other * sn)), dn.bfr(a.add(b).fp32(0.5)));
            }
        };
    }
};

// --- dn_scale_add: x = bf16(x + bf16(g y)), in place -------------------------------------------------------------

pub const ScaleAdd = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *dn.Lib,
    name: []const u8,
    rows: usize = 0,
    /// the stream's rows as they go in, y, then g [1280]
    v: [3][]u16 = .{ &.{}, &.{}, &.{} },
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: the width, 1,280 (the launcher's).
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = dn.noSums, .ops = 1, .mids = 1, .out = .bf16, .in_place = true, .served = &.{.{ .rows = cfg.rows, .n = W, .k = 1 }} };
    }
    const C = Common(ScaleAdd);
    pub const deinit = C.deinit;
    pub const kernel = C.kernel;
    pub const bounds = C.bounds;
    pub const intervals = C.intervals;
    const vt = blk: {
        var hooks = g.hooks(ScaleAdd);
        hooks.overlap = overlap;
        break :blk hooks;
    };
    /// The stream is its own output.
    fn overlap(_: *anyopaque) bool {
        return true;
    }

    pub fn prepare(t: *ScaleAdd, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != W) return false;
        C.fresh(t);
        t.rows = shape.rows;
        const n = shape.rows * W;
        t.v[0] = try t.a.alloc(u16, n);
        t.v[1] = try t.a.alloc(u16, n);
        t.v[2] = try t.a.alloc(u16, W);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        dn.randomBf(rng, t.v[0], -8, 8);
        dn.randomBf(rng, t.v[1], -4, 4);
        dn.randomBf(rng, t.v[2], -2, 2);
        if (fill == .nonfinite) {
            for ([_]u16{ dn.nan, dn.inf, dn.ninf }, 0..) |v, i| {
                t.v[0][(i * 13) % n] = v;
                t.v[1][(i * 17 + 1) % n] = v;
            }
            t.v[2][5] = dn.nan;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.v[1]));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.v[2]));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *ScaleAdd) !void {
        try t.lib.need("dn_scale_add");
        // in place: the stream's rows go back before each independent run
        @memcpy(t.rig.bytes(t.out, t.count() * 2), std.mem.sliceAsBytes(t.v[0]));
        const p = t.rig.begin();
        t.lib.k.scaleAdd(p.e, kref(t.out), kref(t.ins[0]), kref(t.ins[1]), t.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *ScaleAdd) usize {
        return t.rows * W;
    }

    pub fn reference(t: *ScaleAdd, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (t.v[0], t.v[1], 0..) |xb, yb, i| {
            const x: f64 = dn.value(xb);
            const y: f64 = dn.value(yb);
            const gv: f64 = dn.value(t.v[2][i % W]);
            t.ends.put(want, mag, i, bf(x + bf(gv * y)), dn.bfr(prod(gv, y).add(Range.of(x)).fp32(0.5)));
        }
    }
};

// --- dn_swiglu: h = bf16(bf16(silu(gate)) up) from rows [gate | up] -------------------------------------------------

pub const Swiglu = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *dn.Lib,
    name: []const u8,
    rows: usize = 0,
    v: [1][]u16 = .{&.{}},
    ins: [1]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    const F = cfg.ffn;

    /// Rows: positions; n: the MLP's 5,120 (the launcher's).
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = dn.noSums, .ops = dn.exp_ulps + dn.div_ulps + 1, .mids = 1, .out = .bf16, .served = &.{.{ .rows = cfg.rows, .n = F, .k = 1 }} };
    }
    const C = Common(Swiglu);
    pub const deinit = C.deinit;
    pub const kernel = C.kernel;
    pub const bounds = C.bounds;
    pub const intervals = C.intervals;
    const vt = g.hooks(Swiglu);

    pub fn prepare(t: *Swiglu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != F) return false;
        C.fresh(t);
        t.rows = shape.rows;
        const n = shape.rows * 2 * F;
        t.v[0] = try t.a.alloc(u16, n);
        var prng = std.Random.DefaultPrng.init(seed);
        dn.randomBf(prng.random(), t.v[0], -8, 8);
        if (fill == .nonfinite) {
            for ([_]u16{ dn.nan, dn.inf, dn.ninf }, 0..) |v, i| {
                t.v[0][(i * 13) % F] = v;
                t.v[0][2 * F + F + i * 17] = v;
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.v[0]));
        t.out = try t.rig.output(shape.rows * F * 2);
        return true;
    }

    pub fn run(t: *Swiglu) !void {
        try t.lib.need("dn_swiglu");
        const p = t.rig.begin();
        t.lib.k.swiglu(p.e, kref(t.ins[0]), kref(t.out), t.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *Swiglu) usize {
        return t.rows * F;
    }

    pub fn reference(t: *Swiglu, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..t.rows) |r| for (0..F) |j| {
            const gv: f64 = dn.value(t.v[0][r * 2 * F + j]);
            const u: f64 = dn.value(t.v[0][r * 2 * F + F + j]);
            const s = gv / (1 + @exp(-gv));
            // silu's fp32 steps: exp(-g), 1 + it, then g over that at fast math's division
            const den = Range.of(@exp(-gv)).fp32(dn.exp_ulps).add(Range.of(1)).fp32(0.5);
            const sr = dn.bfr(Range.of(gv).mul(den.recip()).fp32(dn.div_ulps));
            const y = Range{ .lo = @min(bf(sr.lo) * u, bf(sr.hi) * u), .hi = @max(bf(sr.lo) * u, bf(sr.hi) * u) };
            t.ends.put(want, mag, r * F + j, bf(bf(s) * u), dn.bfr(y));
        };
    }
};
