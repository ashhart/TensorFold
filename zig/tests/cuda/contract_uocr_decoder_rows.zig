//! tf_uocr_norm and tf_uocr_qk under the contract: RMS norms and rotary at decoder.metal's bf16 points.
const std = @import("std");
const cuda = @import("cuda");
const ud = @import("contract_uocr_decoder.zig");
const rig_ = @import("contract_rig.zig");

const c = ud.c;
const g = ud.g;
const mi = ud.mi;
const Range = mi.Range;
const Ends = mi.Ends;
const bf = mi.bf;
const H = ud.H;
const D = ud.D;

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

// --- tf_uocr_norm: R = bf16(R + A) when adding, Y = bf16(w bf16(r inv)) --------------------------------------------

/// Five values a thread in turn, the warp's 5 shuffles, then the 8 warps' partials through one warp's 5.
fn normDepth(_: c.Shape) usize {
    return 5 + 5 + 5;
}

const norm_served = [_]c.Shape{ .{ .rows = 2048, .n = H, .k = H }, .{ .rows = 37, .n = H, .k = H }, .{ .rows = 1, .n = H, .k = H } };

pub const Norm = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    /// The residual's form: R = bf16(R + A) in place first, R's rows an output too.
    add: bool,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    a: []u16 = &.{},
    w: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    r_out: u64 = 0,
    y: u64 = 0,
    ends: Ends,

    /// Rows: positions; n and k: the hidden width.
    pub fn spec(name: []const u8, add: bool) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = normDepth,
            .ops = 2 * (mi.div_ulps + mi.sqrt_ulps) + 1,
            .mids = 1,
            .out = .bf16,
            .in_place = add,
            .served = &norm_served,
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }

    pub fn deinit(t: *Norm) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Norm) void {
        const a = t.rig.gpu.gpa;
        for ([_][]u16{ t.x, t.a, t.w }) |v| a.free(v);
        t.x = &.{};
        t.a = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Norm) g.Kernel {
        return .{ .spec = spec(t.name, t.add), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Norm);

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k != H) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * H;
        t.x = try a.alloc(u16, n);
        t.a = try a.alloc(u16, n);
        t.w = try a.alloc(u16, H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            mi.integers(rng, t.x, 32);
            mi.integers(rng, t.a, 32);
        } else {
            mi.randomBf16(rng, t.x, -3, 3);
            mi.randomBf16(rng, t.a, -2, 2);
        }
        mi.randomBf16(rng, t.w, 0.25, 2);
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, H / 3 }, .{ 1, H / 2 }, .{ 2, H - 1 }, .{ 3, 0 } };
            for (at, [_]u16{ mi.nan16, mi.inf16, mi.ninf16, mi.inf16 }) |p, v| if (p[0] < shape.rows) {
                t.x[p[0] * H + p[1]] = v;
            };
            if (shape.rows > 4) t.a[4 * H + 7] = mi.nan16;
            t.w[5] = mi.nan16;
        }
        t.ins[0] = try mi.input(&t.rig, t.x);
        t.ins[1] = try mi.input(&t.rig, t.a);
        t.ins[2] = try mi.input(&t.rig, t.w);
        if (t.add) t.r_out = try t.rig.output(n * 2);
        t.y = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Norm) !void {
        var r = t.ins[0];
        if (t.add) {
            // in place: the residual's rows go back before each independent run
            const bytes = t.shape.rows * H * 2;
            try (cuda.DeviceBuffer{ .d = t.rig.gpu.d, .ptr = t.r_out, .len = bytes }).copyFrom(0, t.ins[0], bytes, t.rig.s.handle);
            r = t.r_out;
        }
        try t.k.run("norm", t.rig.s, .{ t.shape.rows, 1, 1 }, 256, .{ r, t.ins[1], t.ins[2], t.y, @as(c_int, @intFromBool(t.add)), ud.eps });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * H * @as(usize, if (t.add) 2 else 1);
    }

    /// Row r's residual values: x, or bf16(x + a).
    fn values(t: *Norm, r: usize, v: *[H]f64) void {
        for (v, 0..) |*o, j| {
            const x: f64 = c.fromBf16(t.x[r * H + j]);
            o.* = if (t.add) bf(x + c.fromBf16(t.a[r * H + j])) else x;
        }
    }

    fn row(t: *Norm, ends: *Ends, r: usize, v: *const [H]f64, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const n = mi.invNorm(ss, ssr, H);
        const base = if (t.add) t.shape.rows * H else 0;
        for (v, 0..) |x, j| {
            if (t.add) ends.put(want, mag, r * H + j, x, Range.of(x));
            const w: f64 = c.fromBf16(t.w[j]);
            const y = bf(bf(x * n.inv) * w);
            const yr = Range.of(x).mul(n.r).fp32(0.5).bf16().mul(Range.of(w)).fp32(0.5).bf16();
            ends.put(want, mag, base + r * H + j, y, yr);
        }
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        var v: [H]f64 = undefined;
        for (0..t.shape.rows) |r| {
            t.values(r, &v);
            var ss: f64 = 0;
            for (v) |x| ss += x * x;
            const e = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(normDepth(t.shape)), c.unit(.f32)) * ss;
            t.row(&t.ends, r, &v, ss, .{ .lo = ss - e, .hi = ss + e }, want, mag);
        }
    }
    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var v: [H]f64 = undefined;
        var terms: [H]f64 = undefined;
        for (0..t.shape.rows) |r| {
            t.values(r, &v);
            for (&terms, v) |*s, x| s.* = x * x;
            const ss = mi.rows.sumIn(acc, order, &terms);
            t.row(&scratch, r, &v, ss, Range.of(ss), out, mag);
        }
    }
};

// --- tf_uocr_qk: q and k heads turned, v copied, the router's rows normed and turned -------------------------------

/// The router norm's sum: one value a thread, a warp's 5 shuffles, then the 4 warps' partials through one warp's 5.
fn qkDepth(_: c.Shape) usize {
    return 1 + 5 + 5;
}

/// The cache position of row 0 the test writes at (the caches' outputs start there).
const qk_pos0 = 4099;

const qk_served = [_]c.Shape{ .{ .rows = 2048, .n = ud.heads, .k = D }, .{ .rows = 37, .n = ud.heads, .k = D }, .{ .rows = 1, .n = ud.heads, .k = D } };

fn clamp(r: Range) Range {
    return .{ .lo = ud.clamp(r.lo), .hi = ud.clamp(r.hi) };
}

pub const Qk = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// [rows, q | k | v] and [rows, router q | router k]
    p: []u16 = &.{},
    rp: []u16 = &.{},
    /// the router's q and k norm weights
    nw: []u16 = &.{},
    /// cos then sin [rows, 64], bf16 bits
    rot: []u16 = &.{},
    ins: [5]u64 = @splat(0),
    outs: [5]u64 = @splat(0),
    ends: Ends,

    /// Rows: positions; n: heads (10); k: the head and router width (128).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = qkDepth,
            .ops = 2 * (mi.div_ulps + mi.sqrt_ulps) + 3,
            .mids = 3,
            .out = .bf16,
            .served = &qk_served,
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }

    pub fn deinit(t: *Qk) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Qk) void {
        const a = t.rig.gpu.gpa;
        for ([_][]u16{ t.p, t.rp, t.nw, t.rot }) |v| a.free(v);
        t.p = &.{};
        t.rp = &.{};
        t.nw = &.{};
        t.rot = &.{};
    }
    pub fn kernel(t: *Qk) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Qk);

    pub fn prepare(t: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != ud.heads or shape.k != D) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        const rows = shape.rows;
        t.p = try a.alloc(u16, rows * 3 * H);
        t.rp = try a.alloc(u16, rows * 2 * D);
        t.nw = try a.alloc(u16, 2 * D);
        t.rot = try a.alloc(u16, 2 * rows * 64);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        mi.randomBf16(rng, t.p, -4, 4);
        if (fill == .stress) mi.integers(rng, t.rp, 128) else mi.randomBf16(rng, t.rp, -4, 4);
        mi.randomBf16(rng, t.nw, -0.75, 1);
        for (0..rows * 64) |i| {
            const angle = rng.float(f32) * 2 * std.math.pi;
            t.rot[i] = c.toBf16(@cos(angle));
            t.rot[rows * 64 + i] = c.toBf16(@sin(angle));
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, 5 }, .{ 1, H + 70 }, .{ 2, 2 * H + 9 }, .{ 3, H + 128 * 3 } };
            for (at, [_]u16{ mi.nan16, mi.inf16, mi.nan16, mi.ninf16 }) |q, v| if (q[0] < rows) {
                t.p[q[0] * 3 * H + q[1]] = v;
            };
            if (rows > 4) t.rp[4 * 2 * D + 9] = mi.nan16;
        }
        t.ins[0] = try mi.input(&t.rig, t.p);
        t.ins[1] = try mi.input(&t.rig, t.rp);
        t.ins[2] = try mi.input(&t.rig, t.rot[0 .. rows * 64]);
        t.ins[3] = try mi.input(&t.rig, t.rot[rows * 64 ..]);
        t.ins[4] = try mi.input(&t.rig, t.nw);
        for (&t.outs, [_]usize{ H, H, H, D, D }) |*o, w| o.* = try t.rig.output(rows * w * 2);
        return true;
    }

    pub fn run(t: *Qk) !void {
        // the caches' outputs hold rows pos0 .. pos0 + rows: their bases sit pos0 rows before them
        const kc = t.outs[1] - qk_pos0 * H * 2;
        const vc = t.outs[2] - qk_pos0 * H * 2;
        const rk = t.outs[4] - qk_pos0 * D * 2;
        const i = t.ins;
        try t.k.run("qk", t.rig.s, .{ t.shape.rows, 1, 1 }, 128, .{ i[0], i[1], i[2], i[3], i[4], i[4] + D * 2, t.outs[0], kc, vc, t.outs[3], rk, @as(c_int, qk_pos0), ud.eps });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Qk) usize {
        return t.shape.rows * (3 * H + 2 * D);
    }

    fn cs(t: *Qk, r: usize, d: usize) [2]f64 {
        return .{ c.fromBf16(t.rot[r * 64 + d % 64]), c.fromBf16(t.rot[t.shape.rows * 64 + r * 64 + d % 64]) };
    }

    /// Rotary of value x at channel d with its partner, both with their ranges.
    fn rope(t: *Qk, r: usize, d: usize, x: f64, xr: Range, p: f64, pr: Range) struct { v: f64, r: Range } {
        const k = t.cs(r, d);
        const sign: f64 = if (d < 64) -1 else 1;
        const v = bf(bf(x * k[0]) + bf(sign * p * k[1]));
        const vr = xr.mul(Range.of(k[0])).fp32(0.5).bf16().add(pr.mul(Range.of(sign * k[1])).fp32(0.5).bf16()).fp32(0.5).bf16();
        return .{ .v = v, .r = vr };
    }

    /// Row r's q, k and v heads into their outputs (no sums).
    fn heads(t: *Qk, ends: *Ends, r: usize, want: []f64, mag: []f64) void {
        const rows = t.shape.rows;
        const x = t.p[r * 3 * H ..][0 .. 3 * H];
        for (0..2) |which| for (0..ud.heads) |h| for (0..D) |d| {
            const o = which * H + h * D;
            const xv: f64 = c.fromBf16(x[o + d]);
            const pv: f64 = c.fromBf16(x[o + (d ^ 64)]);
            const got = t.rope(r, d, xv, Range.of(xv), pv, Range.of(pv));
            ends.put(want, mag, which * rows * H + r * H + h * D + d, got.v, got.r);
        };
        for (0..H) |e| {
            const v: f64 = c.fromBf16(x[2 * H + e]);
            ends.put(want, mag, 2 * rows * H + r * H + e, v, Range.of(v));
        }
    }

    /// Row r's router q (0) or k (1) with its sum of squares `ss` (range `ssr`): normed, scaled, turned.
    fn router(t: *Qk, ends: *Ends, which: usize, r: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const x = t.rp[r * 2 * D + which * D ..][0..D];
        const n = mi.invNorm(ss, ssr, D);
        var v: [D]f64 = undefined;
        var vr: [D]Range = undefined;
        for (x, 0..) |xb, d| {
            const f: f64 = c.fromBf16(xb);
            const s = 1 + @as(f64, c.fromBf16(t.nw[which * D + d]));
            v[d] = bf(ud.clamp(ud.clamp(f * n.inv) * s));
            vr[d] = clamp(clamp(Range.of(f).mul(n.r).fp32(0.5)).mul(Range.of(s)).fp32(0.5)).bf16();
        }
        const base = 3 * t.shape.rows * H + which * t.shape.rows * D;
        for (0..D) |d| {
            const got = t.rope(r, d, v[d], vr[d], v[d ^ 64], vr[d ^ 64]);
            ends.put(want, mag, base + r * D + d, got.v, got.r);
        }
    }

    pub fn reference(t: *Qk, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..t.shape.rows) |r| {
            t.heads(&t.ends, r, want, mag);
            for (0..2) |which| {
                const s = mi.squares(t.rp[r * 2 * D + which * D ..][0..D], qkDepth(t.shape), t.fill == .stress);
                t.router(&t.ends, which, r, s.ss, s.r, want, mag);
            }
        }
    }
    pub fn bounds(t: *Qk, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Qk, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var terms: [D]f64 = undefined;
        for (0..t.shape.rows) |r| {
            t.heads(&scratch, r, out, mag);
            for (0..2) |which| {
                for (&terms, t.rp[r * 2 * D + which * D ..][0..D]) |*v, x| {
                    const f: f64 = c.fromBf16(x);
                    v.* = f * f;
                }
                const ss = mi.rows.sumIn(acc, order, &terms);
                t.router(&scratch, which, r, ss, Range.of(ss), out, mag);
            }
        }
    }
};
