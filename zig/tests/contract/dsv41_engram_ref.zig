//! DeepSeek V4.1 Engram for the contract (model.py:313-323, 343-360): inputs, fp32 oracles, float64 reference.
const std = @import("std");
const contract = @import("contract");

const c = contract.spec;
const rows = contract.rows;
const fp8ref = @import("dsv41_fp8_ref.zig");

pub const head = 256;
pub const hc = 4;
pub const Stage = enum { embed, aq, wkv, gate, add };

pub fn e4m3(code: u8) f64 {
    if (code & 0x7f == 0x7f) return std.math.nan(f64);
    const e: i32 = (code >> 3) & 15;
    const m: f64 = @floatFromInt(code & 7);
    const v = if (e == 0) std.math.ldexp(m, -9) else std.math.ldexp(1 + m / 8, e - 7);
    return if (code & 0x80 != 0) -v else v;
}

pub fn e8m0(code: u8) f64 {
    return if (code == 255) std.math.nan(f64) else std.math.ldexp(@as(f64, 1), @as(i32, code) - 127);
}

/// An f64 value as the kernel stores it: rounded to fp32, then to bf16.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

fn gammaOf(d: usize) f64 {
    return c.gamma(@floatFromInt(d), 0x1p-24);
}

pub const Out = struct { oracle: f32, exact: f64, mag: f64 };
pub const GateOut = struct { oracle: f32, exact: f64, allow: f64 };

/// act_quant of one 32-block as the reference leaves it (ue8m0 scale, ties-to-even E4M3 codes); non-finite: all NaN.
pub fn aqBlock(src: []const u16, dst: []u16) void {
    var amax: f32 = 0;
    var bad = false;
    for (src) |b| {
        const v = c.fromBf16(b);
        bad = bad or !std.math.isFinite(v);
        amax = @max(amax, @abs(v));
    }
    const sb: u32 = @bitCast(@max(amax, 1e-4) * (1.0 / 448.0));
    const e: i32 = @as(i32, @intCast((sb >> 23) & 0xff)) - 127 + @intFromBool(sb & 0x7fffff != 0);
    for (src, dst) |b, *o| {
        if (bad) {
            o.* = c.toBf16(std.math.nan(f32));
            continue;
        }
        const v = c.fromBf16(b);
        const a: f64 = @min(@abs(@as(f64, v)) * std.math.ldexp(@as(f64, 1), -e), 448);
        var best: u8 = 0;
        for (1..127) |code| {
            const d1 = @abs(a - e4m3(@intCast(code)));
            const d0 = @abs(a - e4m3(best));
            if (d1 < d0 or (d1 == d0 and code % 2 == 0)) best = @intCast(code);
        }
        const val = e4m3(best) * std.math.ldexp(@as(f64, 1), e);
        o.* = c.toBf16(@floatCast(if (std.math.signbit(v)) -val else val));
    }
}

/// Host inputs of one stage; shape is {rows, n, k}: embed n = k = cols * 256, wkv k = cols * 256, gate and add k = dim.
pub const Data = struct {
    a: std.mem.Allocator,
    stage: Stage,
    shape: c.Shape,
    stress: bool = false,
    eps: f32 = 1e-6,
    codes: []u8 = &.{},
    scales: []u8 = &.{},
    emb: []u16 = &.{},
    f: ?fp8ref.Fp8 = null,
    x: []u16 = &.{},
    kv: []u16 = &.{},
    q: []u16 = &.{},
    kw: []u16 = &.{},
    mask: []u8 = &.{},
    g: []f32 = &.{},

    pub fn init(a: std.mem.Allocator, stage: Stage, s: c.Shape) !Data {
        var d: Data = .{ .a = a, .stage = stage, .shape = s };
        errdefer d.deinit();
        const r = s.rows;
        switch (stage) {
            .embed => {
                d.codes = try a.alloc(u8, r * s.k);
                d.scales = try a.alloc(u8, r * s.k / 32);
            },
            .aq => d.emb = try a.alloc(u16, r * s.k),
            .wkv => {
                d.emb = try a.alloc(u16, r * s.k);
                d.f = try fp8ref.Fp8.init(a, s);
            },
            .gate, .add => {
                d.x = try a.alloc(u16, r * hc * s.k);
                d.kv = try a.alloc(u16, r * (hc + 1) * s.k);
                if (stage == .gate) {
                    d.q = try a.alloc(u16, hc * s.k);
                    d.kw = try a.alloc(u16, hc * s.k);
                    d.mask = try a.alloc(u8, r);
                } else d.g = try a.alloc(f32, r * hc);
            },
        }
        return d;
    }

    pub fn deinit(d: *Data) void {
        d.a.free(d.codes);
        d.a.free(d.scales);
        d.a.free(d.emb);
        if (d.f) |f| f.deinit();
        d.a.free(d.x);
        d.a.free(d.kv);
        d.a.free(d.q);
        d.a.free(d.kw);
        d.a.free(d.mask);
        d.a.free(d.g);
    }

    pub fn outputs(d: Data) usize {
        const s = d.shape;
        return switch (d.stage) {
            .embed, .aq => s.rows * s.k,
            .wkv => s.rows * s.n,
            .gate => s.rows * hc,
            .add => s.rows * hc * s.k,
        };
    }

    pub fn fill(d: *Data, mode: rows.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        d.stress = mode == .stress;
        switch (d.stage) {
            .embed => d.fillEmbed(rng, mode),
            .aq => d.fillAq(rng, mode),
            .wkv => d.fillWkv(rng, mode),
            .gate => d.fillGate(rng, mode),
            .add => d.fillAdd(rng, mode),
        }
    }

    fn plant(v: []u16, at: usize, kind: usize) void {
        const nan = std.math.nan(f32);
        const put = [_]f32{ nan, std.math.inf(f32), -std.math.inf(f32) };
        v[at] = c.toBf16(put[kind % 3]);
    }

    fn fillEmbed(d: *Data, rng: std.Random, mode: rows.Fill) void {
        const k = d.shape.k;
        for (d.codes) |*q| {
            const b = rng.int(u8);
            q.* = if (b & 0x7f == 0x7f) b & 0xfe else b;
        }
        for (d.scales) |*s| s.* = 119 + rng.uintLessThan(u8, 7);
        if (mode != .nonfinite) return;
        const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
        const code = [_]u8{ 0x7f, 0x7e, 0xfe, 0x01, 0x7e };
        const scale = [_]u8{ 120, 254, 254, 255, 254 }; // 448 * 2^127 overflows to infinity; scale 255 is NaN
        for (at, code, scale) |p, q, s| if (p[0] < d.shape.rows) {
            d.codes[p[0] * k + p[1]] = q;
            d.scales[(p[0] * k + p[1]) / 32] = s;
        };
    }

    fn fillEmb(d: *Data, rng: std.Random, mode: rows.Fill) void {
        const k = d.shape.k;
        for (d.emb, 0..) |*v, i| {
            const big: f32 = if (i % 64 == 5) 60 else 1;
            v.* = if (i % 128 >= 96) 0 else c.toBf16((rng.float(f32) * 8 - 4) * big);
        }
        if (mode != .nonfinite) return;
        const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
        for (at, 0..) |p, i| if (p[0] < d.shape.rows) plant(d.emb, p[0] * k + p[1], i);
    }

    fn fillAq(d: *Data, rng: std.Random, mode: rows.Fill) void {
        d.fillEmb(rng, mode);
    }

    fn fillWkv(d: *Data, rng: std.Random, mode: rows.Fill) void {
        const f = d.f.?;
        f.fill(.random, rng.int(u64));
        d.fillEmb(rng, mode);
        d.quantise(f.x);
    }

    /// act_quant of every 32-block of emb, as the kernel leaves it, into `out`.
    fn quantise(d: Data, out: []u16) void {
        var i: usize = 0;
        while (i < d.emb.len) : (i += 32) aqBlock(d.emb[i..][0..32], out[i..][0..32]);
    }

    fn fillGate(d: *Data, rng: std.Random, mode: rows.Fill) void {
        const dim = d.shape.k;
        for (d.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (d.kv) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (d.q) |*w| w.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (d.kw) |*w| w.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (d.mask, 0..) |*m, r| m.* = @intFromBool(mode != .random or r % 3 != 2);
        if (mode == .nonfinite) {
            const at = [_][2]usize{ .{ 0, dim / 3 }, .{ 1, dim / 2 }, .{ 2, dim - 1 }, .{ 3, 0 }, .{ 3, dim - 1 } };
            for (at, 0..) |p, i| if (p[0] < d.shape.rows) plant(d.x, p[0] * hc * dim + p[1], i);
            if (d.shape.rows > 4) plant(d.kv, 4 * (hc + 1) * dim + dim + 5, 0);
        }
        if (mode != .stress) return;
        const h = dim / 2;
        for (0..hc) |q| for (0..h) |i| {
            const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 4));
            d.q[q * dim + i] = c.toBf16(w);
            d.q[q * dim + h + i] = c.toBf16(w);
            d.kw[q * dim + i] = c.toBf16(1);
            d.kw[q * dim + h + i] = c.toBf16(1);
        };
        for (0..d.shape.rows) |r| for (0..hc) |q| for (0..h) |i| {
            const xv: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 16));
            const kv: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 16));
            d.x[(r * hc + q) * dim + i] = c.toBf16(xv);
            d.x[(r * hc + q) * dim + h + i] = c.toBf16(xv + @as(f32, @floatFromInt(@intFromBool(i < 8))));
            d.kv[r * (hc + 1) * dim + q * dim + i] = c.toBf16(kv);
            d.kv[r * (hc + 1) * dim + q * dim + h + i] = c.toBf16(-kv);
        };
    }

    fn fillAdd(d: *Data, rng: std.Random, mode: rows.Fill) void {
        const dim = d.shape.k;
        for (d.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (d.kv) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (d.g, 0..) |*g, i| g.* = if (i % 7 == 3) 0 else rng.float(f32);
        if (mode != .nonfinite) return;
        const nan = std.math.nan(f32);
        if (d.shape.rows > 4) {
            plant(d.x, 1, 1);
            plant(d.kv, (hc + 1) * dim + 4 * dim + 2, 2);
            plant(d.kv, 2 * (hc + 1) * dim + 4 * dim + 7, 1);
            d.g[3 * hc + 1] = nan;
            d.g[4 * hc] = std.math.inf(f32);
            plant(d.kv, 4 * (hc + 1) * dim + 4 * dim, 0);
        }
    }

    /// weight = q_weight.float() * k_weight.float(): a product of two bf16 values, exact in fp32.
    fn weight(d: Data, h: usize, i: usize) f32 {
        return c.fromBf16(d.q[h * d.shape.k + i]) * c.fromBf16(d.kw[h * d.shape.k + i]);
    }

    const Sums = struct { sh: f64, sk: f64, sd: f64, mag: f64 };

    fn gateSums(d: Data, r: usize, h: usize) Sums {
        const dim = d.shape.k;
        var s: Sums = .{ .sh = 0, .sk = 0, .sd = 0, .mag = 0 };
        for (0..dim) |i| {
            const x: f64 = c.fromBf16(d.x[(r * hc + h) * dim + i]);
            const k: f64 = c.fromBf16(d.kv[r * (hc + 1) * dim + h * dim + i]);
            const w: f64 = d.weight(h, i);
            s.sh += x * x;
            s.sk += k * k;
            s.sd += x * w * k;
            s.mag += @abs(x * w * k);
        }
        return s;
    }

    fn dscale64(dim: usize) f64 {
        return std.math.pow(f64, @floatFromInt(dim), -0.5);
    }

    fn rstdOf(d: Data, sh: f64, sk: f64) f64 {
        const n: f64 = @floatFromInt(d.shape.k);
        return 1 / @sqrt(sh / n + d.eps) * (1 / @sqrt(sk / n + d.eps));
    }

    /// signed sqrt of the clamped |dot| then sigmoid (engram.py gate, clamp_min keeps NaN).
    pub fn gateOf(dot: f64) f64 {
        var m = @abs(dot);
        if (m < 1e-6) m = 1e-6;
        const z = std.math.copysign(@sqrt(m), dot);
        return 1 / (1 + @exp(-z));
    }

    fn gateFrom(d: Data, s: Sums) f64 {
        return gateOf(s.sd * d.rstdOf(s.sh, s.sk) * dscale64(d.shape.k));
    }

    /// Gate (r, h): the fp32 oracle, the float64 gate and the allowance over the dot's error (the gate is monotone).
    pub fn gateOne(d: Data, r: usize, h: usize) GateOut {
        const dim = d.shape.k;
        if (d.mask[r] == 0) return .{ .oracle = 0, .exact = 0, .allow = 0x1p-149 };
        var sh: [32]f32 = @splat(0);
        var sk: [32]f32 = @splat(0);
        var sd: [32]f32 = @splat(0);
        for (0..dim) |i| {
            const x = c.fromBf16(d.x[(r * hc + h) * dim + i]);
            const k = c.fromBf16(d.kv[r * (hc + 1) * dim + h * dim + i]);
            const xx = x * x;
            const kk = k * k;
            const xwk = (x * d.weight(h, i)) * k;
            sh[i % 32] += xx;
            sk[i % 32] += kk;
            sd[i % 32] += xwk;
        }
        var s: usize = 16;
        while (s > 0) : (s /= 2) for (0..s) |l| {
            sh[l] += sh[l + s];
            sk[l] += sk[l + s];
            sd[l] += sd[l + s];
        };
        const fd: f32 = @floatFromInt(dim);
        const rstd = (1 / @sqrt(sh[0] / fd + d.eps)) * (1 / @sqrt(sk[0] / fd + d.eps));
        const dot = (sd[0] * rstd) * @as(f32, @floatCast(dscale64(dim)));
        var m = @abs(dot);
        if (m < 1e-6) m = 1e-6;
        const z = std.math.copysign(@sqrt(m), dot);
        const oracle = 1 / (1 + @exp(-z));
        const e = d.gateSums(r, h);
        const dot64 = e.sd * d.rstdOf(e.sh, e.sk) * dscale64(dim);
        const want = gateOf(dot64);
        const g = gammaOf(dim / 32 + 8);
        const rho: f64 = (if (d.stress) 0 else 2 * g) + 0x1p-19;
        const sum_err = if (d.stress) 0 else g * e.mag * d.rstdOf(e.sh, e.sk) * dscale64(dim);
        const delta = rho * @abs(dot64) + sum_err;
        const spread = @max(want - gateOf(dot64 - delta), gateOf(dot64 + delta) - want);
        return .{ .oracle = oracle, .exact = want, .allow = spread + 0x1p-21 * want + 0x1p-149 };
    }

    pub fn addOne(d: Data, r: usize, i: usize) Out {
        const dim = d.shape.k;
        const h: f64 = c.fromBf16(d.x[r * hc * dim + i]);
        const v: f64 = c.fromBf16(d.kv[r * (hc + 1) * dim + hc * dim + i % dim]);
        const gv = @as(f64, d.g[r * hc + i / dim]) * v;
        return .{ .oracle = 0, .exact = h + gv, .mag = @abs(h) + @abs(gv) };
    }

    /// Float64 outputs `want` and the term magnitudes `mag`; wkv's want is its oracle after a drift check.
    pub fn reference(d: Data, want: []f64, mag: []f64) !void {
        const s = d.shape;
        switch (d.stage) {
            .embed => for (0..s.rows * s.k) |i| {
                want[i] = bf(e4m3(d.codes[i]) * e8m0(d.scales[i / 32]));
                mag[i] = @abs(want[i]);
            },
            .aq => {
                var q: [32]u16 = undefined;
                var i: usize = 0;
                while (i < d.emb.len) : (i += 32) {
                    aqBlock(d.emb[i..][0..32], &q);
                    for (q, 0..) |b, j| want[i + j] = c.fromBf16(b);
                }
                for (mag, want) |*m, w| m.* = @abs(w);
            },
            .wkv => try d.f.?.reference(want, mag, true),
            .gate => for (0..s.rows) |r| for (0..hc) |h| {
                const o = d.gateOne(r, h);
                want[r * hc + h] = o.exact;
                mag[r * hc + h] = 0;
                if (std.math.isFinite(o.exact) and @abs(@as(f64, o.oracle) - o.exact) > o.allow) return error.OracleDrift;
            },
            .add => for (0..s.rows * hc * s.k) |i| {
                const o = d.addOne(i / (hc * s.k), i % (hc * s.k));
                want[i] = o.exact;
                mag[i] = o.mag;
            },
        }
    }

    /// Exact order for embed and wkv (random inputs); the gate's spread; add's roundings.
    pub fn bounds(d: Data, want: []const f64, mag: []const f64, allow: []f64) !void {
        const s = d.shape;
        switch (d.stage) {
            .embed => @memset(allow, 0),
            .aq, .wkv => @memset(allow, 0),
            .gate => for (0..s.rows) |r| for (0..hc) |h| {
                allow[r * hc + h] = d.gateOne(r, h).allow;
            },
            .add => for (allow, want, mag) |*l, w, m| {
                l.* = 0x1p-8 * @abs(w) + 0x1p-22 * m + 0x1p-133;
            },
        }
    }

    /// Gate 4: each output if the sums ran in `acc` in `order` (wkv: the closer of a flat chain and 32 lane chains).
    pub fn simulate(d: Data, acc: c.Accum, order: rows.Order, out: []f64) !void {
        const k = d.shape.k;
        const t = try d.a.alloc(f64, 3 * k);
        defer d.a.free(t);
        if (d.stage == .gate) {
            for (0..d.shape.rows) |r| for (0..hc) |h| {
                const dim = k;
                for (0..dim) |i| {
                    const x: f64 = c.fromBf16(d.x[(r * hc + h) * dim + i]);
                    const key: f64 = c.fromBf16(d.kv[r * (hc + 1) * dim + h * dim + i]);
                    t[i] = x * x;
                    t[dim + i] = key * key;
                    t[2 * dim + i] = x * d.weight(h, i) * key;
                }
                const s: Sums = .{ .sh = rows.sumIn(acc, order, t[0..dim]), .sk = rows.sumIn(acc, order, t[dim .. 2 * dim]), .sd = rows.sumIn(acc, order, t[2 * dim ..]), .mag = 0 };
                out[r * hc + h] = d.gateFrom(s);
            };
            return;
        }
        return error.NoSums;
    }
};
