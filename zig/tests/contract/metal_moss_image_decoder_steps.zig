//! md_unpack, md_conv (1x1) and md_norm under the contract: fp32 steps against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const mi = @import("metal_moss_image_common.zig");

const c = mi.c;
const g = mi.g;
const Range = mi.Range;
const Ends = mi.Ends;
const dec = tf.moss_image.decoder;

pub fn ref(r: rig_.Ref) mi.ks.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

pub fn uniform(rng: std.Random, out: []f32, lo: f32, hi: f32) void {
    for (out) |*v| v.* = lo + rng.float(f32) * (hi - lo);
}

pub const nan = std.math.nan(f32);
pub const inf = std.math.inf(f32);

/// NaN, +Inf, -Inf at three places, and +Inf with -Inf at two places `apart` further on (one row holding both).
pub fn plant(x: []f32, apart: usize) void {
    for ([_]usize{ 0, 1, 2 }, [_]f32{ nan, inf, -inf }) |at, v| x[(at * 7) % x.len] = v;
    if (x.len > 3 * apart) {
        x[3 * apart] = inf;
        x[3 * apart + 1] = -inf;
    }
}

// --- md_unpack: z = x s + m, each 2 x 2 patch of a latent row unpacked ---------------------------------------------

pub const Unpack = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    s: [128]f32 = undefined,
    m: [128]f32 = undefined,
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: the latent grid's rows; n: its columns (k unused).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            .ops = 2,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{ .size = 8 }, .{} },
            .served = &.{ .{ .rows = 64, .n = 64, .k = 1 }, .{ .rows = 55, .n = 75, .k = 1 }, .{ .rows = 48, .n = 84, .k = 1 } },
        };
    }

    pub fn deinit(t: *Unpack) void {
        t.a.free(t.x);
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Unpack, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Unpack);

    pub fn prepare(t: *Unpack, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.rows * shape.n * 128);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        uniform(rng, t.x, -3, 3);
        uniform(rng, &t.s, 0.5, 2);
        uniform(rng, &t.m, -1, 1);
        if (fill == .nonfinite) plant(t.x, 128);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(&t.s));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(&t.m));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Unpack) !void {
        const p = t.rig.begin();
        dec.unpack(p.e, t.pipe, ref(t.ins[0]), ref(t.ins[1]), ref(t.ins[2]), ref(t.out), t.shape.rows, t.shape.n);
        try t.rig.finish(p);
    }

    pub fn count(t: *Unpack) usize {
        return t.shape.rows * t.shape.n * 128;
    }

    pub fn reference(t: *Unpack, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const w2 = t.shape.n * 2;
        for (0..t.shape.rows * 2 * w2) |px| for (0..32) |ch| {
            const py = px / w2;
            const pxx = px % w2;
            const k = ch * 4 + (py % 2) * 2 + pxx % 2;
            const x: f64 = t.x[((py / 2) * t.shape.n + pxx / 2) * 128 + k];
            const r = Range.of(x * t.s[k]).fp32(0.5).add(Range.of(t.m[k])).fp32(0.5);
            t.ends.put(want, mag, px * 32 + ch, x * t.s[k] + t.m[k], r);
        };
    }
    pub fn bounds(t: *Unpack, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Unpack, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

// --- md_conv: a 1x1 convolution, fp32 sums, + bias (+ a residual) ------------------------------------------------

/// A 1x1 form the decoder launches: whether it adds a residual, and its served shapes.
pub const Form = struct { label: []const u8, add: bool = false, served: []const c.Shape };

pub const conv_forms = [_]Form{
    .{ .label = "1x1", .served = &.{ .{ .rows = 4100, .n = 1152, .k = 384 }, .{ .rows = 517, .n = 32, .k = 32 }, .{ .rows = 517, .n = 192, .k = 384 }, .{ .rows = 517, .n = 96, .k = 192 } } },
    .{ .label = "1x1 + residual", .add = true, .served = &.{.{ .rows = 4100, .n = 384, .k = 384 }} },
};

pub const Conv = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    form: Form,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    r: []f32 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Each output sums cin products, then the bias and the residual.
    fn depth(s: c.Shape) usize {
        return s.k + 2;
    }

    /// Rows: pixels; n: output channels; k: input channels.
    pub fn spec(name: []const u8, f: Form) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = .{ .{ .size = 64 }, .{ .size = 64 }, .{ .size = 32, .step = 32, .min = 32 } },
            .served = f.served,
        };
    }

    pub fn deinit(t: *Conv) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Conv) void {
        inline for (.{ &t.x, &t.w, &t.b, &t.r }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }
    pub fn kernel(t: *Conv, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Conv);

    fn ldw(t: *const Conv) usize {
        return std.mem.alignForward(usize, t.shape.n, 64);
    }

    pub fn prepare(t: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const L = t.ldw();
        t.x = try t.a.alloc(f32, shape.rows * shape.k);
        t.w = try t.a.alloc(f32, shape.k * L);
        t.b = try t.a.alloc(f32, shape.n);
        t.r = try t.a.alloc(f32, if (t.form.add) shape.rows * shape.n else 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(t.w, 0);
        if (fill == .stress) {
            // the channels' second half mirroring the first, weights 1 then -1: partial sums reach ~2^18 and cancel
            const half = shape.k / 2;
            for (0..shape.rows) |p| for (0..shape.k) |ci| {
                const at = if (ci < half) ci else ci - half;
                const v: f32 = if (at % 2 == 0) 127 else 2041;
                t.x[p * shape.k + ci] = if (ci < half) v else v + @as(f32, if (at < 4) 1 else 0);
            };
            for (0..shape.k) |k| for (0..shape.n) |o| {
                t.w[k * L + o] = if (k < half) 1 else -1;
            };
            @memset(t.b, 0);
            for (t.r) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
        } else {
            uniform(rng, t.x, -2, 2);
            for (0..shape.k) |k| uniform(rng, t.w[k * L ..][0..shape.n], -0.1, 0.1);
            uniform(rng, t.b, -1, 1);
            uniform(rng, t.r, -3, 3);
            if (fill == .nonfinite) plant(t.x, shape.k);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        if (t.form.add) t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.r));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Conv) !void {
        const p = t.rig.begin();
        const s: dec.ConvShape = .{ .rows = t.shape.rows, .cols = 1, .cin = t.shape.k, .cout = t.shape.n, .ldw = t.ldw() };
        dec.convolve(p.e, t.pipe, ref(t.ins[0]), ref(t.ins[1]), ref(t.ins[2]), if (t.form.add) ref(t.ins[3]) else null, ref(t.out), s);
        try t.rig.finish(p);
    }

    pub fn count(t: *Conv) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Each output's products, in the kernel's order over the input channels.
    fn terms(t: *const Conv, p: usize, o: usize, out: []f64) void {
        for (out, 0..) |*v, ci| v.* = @as(f64, t.x[p * t.shape.k + ci]) * t.w[ci * t.ldw() + o];
    }

    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        for (0..t.shape.rows) |p| for (0..t.shape.n) |o| {
            t.terms(p, o, buf);
            var sum: f64 = 0;
            var m: f64 = 0;
            for (buf) |v| {
                sum += v;
                m += @abs(v);
            }
            const i = p * t.shape.n + o;
            const r: f64 = if (t.form.add) t.r[i] else 0;
            want[i] = sum + t.b[o] + r;
            m += @abs(@as(f64, t.b[o])) + @abs(r);
            mag[i] = if (std.math.isFinite(m)) m else 0;
        };
    }

    pub fn simulate(t: *Conv, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        for (0..t.shape.rows) |p| for (0..t.shape.n) |o| {
            t.terms(p, o, buf);
            const i = p * t.shape.n + o;
            out[i] = contract.rows.sumIn(acc, order, buf) + t.b[o] + (if (t.form.add) t.r[i] else 0);
        };
    }
};

// --- md_norm: GroupNorm by given statistics, then SiLU or not ------------------------------------------------------

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    silu: bool,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    stats: [64]f32 = undefined,
    gamma: []f32 = &.{},
    beta: []f32 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: pixels; n: channels (32 groups of n / 32).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            .ops = 4 + mi.exp_ulps + mi.div_ulps + 1,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{ .size = 96, .step = 32, .min = 32 }, .{} },
            .served = &.{ .{ .rows = 4100, .n = 384, .k = 1 }, .{ .rows = 4100, .n = 192, .k = 1 }, .{ .rows = 4100, .n = 96, .k = 1 } },
        };
    }

    pub fn deinit(t: *Norm) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Norm) void {
        inline for (.{ &t.x, &t.gamma, &t.beta }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }
    pub fn kernel(t: *Norm, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Norm);

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.rows * shape.n);
        t.gamma = try t.a.alloc(f32, shape.n);
        t.beta = try t.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // values stay within +-60 after the norm, short of fp32 exp's overflow in the SiLU
        uniform(rng, t.x, -4, 4);
        uniform(rng, t.stats[0..32], -1, 1);
        uniform(rng, t.stats[32..], 0.2, 3);
        uniform(rng, t.gamma, -2, 2);
        uniform(rng, t.beta, -1, 1);
        if (fill == .nonfinite) plant(t.x, shape.n);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(&t.stats));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.gamma));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.beta));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Norm) !void {
        const p = t.rig.begin();
        dec.normed(p.e, t.pipe, ref(t.ins[0]), ref(t.ins[1]), ref(t.ins[2]), ref(t.ins[3]), ref(t.out), t.shape.rows, t.shape.n, t.silu);
        try t.rig.finish(p);
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const size = t.shape.n / 32;
        for (t.x, 0..) |xv, i| {
            const ch = i % t.shape.n;
            const grp = ch / size;
            const x: f64 = xv;
            const m: f64 = t.stats[grp];
            const r: f64 = t.stats[32 + grp];
            const gm: f64 = t.gamma[ch];
            const b: f64 = t.beta[ch];
            const v = (x - m) * r * gm + b;
            var vr = Range.of(x - m).fp32(0.5).mul(Range.of(r)).fp32(0.5).mul(Range.of(gm)).fp32(0.5).add(Range.of(b)).fp32(0.5);
            var w = v;
            if (t.silu) {
                w = v / (1 + @exp(-v));
                const e = (Range{ .lo = @exp(-vr.hi), .hi = @exp(-vr.lo) }).fp32(mi.exp_ulps);
                vr = vr.mul(e.add(Range.of(1)).fp32(0.5).recip()).fp32(mi.div_ulps);
            }
            t.ends.put(want, mag, i, w, vr);
        }
    }
    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};
