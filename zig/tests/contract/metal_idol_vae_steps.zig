//! iv_conv_in, iv_conv, iv_norm and iv_tokens under the contract: Idol's image encoder's fp32 steps against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");

const c = id.c;
const g = id.g;
const Range = id.Range;
const Ends = id.Ends;
pub const ks = tf.idol.vae_kernels;

pub fn ref(r: rig_.Ref) ks.Ref {
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
    if (x.len > 3 * apart + 1) {
        x[3 * apart] = inf;
        x[3 * apart + 1] = -inf;
    }
}

// --- iv_conv_in: conv3x3(2 x - 1) + b of the picture, 3 channels to 128 ----------------------------------------------

pub const ConvIn = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// The 27 taps' products, 2 x - 1's rounding and the bias.
    fn depth(_: c.Shape) usize {
        return 27 + 2;
    }

    /// Rows: the picture's rows; n: the 128 output channels; k: its columns (the encoder's picture is 1024 x 1024).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = .{ .{ .size = 2 }, .{}, .{ .size = 2 } },
            .served = &.{ .{ .rows = 12, .n = 128, .k = 1024 }, .{ .rows = 1024, .n = 128, .k = 12 }, .{ .rows = 37, .n = 128, .k = 41 } },
        };
    }

    pub fn deinit(t: *ConvIn) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *ConvIn) void {
        inline for (.{ &t.x, &t.w, &t.b }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }
    pub fn kernel(t: *ConvIn, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(ConvIn);

    pub fn prepare(t: *ConvIn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n != 128) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.rows * shape.k * 3);
        t.w = try t.a.alloc(f32, 27 * 128);
        t.b = try t.a.alloc(f32, 128);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // 2 x - 1 an odd integer (127 or 2041), channel 1 two more, channel 2 zero; weights 509, -509, 1 cancel
            for (0..shape.rows * shape.k) |p| {
                const v: f32 = if (p % 2 == 0) 2041 else 127;
                t.x[p * 3] = (v + 1) / 2;
                t.x[p * 3 + 1] = (v + 3) / 2;
                t.x[p * 3 + 2] = 0.5;
            }
            for (0..27) |r| @memset(t.w[r * 128 ..][0..128], switch (r % 3) {
                0 => 509,
                1 => -509,
                else => 1,
            });
            @memset(t.b, 0);
        } else {
            uniform(rng, t.x, 0, 1);
            uniform(rng, t.w, -0.5, 0.5);
            uniform(rng, t.b, -1, 1);
            if (fill == .nonfinite) plant(t.x, shape.k * 3);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *ConvIn) !void {
        const p = t.rig.begin();
        ks.convIn(p.e, t.pipe, ref(t.ins[0]), ref(t.ins[1]), ref(t.ins[2]), ref(t.out), t.shape.rows, t.shape.k);
        try t.rig.finish(p);
    }

    pub fn count(t: *ConvIn) usize {
        return t.shape.rows * t.shape.k * 128;
    }

    /// Output (pixel, o)'s 27 products in the kernel's order (row, column, channel), zero past the picture's edge.
    fn terms(t: *const ConvIn, px: usize, o: usize, out: *[27]f64) void {
        const py: isize = @intCast(px / t.shape.k);
        const pxx: isize = @intCast(px % t.shape.k);
        for (0..3) |ky| for (0..3) |kx| for (0..3) |ch| {
            const iy = py + @as(isize, @intCast(ky)) - 1;
            const ix = pxx + @as(isize, @intCast(kx)) - 1;
            const at = (ky * 3 + kx) * 3 + ch;
            out[at] = if (iy < 0 or ix < 0 or iy >= t.shape.rows or ix >= t.shape.k) 0 else blk: {
                const v: f64 = t.x[(@as(usize, @intCast(iy)) * t.shape.k + @as(usize, @intCast(ix))) * 3 + ch];
                break :blk (2 * v - 1) * t.w[at * 128 + o];
            };
        };
    }

    pub fn reference(t: *ConvIn, want: []f64, mag: []f64) !void {
        var buf: [27]f64 = undefined;
        for (0..t.shape.rows * t.shape.k) |px| for (0..128) |o| {
            t.terms(px, o, &buf);
            var s: f64 = 0;
            var m: f64 = 0;
            for (buf) |v| {
                s += v;
                m += @abs(v);
            }
            want[px * 128 + o] = s + t.b[o];
            m += @abs(@as(f64, t.b[o]));
            mag[px * 128 + o] = if (std.math.isFinite(m)) m else 0;
        };
    }

    pub fn simulate(t: *ConvIn, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [27]f64 = undefined;
        for (0..t.shape.rows * t.shape.k) |px| for (0..128) |o| {
            t.terms(px, o, &buf);
            out[px * 128 + o] = contract.rows.sumIn(acc, order, &buf) + t.b[o];
        };
    }
};

// --- iv_conv: a 1x1 or stride-2 3x3 convolution as an implicit GEMM, fp32 sums, + bias (+ a residual) ---------------

/// A form the encoder launches: its taps (1, or 9 at stride 2), a residual, and its served shapes.
pub const Form = struct { label: []const u8, taps: usize = 1, add: bool = false, served: []const c.Shape };

/// A 1x1's rows are pixels (fewer than the engine's: rows only tile); a 3x3's are output rows of 9 columns.
pub const conv_forms = [_]Form{
    .{ .label = "iv_conv 1x1", .served = &.{ .{ .rows = 1100, .n = 1536, .k = 512 }, .{ .rows = 1100, .n = 256, .k = 128 }, .{ .rows = 1100, .n = 512, .k = 256 }, .{ .rows = 1100, .n = 64, .k = 64 } } },
    .{ .label = "iv_conv 1x1 + residual", .add = true, .served = &.{.{ .rows = 1100, .n = 512, .k = 512 }} },
    .{ .label = "iv_conv 3x3 stride 2", .taps = 9, .served = &.{ .{ .rows = 8, .n = 128, .k = 128 }, .{ .rows = 8, .n = 256, .k = 256 }, .{ .rows = 8, .n = 512, .k = 512 } } },
};

const out_cols = 9;

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

    /// Each output sums taps * cin products, then the bias and the residual.
    fn depth1(s: c.Shape) usize {
        return s.k + 2;
    }
    fn depth9(s: c.Shape) usize {
        return 9 * s.k + 2;
    }

    /// Rows: pixels (1x1) or output rows (3x3); n: output channels; k: input channels.
    pub fn spec(name: []const u8, f: Form) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = if (f.taps == 9) depth9 else depth1,
            .out = .f32,
            .tiles = .{ if (f.taps == 9) .{ .size = 8 } else .{ .size = 64 }, .{ .size = 64 }, .{ .size = 32, .step = 32, .min = 32 } },
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
    fn cols(t: *const Conv) usize {
        return if (t.form.taps == 9) out_cols else 1;
    }
    fn pixels(t: *const Conv) usize {
        return t.shape.rows * t.cols();
    }
    fn inRows(t: *const Conv) usize {
        return if (t.form.taps == 9) 2 * t.shape.rows else t.shape.rows;
    }
    fn inCols(t: *const Conv) usize {
        return if (t.form.taps == 9) 2 * out_cols else 1;
    }

    pub fn prepare(t: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const L = t.ldw();
        const K = t.form.taps * shape.k;
        const in_px = t.inRows() * t.inCols();
        t.x = try t.a.alloc(f32, in_px * shape.k);
        t.w = try t.a.alloc(f32, K * L);
        t.b = try t.a.alloc(f32, shape.n);
        t.r = try t.a.alloc(f32, if (t.form.add) t.pixels() * shape.n else 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(t.w, 0);
        if (fill == .stress) {
            // the channels' second half mirroring the first, weights 1 then -1: partial sums reach ~2^18 and cancel
            const half = shape.k / 2;
            for (0..in_px) |p| for (0..shape.k) |ci| {
                const at = if (ci < half) ci else ci - half;
                const v: f32 = if (at % 2 == 0) 127 else 2041;
                t.x[p * shape.k + ci] = if (ci < half) v else v + @as(f32, if (at < 4) 1 else 0);
            };
            for (0..K) |k| for (0..shape.n) |o| {
                t.w[k * L + o] = if (k % shape.k < half) 1 else -1;
            };
            @memset(t.b, 0);
            for (t.r) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
        } else {
            uniform(rng, t.x, -2, 2);
            for (0..K) |k| uniform(rng, t.w[k * L ..][0..shape.n], -0.1, 0.1);
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
        const s: ks.ConvShape = .{ .rows = t.shape.rows, .cols = t.cols(), .in_rows = t.inRows(), .in_cols = t.inCols(), .cin = t.shape.k, .cout = t.shape.n, .ldw = t.ldw(), .taps = t.form.taps };
        ks.conv(p.e, t.pipe, ref(t.ins[0]), ref(t.ins[1]), ref(t.ins[2]), if (t.form.add) ref(t.ins[3]) else null, ref(t.out), s);
        try t.rig.finish(p);
    }

    pub fn count(t: *Conv) usize {
        return t.pixels() * t.shape.n;
    }

    /// Output (p, o)'s products in the kernel's order, tap then input channel; a tap past the right or lower edge is 0.
    fn terms(t: *const Conv, p: usize, o: usize, out: []f64) void {
        const K = t.shape.k;
        const oy = p / t.cols();
        const ox = p % t.cols();
        const stride: usize = if (t.form.taps == 9) 2 else 1;
        for (0..t.form.taps) |tap| {
            const iy = oy * stride + tap / 3;
            const ix = ox * stride + tap % 3;
            for (0..K) |ci| {
                const inside = iy < t.inRows() and ix < t.inCols();
                out[tap * K + ci] = if (inside) @as(f64, t.x[(iy * t.inCols() + ix) * K + ci]) * t.w[(tap * K + ci) * t.ldw() + o] else 0;
            }
        }
    }

    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        const buf = try t.a.alloc(f64, t.form.taps * t.shape.k);
        defer t.a.free(buf);
        for (0..t.pixels()) |p| for (0..t.shape.n) |o| {
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
        const buf = try t.a.alloc(f64, t.form.taps * t.shape.k);
        defer t.a.free(buf);
        for (0..t.pixels()) |p| for (0..t.shape.n) |o| {
            t.terms(p, o, buf);
            const i = p * t.shape.n + o;
            out[i] = contract.rows.sumIn(acc, order, buf) + t.b[o] + (if (t.form.add) t.r[i] else 0);
        };
    }
};

// --- iv_norm: GroupNorm by given statistics, then SiLU or not -------------------------------------------------------

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
    pub fn spec(name: []const u8, silu: bool) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = id.noSums,
            .ops = 4 + id.exp_ulps + id.div_ulps + 1,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{ .size = 128, .step = 32, .min = 32 }, .{} },
            .served = if (silu) &.{ .{ .rows = 4100, .n = 128, .k = 1 }, .{ .rows = 4100, .n = 256, .k = 1 }, .{ .rows = 4100, .n = 512, .k = 1 } } else &.{.{ .rows = 4100, .n = 512, .k = 1 }},
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
        return .{ .spec = spec(t.name, t.silu), .ctx = t, .vt = &vt, .control = control };
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
        ks.normed(p.e, t.pipe, ref(t.ins[0]), ref(t.ins[1]), ref(t.ins[2]), ref(t.ins[3]), ref(t.out), t.shape.rows, t.shape.n, t.silu);
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
                const e = (Range{ .lo = @exp(-vr.hi), .hi = @exp(-vr.lo) }).fp32(id.exp_ulps);
                vr = vr.mul(e.add(Range.of(1)).fp32(0.5).recip()).fp32(id.div_ulps);
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

// --- iv_tokens: the posterior's sample, patchified and normalized by the batch norm's statistics --------------------

pub const Tokens = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    moments: []f32 = &.{},
    noise: []f32 = &.{},
    m: [128]f32 = undefined,
    s: [128]f32 = undefined,
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    const pad = 5;

    /// Rows: the latent grid's rows; n: the 128 channels of a row; k: the grid's columns.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = id.noSums,
            .ops = id.exp_ulps + id.div_ulps + 3,
            .out = .f32,
            .tiles = .{ .{ .size = 2, .step = 2, .min = 2 }, .{}, .{ .size = 2, .step = 2, .min = 2 } },
            .served = &.{ .{ .rows = 128, .n = 128, .k = 128 }, .{ .rows = 6, .n = 128, .k = 10 } },
        };
    }

    pub fn deinit(t: *Tokens) void {
        t.a.free(t.moments);
        t.a.free(t.noise);
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Tokens, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Tokens);

    fn latents(t: *const Tokens) usize {
        return t.shape.rows * t.shape.k;
    }

    pub fn prepare(t: *Tokens, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != 128 or shape.rows == 0 or shape.k == 0 or shape.rows % 2 != 0 or shape.k % 2 != 0) return false;
        t.rig.reset();
        t.a.free(t.moments);
        t.a.free(t.noise);
        t.shape = shape;
        const px = t.latents();
        t.moments = try t.a.alloc(f32, px * 64);
        t.noise = try t.a.alloc(f32, px * 32);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..px) |p| {
            uniform(rng, t.moments[p * 64 ..][0..32], -3, 3);
            // logvars past both of the clamp's ends (-30, 20) as well as inside them
            uniform(rng, t.moments[p * 64 + 32 ..][0..32], -40, 30);
        }
        uniform(rng, t.noise, -3, 3);
        uniform(rng, &t.m, -1, 1);
        uniform(rng, &t.s, 0.5, 4);
        if (fill == .nonfinite) {
            plant(t.moments, 64);
            plant(t.noise, 32);
            t.moments[5 * 64 + 40] = nan;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.moments));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.noise));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(&t.m));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(&t.s));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Tokens) !void {
        const p = t.rig.begin();
        ks.tokens(p.e, t.pipe, ref(t.ins[0]), ref(t.ins[1]), ref(t.ins[2]), ref(t.ins[3]), ref(t.out), t.shape.rows, t.shape.k, pad);
        try t.rig.finish(p);
    }

    pub fn count(t: *Tokens) usize {
        return (pad + t.latents() / 4) * 128;
    }

    /// torch.clamp(x, -30, 20): NaN stays NaN.
    fn clamp(x: f64) f64 {
        return if (std.math.isNan(x)) x else std.math.clamp(x, -30, 20);
    }

    pub fn reference(t: *Tokens, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..pad * 128) |i| t.ends.put(want, mag, i, 0, Range.of(0));
        const w2 = t.shape.k / 2;
        for (0..t.latents() / 4) |patch| for (0..128) |k| {
            const ch = k / 4;
            const py = (patch / w2) * 2 + (k % 4) / 2;
            const px = (patch % w2) * 2 + k % 2;
            const p = py * t.shape.k + px;
            const mean: f64 = t.moments[p * 64 + ch];
            const lv = clamp(t.moments[p * 64 + 32 + ch]);
            const n: f64 = t.noise[p * 32 + ch];
            const sd = @exp(0.5 * lv);
            const z = mean + sd * n;
            const v = (z - t.m[k]) / t.s[k];
            const er = (Range{ .lo = @exp(0.5 * lv), .hi = @exp(0.5 * lv) }).fp32(id.exp_ulps);
            const zr = er.mul(Range.of(n)).fp32(0.5).add(Range.of(mean)).fp32(0.5);
            const vr = zr.sub(Range.of(t.m[k])).fp32(0.5).mul(Range.of(1 / @as(f64, t.s[k]))).fp32(id.div_ulps);
            t.ends.put(want, mag, (pad + patch) * 128 + k, v, vr);
        };
    }
    pub fn bounds(t: *Tokens, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Tokens, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};
