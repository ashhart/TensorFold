//! The encoder's me_gdn_prep, me_gdn and me_attention under the contract, against float64 within their fp32 error.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const mi = @import("metal_moss_image_common.zig");

const c = mi.c;
const g = mi.g;
const ks = mi.ks;
const ec = tf.moss_image.encoder_config;
const Range = mi.Range;
const Ends = mi.Ends;
const bf = mi.bf;
const kref = mi.kref;
const K = ec.k_dim;
const u = c.unit(.f32);

fn exp(x: f64) f64 {
    return @exp(x);
}
fn log(x: f64) f64 {
    return @log(x);
}

// --- me_gdn_prep: q, k L2-normed (q scaled), beta = bf16(sigmoid(b)), g = -exp(A_log) softplus(a + dt_bias) ---------

fn prepDepth(_: c.Shape) usize {
    return 4 + 5;
}

pub const Prep = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    mix: []u16 = &.{},
    /// a, then b [rows, vheads]
    ab: []u16 = &.{},
    /// A_log, then dt_bias [vheads]
    decay: []u16 = &.{},
    ins: [5]rig_.Ref = undefined,
    outs: [4]rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: key heads (16, two value heads each); k: the key width.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = prepDepth,
            .ops = 2 * (mi.rsqrt_ulps + mi.exp_ulps + mi.log_ulps) + 4,
            .out = .f32,
            .served = &.{ .{ .rows = 63, .n = ec.k_heads, .k = K }, .{ .rows = 10, .n = ec.k_heads, .k = K }, .{ .rows = 4096, .n = ec.k_heads, .k = K } },
        };
    }

    pub fn deinit(t: *Prep) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Prep) void {
        for ([_]*[]u16{ &t.mix, &t.ab, &t.decay }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }
    pub fn kernel(t: *Prep, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Prep);

    fn vheads(t: *const Prep) usize {
        return 2 * t.shape.n;
    }
    fn stride(t: *const Prep) usize {
        return 2 * t.shape.n * K + t.vheads() * K;
    }
    fn params(t: *const Prep) ks.Gdn {
        return .{ .rows = @intCast(t.shape.rows), .kheads = @intCast(t.shape.n), .vheads = @intCast(t.vheads()), .mix_stride = @intCast(t.stride()), .scale = 1.0 / @sqrt(@as(f32, K)) };
    }

    pub fn prepare(t: *Prep, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != K) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const vh = t.vheads();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.mix = try t.a.alloc(u16, shape.rows * t.stride());
        if (fill == .stress) mi.integers(rng, t.mix, 128) else mi.randomBf16(rng, t.mix, -3, 3);
        t.ab = try t.a.alloc(u16, 2 * shape.rows * vh);
        mi.randomBf16(rng, t.ab, -6, 6);
        t.decay = try t.a.alloc(u16, 2 * vh);
        mi.randomBf16(rng, t.decay[0..vh], -4, 3);
        mi.randomBf16(rng, t.decay[vh..], -2, 2);
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
                t.mix[i * t.stride() + i * 37] = v;
                t.mix[(i + 1) * t.stride() + shape.n * K + i * 53] = v;
                t.ab[(i + 2) * vh + i] = v;
                t.ab[shape.rows * vh + (i + 3) * vh + i] = v;
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.mix));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.ab[0 .. shape.rows * vh]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.ab[shape.rows * vh ..]));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.decay[0..vh]));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.decay[vh..]));
        const qk = shape.rows * shape.n * K * 4;
        for ([_]usize{ qk, qk, shape.rows * vh * 4, shape.rows * vh * 4 }, &t.outs) |n, *o| o.* = try t.rig.output(n);
        return true;
    }

    pub fn run(t: *Prep) !void {
        const p = t.rig.begin();
        ks.encoderGdnPrep(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.ins[3]), kref(t.ins[4]), kref(t.outs[0]), kref(t.outs[1]), kref(t.outs[2]), kref(t.outs[3]), t.params());
        try t.rig.finish(p);
    }

    pub fn count(t: *Prep) usize {
        return 2 * t.shape.rows * t.shape.n * K + 2 * t.shape.rows * t.vheads();
    }

    /// Row r's key head h of q (0) or k (1), normed from its sum of squares `ss` (range `ssr`).
    fn head(t: *Prep, ends: *Ends, which: usize, r: usize, h: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const x = t.mix[r * t.stride() + which * t.shape.n * K + h * K ..][0..K];
        const inv = 1 / @sqrt(ss + 1e-6);
        const invr = ssr.add(Range.of(@as(f32, 1e-6))).fp32(0.5).rsqrt().fp32(mi.rsqrt_ulps);
        const scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, K)));
        const base = which * t.shape.rows * t.shape.n * K + (r * t.shape.n + h) * K;
        for (x, 0..) |v, i| {
            const f: f64 = c.fromBf16(v);
            var y = f * inv;
            var yr = Range.of(f).mul(invr).fp32(0.5);
            if (which == 0) {
                y *= scale;
                yr = yr.mul(Range.of(scale)).fp32(0.5);
            }
            ends.put(want, mag, base + i, y, yr);
        }
    }

    /// beta and g of every row and value head.
    fn gates(t: *Prep, ends: *Ends, want: []f64, mag: []f64) void {
        const vh = t.vheads();
        const n = t.shape.rows * vh;
        const at = 2 * t.shape.rows * t.shape.n * K;
        for (0..n) |i| {
            const v = i % vh;
            const b: f64 = c.fromBf16(t.ab[n + i]);
            ends.put(want, mag, at + n + i, bf(1 / (1 + @exp(-b))), mi.sigmoid(b).bf16());
            const x: f64 = @as(f32, @floatCast(@as(f64, c.fromBf16(t.ab[i])) + c.fromBf16(t.decay[vh + v])));
            const a_log: f64 = c.fromBf16(t.decay[v]);
            const soft = if (x > 20) x else @log(1 + @exp(x));
            const xr = Range.of(@as(f64, c.fromBf16(t.ab[i])) + c.fromBf16(t.decay[vh + v])).fp32(0.5);
            const sr = if (x > 20) xr else xr.up(exp).fp32(mi.exp_ulps).add(Range.of(1)).fp32(0.5).up(log).fp32(mi.log_ulps);
            const er = Range.of(@exp(a_log)).fp32(mi.exp_ulps);
            ends.put(want, mag, at + i, -@exp(a_log) * soft, (Range{ .lo = -er.hi, .hi = -er.lo }).mul(sr).fp32(0.5));
        }
    }

    pub fn reference(t: *Prep, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..2) |which| for (0..t.shape.rows) |r| for (0..t.shape.n) |h| {
            const s = mi.squares(t.mix[r * t.stride() + which * t.shape.n * K + h * K ..][0..K], prepDepth(t.shape), t.fill == .stress);
            t.head(&t.ends, which, r, h, s.ss, s.r, want, mag);
        };
        t.gates(&t.ends, want, mag);
    }
    pub fn bounds(t: *Prep, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Prep, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Prep, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var terms: [K]f64 = undefined;
        for (0..2) |which| for (0..t.shape.rows) |r| for (0..t.shape.n) |h| {
            for (&terms, t.mix[r * t.stride() + which * t.shape.n * K + h * K ..][0..K]) |*v, x| {
                const f: f64 = c.fromBf16(x);
                v.* = f * f;
            }
            const ss = contract.rows.sumIn(acc, order, &terms);
            t.head(&scratch, which, r, h, ss, Range.of(ss), out, mag);
        };
        t.gates(&scratch, out, mag);
    }
};

// --- me_gdn: S = S exp(g); d = (v - S^T k) beta; S += k d^T; o = S^T q, over the rows in order -----------------------

/// A step's two 128-term sums, the update and the decay, every row before it.
fn gdnDepth(s: c.Shape) usize {
    return s.rows * (2 * K + 8);
}

pub const Gdn = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    stress: bool = false,
    /// q then k, fp32 [rows, kheads, 128]
    qk: []f32 = &.{},
    mix: []u16 = &.{},
    /// g then beta, fp32 [rows, vheads]
    gb: []f32 = &.{},
    ins: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: positions; n: value heads (two a key head; each its own threadgroup, so 4 stand for 32); k: the width.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = gdnDepth,
            .ops = 2 * mi.exp_ulps,
            .out = .bf16,
            .served = &.{ .{ .rows = 63, .n = ec.v_heads, .k = K }, .{ .rows = 10, .n = ec.v_heads, .k = K }, .{ .rows = 4096, .n = 4, .k = K } },
        };
    }

    pub fn deinit(t: *Gdn) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Gdn) void {
        t.a.free(t.qk);
        t.a.free(t.mix);
        t.a.free(t.gb);
        t.qk = &.{};
        t.mix = &.{};
        t.gb = &.{};
    }
    pub fn kernel(t: *Gdn, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Gdn);

    fn kheads(t: *const Gdn) usize {
        return t.shape.n / 2;
    }
    fn stride(t: *const Gdn) usize {
        return 2 * t.kheads() * K + t.shape.n * K;
    }
    fn params(t: *const Gdn) ks.Gdn {
        return .{ .rows = @intCast(t.shape.rows), .kheads = @intCast(t.kheads()), .vheads = @intCast(t.shape.n), .mix_stride = @intCast(t.stride()), .scale = 1 };
    }

    pub fn prepare(t: *Gdn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 2 != 0 or shape.k != K) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.stress = fill == .stress;
        const T = shape.rows;
        const kh = t.kheads();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.qk = try t.a.alloc(f32, 2 * T * kh * K);
        t.mix = try t.a.alloc(u16, T * t.stride());
        t.gb = try t.a.alloc(f32, 2 * T * shape.n);
        mi.randomBf16(rng, t.mix, -2, 2);
        if (t.stress) {
            // no decay, beta 1, one-hot keys: columns 0, 1, 2 hold j, 2^20, -2^20; q sums them back to j, exact in fp32
            @memset(t.qk, 0);
            @memset(t.gb[0 .. T * shape.n], 0);
            @memset(t.gb[T * shape.n ..], 1);
            for (0..T) |r| for (0..kh) |h| {
                const at = (r * kh + h) * K;
                for (0..3) |i| t.qk[at + i] = 1;
                t.qk[T * kh * K + at + r % 3] = 1;
            };
            for (0..T) |r| for (0..shape.n) |h| for (0..K) |j| {
                const v: f32 = switch (r % 3) {
                    0 => @floatFromInt(j),
                    1 => 0x1p20,
                    else => -0x1p20,
                };
                t.mix[r * t.stride() + 2 * kh * K + h * K + j] = c.toBf16(v);
            };
        } else {
            // unit keys and queries (the prep's output), decays in (0, 1), beta in (0, 1)
            var row: usize = 0;
            while (row < 2 * T * kh) : (row += 1) {
                const x = t.qk[row * K ..][0..K];
                var ss: f32 = 0;
                for (x) |*v| {
                    v.* = rng.float(f32) * 2 - 1;
                    ss += v.* * v.*;
                }
                for (x) |*v| v.* /= @sqrt(ss);
            }
            for (t.gb[0 .. T * shape.n]) |*v| v.* = -rng.float(f32) * 2;
            for (t.gb[T * shape.n ..]) |*v| v.* = c.fromBf16(c.toBf16(rng.float(f32)));
        }
        if (fill == .nonfinite) {
            t.mix[2 * t.stride() + 2 * kh * K + 5] = mi.nan16;
            t.mix[3 * t.stride() + 2 * kh * K + K + 9] = mi.inf16;
            t.qk[(4 * kh + 0) * K + 7] = std.math.inf(f32);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.qk[0 .. T * kh * K]));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.qk[T * kh * K ..]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.mix));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.gb[0 .. T * shape.n]));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.gb[T * shape.n ..]));
        t.out = try t.rig.output(T * shape.n * K * 2);
        return true;
    }

    pub fn run(t: *Gdn) !void {
        const p = t.rig.begin();
        ks.encoderGdn(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.ins[3]), kref(t.ins[4]), kref(t.out), t.params());
        try t.rig.finish(p);
    }

    pub fn count(t: *Gdn) usize {
        return t.shape.rows * t.shape.n * K;
    }

    /// The recurrence in float64 (sums in `acc`, `order` if given); each allowance carries every earlier step's error.
    fn recur(t: *Gdn, want: []f64, allow: ?[]f64, sums: ?struct { acc: c.Accum, order: g.Order }) !void {
        const T = t.shape.rows;
        const vh = t.shape.n;
        const kh = t.kheads();
        const s = try t.a.alloc(f64, 2 * K * K);
        defer t.a.free(s);
        const st = s[0 .. K * K];
        const mag = s[K * K ..];
        var err: [K]f64 = undefined;
        var terms: [K]f64 = undefined;
        for (0..vh) |h| {
            @memset(s, 0);
            @memset(&err, 0);
            for (0..T) |r| {
                const q = t.qk[(r * kh + h / 2) * K ..][0..K];
                const k = t.qk[T * kh * K + (r * kh + h / 2) * K ..][0..K];
                const decay = @exp(@as(f64, t.gb[r * vh + h]));
                const beta: f64 = t.gb[T * vh + r * vh + h];
                var qn: f64 = 0;
                var kn: f64 = 0;
                for (q, k) |a, b| {
                    qn += @as(f64, a) * a;
                    kn += @as(f64, b) * b;
                }
                for (0..K) |j| {
                    const v: f64 = c.fromBf16(t.mix[r * t.stride() + 2 * kh * K + h * K + j]);
                    var kv: f64 = 0;
                    var kv_mag: f64 = 0;
                    for (0..K) |i| {
                        st[i * K + j] *= decay;
                        mag[i * K + j] *= decay;
                        terms[i] = st[i * K + j] * k[i];
                        kv_mag += mag[i * K + j] * @abs(k[i]);
                    }
                    kv = if (sums) |x| contract.rows.sumIn(x.acc, x.order, &terms) else blk: {
                        var sum: f64 = 0;
                        for (terms) |v2| sum += v2;
                        break :blk sum;
                    };
                    const delta = (v - kv) * beta;
                    const delta_mag = (@abs(v) + kv_mag) * beta;
                    var col: f64 = 0;
                    var o_mag: f64 = 0;
                    for (0..K) |i| {
                        st[i * K + j] += k[i] * delta;
                        mag[i * K + j] += @abs(k[i]) * delta_mag;
                        col += mag[i * K + j] * mag[i * K + j];
                        terms[i] = st[i * K + j] * q[i];
                        o_mag += mag[i * K + j] * @abs(q[i]);
                    }
                    const o = if (sums) |x| contract.rows.sumIn(x.acc, x.order, &terms) else blk: {
                        var sum: f64 = 0;
                        for (terms) |v2| sum += v2;
                        break :blk sum;
                    };
                    const at = (r * vh + h) * K + j;
                    want[at] = o;
                    if (allow) |al| {
                        // this step's error in the column (2-norm): the decay, the key sum, the delta and the update
                        err[j] += (2 * mi.exp_ulps * u + 3 * u) * @sqrt(col) + c.gamma(K, u) * kv_mag * beta * @sqrt(kn) + u * delta_mag * @sqrt(kn);
                        const work = err[j] * @sqrt(qn) + c.gamma(K, u) * o_mag;
                        al[at] = if (std.math.isFinite(o)) work + 0x1p-8 * (@abs(o) + work) + 0x1p-133 else 0;
                    }
                }
            }
        }
    }

    pub fn reference(t: *Gdn, want: []f64, mag: []f64) !void {
        try t.recur(want, mag, null);
    }
    pub fn bounds(t: *Gdn, want: []const f64, tolerance: []const f64, allow: []f64) !void {
        for (want, tolerance, allow) |w, tl, *l| l.* = if (t.stress) c.boundExact(spec(t.name), w) else tl;
    }
    pub fn simulate(t: *Gdn, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!t.stress) return error.GdnStressNeedsExactSums;
        try t.recur(out, null, .{ .acc = acc, .order = order });
    }
};

// --- me_attention: causal softmax(q k^T / 16) v in fp32 over each row's keys, against attention_ref -----------------

fn attentionDepth(s: c.Shape) usize {
    return ec.head_dim + s.k;
}

pub const Attention = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    stress: bool = false,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: query heads (four a key head; each its own threadgroups, so 8 stand for 16); n: head width; k: rows L.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = attentionDepth,
            .ops = 8,
            .out = .bf16,
            .served = &.{ .{ .rows = 8, .n = ec.head_dim, .k = 63 }, .{ .rows = 8, .n = ec.head_dim, .k = 10 }, .{ .rows = 8, .n = ec.head_dim, .k = 34 }, .{ .rows = 8, .n = ec.head_dim, .k = 4096 } },
        };
    }

    pub fn deinit(t: *Attention) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Attention) void {
        t.a.free(t.q);
        t.a.free(t.k);
        t.a.free(t.v);
        t.q = &.{};
        t.k = &.{};
        t.v = &.{};
    }
    pub fn kernel(t: *Attention, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Attention);

    fn kvHeads(t: *const Attention) usize {
        return t.shape.rows / 4;
    }

    pub fn prepare(t: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows % 4 != 0 or shape.n != ec.head_dim or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.stress = fill == .stress;
        const L = shape.k;
        const D = ec.head_dim;
        t.q = try t.a.alloc(u16, shape.rows * L * D);
        t.k = try t.a.alloc(u16, t.kvHeads() * L * D);
        t.v = try t.a.alloc(u16, t.kvHeads() * L * D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // equal scores; each column's values partial sums reaching ~2^22 that cancel to 4 over the whole row
            @memset(t.q, 0);
            @memset(t.k, 0);
            const half = L / 2;
            for (0..t.kvHeads()) |h| for (0..L) |j| {
                const at = if (j < half) j else j - half;
                const positive: f32 = if (at % 2 == 0) 127 else 4064;
                const residual: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
                const v: f32 = if (j < half) positive else if (j < 2 * half) residual - positive else 0;
                @memset(t.v[(h * L + j) * D ..][0..D], c.toBf16(v));
            };
        } else {
            mi.randomBf16(rng, t.q, -1, 1);
            mi.randomBf16(rng, t.k, -1, 1);
            mi.randomBf16(rng, t.v, -1, 1);
        }
        if (fill == .nonfinite) {
            for ([_][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 }, .{ 4, 3 } }, [_]u16{ mi.nan16, mi.inf16, mi.ninf16, mi.inf16, mi.ninf16 }) |p, v| if (p[0] < L) {
                t.v[p[0] * D + p[1]] = v;
            };
            if (L > 2) t.q[(L + 2) * D + 5] = mi.nan16;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.v));
        t.out = try t.rig.output(L * shape.rows * D * 2);
        return true;
    }

    pub fn run(t: *Attention) !void {
        const p = t.rig.begin();
        ks.encoderAttention(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out), t.shape.k, t.shape.rows, t.kvHeads(), 1.0 / @sqrt(@as(f32, ec.head_dim)));
        try t.rig.finish(p);
    }

    pub fn count(t: *Attention) usize {
        return t.shape.k * t.shape.rows * ec.head_dim;
    }

    /// attention_ref.head over each row's keys 0..i; O in [L, heads, 256].
    pub fn reference(t: *Attention, want: []f64, mag: []f64) !void {
        const L = t.shape.k;
        const D = ec.head_dim;
        const heads = t.shape.rows;
        const kf = try t.a.alloc(f64, 2 * L * D);
        defer t.a.free(kf);
        const vf = kf[L * D ..];
        const keys = try t.a.alloc(usize, L);
        defer t.a.free(keys);
        for (keys, 0..) |*k, i| k.* = i;
        var qf: [D]f64 = undefined;
        const scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, D)));
        for (0..t.kvHeads()) |kh| {
            for (kf[0 .. L * D], vf, t.k[kh * L * D ..][0 .. L * D], t.v[kh * L * D ..][0 .. L * D]) |*a, *b, x, y| {
                a.* = c.fromBf16(x);
                b.* = c.fromBf16(y);
            }
            if (t.stress) {
                // q = k = 0: every score 0, so each row is the mean of the values so far (the same in every column)
                var sum: f64 = 0;
                for (0..L) |i| {
                    sum += vf[i * D];
                    for (kh * 4..kh * 4 + 4) |h| {
                        @memset(want[(i * heads + h) * D ..][0..D], sum / @as(f64, @floatFromInt(i + 1)));
                        @memset(mag[(i * heads + h) * D ..][0..D], 0);
                    }
                }
                continue;
            }
            for (kh * 4..kh * 4 + 4) |h| for (0..L) |i| {
                for (&qf, t.q[(h * L + i) * D ..][0..D]) |*x, b| x.* = c.fromBf16(b);
                const at = (i * heads + h) * D;
                tf.attention_ref.head(&qf, kf[0 .. L * D], vf, keys[0 .. i + 1], D, scale, want[at..][0..D], mag[at..][0..D]);
            };
        }
    }

    pub fn bounds(t: *Attention, want: []const f64, tolerance: []const f64, allow: []f64) !void {
        for (want, tolerance, allow) |w, tl, *l| l.* = if (t.stress) c.boundExact(spec(t.name), w) else if (std.math.isFinite(w)) tl else 0;
    }

    pub fn simulate(t: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!t.stress) return error.AttentionStressNeedsEqualScores;
        const L = t.shape.k;
        const D = ec.head_dim;
        const terms = try t.a.alloc(f64, L);
        defer t.a.free(terms);
        const ones = try t.a.alloc(f64, L);
        defer t.a.free(ones);
        @memset(ones, 1);
        // every column of a key head holds the same values: one column's sums stand for all, and for its query heads
        for (0..t.kvHeads()) |kh| {
            for (terms, 0..) |*v, j| v.* = c.fromBf16(t.v[(kh * L + j) * D]);
            for (0..L) |i| {
                const o = contract.rows.sumIn(acc, order, terms[0 .. i + 1]) / contract.rows.sumIn(acc, order, ones[0 .. i + 1]);
                for (kh * 4..kh * 4 + 4) |h| @memset(out[(i * t.shape.rows + h) * D ..][0..D], o);
            }
        }
    }
};
