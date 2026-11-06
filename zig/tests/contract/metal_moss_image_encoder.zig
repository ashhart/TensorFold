//! The encoder's me_head, me_gated_norm, me_gate, me_silu_mul and me_conv under the contract, at torch's bf16 points.
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
const W = ec.head_dim;

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

// --- me_head: 256-wide heads normed (1 + w), the first 64 channels turned, laid out [heads, rows, 256]; v copied ---

/// Eight channels a lane in turn, then the simdgroup tree.
fn headDepth(_: c.Shape) usize {
    return 8 + 5;
}

/// q: heads 512 apart in q_proj's rows (each followed by its gate); k: packed heads; v: packed heads, copied.
pub const HeadForm = enum { q, k, v };

pub const Head = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    form: HeadForm,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    w: []u16 = &.{},
    rot: []f32 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: heads (16 q, 4 k and v); k: the head width (256).
    pub fn spec(name: []const u8, form: HeadForm) c.Spec {
        const served: []const c.Shape = switch (form) {
            .q => &.{ .{ .rows = 63, .n = ec.heads, .k = W }, .{ .rows = 10, .n = ec.heads, .k = W }, .{ .rows = 4096, .n = ec.heads, .k = W } },
            .k, .v => &.{ .{ .rows = 63, .n = ec.kv_heads, .k = W }, .{ .rows = 10, .n = ec.kv_heads, .k = W }, .{ .rows = 4096, .n = ec.kv_heads, .k = W } },
        };
        return .{
            .name = name,
            .accum = if (form == .v) .none else .f32,
            .depth = if (form == .v) mi.noSums else headDepth,
            .ops = 2 * (mi.div_ulps + mi.rsqrt_ulps) + 3,
            .mids = 3,
            .out = .bf16,
            .served = served,
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }

    pub fn deinit(t: *Head) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Head) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.a.free(t.rot);
        t.x = &.{};
        t.w = &.{};
        t.rot = &.{};
    }
    pub fn kernel(t: *Head, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Head);

    fn stride(t: *const Head) usize {
        return if (t.form == .q) 2 * t.shape.n * W else t.shape.n * W;
    }
    fn headStride(t: *const Head) usize {
        return if (t.form == .q) 2 * W else W;
    }

    pub fn prepare(t: *Head, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != W) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, shape.rows * t.stride());
        if (fill == .stress) mi.integers(rng, t.x, 128) else mi.randomBf16(rng, t.x, -4, 4);
        t.w = try t.a.alloc(u16, W);
        mi.randomBf16(rng, t.w, -0.75, 1);
        const half = shape.rows * 32;
        t.rot = try t.a.alloc(f32, 2 * half);
        for (0..half) |i| {
            const angle = rng.float(f32) * 2 * std.math.pi;
            t.rot[i] = c.fromBf16(c.toBf16(@cos(angle)));
            t.rot[half + i] = c.fromBf16(c.toBf16(@sin(angle)));
        }
        if (fill == .nonfinite) {
            const at = [_][3]usize{ .{ 0, 0, 5 }, .{ 1, 1, 40 }, .{ 2, 2, 200 }, .{ 3, 0, 0 }, .{ 4, 3, 63 } };
            for (at, [_]u16{ mi.nan16, mi.inf16, mi.ninf16, mi.nan16, mi.ninf16 }) |p, v| if (p[0] < shape.rows and p[1] < shape.n) {
                t.x[p[0] * t.stride() + p[1] * t.headStride() + p[2]] = v;
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.rot[0..half]));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.rot[half..]));
        t.out = try t.rig.output(shape.rows * shape.n * W * 2);
        return true;
    }

    pub fn run(t: *Head) !void {
        const p = t.rig.begin();
        ks.encoderHeads(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.ins[3]), kref(t.out), t.shape.rows, t.shape.n, t.stride(), t.headStride(), ec.eps, t.form != .v);
        try t.rig.finish(p);
    }

    pub fn count(t: *Head) usize {
        return t.shape.rows * t.shape.n * W;
    }

    fn head(t: *Head, ends: *Ends, r: usize, h: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const rows = t.shape.rows;
        const x = t.x[r * t.stride() + h * t.headStride() ..][0..W];
        const base = (h * rows + r) * W;
        if (t.form == .v) {
            for (x, 0..) |v, ch| ends.put(want, mag, base + ch, c.fromBf16(v), Range.of(c.fromBf16(v)));
            return;
        }
        const n = mi.invNorm(ss, ssr, W);
        var normed: [W]f64 = undefined;
        var nr: [W]Range = undefined;
        for (x, t.w, &normed, &nr) |xv, wv, *o, *orr| {
            const f: f64 = c.fromBf16(xv);
            const one_w = 1 + @as(f64, c.fromBf16(wv));
            o.* = bf(f * n.inv * one_w);
            orr.* = Range.of(f).mul(n.r).fp32(0.5).mul(Range.of(one_w).fp32(0.5)).fp32(0.5).bf16();
        }
        const half = rows * 32;
        for (0..W) |ch| {
            if (ch >= ec.rotary) {
                ends.put(want, mag, base + ch, normed[ch], nr[ch]);
                continue;
            }
            const j = ch % 32;
            const partner = if (ch < 32) ch + 32 else ch - 32;
            const sign: f64 = if (ch < 32) -1 else 1;
            const cs: f64 = t.rot[r * 32 + j];
            const sn: f64 = t.rot[half + r * 32 + j];
            const v = bf(bf(normed[ch] * cs) + bf(sign * normed[partner] * sn));
            const vr = nr[ch].mul(Range.of(cs)).fp32(0.5).bf16().add(nr[partner].mul(Range.of(sign * sn)).fp32(0.5).bf16()).fp32(0.5).bf16();
            ends.put(want, mag, base + ch, v, vr);
        }
    }

    pub fn reference(t: *Head, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..t.shape.rows) |r| for (0..t.shape.n) |h| {
            const s = mi.squares(t.x[r * t.stride() + h * t.headStride() ..][0..W], headDepth(t.shape), t.fill == .stress);
            t.head(&t.ends, r, h, s.ss, s.r, want, mag);
        };
    }
    pub fn bounds(t: *Head, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Head, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Head, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var terms: [W]f64 = undefined;
        for (0..t.shape.rows) |r| for (0..t.shape.n) |h| {
            for (&terms, t.x[r * t.stride() + h * t.headStride() ..][0..W]) |*v, x| {
                const f: f64 = c.fromBf16(x);
                v.* = f * f;
            }
            const ss = contract.rows.sumIn(acc, order, &terms);
            t.head(&scratch, r, h, ss, Range.of(ss), out, mag);
        };
    }
};

// --- me_gated_norm: bf16(bf16(w bf16(x rsqrt(mean x^2 + eps))) silu(z)) by 128-wide head -----------------------------

/// Four channels a lane in turn, then the simdgroup tree.
fn gatedDepth(_: c.Shape) usize {
    return 4 + 5;
}

pub const GatedNorm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    z: []u16 = &.{},
    w: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: the value width (32 heads of 128); k: the head width.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = gatedDepth,
            .ops = 2 * (mi.div_ulps + mi.rsqrt_ulps + mi.exp_ulps + mi.div_ulps) + 2,
            .mids = 2,
            .out = .bf16,
            .served = &.{ .{ .rows = 63, .n = ec.hidden, .k = ec.v_dim }, .{ .rows = 10, .n = ec.hidden, .k = ec.v_dim }, .{ .rows = 4096, .n = ec.hidden, .k = ec.v_dim } },
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }

    pub fn deinit(t: *GatedNorm) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *GatedNorm) void {
        for ([_]*[]u16{ &t.x, &t.z, &t.w }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }
    pub fn kernel(t: *GatedNorm, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(GatedNorm);

    pub fn prepare(t: *GatedNorm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != ec.v_dim or shape.n == 0 or shape.n % ec.v_dim != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const n = shape.rows * shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, n);
        t.z = try t.a.alloc(u16, n);
        t.w = try t.a.alloc(u16, ec.v_dim);
        if (fill == .stress) mi.integers(rng, t.x, 128) else mi.randomBf16(rng, t.x, -2, 2);
        mi.randomBf16(rng, t.z, -12, 12);
        mi.randomBf16(rng, t.w, 0.25, 2);
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
                t.x[(i * 131 + 7) % n] = v;
                t.z[(i * 257 + 3) % n] = v;
            }
            t.w[9] = mi.nan16;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.z));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *GatedNorm) !void {
        const p = t.rig.begin();
        ks.encoderGatedNorm(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out), t.shape.rows, t.shape.n, ec.eps);
        try t.rig.finish(p);
    }

    pub fn count(t: *GatedNorm) usize {
        return t.shape.rows * t.shape.n;
    }

    fn head(t: *GatedNorm, ends: *Ends, at: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const n = mi.invNorm(ss, ssr, ec.v_dim);
        for (0..ec.v_dim) |i| {
            const x: f64 = c.fromBf16(t.x[at + i]);
            const z: f64 = c.fromBf16(t.z[at + i]);
            const w: f64 = c.fromBf16(t.w[i]);
            const y = bf(bf(w * bf(x * n.inv)) * (z / (1 + @exp(-z))));
            const yr = Range.of(w).mul(Range.of(x).mul(n.r).fp32(0.5).bf16()).fp32(0.5).bf16().mul(mi.silu(z)).fp32(0.5).bf16();
            ends.put(want, mag, at + i, y, yr);
        }
    }

    pub fn reference(t: *GatedNorm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        var at: usize = 0;
        while (at < t.count()) : (at += ec.v_dim) {
            const s = mi.squares(t.x[at..][0..ec.v_dim], gatedDepth(t.shape), t.fill == .stress);
            t.head(&t.ends, at, s.ss, s.r, want, mag);
        }
    }
    pub fn bounds(t: *GatedNorm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *GatedNorm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *GatedNorm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var terms: [ec.v_dim]f64 = undefined;
        var at: usize = 0;
        while (at < t.count()) : (at += ec.v_dim) {
            for (&terms, t.x[at..][0..ec.v_dim]) |*v, x| {
                const f: f64 = c.fromBf16(x);
                v.* = f * f;
            }
            const ss = contract.rows.sumIn(acc, order, &terms);
            t.head(&scratch, at, ss, Range.of(ss), out, mag);
        }
    }
};

// --- me_gate, me_silu_mul and me_conv: elementwise steps at the publisher's bf16 points ----------------------------

pub const StepKind = enum { gate, silu_mul, conv };

/// The conv's four taps, summed in turn.
fn convDepth(_: c.Shape) usize {
    return ec.conv_taps;
}

pub const Step = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kind: StepKind,
    name: []const u8,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// gate: the attention rows; silu_mul: the gate; conv: the rows
    x: []u16 = &.{},
    /// gate: q_proj's rows (twice as wide); silu_mul: up; conv: the taps [width, 4]
    y: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n: the row width; k: the conv's taps (else 1).
    pub fn spec(name: []const u8, kind: StepKind) c.Spec {
        return switch (kind) {
            .gate => .{ .name = name, .accum = .none, .depth = mi.noSums, .ops = 2 * (mi.exp_ulps + mi.div_ulps) + 2, .mids = 1, .out = .bf16, .tiles = .{ .{}, .{ .size = 512, .step = 256, .min = 256 }, .{} }, .served = &.{ .{ .rows = 63, .n = ec.hidden, .k = 1 }, .{ .rows = 4096, .n = ec.hidden, .k = 1 } } },
            .silu_mul => .{ .name = name, .accum = .none, .depth = mi.noSums, .ops = 2 * (mi.exp_ulps + mi.div_ulps) + 2, .mids = 1, .out = .bf16, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 63, .n = ec.ffn, .k = 1 }, .{ .rows = 4096, .n = ec.ffn, .k = 1 } } },
            .conv => .{ .name = name, .accum = .f32, .depth = convDepth, .ops = 2 * (mi.exp_ulps + mi.div_ulps) + 2, .mids = 1, .out = .bf16, .tiles = .{ .{ .size = 4 }, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 63, .n = ec.mix, .k = ec.conv_taps }, .{ .rows = 10, .n = ec.mix, .k = ec.conv_taps }, .{ .rows = 4096, .n = ec.mix, .k = ec.conv_taps } } },
        };
    }

    pub fn deinit(t: *Step) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Step) void {
        t.a.free(t.x);
        t.a.free(t.y);
        t.x = &.{};
        t.y = &.{};
    }
    pub fn kernel(t: *Step, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.kind), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Step);

    pub fn prepare(t: *Step, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        if (t.kind == .gate and shape.n % W != 0) return false;
        if (t.kind == .conv and shape.k != ec.conv_taps) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const n = shape.rows * shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, n);
        t.y = try t.a.alloc(u16, switch (t.kind) {
            .gate => 2 * n,
            .silu_mul => n,
            .conv => shape.n * ec.conv_taps,
        });
        mi.randomBf16(rng, t.x, -8, 8);
        mi.randomBf16(rng, t.y, -8, 8);
        if (fill == .stress) {
            // the taps 256, 1, 1, -256 over rows repeating 256, 1, 1, 256: 2 at every fourth row, exact only in fp32
            for (0..shape.n) |ch| for ([_]f32{ 256, 1, 1, -256 }, 0..) |v, k| {
                t.y[ch * ec.conv_taps + k] = c.toBf16(v);
            };
            for (0..shape.rows) |r| @memset(t.x[r * shape.n ..][0..shape.n], c.toBf16(if (r % 4 == 0 or r % 4 == 3) 256 else 1));
        }
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, i| {
                t.x[(i * 131 + 7) % n] = v;
                t.y[(i * 257 + 3) % t.y.len] = v;
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.y));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Step) !void {
        const p = t.rig.begin();
        ks.encoderRows(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.out), t.shape.rows, t.shape.n);
        try t.rig.finish(p);
    }

    pub fn count(t: *Step) usize {
        return t.shape.rows * t.shape.n;
    }

    /// The conv's terms for output (r, ch): the taps over rows r - 3 .. r, zero before row 0.
    fn taps(t: *const Step, r: usize, ch: usize, terms: *[ec.conv_taps]f64) []f64 {
        var n: usize = 0;
        for (0..ec.conv_taps) |k| if (r + k >= 3) {
            terms[n] = @as(f64, c.fromBf16(t.y[ch * ec.conv_taps + k])) * c.fromBf16(t.x[(r + k - 3) * t.shape.n + ch]);
            n += 1;
        };
        return terms[0..n];
    }

    fn convOut(sum: Range) Range {
        const lo = bf(sum.lo);
        const hi = bf(sum.hi);
        return mi.silu(lo).bf16().join(mi.silu(hi).bf16());
    }

    pub fn reference(t: *Step, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const w = t.shape.n;
        for (0..t.count()) |i| {
            const r = i / w;
            const ch = i % w;
            const x: f64 = c.fromBf16(t.x[i]);
            switch (t.kind) {
                .gate => {
                    const gv: f64 = c.fromBf16(t.y[r * 2 * w + (ch / W) * 2 * W + W + ch % W]);
                    t.ends.put(want, mag, i, bf(x * bf(1 / (1 + @exp(-gv)))), Range.of(x).mul(mi.sigmoid(gv).bf16()).fp32(0.5).bf16());
                },
                .silu_mul => {
                    const u: f64 = c.fromBf16(t.y[i]);
                    t.ends.put(want, mag, i, bf(bf(x / (1 + @exp(-x))) * u), mi.silu(x).bf16().mul(Range.of(u)).fp32(0.5).bf16());
                },
                .conv => {
                    var buf: [ec.conv_taps]f64 = undefined;
                    const terms = t.taps(r, ch, &buf);
                    var sum: f64 = 0;
                    var m: f64 = 0;
                    for (terms) |v| {
                        sum += v;
                        m += @abs(v);
                    }
                    const e = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(ec.conv_taps), c.unit(.f32)) * m;
                    const cv = bf(sum);
                    t.ends.put(want, mag, i, bf(cv / (1 + @exp(-cv))), convOut(.{ .lo = sum - e, .hi = sum + e }));
                },
            }
        }
    }
    pub fn bounds(t: *Step, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Step, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Step, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (t.kind != .conv) return error.NoSums;
        const w = t.shape.n;
        for (out, 0..) |*o, i| {
            var buf: [ec.conv_taps]f64 = undefined;
            const cv = bf(contract.rows.sumIn(acc, order, t.taps(i / w, i % w, &buf)));
            o.* = bf(cv / (1 + @exp(-cv)));
        }
    }
};
