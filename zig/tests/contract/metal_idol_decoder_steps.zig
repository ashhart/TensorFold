//! dec_embed, dec_half, dec_kv and dec_heads under the contract: the decoders' steps at idol-sdk's rounding points.
const std = @import("std");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const id = @import("metal_idol_common.zig");
const sources = @import("kernel_sources");

const c = id.c;
const g = id.g;
const H = id.H;
const D = id.D;
const Range = id.Range;
const Ends = id.Ends;
const hf = id.hf;
const kref = id.kref;
const dk = id.tf.idol.decoder_kernels;
const dc = id.tf.idol.decoder_config;

/// The decoders' library: idol_decoder.metal through decoder_kernels.zig's launchers.
pub const Lib = id.LibOf(dk.Kernels, &dk.names, sources.idol_decoder);

/// The steps' common parts: the outputs' ends, and its bounds and intervals from them.
fn Common(comptime T: type) type {
    return struct {
        pub fn bounds(t: *T, want: []const f64, _: []const f64, allow: []f64) !void {
            t.ends.bounds(want, allow);
        }
        pub fn intervals(t: *T, low: []f64, high: []f64) !void {
            t.ends.intervals(low, high);
        }
        pub fn simulate(t: *T, acc: c.Accum, order: g.Order, out: []f64) !void {
            const mag = try t.a.alloc(f64, out.len);
            defer t.a.free(mag);
            var scratch: Ends = .{ .a = t.a };
            defer scratch.deinit();
            try scratch.size(t.count());
            t.each(&scratch, out, mag, acc, order);
        }
    };
}

/// A value rounded once to fp32.
fn f32r(v: f64) f64 {
    return @as(f32, @floatCast(v));
}

/// The rows the decoders call a step at: one octree parent, the fewest past the short attention, 32k, 262k and 1M.
const served_rows = [_]usize{ 1024, 1, 9, 8192, 32768 };

/// Those rows at `n` outputs a row and `k` inputs.
fn servedAt(comptime n: usize, comptime k: usize) [served_rows.len]c.Shape {
    var out: [served_rows.len]c.Shape = undefined;
    for (&out, served_rows) |*s, r| s.* = .{ .rows = r, .n = n, .k = k };
    return out;
}

const embed_served = servedAt(H, 3);
const half_served = servedAt(H, 1);
const heads_served = servedAt(16, D);

// --- dec_embed: h = fp32(fma(x2, w2, fma(x1, w1, x0 w0)) + b) + pe, pe = sin or cos of fp32(fp32(x f) pi) ------

/// The three-input FMA chain, then the bias.
fn embedDepth(_: c.Shape) usize {
    return 4;
}

pub const Embed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *Lib,
    name: []const u8,
    rows: usize = 0,
    stress: bool = false,
    p: []f32 = &.{},
    /// w [1024, 3], b [1024], then the frequencies [170]
    wbf: []f32 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    const pi32: f64 = @as(f32, std.math.pi);

    /// Rows: points; n: the stream's 1,024; k: the three coordinates.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = embedDepth, .ops = 2 + id.trig_ulps, .out = .f32, .served = &embed_served };
    }

    pub fn deinit(t: *Embed) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Embed) void {
        t.a.free(t.p);
        t.a.free(t.wbf);
        t.p = &.{};
        t.wbf = &.{};
    }
    pub fn kernel(t: *Embed, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Embed);
    pub const bounds = Common(Embed).bounds;
    pub const intervals = Common(Embed).intervals;
    pub const simulate = Common(Embed).simulate;

    pub fn prepare(t: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k != 3) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.stress = fill == .stress;
        t.p = try t.a.alloc(f32, t.rows * 3);
        t.wbf = try t.a.alloc(f32, H * 4 + dc.pe_freqs);
        const w = t.wbf[0 .. H * 3];
        const b = t.wbf[H * 3 .. H * 4];
        const f = t.wbf[H * 4 ..][0..dc.pe_freqs];
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // x0 w0 and x1 w1 near +-2^16 (past fp16's range) cancelling to a few hundred; no embedding (frequencies 0)
            for (0..t.rows) |r| {
                t.p[r * 3] = id.int(rng, 200, 300);
                t.p[r * 3 + 1] = t.p[r * 3];
                t.p[r * 3 + 2] = id.int(rng, 1, 8);
            }
            for (0..H) |o| {
                w[o * 3] = id.sign(rng) * id.int(rng, 220, 300);
                w[o * 3 + 1] = -w[o * 3] + id.int(rng, -1, 1);
                w[o * 3 + 2] = id.int(rng, -8, 8);
            }
            for (b) |*v| v.* = id.int(rng, -8, 8);
            @memset(f, 0);
        } else {
            id.randomF32(rng, t.p, 0, 1);
            id.randomF32(rng, w, -1, 1);
            id.randomF32(rng, b, -0.5, 0.5);
            id.tf.idol.octree.pointFreqs(f);
        }
        if (fill == .nonfinite) {
            for ([_]f32{ id.nan32, id.inf32, -id.inf32 }, 0..) |v, i| t.p[i * 3 + i] = v;
            b[9] = id.nan32;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.p));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(b));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(f));
        t.out = try t.rig.output(t.rows * H * 4);
        return true;
    }

    pub fn run(t: *Embed) !void {
        try t.lib.need("dec_embed");
        const p = t.rig.begin();
        t.lib.k.embed(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.ins[3]), kref(t.out), t.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *Embed) usize {
        return t.rows * H;
    }

    fn each(t: *Embed, ends: *Ends, want: []f64, mag: []f64, acc: ?c.Accum, order: g.Order) void {
        const gd = c.gamma(3, c.unit(.f32));
        const w = t.wbf[0 .. H * 3];
        const b = t.wbf[H * 3 .. H * 4];
        const f = t.wbf[H * 4 ..][0..dc.pe_freqs];
        var terms: [3]f64 = undefined;
        for (0..t.rows) |r| for (0..H) |o| {
            const x = t.p[r * 3 ..][0..3];
            var s: f64 = 0;
            var abs: f64 = 0;
            for (&terms, x, w[o * 3 ..][0..3]) |*v, xv, wv| {
                v.* = @as(f64, xv) * @as(f64, wv);
                s += v.*;
                abs += @abs(v.*);
            }
            var sr = if (t.stress) Range.of(s) else Range.of(s).widen(gd * abs);
            if (acc) |y| {
                s = contract.rows.sumIn(y, order, &terms);
                sr = Range.of(s);
            }
            const bv: f64 = b[o];
            var pe: f64 = 0;
            var per = Range.of(0);
            if (o < 1020) {
                const axis = o / 340;
                const j = o % 340;
                const xf = @as(f64, x[axis]) * @as(f64, f[j % 170]);
                // torch's fp32 angle; the kernel's within an ulp of it whichever way its fast math associates
                const ang = f32r(f32r(xf) * pi32);
                const ar = Range.of(xf).fp32(0.5).mul(Range.of(pi32)).fp32(1);
                if (j < 170) {
                    pe = @sin(ang);
                    per = id.sinOver(ar.lo, ar.hi).fp32(id.trig_ulps);
                } else {
                    pe = @cos(ang);
                    per = id.cosOver(ar.lo, ar.hi).fp32(id.trig_ulps);
                }
                if (std.math.isNan(ang)) per = Range.of(ang);
            }
            const y = sr.add(Range.of(bv)).fp32(0.5).add(per).fp32(0.5);
            ends.put(want, mag, r * H + o, s + bv + pe, y);
        };
    }

    pub fn reference(t: *Embed, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        t.each(&t.ends, want, mag, null, .sequential);
    }
};

// --- dec_half: o = fp16(y + b) by column ------------------------------------------------------------------------

pub const Half = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *Lib,
    name: []const u8,
    rows: usize = 0,
    /// y [rows, 1024] then b [1024]
    yb: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: points; n: the stream's 1,024 (the stem's only width).
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = id.noSums, .ops = 1, .out = .f16, .served = &half_served };
    }

    pub fn deinit(t: *Half) void {
        t.a.free(t.yb);
        t.yb = &.{};
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Half, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Half);
    pub const bounds = Common(Half).bounds;
    pub const intervals = Common(Half).intervals;

    pub fn prepare(t: *Half, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H) return false;
        t.rig.reset();
        t.a.free(t.yb);
        t.rows = shape.rows;
        const n = t.rows * H;
        t.yb = try t.a.alloc(f32, n + H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        id.randomF32(rng, t.yb[0..n], -8, 8);
        id.randomF32(rng, t.yb[n..], -2, 2);
        if (fill == .nonfinite) {
            for ([_]f32{ id.nan32, id.inf32, -id.inf32 }, 0..) |v, i| t.yb[(i * 13) % n] = v;
            t.yb[n + 6] = id.inf32;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.yb[0..n]));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.yb[n..]));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Half) !void {
        try t.lib.need("dec_half");
        const p = t.rig.begin();
        t.lib.k.toHalf(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.out), t.rows, H);
        try t.rig.finish(p);
    }

    pub fn count(t: *Half) usize {
        return t.rows * H;
    }

    pub fn reference(t: *Half, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const n = t.count();
        for (0..n) |i| {
            const v = @as(f64, t.yb[i]) + @as(f64, t.yb[n + i % H]);
            t.ends.put(want, mag, i, hf(v), Range.of(v).fp32(0.5).fp16());
        }
    }
};

// --- dec_kv: kv = fp16(ctx w^T + b), the latent's 16 channels in fp32 FMA -----------------------------------------

const C = dc.context_rows;
const K = dc.context_channels;

/// The 16 inputs' FMA chain, then the bias.
fn kvDepth(_: c.Shape) usize {
    return K + 1;
}

pub const Kv = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *Lib,
    name: []const u8,
    stress: bool = false,
    /// ctx [8192, 16], w [2048, 16], b [2048]
    v: []f16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    const N = 2 * H;

    /// Rows: the latent's 8,192, which the launcher fixes (a gate's fewer rows run all of them); k: its 16 channels.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = kvDepth, .ops = 1, .out = .f16, .served = &.{.{ .rows = C, .n = N, .k = K }} };
    }

    pub fn deinit(t: *Kv) void {
        t.a.free(t.v);
        t.v = &.{};
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Kv, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Kv);
    pub const bounds = Common(Kv).bounds;
    pub const intervals = Common(Kv).intervals;
    pub const simulate = Common(Kv).simulate;

    pub fn prepare(t: *Kv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != N or shape.k != K) return false;
        t.rig.reset();
        t.a.free(t.v);
        t.stress = fill == .stress;
        t.v = try t.a.alloc(f16, C * K + N * K + N);
        const ctx = t.v[0 .. C * K];
        const w = t.v[C * K ..][0 .. N * K];
        const b = t.v[C * K + N * K ..];
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // products to 2^16 whose halves cancel but for the first: partial sums near 2^19, exact in fp32
            for (0..C) |r| for (0..K / 2) |k| {
                ctx[r * K + k] = @floatCast(id.int(rng, 1, 255));
                ctx[r * K + K / 2 + k] = @floatCast(@as(f32, ctx[r * K + k]) + @as(f32, if (k == 0) 1 else 0));
            };
            for (0..N) |o| for (0..K / 2) |k| {
                w[o * K + k] = @floatCast(id.sign(rng) * id.int(rng, 1, 255));
                w[o * K + K / 2 + k] = -w[o * K + k];
            };
            for (b) |*x| x.* = @floatCast(id.int(rng, -8, 8));
        } else {
            id.randomF16(rng, ctx, -3, 3);
            id.randomF16(rng, w, -0.5, 0.5);
            id.randomF16(rng, b, -0.5, 0.5);
        }
        if (fill == .nonfinite) {
            for ([_]f16{ id.nan16, id.inf16, -id.inf16 }, 0..) |x, i| ctx[i * K + 3 * i] = x;
            b[9] = id.nan16;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(ctx));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(b));
        t.out = try t.rig.output(C * N * 2);
        return true;
    }

    pub fn run(t: *Kv) !void {
        try t.lib.need("dec_kv");
        const p = t.rig.begin();
        t.lib.k.kv(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out));
        try t.rig.finish(p);
    }

    pub fn count(_: *Kv) usize {
        return C * N;
    }

    fn each(t: *Kv, ends: *Ends, want: []f64, mag: []f64, acc: ?c.Accum, order: g.Order) void {
        const gd = c.gamma(K, c.unit(.f32));
        const ctx = t.v[0 .. C * K];
        const w = t.v[C * K ..][0 .. N * K];
        const b = t.v[C * K + N * K ..];
        var terms: [K]f64 = undefined;
        for (0..C) |r| for (0..N) |o| {
            var s: f64 = 0;
            var abs: f64 = 0;
            for (&terms, ctx[r * K ..][0..K], w[o * K ..][0..K]) |*x, z, wv| {
                x.* = @as(f64, z) * @as(f64, wv);
                s += x.*;
                abs += @abs(x.*);
            }
            var sr = if (t.stress) Range.of(s) else Range.of(s).widen(gd * abs);
            if (acc) |y| {
                s = contract.rows.sumIn(y, order, &terms);
                sr = Range.of(s);
            }
            const bv: f64 = b[o];
            ends.put(want, mag, r * N + o, hf(s + bv), sr.add(Range.of(bv)).fp32(0.5).fp16());
        };
    }

    pub fn reference(t: *Kv, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        t.each(&t.ends, want, mag, null, .sequential);
    }
};

// --- dec_heads: q and k heads F.normalize(x) times gamma and 8, rounded; v copied -----------------------------------

const heads_why = "an fp16 output follows the head's sum of squares as its square root: a bf16 sum is off a few ulps";

/// A lane's two squares, then the simdgroup tree.
fn headsDepth(_: c.Shape) usize {
    return 1 + 5;
}

pub const Heads = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *Lib,
    name: []const u8,
    rows: usize = 0,
    stress: bool = false,
    /// packed q, k, v rows [rows, 3072], as to_qkv writes them
    qkv: []f16 = &.{},
    /// gq then gk [1024] each
    gam: []f32 = &.{},
    ins: [3]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    ends: Ends,

    /// Rows: points; n: heads; k: the head width.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = headsDepth,
            .ops = id.sqrt_ulps + id.div_ulps + 2,
            .out = .f16,
            .served = &heads_served,
            .sums = .{ .bf16 = 0, .why = heads_why },
        };
    }

    pub fn deinit(t: *Heads) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Heads) void {
        t.a.free(t.qkv);
        t.a.free(t.gam);
        t.qkv = &.{};
        t.gam = &.{};
    }
    pub fn kernel(t: *Heads, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Heads);
    pub const bounds = Common(Heads).bounds;
    pub const intervals = Common(Heads).intervals;

    pub fn prepare(t: *Heads, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 16 or shape.k != D) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.stress = fill == .stress;
        const n = shape.rows;
        t.qkv = try t.a.alloc(f16, n * 3 * H);
        t.gam = try t.a.alloc(f32, 2 * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        id.randomF16(rng, t.qkv, -4, 4);
        id.randomF32(rng, t.gam, 0.25, 2);
        // integer heads, whose squares sum to ~2^20 exactly
        if (t.stress) for (t.qkv) |*v| {
            v.* = @floatCast(id.int(rng, -128, 128));
        };
        if (fill == .nonfinite) {
            // q NaN, k +inf, v -inf, in turn; then a NaN gamma
            const at = [_]usize{ 0 * 3 * H + 0 * D + 5, 1 * 3 * H + H + 2 * D + 9, 2 * 3 * H + 2 * H + 3 * D + 1 };
            for (at, [_]f16{ id.nan16, id.inf16, -id.inf16 }) |p, v| if (p < t.qkv.len) {
                t.qkv[p] = v;
            };
            t.gam[H + 4 * D + 6] = id.nan32;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.gam[0..H]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.gam[H..]));
        for (&t.outs) |*o| o.* = try t.rig.output(n * H * 2);
        return true;
    }

    pub fn run(t: *Heads) !void {
        try t.lib.need("dec_heads");
        const p = t.rig.begin();
        const k = &t.lib.k;
        k.heads(p.e, kref(t.ins[0]), kref(t.ins[1]), kref(t.outs[0]), t.rows, 3 * H, 0);
        k.heads(p.e, kref(t.ins[0]), kref(t.ins[2]), kref(t.outs[1]), t.rows, 3 * H, H);
        k.heads(p.e, kref(t.ins[0]), null, kref(t.outs[2]), t.rows, 3 * H, 2 * H);
        try t.rig.finish(p);
    }

    pub fn count(t: *Heads) usize {
        return 3 * t.rows * H;
    }

    /// The head's outputs from its sum of squares `ss` (range `ssr`).
    fn head(t: *Heads, ends: *Ends, which: usize, r: usize, h: usize, ss: f64, ssr: Range, want: []f64, mag: []f64) void {
        const e12: f64 = @as(f32, 1e-12);
        // torch's clamp_min keeps a NaN norm (@max would drop it)
        const norm = if (std.math.isNan(ss)) ss else @max(@sqrt(ss), e12);
        const nr0 = ssr.up(id.sqrt).fp32(id.sqrt_ulps);
        const nr = Range{ .lo = @max(nr0.lo, e12), .hi = @max(nr0.hi, e12) };
        const x = t.qkv[r * 3 * H + which * H + h * D ..][0..D];
        const gam = t.gam[which * H + h * D ..][0..D];
        const base = which * t.rows * H + (h * t.rows + r) * D;
        for (0..D) |i| {
            const v: f64 = x[i];
            const gv: f64 = gam[i];
            const rr = Range.of(v).mul(nr.recip()).fp32(id.div_ulps).mul(Range.of(gv)).fp32(0.5).mul(Range.of(8)).fp32(0.5).fp16();
            ends.put(want, mag, base + i, hf(v / norm * gv * 8), rr);
        }
    }

    /// v's heads, copied bit for bit.
    fn values(t: *Heads, ends: *Ends, want: []f64, mag: []f64) void {
        for (0..t.rows) |r| for (0..16) |h| for (0..D) |i| {
            const f: f64 = t.qkv[r * 3 * H + 2 * H + h * D + i];
            ends.put(want, mag, 2 * t.rows * H + (h * t.rows + r) * D + i, f, Range.of(f));
        };
    }

    fn each(t: *Heads, ends: *Ends, want: []f64, mag: []f64, acc: ?c.Accum, order: g.Order) void {
        const gd = c.gamma(@floatFromInt(headsDepth(.{ .rows = 1, .n = 1, .k = 1 })), c.unit(.f32));
        var terms: [D]f64 = undefined;
        for (0..2) |which| for (0..t.rows) |r| for (0..16) |h| {
            var ss: f64 = 0;
            for (&terms, t.qkv[r * 3 * H + which * H + h * D ..][0..D]) |*s, x| {
                s.* = @as(f64, x) * @as(f64, x);
                ss += s.*;
            }
            var ssr = if (t.stress) Range.of(ss) else Range{ .lo = ss * (1 - gd), .hi = ss * (1 + gd) };
            if (acc) |y| {
                ss = contract.rows.sumIn(y, order, &terms);
                ssr = Range.of(ss);
            }
            t.head(ends, which, r, h, ss, ssr, want, mag);
        };
        t.values(ends, want, mag);
    }

    pub fn reference(t: *Heads, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        t.each(&t.ends, want, mag, null, .sequential);
    }

    pub const simulate = Common(Heads).simulate;
};
