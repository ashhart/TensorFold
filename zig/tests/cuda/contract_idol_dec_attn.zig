//! The decoders' cross attention under the contract: tiled (P to nearest fp16, as torch's CUDA kernels) or fp32 short.
const std = @import("std");
const cuda = @import("cuda");
const dk = @import("idol_decoder_cuda");
const rig_ = @import("contract_rig.zig");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const g = r.g;
const D = r.D;
const heads = r.heads;
const H = r.hidden;
const C = r.latent;
const scale: f64 = 0.125;

/// Query rows repeat at this period (prime to the 128-row blocks): float64 computes 89 queries a head over every key.
const query_period = 89;

/// Heads from here on make O one P's fp16 remainder alone (see `exact`): with P rounded to fp16 they give 0.
const exact_heads = 8;
const p_dot: f64 = -1;
const p_hi: f64 = @as(f16, @floatCast(@exp(-1.0 / 8.0)));

/// rows: the 16 heads, n: Lq, k: Lk.
pub const forms = [2]struct { form: dk.Form, served: []const c.Shape, tiles: [3]c.Tile }{
    .{
        .form = .tiled,
        // octree parents to the latent (Lk always 8,192: the key tail checked at the shapes gate's made-up edges)
        .served = &.{ .{ .rows = heads, .n = 1000, .k = C }, .{ .rows = heads, .n = 9, .k = C }, .{ .rows = heads, .n = 4096, .k = C } },
        .tiles = .{ .{}, .{ .size = 16 * dk.attention_warps }, .{ .size = 64 } },
    },
    .{
        .form = .few,
        // the first octree levels' 1 to 8 parents against the latent
        .served = &.{ .{ .rows = heads, .n = dk.few_rows, .k = C }, .{ .rows = heads, .n = 1, .k = C } },
        .tiles = .{ .{}, .{ .size = dk.few_rows }, .{ .size = 256 } },
    },
};

/// The keys' sums: four mma chunks and a rescale a 64-key tile (tiled); 32 keys, a warp's and 8 warps' folds (short).
fn depthTiled(s: c.Shape) usize {
    return 5 * ((s.k + 63) / 64) + 16;
}

fn depthFew(_: c.Shape) usize {
    return 32 + 5 + 8;
}

const depths = [2]*const fn (c.Shape) usize{ depthTiled, depthFew };

fn half(x: f64) f16 {
    return @floatCast(@as(f32, @floatCast(x)));
}

/// The spacing of fp16 values at |x| (subnormals 2^-24).
fn ulp16(x: f64) f64 {
    const a = @abs(x);
    if (a < 0x1p-14) return 0x1p-24;
    return std.math.pow(f64, 2, @floor(std.math.log2(a)) - 10);
}

pub const Attention = struct {
    rig: rig_.Rig,
    k: *const dk.Kernels,
    which: usize,
    name: []const u8,
    control: ?g.Gate = null,
    lq: usize = 0,
    lk: usize = 0,
    stress: bool = false,
    /// q [lq, 1024] and kv [lk, 2048] (k heads then v heads), as the decoders' buffers hold them
    q: std.ArrayList(f16) = .empty,
    kv: std.ArrayList(f16) = .empty,
    dev: [2]u64 = .{ 0, 0 },
    out: u64 = 0,

    pub fn spec(name: []const u8, which: usize) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depths[which], .ops = 8, .out = .f16, .tiles = forms[which].tiles, .served = forms[which].served };
    }

    pub fn deinit(t: *Attention) void {
        t.q.deinit(t.rig.gpu.gpa);
        t.kv.deinit(t.rig.gpu.gpa);
        t.rig.deinit();
    }

    pub fn kernel(t: *Attention) g.Kernel {
        return .{ .spec = spec(t.name, t.which), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Attention);

    fn rounded(t: *const Attention) bool {
        return forms[t.which].form == .tiled;
    }

    fn qAt(t: *Attention, row: usize, hd: usize) []f16 {
        return t.q.items[row * H + hd * D ..][0..D];
    }

    fn kAt(t: *Attention, key: usize, hd: usize) []f16 {
        return t.kv.items[key * 2 * H + hd * D ..][0..D];
    }

    fn vAt(t: *Attention, key: usize, hd: usize) []f16 {
        return t.kv.items[key * 2 * H + H + hd * D ..][0..D];
    }

    pub fn prepare(t: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // always the decoders' 16 heads: the single-shape gates' fewer `rows` run all of them
        if (shape.n == 0 or shape.k == 0) return false;
        if (forms[t.which].form == .few and (shape.n > dk.few_rows or shape.k > dk.few_keys)) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        t.lq = shape.n;
        t.lk = shape.k;
        t.stress = fill == .stress;
        try t.q.resize(a, t.lq * H);
        try t.kv.resize(a, t.lk * 2 * H);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const L = t.lk;
        if (t.stress) {
            // q = k = 0; odd fp16 integers, partial sums under 2^24 that cancel to 4; an fp16 chain loses bits
            @memset(t.q.items, 0);
            @memset(t.kv.items, 0);
            const hl = L / 2;
            const big: f64 = if (hl / 2 * (127 + 2047) < 1 << 24) 2047 else 1023;
            for (0..heads) |hd| for (0..L) |j| {
                const at = if (j < hl) j else j - hl;
                const positive: f64 = if (at % 2 == 0) 127 else big;
                const residual: f64 = if (at < 8 and at % 2 == 0) 1 else 0;
                const v: f64 = if (j < hl) positive else if (j < 2 * hl) residual - positive else 0;
                @memset(t.vAt(j, hd), half(v));
            };
        } else {
            for (t.q.items) |*x| x.* = half(-2 + 4 * rng.float(f64));
            for (t.kv.items, 0..) |*x, i| x.* = half(if (i % (2 * H) < H) -2 + 4 * rng.float(f64) else -1 + 2 * rng.float(f64));
            if (L >= 2) for (exact_heads..heads) |hd| t.exact(rng, hd);
        }
        if (fill == .nonfinite) {
            // head 0: values NaN, +inf, -inf, and both infinities in one column; head 1: a NaN query, an infinite key
            for ([_][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 }, .{ 4, 3 } }, [_]f64{ r.nan, r.inf, -r.inf, r.inf, -r.inf }) |p, v| if (p[0] < L) {
                t.vAt(p[0], 0)[p[1]] = half(v);
            };
            t.qAt(0, 1)[5] = half(r.nan);
            if (L > 7) t.kAt(7, 1)[3] = half(r.inf);
        }
        // every head's queries repeat at the period, so float64 computes each distinct one once
        const p = @min(t.lq, query_period);
        for (p..t.lq) |i| @memcpy(t.q.items[i * H ..][0..H], t.q.items[(i % p) * H ..][0..H]);
        t.dev[0] = try t.rig.input(std.mem.sliceAsBytes(t.q.items));
        t.dev[1] = try t.rig.input(std.mem.sliceAsBytes(t.kv.items));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    /// Head hd: scores 0, -1/8 (last key) and -256 (P = 1, e^-1/8, 0), values -2^m fp16(P) and 2^m: O = 2^m lo / l.
    fn exact(t: *Attention, rng: std.Random, hd: usize) void {
        const L = t.lk;
        for (0..t.lq) |i| {
            const q = t.qAt(i, hd);
            @memset(q, 0);
            q[0] = 1;
        }
        for (0..L) |j| {
            const k = t.kAt(j, hd);
            @memset(k, 0);
            k[0] = half(if (j == 0) 0 else if (j == L - 1) p_dot else -2048);
        }
        for (0..D) |e| {
            const m: f64 = @floatFromInt(@as(u32, 1) << @intCast(rng.uintLessThan(u32, 11)));
            const s = (if (rng.boolean()) @as(f64, 1) else -1) * m;
            t.vAt(0, hd)[e] = half(-s * p_hi);
            t.vAt(L - 1, hd)[e] = half(s);
        }
    }

    pub fn run(t: *Attention) !void {
        try dk.attentionAs(t.k, t.rig.s, forms[t.which].form, t.dev[0], H, t.dev[1], 2 * H, t.out, t.lq, t.lk, @floatCast(scale));
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Attention) usize {
        return t.lq * H;
    }

    /// Float64 O at the kernel's P; `mag` the allowance: fp32 steps, P's rounding near a midpoint, O's own rounding.
    pub fn reference(t: *Attention, want: []f64, mag: []f64) !void {
        const a = t.rig.gpu.gpa;
        const L = t.lk;
        const s = try a.alloc(f64, L);
        defer a.free(s);
        const ds = try a.alloc(f64, L);
        defer a.free(ds);
        const kf = try a.alloc(f64, L * D);
        defer a.free(kf);
        const sums = c.gamma(@floatFromInt(depths[t.which](.{ .rows = heads, .n = t.lq, .k = L })), c.unit(.f32));
        var qf: [D]f64 = undefined;
        var out: [D]f64 = undefined;
        var tol: [D]f64 = undefined;
        var flip: [D]f64 = undefined;
        const p = @min(t.lq, query_period);
        for (0..heads) |hd| {
            for (0..L) |j| for (t.kAt(j, hd), 0..) |x, e| {
                kf[j * D + e] = x;
            };
            for (0..p) |i| {
                for (&qf, t.qAt(i, hd)) |*x, b| x.* = b;
                var m: f64 = -std.math.inf(f64);
                var top: usize = 0;
                var nan = false;
                for (s, ds, 0..) |*sj, *dj, j| {
                    const Vec = @Vector(8, f64);
                    var acc: Vec = @splat(0);
                    var mag_acc: Vec = @splat(0);
                    var e: usize = 0;
                    while (e < D) : (e += 8) {
                        const prod = @as(Vec, qf[e..][0..8].*) * @as(Vec, kf[j * D + e ..][0..8].*);
                        acc += prod;
                        mag_acc += @abs(prod);
                    }
                    sj.* = @reduce(.Add, acc) * scale;
                    dj.* = c.gamma(D, c.unit(.f32)) * @reduce(.Add, mag_acc) * scale;
                    nan = nan or std.math.isNan(sj.*);
                    if (sj.* > m) {
                        m = sj.*;
                        top = j;
                    }
                }
                if (nan) m = std.math.nan(f64);
                var l: f64 = 0;
                @memset(&out, 0);
                @memset(&tol, 0);
                @memset(&flip, 0);
                for (s, ds, 0..) |sj, dj, j| {
                    const pj = @exp(sj - m);
                    l += pj;
                    // P's relative error: its score's and the top score's sums, exp2's approximation
                    const rel = dj + ds[top] + 0x1p-21;
                    var used = pj;
                    var turn: f64 = 0;
                    if (t.rounded()) {
                        used = half(pj);
                        const u = ulp16(pj);
                        const mid = (@floor(pj / u) + 0.5) * u;
                        if (@abs(pj - mid) <= pj * rel + 0x1p-60) turn = u;
                    }
                    for (&out, &tol, &flip, t.vAt(j, hd)) |*o, *tl, *fl, vh| {
                        const vv: f64 = vh;
                        o.* += used * vv;
                        tl.* += pj * @abs(vv) * (rel + sums);
                        fl.* += turn * @abs(vv);
                    }
                }
                var row = i;
                while (row < t.lq) : (row += p) {
                    const at = row * H + hd * D;
                    for (out, tol, flip, 0..) |o, tl, fl, e| {
                        want[at + e] = o / l;
                        mag[at + e] = (tl + fl) / l + @abs(o / l) * 0x1p-10 + 0x1p-24;
                    }
                }
            }
        }
    }

    pub fn bounds(t: *Attention, want: []const f64, tolerance: []const f64, allow: []f64) !void {
        for (want, tolerance, allow) |w, tl, *l| l.* = if (t.stress) c.boundExact(spec(t.name, t.which), w) else if (std.math.isFinite(w)) tl else 0;
    }

    pub fn simulate(t: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!t.stress) return error.AttentionStressNeedsEqualScores;
        const a = t.rig.gpu.gpa;
        const L = t.lk;
        const terms = try a.alloc(f64, L);
        defer a.free(terms);
        const ones = try a.alloc(f64, L);
        defer a.free(ones);
        @memset(ones, 1);
        const l = r.rows.sumIn(acc, order, ones);
        for (0..heads) |hd| for (0..D) |e| {
            for (terms, 0..) |*v, j| v.* = t.vAt(j, hd)[e];
            const o = r.rows.sumIn(acc, order, terms) / l;
            for (0..t.lq) |i| out[i * H + hd * D + e] = o;
        };
    }
};

test "the exact heads' P has an fp16 remainder far from both zero and a tie" {
    const p = @exp(p_dot / 8);
    const lo = (p - p_hi) / 0x1p-11;
    try std.testing.expect(@abs(lo) > 0.25 and @abs(lo) < 0.4);
    try std.testing.expectEqual(@as(f64, 0x1p-11), ulp16(0.9));
    try std.testing.expectEqual(@as(f64, 0x1p-24), ulp16(0x1p-20));
}

test "an exact head's fp32 remainder lies past the short form's allowance, and its fp16 P past no midpoint" {
    // the exact heads' O: 0 with P rounded to fp16, 2^m lo / l in fp32 (m = 0 here)
    const p = @exp(p_dot / 8);
    const l = 1 + p;
    const remainder = (p - p_hi) / l;
    const sums = c.gamma(@floatFromInt(depthFew(.{ .rows = heads, .n = 8, .k = C })), c.unit(.f32));
    const rel = c.gamma(D, c.unit(.f32)) * scale + 0x1p-21;
    const allow = (1 * p_hi + p) * (rel + sums) / l + @abs(remainder) * 0x1p-10 + 0x1p-24;
    try std.testing.expect(@abs(remainder) > 4 * allow);
    // the tiled form's sums allowance at 8,192 keys still leaves the remainder outside it: a fp32 P would go red
    const tiled = c.gamma(@floatFromInt(depthTiled(.{ .rows = heads, .n = 1000, .k = C })), c.unit(.f32));
    try std.testing.expect(@abs(remainder) > (1 * p_hi + p) * (rel + tiled) / l);
    const u = ulp16(p);
    try std.testing.expect(@abs(p - (@floor(p / u) + 0.5) * u) > p * 2 * rel);
}
