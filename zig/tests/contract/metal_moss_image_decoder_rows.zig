//! md_group_part, md_group_finish and md_attention under the contract: fp32 sums against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const mi = @import("metal_moss_image_common.zig");
const st = @import("metal_moss_image_decoder_steps.zig");

const c = mi.c;
const g = mi.g;
const Range = mi.Range;
const Ends = mi.Ends;
const dec = tf.moss_image.decoder;
const ref = st.ref;
const uniform = st.uniform;
const plant = st.plant;
const nan = st.nan;
const inf = st.inf;

// --- md_group_part: GroupNorm's chunk sums, of the values or of their squared deviations ---------------------------

pub const GroupPart = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    centered: bool,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    stats: [64]f32 = undefined,
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// A thread's chunk / 256 pixels of the group's channels in turn, then the simdgroup tree and the 8 simdgroups.
    fn depth(s: c.Shape) usize {
        return (@min(s.k, dec.chunk) + 255) / 256 * (s.n / 32) + 5 + 8 + 2;
    }

    /// Rows unused (1); n: channels (32 groups); k: pixels, summed in chunks.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = .{ .{}, .{ .size = 96, .step = 32, .min = 32 }, .{ .size = dec.chunk } },
            .served = &.{ .{ .rows = 1, .n = 384, .k = 16384 }, .{ .rows = 1, .n = 384, .k = 16500 }, .{ .rows = 1, .n = 192, .k = 262144 }, .{ .rows = 1, .n = 96, .k = 1048576 } },
        };
    }

    pub fn deinit(t: *GroupPart) void {
        t.a.free(t.x);
        t.rig.deinit();
    }
    pub fn kernel(t: *GroupPart, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(GroupPart);

    pub fn prepare(t: *GroupPart, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n == 0 or shape.n % 32 != 0 or shape.k == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.k * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integers: plain sums reach ~2^21 and cancel; deviations within 18 keep squares' sums under 2^24
            for (0..32) |grp| t.stats[grp] = @floatFromInt(rng.intRangeAtMost(i32, -4, 4));
            for (0..shape.k) |p| for (0..shape.n) |ch| {
                const grp = ch / (shape.n / 32);
                const in = p % dec.chunk;
                const half = dec.chunk / 2;
                const v: f32 = if (t.centered)
                    t.stats[grp] + @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -18, 18)))
                else if (in < half) @as(f32, if (in % 2 == 0) 127 else 511) else -@as(f32, if ((in - half) % 2 == 0) 127 else 511) + @as(f32, if (in - half < 4) 1 else 0);
                t.x[p * shape.n + ch] = v;
            };
        } else {
            uniform(rng, t.x, -3, 3);
            uniform(rng, t.stats[0..32], -0.5, 0.5);
            if (fill == .nonfinite) plant(t.x, shape.n);
        }
        uniform(rng, t.stats[32..], 0.5, 2);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(&t.stats));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *GroupPart) !void {
        const p = t.rig.begin();
        dec.groupPart(p.e, t.pipe, ref(t.ins[0]), ref(t.ins[1]), ref(t.out), t.shape.k, t.shape.n, t.centered);
        try t.rig.finish(p);
    }

    pub fn count(t: *GroupPart) usize {
        return 32 * dec.chunks(t.shape.k);
    }

    /// Output (group, chunk)'s terms: each pixel's channels of the group, or their squared deviations.
    fn terms(t: *const GroupPart, grp: usize, ck: usize, out: []f64) usize {
        const size = t.shape.n / 32;
        var n: usize = 0;
        for (ck * dec.chunk..@min((ck + 1) * dec.chunk, t.shape.k)) |p| for (0..size) |i| {
            const v: f64 = t.x[p * t.shape.n + grp * size + i];
            const d = v - t.stats[grp];
            out[n] = if (t.centered) d * d else v;
            n += 1;
        };
        return n;
    }

    pub fn reference(t: *GroupPart, want: []f64, mag: []f64) !void {
        const buf = try t.a.alloc(f64, dec.chunk * (t.shape.n / 32));
        defer t.a.free(buf);
        const chunks = dec.chunks(t.shape.k);
        for (0..32) |grp| for (0..chunks) |ck| {
            const n = t.terms(grp, ck, buf);
            var s: f64 = 0;
            var m: f64 = 0;
            for (buf[0..n]) |v| {
                s += v;
                m += @abs(v);
            }
            want[grp * chunks + ck] = s;
            mag[grp * chunks + ck] = if (std.math.isFinite(m)) m else 0;
        };
    }

    pub fn simulate(t: *GroupPart, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try t.a.alloc(f64, dec.chunk * (t.shape.n / 32));
        defer t.a.free(buf);
        const chunks = dec.chunks(t.shape.k);
        for (0..32) |grp| for (0..chunks) |ck| {
            const n = t.terms(grp, ck, buf);
            out[grp * chunks + ck] = contract.rows.sumIn(acc, order, buf[0..n]);
        };
    }
};

// --- md_group_finish: the 32 means, then the 32 rstd, from the chunk sums -------------------------------------------

pub const GroupFinish = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    part: []f32 = &.{},
    ins: [1]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ends: Ends,

    /// Each lane's chunks / 32 sums in turn, then the simdgroup tree.
    fn depth(s: c.Shape) usize {
        return (s.k + 31) / 32 + 5;
    }

    /// Rows unused (1); n: the 32 groups; k: the chunks. The group size is the widest stage's, 384 / 32.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = .{ .{}, .{}, .{ .size = 32 } },
            .served = &.{ .{ .rows = 1, .n = 32, .k = 4 }, .{ .rows = 1, .n = 32, .k = 5 }, .{ .rows = 1, .n = 32, .k = 64 }, .{ .rows = 1, .n = 32, .k = 256 } },
        };
    }

    pub fn deinit(t: *GroupFinish) void {
        t.a.free(t.part);
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *GroupFinish, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(GroupFinish);

    fn pixels(t: *const GroupFinish) usize {
        return t.shape.k * dec.chunk;
    }

    pub fn prepare(t: *GroupFinish, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != 32 or shape.k == 0) return false;
        t.rig.reset();
        t.a.free(t.part);
        t.shape = shape;
        t.part = try t.a.alloc(f32, 32 * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integer sums near 2^16 (past fp16's range), the second half of chunks their negatives plus 64
            for (0..32) |grp| for (0..shape.k) |ck| {
                const half = shape.k / 2;
                const v: f32 = if (ck < half) 65533 + @as(f32, @floatFromInt(ck % 7)) else if (ck < 2 * half) -65533 - @as(f32, @floatFromInt((ck - half) % 7)) + 64 else 64;
                t.part[grp * shape.k + ck] = v;
            };
        } else {
            uniform(rng, t.part, 0, 5000);
            if (fill == .nonfinite) plant(t.part, shape.k);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.part));
        t.out = try t.rig.output(64 * 4);
        return true;
    }

    pub fn run(t: *GroupFinish) !void {
        const p = t.rig.begin();
        for ([_]bool{ false, true }) |centered| dec.groupFinish(p.e, t.pipe, ref(t.ins[0]), ref(t.out), t.pixels(), 384, centered);
        try t.rig.finish(p);
    }

    pub fn count(_: *GroupFinish) usize {
        return 64;
    }

    fn outputs(t: *GroupFinish, sums: [32]f64, mags: [32]f64, want: []f64, mag: []f64, ends: *Ends) void {
        const n: f64 = @floatFromInt(t.pixels() * 12);
        const nf: f64 = @as(f32, @floatCast(n));
        const eps: f64 = dec.eps;
        for (0..32) |grp| {
            const s = sums[grp];
            const e = c.gamma(@floatFromInt(depth(t.shape)), c.unit(.f32)) * mags[grp];
            const sr = Range{ .lo = s - e, .hi = s + e };
            const mean = (Range{ .lo = sr.lo / nf, .hi = sr.hi / nf }).fp32(mi.div_ulps);
            ends.put(want, mag, grp, s / n, mean);
            const pos = mean.add(Range.of(eps)).fp32(0.5);
            const inv = if (pos.lo > 0) pos.rsqrt().fp32(mi.rsqrt_ulps) else Range.of(1 / @sqrt(s / n + eps));
            ends.put(want, mag, 32 + grp, 1 / @sqrt(s / n + eps), inv);
        }
    }

    pub fn reference(t: *GroupFinish, want: []f64, mag: []f64) !void {
        try t.ends.size(64);
        var sums: [32]f64 = @splat(0);
        var mags: [32]f64 = @splat(0);
        for (0..32) |grp| for (t.part[grp * t.shape.k ..][0..t.shape.k]) |v| {
            sums[grp] += v;
            mags[grp] += @abs(@as(f64, v));
        };
        t.outputs(sums, mags, want, mag, &t.ends);
    }
    pub fn bounds(t: *GroupFinish, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *GroupFinish, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *GroupFinish, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(terms);
        const n: f64 = @floatFromInt(t.pixels() * 12);
        for (0..32) |grp| {
            for (terms, t.part[grp * t.shape.k ..][0..t.shape.k]) |*v, p| v.* = p;
            const s = contract.rows.sumIn(acc, order, terms);
            out[grp] = s / n;
            out[32 + grp] = 1 / @sqrt(s / n + dec.eps);
        }
    }
};

// --- md_attention: softmax(q k^T / sqrt(384)) v, one head of 384, fp32 throughout ----------------------------------

pub const Attention = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    name: []const u8,
    shape: c.Shape = undefined,
    stress: bool = false,
    qkv: []f32 = &.{},
    tol: []f64 = &.{},
    ins: [1]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    const D = 384;
    const ld = 3 * D;

    /// The scores' 384-deep sums, the keys' sums and one rescale a block of 32 keys.
    fn depth(s: c.Shape) usize {
        return D + s.k + s.k / 32 + 8;
    }

    /// Rows: the queries compared (the first `rows` of all k); n: the head width; k: the positions L.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = mi.div_ulps + 2,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{}, .{ .size = 32 } },
            .served = &.{ .{ .rows = 24, .n = D, .k = 16384 }, .{ .rows = 24, .n = D, .k = 16500 }, .{ .rows = 24, .n = D, .k = 16128 }, .{ .rows = 40, .n = D, .k = 40 } },
        };
    }

    pub fn deinit(t: *Attention) void {
        t.a.free(t.qkv);
        t.a.free(t.tol);
        t.rig.deinit();
    }
    pub fn kernel(t: *Attention, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Attention);

    pub fn prepare(t: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != D or shape.k == 0 or shape.rows == 0 or shape.rows > shape.k) return false;
        t.rig.reset();
        t.a.free(t.qkv);
        t.shape = shape;
        t.stress = fill == .stress;
        const L = shape.k;
        const padded = std.mem.alignForward(usize, L, 32);
        t.qkv = try t.a.alloc(f32, padded * ld);
        @memset(t.qkv, 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // equal scores (q = k = 0); each column's values integers whose partial sums reach ~2^23 and cancel to 4
            const half = L / 2;
            for (0..L) |j| {
                const at = if (j < half) j else j - half;
                const positive: f32 = if (at % 2 == 0) 127 else 2041;
                const residual: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
                const v: f32 = if (j < half) positive else if (j < 2 * half) residual - positive else 0;
                @memset(t.qkv[j * ld + 2 * D ..][0..D], v);
            }
        } else {
            for (0..L) |j| uniform(rng, t.qkv[j * ld ..][0..ld], -1, 1);
            if (fill == .nonfinite) {
                // values NaN, +Inf, -Inf, and both infinities in one column; a NaN query, an infinite key
                for ([_][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 }, .{ 4, 3 } }, [_]f32{ nan, inf, -inf, inf, -inf }) |p, v| if (p[0] < L) {
                    t.qkv[p[0] * ld + 2 * D + p[1]] = v;
                };
                if (L > 6) t.qkv[1 * ld + 5] = nan;
                if (L > 7) t.qkv[7 * ld + D + 3] = inf;
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.out = try t.rig.output(padded * D * 4);
        return true;
    }

    pub fn run(t: *Attention) !void {
        const p = t.rig.begin();
        dec.attend(p.e, t.pipe, ref(t.ins[0]), ref(t.out), t.shape.k);
        try t.rig.finish(p);
    }

    pub fn count(t: *Attention) usize {
        return t.shape.rows * D;
    }

    /// Each compared query's output in float64, and its bound: the scores' error moved through exp, then the sums'.
    pub fn reference(t: *Attention, want: []f64, mag: []f64) !void {
        const L = t.shape.k;
        const scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, D)));
        const u = c.unit(.f32);
        t.a.free(t.tol);
        t.tol = try t.a.alloc(f64, t.count());
        const s = try t.a.alloc(f64, L);
        defer t.a.free(s);
        const sabs = try t.a.alloc(f64, L);
        defer t.a.free(sabs);
        var out: [D]f64 = undefined;
        var spread: [D]f64 = undefined;
        for (0..t.shape.rows) |i| {
            const q = t.qkv[i * ld ..][0..D];
            var m: f64 = -std.math.inf(f64);
            var nan_row = false;
            var worst: f64 = 0;
            for (s, sabs, 0..) |*sj, *aj, j| {
                const k = t.qkv[j * ld + D ..][0..D];
                var dot: f64 = 0;
                var mag_: f64 = 0;
                for (q, k) |a, b| {
                    dot += @as(f64, a) * b;
                    mag_ += @abs(@as(f64, a) * b);
                }
                sj.* = dot * scale;
                aj.* = mag_ * scale;
                nan_row = nan_row or std.math.isNan(sj.*);
                m = @max(m, sj.*);
                worst = @max(worst, aj.*);
            }
            if (nan_row) m = std.math.nan(f64);
            // a score's error: its sums and the scale, then the subtraction of the maximum
            const es = c.gamma(D + 4, u) * worst + 2 * u * (@abs(m) + worst);
            const rel = 2 * es + (mi.exp_ulps + 1) * 0x1p-24;
            var l: f64 = 0;
            @memset(&out, 0);
            @memset(&spread, 0);
            for (s, 0..) |sj, j| {
                const p = @exp(sj - m);
                l += p;
                for (&out, &spread, t.qkv[j * ld + 2 * D ..][0..D]) |*o, *sp, v| {
                    o.* += p * v;
                    sp.* += p * @abs(@as(f64, v));
                }
            }
            const sums = c.gamma(@floatFromInt(depth(t.shape)), u);
            for (out, spread, 0..) |o, sp, e| {
                const at = i * D + e;
                want[at] = o / l;
                mag[at] = sp / l;
                t.tol[at] = (2 * rel + 2 * sums) * sp / l;
            }
        }
    }

    pub fn bounds(t: *Attention, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, t.tol) |*l, w, tl| l.* = if (t.stress) c.boundExact(spec(t.name), w) else if (std.math.isFinite(w)) tl + c.boundExact(spec(t.name), w) else 0;
    }

    pub fn simulate(t: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!t.stress) return error.AttentionStressNeedsEqualScores;
        const L = t.shape.k;
        const terms = try t.a.alloc(f64, L);
        defer t.a.free(terms);
        const ones = try t.a.alloc(f64, L);
        defer t.a.free(ones);
        @memset(ones, 1);
        const l = contract.rows.sumIn(acc, order, ones);
        for (0..D) |e| {
            for (terms, 0..) |*v, j| v.* = t.qkv[j * ld + 2 * D + e];
            const o = contract.rows.sumIn(acc, order, terms) / l;
            for (0..t.shape.rows) |i| out[i * D + e] = o;
        }
    }
};
