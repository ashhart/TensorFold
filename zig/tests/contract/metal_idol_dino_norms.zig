//! dn_norm and dn_final under the contract: nn.LayerNorm of bf16 rows in fp32, then (dn_final) F.layer_norm in fp32.
const std = @import("std");
const rig_ = @import("rig.zig");
const dn = @import("metal_idol_dino_common.zig");

const c = dn.c;
const g = dn.g;
const cfg = dn.cfg;
const W = dn.W;
const Range = dn.Range;
const Ends = dn.Ends;
const bf = dn.bf;
const kref = dn.kref;

pub const Kind = enum { norm, final };

const norm_why = "a bf16 output follows the sums as var^-1/2: a bf16 sum's error is a few output ulps";

/// A thread's 5 values of the 1,280 in turn, the simdgroup tree, then the tree over the threadgroup's 8 simdgroups.
const sum_depth = W / 256 + 5 + 5;

fn normDepth(_: c.Shape) usize {
    return sum_depth + 3;
}

/// A row's statistics from the kernel's mean (range `mr`) and squared deviations (range `qr`): 1 / sqrt(q / n + eps).
fn invRange(qr: Range) Range {
    const n: f64 = W;
    return (Range{ .lo = qr.lo / n, .hi = qr.hi / n }).add(Range.of(cfg.eps)).fp32(0.5).rsqrt().fp32(dn.rsqrt_ulps);
}

/// The mean's and squares' ranges of a row of exact values `xs` under fp32 sums (or exact ones, `exact`).
fn stats(xs: []const f64, exact: bool) struct { mean: f64, q: f64, mr: Range, qr: Range } {
    const n: f64 = W;
    const gs = c.gamma(sum_depth, c.unit(.f32));
    const gq = c.gamma(sum_depth + 3, c.unit(.f32));
    var s: f64 = 0;
    var abs: f64 = 0;
    for (xs) |v| {
        s += v;
        abs += @abs(v);
    }
    const mean = s / n;
    var q: f64 = 0;
    for (xs) |v| q += (v - mean) * (v - mean);
    if (exact) return .{ .mean = mean, .q = q, .mr = Range.of(mean), .qr = Range.of(q) };
    const mr = (Range{ .lo = (s - gs * abs) / n, .hi = (s + gs * abs) / n }).fp32(dn.div_ulps);
    // the squares about the kernel's mean exceed float64's by n (m' - mean)^2, then fp32 rounds them
    const dm = @max(@abs(mr.lo - mean), @abs(mr.hi - mean));
    return .{ .mean = mean, .q = q, .mr = mr, .qr = .{ .lo = q * (1 - gq), .hi = (q + n * dm * dm) * (1 + gq) } };
}

pub const Norm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    lib: *dn.Lib,
    kind: Kind,
    name: []const u8,
    shape: c.Shape = undefined,
    stress: bool = false,
    x: []u16 = &.{},
    /// w then b, bf16
    wb: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    /// dn_final's bf16 rows, working memory
    hidden: rig_.Ref = undefined,
    ends: Ends,

    /// Rows: positions; n and k: the width, 1,280 (the launcher's).
    pub fn spec(name: []const u8, kind: Kind) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = normDepth,
            .ops = @as(f64, if (kind == .final) 2 else 1) * (2 * (dn.div_ulps + dn.rsqrt_ulps) + 3),
            .out = if (kind == .final) .f32 else .bf16,
            .served = &.{.{ .rows = cfg.rows, .n = W, .k = W }},
            .sums = if (kind == .norm) .{ .bf16 = 0, .why = norm_why } else .{},
        };
    }

    pub fn deinit(t: *Norm) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Norm) void {
        t.a.free(t.x);
        t.a.free(t.wb);
        t.x = &.{};
        t.wb = &.{};
    }
    pub fn kernel(t: *Norm, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.kind), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Norm);

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != W or shape.k != W) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.stress = fill == .stress;
        t.x = try t.a.alloc(u16, shape.rows * W);
        t.wb = try t.a.alloc(u16, 2 * W);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // integers mirrored in each row's halves: a mean of exactly 0 and squares summing to ~2^22, exact in fp32
            for (0..shape.rows) |r| for (0..W / 2) |i| {
                const v = dn.int(rng, -64, 64);
                t.x[r * W + i] = dn.bits(v);
                t.x[r * W + W / 2 + i] = dn.bits(-v);
            };
        } else dn.randomBf(rng, t.x, -3, 3);
        dn.randomBf(rng, t.wb[0..W], 0.25, 2);
        dn.randomBf(rng, t.wb[W..], -1, 1);
        if (t.stress and t.kind == .final) {
            // w = 0: the norm's rows are b, integers mirrored too, so the second norm's sums are exact as well
            @memset(t.wb[0..W], 0);
            for (0..W / 2) |i| {
                const v = dn.int(rng, -64, 64);
                t.wb[W + i] = dn.bits(v);
                t.wb[W + W / 2 + i] = dn.bits(-v);
            }
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, W / 3 }, .{ 1, W / 2 }, .{ 2, W - 1 }, .{ 3, 0 }, .{ 3, W - 1 } };
            for (at, [_]u16{ dn.nan, dn.inf, dn.ninf, dn.inf, dn.ninf }) |p, v| if (p[0] < shape.rows) {
                t.x[p[0] * W + p[1]] = v;
            };
            // dn_final's rows each fold every column into F.layer_norm: a non-finite w or b would reach every row
            if (t.kind == .norm) {
                t.wb[5] = dn.nan;
                t.wb[W + 6] = dn.inf;
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wb[0..W]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.wb[W..]));
        if (t.kind == .final) {
            // the bf16 last_hidden_state is working memory here: the fp32 rows check its rounding
            t.hidden = try t.rig.scratch(shape.rows * W * 2);
            t.out = try t.rig.output(shape.rows * W * 4);
        } else t.out = try t.rig.output(shape.rows * W * 2);
        return true;
    }
    pub fn run(t: *Norm) !void {
        const k = &t.lib.k;
        try t.lib.need(if (t.kind == .final) "dn_final" else "dn_norm");
        const p = t.rig.begin();
        const in = t.ins;
        if (t.kind == .final) {
            k.final(p.e, kref(in[0]), kref(in[1]), kref(in[2]), kref(t.hidden), kref(t.out), t.shape.rows);
        } else k.norm(p.e, kref(in[0]), kref(in[1]), kref(in[2]), kref(t.out), t.shape.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * W;
    }

    /// Row r's outputs, its sums (both norms') taken from `sums` when replayed, else bounded as fp32's.
    fn row(t: *Norm, ends: *Ends, r: usize, want: []f64, mag: []f64, replay: ?Replay) void {
        var xs: [W]f64 = undefined;
        for (&xs, t.x[r * W ..][0..W]) |*o, v| o.* = dn.value(v);
        var st = stats(&xs, t.stress);
        if (replay) |rp| {
            st.mean = rp.sum(&xs) / W;
            var d: [W]f64 = undefined;
            for (&d, xs) |*o, x| o.* = (x - st.mean) * (x - st.mean);
            st.q = rp.sum(&d);
            st.mr = Range.of(st.mean);
            st.qr = Range.of(st.q);
        }
        const inv = 1 / @sqrt(st.q / W + cfg.eps);
        const ir = invRange(st.qr);
        var hs: [W]f64 = undefined;
        var hr: [W]Range = undefined;
        for (0..W) |i| {
            const w: f64 = dn.value(t.wb[i]);
            const b: f64 = dn.value(t.wb[W + i]);
            const vr = Range.of(xs[i]).sub(st.mr).fp32(0.5).mul(ir).fp32(0.5).mul(Range.of(w)).fp32(0.5).add(Range.of(b)).fp32(0.5);
            hs[i] = bf((xs[i] - st.mean) * inv * w + b);
            hr[i] = dn.bfr(vr);
            if (t.kind == .norm) ends.put(want, mag, r * W + i, hs[i], hr[i]);
        }
        if (t.kind == .norm) return;
        // F.layer_norm of the bf16 rows: each value in its range, the statistics over those ranges
        var s2 = stats(&hs, t.stress);
        if (replay) |rp| {
            s2.mean = rp.sum(&hs) / W;
            var d: [W]f64 = undefined;
            for (&d, hs) |*o, x| o.* = (x - s2.mean) * (x - s2.mean);
            s2.q = rp.sum(&d);
            s2.mr = Range.of(s2.mean);
            s2.qr = Range.of(s2.q);
        }
        var lo: f64 = 0;
        var hi: f64 = 0;
        for (hr) |h| {
            lo += h.lo;
            hi += h.hi;
        }
        const spread = Range{ .lo = s2.mr.lo + (lo - s2.mean * W) / W, .hi = s2.mr.hi + (hi - s2.mean * W) / W };
        const mr = Range{ .lo = @min(s2.mr.lo, spread.lo), .hi = @max(s2.mr.hi, spread.hi) };
        var qlo: f64 = 0;
        var qhi: f64 = 0;
        for (hr) |h| {
            const d = h.sub(mr).square();
            qlo += d.lo;
            qhi += d.hi;
        }
        const qr = Range{ .lo = @min(s2.qr.lo, qlo * (1 - c.gamma(sum_depth + 3, c.unit(.f32)))), .hi = @max(s2.qr.hi, qhi * (1 + c.gamma(sum_depth + 3, c.unit(.f32)))) };
        const inv2 = 1 / @sqrt(s2.q / W + cfg.eps);
        const ir2 = invRange(if (t.stress and replay == null) Range.of(s2.q) else qr);
        for (0..W) |i| {
            const y = (hs[i] - s2.mean) * inv2;
            const yr = if (t.stress and replay == null) Range.of(hs[i]).sub(Range.of(s2.mean)).mul(ir2).fp32(0.5) else hr[i].sub(mr).fp32(0.5).mul(ir2).fp32(0.5);
            ends.put(want, mag, r * W + i, y, yr);
        }
    }

    const Replay = struct {
        acc: c.Accum,
        order: g.Order,
        fn sum(rp: Replay, terms: []f64) f64 {
            return dn.rows.sumIn(rp.acc, rp.order, terms);
        }
    };

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..t.shape.rows) |r| t.row(&t.ends, r, want, mag, null);
    }
    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        var scratch: Ends = .{ .a = t.a };
        defer scratch.deinit();
        try scratch.size(t.count());
        for (0..t.shape.rows) |r| t.row(&scratch, r, out, mag, .{ .acc = acc, .order = order });
    }
};
