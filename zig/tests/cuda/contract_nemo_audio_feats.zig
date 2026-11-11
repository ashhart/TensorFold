//! tf_nemo_mel and tf_nemo_norm under the contract: log-mel frames and their per-feature normalisation.
const std = @import("std");
const nemo_audio = @import("nemo_audio");
const rig_ = @import("contract_rig.zig");
const r = @import("contract_nemo_audio_ref.zig");

const c = r.c;
const g = r.g;
const Range = r.Range;
const Ends = r.Ends;
const bf = r.bf;
const K = nemo_audio.kernels;

const fft = 512;
const bins = 257;
const win = 400;
const mels = 128;
const preemph: f32 = 0.97;
const guard: f32 = 1.0 / 16777216.0;

// --- tf_nemo_mel: pre-emphasis, window, |DFT|^2, mel filters, log, one frame a block ------------------------------

fn melDepth(_: c.Shape) usize {
    return win;
}

/// Rows: frames (the last one past the valid ones), n: mel bins, k: the window.
const mel_served = [_]c.Shape{ .{ .rows = 192, .n = mels, .k = win }, .{ .rows = 586, .n = mels, .k = win }, .{ .rows = 4096, .n = mels, .k = win } };

pub const Mel = struct {
    rig: rig_.Rig,
    k: K.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []u16 = &.{}, // the window [400], then the filters [128][257]
    cs: [fft]f32 = undefined,
    sn: [fft]f32 = undefined,
    ins: [4]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = melDepth, .ops = 8, .out = .bf16, .served = &mel_served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "a log of filtered powers: the log shrinks a sum's error to a few bf16 output ulps" } };
    }
    pub fn deinit(t: *Mel) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Mel) void {
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        a.free(t.w);
        t.x = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Mel) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Mel);

    pub fn prepare(t: *Mel, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 2) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * 160;
        t.x = try a.alloc(f32, n);
        t.w = try a.alloc(u16, win + mels * bins);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        r.randomF32(rng, t.x, if (fill == .stress) -64 else -1, if (fill == .stress) 64 else 1);
        r.randomBf16(rng, t.w[0..win], 0, 1);
        r.randomBf16(rng, t.w[win..], 0, 0.05);
        if (fill == .nonfinite) {
            t.x[5] = std.math.nan(f32);
            t.x[(shape.rows - 2) * 160] = std.math.inf(f32);
            t.x[(shape.rows - 2) * 160 + 7] = -std.math.inf(f32);
        }
        for (&t.cs, &t.sn, 0..) |*x, *y, j| {
            const ang = 2.0 * std.math.pi * @as(f64, @floatFromInt(j)) / fft;
            x.* = @floatCast(@cos(ang));
            y.* = @floatCast(@sin(ang));
        }
        t.ins[0] = try r.input(&t.rig, t.x);
        t.ins[1] = try r.input(&t.rig, t.w);
        t.ins[2] = try r.input(&t.rig, &t.cs);
        t.ins[3] = try r.input(&t.rig, &t.sn);
        t.out = try t.rig.output(shape.rows * mels * 2);
        return true;
    }

    pub fn run(t: *Mel) !void {
        const n = t.shape.rows * 160;
        try t.k.mel(t.rig.s, t.ins[0], 0, n, t.ins[1], t.ins[1] + 2 * win, t.ins[2], t.ins[3], t.out, 0, t.shape.rows, t.shape.rows - 1, preemph, guard);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Mel) usize {
        return t.shape.rows * mels;
    }

    /// Frame f's windowed samples, as the kernel's fp32 steps make them.
    fn frame(t: *Mel, f: usize, xw: *[win]f32) void {
        for (xw, 0..) |*o, i| {
            const at = @as(isize, @intCast(f * 160 + i)) - 200;
            var y: f32 = 0;
            if (at >= 0 and at < t.x.len) {
                const u: usize = @intCast(at);
                y = if (u == 0) t.x[0] else t.x[u] - preemph * t.x[u - 1];
            }
            o.* = y * c.fromBf16(t.w[i]);
        }
    }

    /// The frame's powers: each bin's re and im sums in `acc`/`order` (null: float64 with its fp32 range).
    fn powers(t: *Mel, xw: *const [win]f32, how: ?struct { c.Accum, g.Order }, pw: *[bins]f64, pr: *[bins]Range) void {
        var terms: [2][win]f64 = undefined;
        for (0..bins) |k| {
            var part: [2]Range = undefined;
            var v: [2]f64 = undefined;
            for (0..2) |q| {
                var s: f64 = 0;
                var mag: f64 = 0;
                for (0..win) |i| {
                    const j = (k * (i + 56)) % fft;
                    terms[q][i] = @as(f64, xw[i]) * @as(f64, if (q == 0) t.cs[j] else t.sn[j]);
                    s += terms[q][i];
                    mag += @abs(terms[q][i]);
                }
                v[q] = if (how) |h| r.rows.sumIn(h[0], h[1], &terms[q]) else s;
                part[q] = r.sumRange(s, mag, win);
            }
            pw[k] = v[0] * v[0] + v[1] * v[1];
            pr[k] = r.sq(part[0]).fp32(0.5).add(r.sq(part[1]).fp32(0.5)).fp32(0.5);
        }
    }

    fn values(t: *Mel, how: ?struct { c.Accum, g.Order }, want: []f64, mag: []f64, ends: *Ends) void {
        var xw: [win]f32 = undefined;
        var pw: [bins]f64 = undefined;
        var pr: [bins]Range = undefined;
        var terms: [bins]f64 = undefined;
        for (0..t.shape.rows) |f| {
            if (f + 1 >= t.shape.rows) {
                for (0..mels) |m| ends.put(want, mag, f * mels + m, 0, Range.of(0));
                continue;
            }
            t.frame(f, &xw);
            t.powers(&xw, how, &pw, &pr);
            for (0..mels) |m| {
                var s: f64 = 0;
                var lo: f64 = 0;
                var hi: f64 = 0;
                for (0..bins) |k| {
                    const w: f64 = c.fromBf16(t.w[win + m * bins + k]);
                    terms[k] = w * pw[k];
                    s += terms[k];
                    lo += w * pr[k].lo;
                    hi += w * pr[k].hi;
                }
                if (how) |h| s = r.rows.sumIn(h[0], h[1], &terms);
                const e = c.gamma(bins, 0x1p-24) * hi;
                const rr = (Range{ .lo = @max(lo - e, 0), .hi = hi + e }).add(Range.of(guard)).fp32(0.5).up(log).fp32(r.log_ulps).bf16();
                ends.put(want, mag, f * mels + m, bf(@log(s + guard)), rr);
            }
        }
    }

    pub fn reference(t: *Mel, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        t.values(null, want, mag, &t.ends);
    }
    pub fn bounds(t: *Mel, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Mel, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
    pub fn simulate(t: *Mel, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        t.values(.{ acc, order }, out, mag, &scratch);
    }
};

fn log(x: f64) f64 {
    return @log(x);
}

// --- tf_nemo_norm: each feature over the valid frames, (x - mean) / (sd + eps) -------------------------------------

/// A warp's frames in turn (one in eight), then the eight warps' partials.
fn normDepth(s: c.Shape) usize {
    return (s.k + 7) / 8 + 8;
}

/// Rows: frames written (eight to a row), k: the valid ones, n: features.
const norm_served = [_]c.Shape{ .{ .rows = 592, .n = mels, .k = 585 }, .{ .rows = 3424, .n = mels, .k = 3421 }, .{ .rows = 13048, .n = mels, .k = 13047 } };

pub const Norm = struct {
    rig: rig_.Rig,
    k: K.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    ins: [1]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = normDepth, .ops = 6, .out = .bf16, .served = &norm_served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the outputs are the sums' own quotients: a feature's mean and spread cancel their scale, leaving a sum's error a few bf16 ulps" } };
    }
    pub fn deinit(t: *Norm) void {
        t.rig.gpu.gpa.free(t.x);
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Norm) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Norm);

    pub fn prepare(t: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        var s = shape;
        if (s.k >= s.rows) s.k = s.rows - 1; // the single-shape gates' clamped rows: one frame past the valid ones
        if (s.k < 2) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        t.shape = s;
        t.x = try a.alloc(u16, s.rows * mels);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        r.randomBf16(rng, t.x, if (fill == .stress) -24 else -12, 4);
        if (fill == .nonfinite) {
            t.x[0 * mels + 3] = r.nan16;
            t.x[1 * mels + 9] = r.inf16;
            t.x[2 * mels + 9] = r.ninf16;
            t.x[1 * mels + 20] = r.ninf16;
        }
        t.ins[0] = try r.input(&t.rig, t.x);
        t.out = try t.rig.output(s.rows * mels * 2);
        return true;
    }

    pub fn run(t: *Norm) !void {
        try t.k.norm(t.rig.s, t.ins[0], t.out, t.shape.rows, t.shape.k, 1e-5);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Norm) usize {
        return t.shape.rows * mels;
    }

    fn values(t: *Norm, how: ?struct { c.Accum, g.Order }, want: []f64, mag: []f64, ends: *Ends) !void {
        const a = t.rig.gpu.gpa;
        const valid = t.shape.k;
        const depth = normDepth(t.shape);
        const col = try a.alloc(f64, valid);
        defer a.free(col);
        const terms = try a.alloc(f64, valid);
        defer a.free(terms);
        const nv: f64 = @floatFromInt(valid);
        for (0..mels) |m| {
            var s: f64 = 0;
            var abs: f64 = 0;
            for (col, 0..) |*v, f| {
                v.* = c.fromBf16(t.x[f * mels + m]);
                s += v.*;
                abs += @abs(v.*);
            }
            if (how) |h| s = r.rows.sumIn(h[0], h[1], col);
            const mean = s / nv;
            const mr = r.scale(r.sumRange(s, abs, depth), 1 / nv).fp32(0.5);
            var ss: f64 = 0;
            var hi: f64 = 0;
            var lo: f64 = 0;
            for (col, terms) |v, *d| {
                d.* = (v - mean) * (v - mean);
                ss += d.*;
                const dr = r.sq(r.sub(Range.of(v), mr).fp32(0.5));
                lo += dr.lo;
                hi += dr.hi;
            }
            if (how) |h| ss = r.rows.sumIn(h[0], h[1], terms);
            const e = c.gamma(@floatFromInt(depth), 0x1p-24) * hi;
            const vr = r.scale(Range{ .lo = lo - e, .hi = hi + e }, 1 / (nv - 1)).fp32(0.5);
            const sdr = (Range{ .lo = @sqrt(@max(vr.lo, 0)), .hi = @sqrt(vr.hi) }).fp32(0.5).add(Range.of(1e-5)).fp32(0.5);
            const sd = @sqrt(ss / (nv - 1)) + 1e-5;
            for (0..t.shape.rows) |f| {
                const i = f * mels + m;
                if (f >= valid) {
                    ends.put(want, mag, i, 0, Range.of(0));
                    continue;
                }
                const v = col[f];
                ends.put(want, mag, i, bf((v - mean) / sd), r.div(r.sub(Range.of(v), mr).fp32(0.5), sdr).bf16());
            }
        }
    }

    pub fn reference(t: *Norm, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        try t.values(null, want, mag, &t.ends);
    }
    pub fn bounds(t: *Norm, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Norm, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
    pub fn simulate(t: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        try t.values(.{ acc, order }, out, mag, &scratch);
    }
};
