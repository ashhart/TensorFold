//! tf_nemo_qk, _taps, _relu, _preds and _fuse under the contract: row steps at their bf16 rounding points.
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

fn noSums(_: c.Shape) usize {
    return 0;
}

fn served(comptime list: []const usize, n: usize, k: usize) [list.len]c.Shape {
    var out: [list.len]c.Shape = undefined;
    for (&out, list) |*o, x| o.* = .{ .rows = x, .n = n, .k = k };
    return out;
}

/// Rows the speaker model's calls take (a short clip, the first window, a full one) and the ASR windows'.
const spk_calls = [_]usize{ 74, 550, 828 };
const asr_calls = [_]usize{ 74, 550, 600 };
const kept = [_]usize{ 74, 500, 131 };

// --- tf_nemo_qk: q and k heads LayerNormed (the ASR stack), then dims (i, i + 32) turned -----------------------------

/// A lane's four values in turn, then the warp's five shuffles.
fn qkDepth(_: c.Shape) usize {
    return 4 + 5;
}

const qk_asr = served(&asr_calls, 10, 128);
const qk_spk = served(&spk_calls, 8, 64);

pub const Qk = struct {
    rig: rig_.Rig,
    k: K.Kernels,
    normed: bool, // the ASR stack's form: 10 heads of 128 with q/k LayerNorm; else 8 heads of 64
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    v: []u16 = &.{}, // q weight, q bias, k weight, k bias [dim]; then cos, sin [rows][32]
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    pub fn spec(name: []const u8, normed: bool) c.Spec {
        return .{ .name = name, .accum = if (normed) .f32 else .none, .depth = if (normed) qkDepth else noSums, .ops = 8, .mids = 3, .out = .bf16, .served = if (normed) &qk_asr else &qk_spk, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = r.few_terms } };
    }
    pub fn deinit(t: *Qk) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Qk) void {
        t.rig.gpu.gpa.free(t.x);
        t.rig.gpu.gpa.free(t.v);
        t.x = &.{};
        t.v = &.{};
    }
    pub fn kernel(t: *Qk) g.Kernel {
        return .{ .spec = spec(t.name, t.normed), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Qk);

    pub fn prepare(t: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const d = shape.k;
        const w = 3 * shape.n * d;
        t.x = try a.alloc(u16, shape.rows * w);
        t.v = try a.alloc(u16, 4 * d + 2 * 32 * shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        r.randomBf16(rng, t.x, if (fill == .stress) -40 else -3, if (fill == .stress) 40 else 3);
        for (0..4) |i| r.randomBf16(rng, t.v[i * d ..][0..d], if (i % 2 == 0) 0.5 else -0.5, if (i % 2 == 0) 1.5 else 0.5);
        r.randomBf16(rng, t.v[4 * d ..], -1, 1);
        if (fill == .nonfinite) {
            t.x[5] = r.nan16; // row 0, q head 0
            t.x[w + shape.n * d + 3] = r.inf16; // row 1, k head 0
            t.x[2 * w + 2 * shape.n * d + 1] = r.ninf16; // row 2, v: copied as it is
            t.x[3 * w + d + 40] = r.ninf16; // row 3, q head 1
        }
        t.ins[0] = try r.input(&t.rig, t.x);
        t.ins[1] = try r.input(&t.rig, t.v);
        t.out = try t.rig.output(t.x.len * 2);
        return true;
    }

    /// In place on a copy of the rows: the output buffer takes them first.
    pub fn run(t: *Qk) !void {
        const d = t.shape.k;
        try t.rig.gpu.d.check(t.rig.gpu.d.api.cuMemcpyDtoDAsync_v2(t.out, t.ins[0], t.x.len * 2, t.rig.s.handle), "cuMemcpyDtoDAsync");
        const v = t.ins[1];
        const norms: ?[4]u64 = if (t.normed) .{ v, v + 2 * d, v + 4 * d, v + 6 * d } else null;
        try t.k.qk(t.rig.s, t.out, norms, v + 8 * d, v + 8 * d + 64 * t.shape.rows, t.shape.rows, t.shape.n, d, r.eps);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Qk) usize {
        return t.x.len;
    }

    fn values(t: *Qk, how: ?struct { c.Accum, g.Order }, want: []f64, mag: []f64, ends: *Ends) !void {
        const a = t.rig.gpu.gpa;
        const d = t.shape.k;
        const h = t.shape.n;
        const w = 3 * h * d;
        const vals = try r.bfs(a, t.v);
        defer a.free(vals);
        const x = try a.alloc(f64, d);
        defer a.free(x);
        const y = try a.alloc(f64, d);
        defer a.free(y);
        const yr = try a.alloc(Range, d);
        defer a.free(yr);
        for (0..t.shape.rows) |row| {
            for (0..3 * h) |slot| {
                const at = row * w + slot * d;
                for (x, t.x[at..][0..d]) |*o, b| o.* = c.fromBf16(b);
                if (slot >= 2 * h) {
                    for (x, 0..) |v, i| ends.put(want, mag, at + i, v, Range.of(v));
                    continue;
                }
                if (t.normed) {
                    const nw = vals[if (slot < h) 0 else 2 * d..][0..d];
                    const nb = vals[if (slot < h) d else 3 * d..][0..d];
                    r.layerNorm(x, nw, nb, qkDepth(t.shape), y, yr);
                    if (how) |m| try r.layerNormIn(m[0], m[1], a, x, nw, nb, y);
                } else for (x, y, yr) |v, *o, *q| {
                    o.* = v;
                    q.* = Range.of(v);
                }
                for (0..d) |i| {
                    if (i >= 64) {
                        ends.put(want, mag, at + i, y[i], yr[i]);
                        continue;
                    }
                    const l = i % 32;
                    const cs = vals[4 * d + row * 32 + l];
                    const sn = vals[4 * d + 32 * t.shape.rows + row * 32 + l];
                    const p = if (i < 32) .{ y[i], y[i + 32] } else .{ y[i - 32], y[i] };
                    const pr = if (i < 32) .{ yr[i], yr[i + 32] } else .{ yr[i - 32], yr[i] };
                    const C = Range.of(cs);
                    const S = Range.of(sn);
                    if (i < 32) ends.put(want, mag, at + i, bf(bf(p[0] * cs) - bf(p[1] * sn)), r.sub(pr[0].mul(C).fp32(0.5).bf16(), pr[1].mul(S).fp32(0.5).bf16()).fp32(0.5).bf16()) else ends.put(want, mag, at + i, bf(bf(p[1] * cs) + bf(p[0] * sn)), pr[1].mul(C).fp32(0.5).bf16().add(pr[0].mul(S).fp32(0.5).bf16()).fp32(0.5).bf16());
                }
            }
        }
    }

    pub fn reference(t: *Qk, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        try t.values(null, want, mag, &t.ends);
    }
    pub fn bounds(t: *Qk, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Qk, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
    pub fn simulate(t: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        var scratch: Ends = .{ .a = t.rig.gpu.gpa };
        defer scratch.deinit();
        try scratch.size(t.count());
        const mag = try t.rig.gpu.gpa.alloc(f64, out.len);
        defer t.rig.gpu.gpa.free(mag);
        try t.values(.{ acc, order }, out, mag, &scratch);
    }
};

// --- tf_nemo_taps and tf_nemo_relu: copies, exact ---

const taps_served = served(&spk_calls, 192, 3);
const relu_served = served(&spk_calls, 1536, 1);

pub const Copy = struct {
    rig: rig_.Rig,
    k: K.Kernels,
    relu: bool, // tf_nemo_relu in place on a copy; else tf_nemo_taps
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    pub fn spec(name: []const u8, relu: bool) c.Spec {
        return .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .served = if (relu) &relu_served else &taps_served };
    }
    pub fn deinit(t: *Copy) void {
        t.rig.gpu.gpa.free(t.x);
        t.rig.deinit();
    }
    pub fn kernel(t: *Copy) g.Kernel {
        return .{ .spec = spec(t.name, t.relu), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Copy);

    pub fn prepare(t: *Copy, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        t.shape = shape;
        t.x = try a.alloc(u16, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        r.randomBf16(prng.random(), t.x, -4, 4);
        if (fill == .nonfinite) {
            t.x[1] = r.nan16;
            t.x[shape.n + 2] = r.inf16;
            t.x[2 * shape.n + 3] = r.ninf16;
        }
        t.in = try r.input(&t.rig, t.x);
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Copy) !void {
        if (t.relu) {
            try t.rig.gpu.d.check(t.rig.gpu.d.api.cuMemcpyDtoDAsync_v2(t.out, t.in, t.x.len * 2, t.rig.s.handle), "cuMemcpyDtoDAsync");
            try t.k.relu(t.rig.s, t.out, t.x.len);
        } else try t.k.taps(t.rig.s, t.in, t.out, t.shape.rows, t.shape.n);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Copy) usize {
        return if (t.relu) t.x.len else 3 * t.x.len;
    }

    pub fn reference(t: *Copy, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*w, *m, i| {
            if (t.relu) {
                const v: f64 = c.fromBf16(t.x[i]);
                w.* = if (v < 0) 0 else v;
            } else {
                const row = i / (3 * n);
                const from = @as(isize, @intCast(row + i % 3)) - 1;
                w.* = if (from >= 0 and from < t.shape.rows) c.fromBf16(t.x[@as(usize, @intCast(from)) * n + (i % (3 * n)) / 3]) else 0;
            }
            m.* = if (std.math.isFinite(w.*)) @abs(w.*) else 0;
        }
    }
};

// --- tf_nemo_preds: the mean of eight bf16 sigmoids ----------------------------------------------------------------

fn predsDepth(_: c.Shape) usize {
    return 8;
}

const preds_served = served(&spk_calls, 8, 8);

pub const Preds = struct {
    rig: rig_.Rig,
    k: K.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,
    ends: Ends,

    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = predsDepth, .ops = 4, .out = .f32, .store = .bf16, .served = &preds_served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = r.few_terms } };
    }
    pub fn deinit(t: *Preds) void {
        t.rig.gpu.gpa.free(t.x);
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Preds) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Preds);

    pub fn prepare(t: *Preds, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        a.free(t.x);
        t.shape = shape;
        t.x = try a.alloc(u16, shape.rows * 64);
        var prng = std.Random.DefaultPrng.init(seed);
        r.randomBf16(prng.random(), t.x, -8, 8);
        if (fill == .nonfinite) {
            t.x[3] = r.nan16;
            t.x[64 + 9] = r.inf16;
            t.x[128 + 10] = r.ninf16;
        }
        t.in = try r.input(&t.rig, t.x);
        t.out = try t.rig.output(shape.rows * 8 * 4);
        return true;
    }

    pub fn run(t: *Preds) !void {
        try t.k.preds(t.rig.s, t.in, t.out, t.shape.rows, 8, 8);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Preds) usize {
        return t.shape.rows * 8;
    }

    fn values(t: *Preds, how: ?struct { c.Accum, g.Order }, want: []f64, mag: []f64, ends: *Ends) void {
        var p: [8]f64 = undefined;
        for (0..t.shape.rows) |row| for (0..8) |s| {
            var lo: f64 = 0;
            var hi: f64 = 0;
            var sum: f64 = 0;
            for (&p, 0..) |*o, j| {
                const x: f64 = c.fromBf16(t.x[(row * 8 + j) * 8 + s]);
                o.* = bf(1 / (1 + @exp(-x)));
                const er = Range.of(@exp(-x)).fp32(r.exp_ulps).add(Range.of(1)).fp32(0.5).recip().fp32(0.5).bf16();
                lo += er.lo;
                hi += er.hi;
                sum += o.*;
            }
            if (how) |h| sum = r.rows.sumIn(h[0], h[1], &p);
            const e = c.gamma(8, 0x1p-24) * hi;
            ends.put(want, mag, row * 8 + s, bf(sum / 8), r.scale(Range{ .lo = lo - e, .hi = hi + e }, 0.125).bf16());
        };
    }

    pub fn reference(t: *Preds, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        t.values(null, want, mag, &t.ends);
    }
    pub fn bounds(t: *Preds, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Preds, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
    pub fn simulate(t: *Preds, acc: c.Accum, order: g.Order, out: []f64) !void {
        var scratch: Ends = .{ .a = t.rig.gpu.gpa };
        defer scratch.deinit();
        try scratch.size(t.count());
        const mag = try t.rig.gpu.gpa.alloc(f64, out.len);
        defer t.rig.gpu.gpa.free(mag);
        t.values(.{ acc, order }, out, mag, &scratch);
    }
};

// --- tf_nemo_fuse: the normed ASR row plus the speakers' normed activity through the kernel --------------------------

fn fuseDepth(_: c.Shape) usize {
    return 8;
}

const width = 1280;
const fuse_served = served(&kept, width, 8);

pub const Fuse = struct {
    rig: rig_.Rig,
    k: K.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    x: []u16 = &.{}, // the normed ASR rows [rows][1280], then the norm's weight and bias [8], then the kernel [8][1280]
    p: []f32 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = fuseDepth, .ops = 8, .mids = 2, .out = .bf16, .served = &fuse_served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = r.few_terms } };
    }
    pub fn deinit(t: *Fuse) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Fuse) void {
        t.rig.gpu.gpa.free(t.x);
        t.rig.gpu.gpa.free(t.p);
        t.x = &.{};
        t.p = &.{};
    }
    pub fn kernel(t: *Fuse) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Fuse);

    pub fn prepare(t: *Fuse, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * width;
        t.x = try a.alloc(u16, n + 16 + 8 * width);
        t.p = try a.alloc(f32, shape.rows * 8);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        r.randomBf16(rng, t.x[0..n], -3, 3);
        r.randomBf16(rng, t.x[n .. n + 8], 0.5, 1.5);
        r.randomBf16(rng, t.x[n + 8 .. n + 16], -0.5, 0.5);
        r.randomBf16(rng, t.x[n + 16 ..], -1, 1);
        for (t.p, 0..) |*v, i| v.* = if (i % 5 == 0) 0.5 else rng.float(f32); // 0.5 itself is not active
        if (fill == .nonfinite) {
            t.x[2] = r.nan16;
            t.x[width + 4] = r.inf16;
            t.x[2 * width + 4] = r.ninf16;
            t.x[n + 16 + 3 * width + 7] = r.nan16; // a kernel column: every row's output there
            t.p[9] = std.math.nan(f32); // a NaN prediction is not active
        }
        t.ins[0] = try r.input(&t.rig, t.x);
        t.ins[1] = try r.input(&t.rig, t.p);
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Fuse) !void {
        const n = t.shape.rows * width;
        const x = t.ins[0];
        try t.k.fuse(t.rig.s, x, t.ins[1], x + 2 * n, x + 2 * n + 16, x + 2 * n + 32, t.out, null, t.shape.rows, width, 0.5, r.eps);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Fuse) usize {
        return t.shape.rows * width;
    }

    fn values(t: *Fuse, how: ?struct { c.Accum, g.Order }, want: []f64, mag: []f64, ends: *Ends) !void {
        const a = t.rig.gpu.gpa;
        const n = t.shape.rows * width;
        const v = try r.bfs(a, t.x[n..]);
        defer a.free(v);
        var act: [8]f64 = undefined;
        var y: [8]f64 = undefined;
        var yr: [8]Range = undefined;
        var terms: [8]f64 = undefined;
        for (0..t.shape.rows) |row| {
            for (&act, t.p[row * 8 ..][0..8]) |*o, p| o.* = if (p > 0.5) 1 else 0;
            r.layerNorm(&act, v[0..8], v[8..16], 8, &y, &yr);
            if (how) |h| try r.layerNormIn(h[0], h[1], a, &act, v[0..8], v[8..16], &y);
            for (0..width) |col| {
                var s: f64 = 0;
                var pr = Range.of(0);
                var abs: f64 = 0;
                for (0..8) |i| {
                    const kv = v[16 + i * width + col];
                    terms[i] = y[i] * kv;
                    s += terms[i];
                    pr = pr.add(yr[i].mul(Range.of(kv)));
                    abs += @max(@abs(yr[i].lo), @abs(yr[i].hi)) * @abs(kv);
                }
                if (how) |h| s = r.rows.sumIn(h[0], h[1], &terms);
                const e = c.gamma(8, 0x1p-24) * abs;
                const an: f64 = c.fromBf16(t.x[row * width + col]);
                const rr = Range.of(an).add((Range{ .lo = pr.lo - e, .hi = pr.hi + e }).bf16()).fp32(0.5).bf16();
                ends.put(want, mag, row * width + col, bf(an + bf(s)), rr);
            }
        }
    }

    pub fn reference(t: *Fuse, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        try t.values(null, want, mag, &t.ends);
    }
    pub fn bounds(t: *Fuse, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Fuse, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
    pub fn simulate(t: *Fuse, acc: c.Accum, order: g.Order, out: []f64) !void {
        var scratch: Ends = .{ .a = t.rig.gpu.gpa };
        defer scratch.deinit();
        try scratch.size(t.count());
        const mag = try t.rig.gpu.gpa.alloc(f64, out.len);
        defer t.rig.gpu.gpa.free(mag);
        try t.values(.{ acc, order }, out, mag, &scratch);
    }
};
