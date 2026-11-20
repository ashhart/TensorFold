//! Inkling 2's DSpark norms under the contract: tf_ink2_dsp_mix, tf_ink2_dsp_addnorm and tf_ink2_dsp_qk.
const std = @import("std");
const core = @import("core");
const dk = @import("inkling2").dspark_ops;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

const hidden = 4608;
const taps = 8;
const D = 128;
const eps: f32 = 1e-6;
const theta: f32 = 1e6;

/// A thread's chain over the row (512 threads), the warp's 5 shuffles, warp 0's 5 over the partials, the square.
fn rowDepth(s: c.Shape) usize {
    return (s.n + 511) / 512 + 5 + 5 + 1;
}

/// x * y in fp32: exact for gate 4's integers (no spread), else one rounding.
fn times(s: Sums, x: Iv, y: Iv) Iv {
    if (s.exact) return Iv.of(x.m * y.m);
    return x.mul(y);
}

fn square(s: Sums, v: Iv) Iv {
    if (s.exact) return Iv.of(v.m * v.m);
    const lo = @min(@abs(v.lo), @abs(v.hi));
    const hi = @max(@abs(v.lo), @abs(v.hi));
    return (Iv{ .m = v.m * v.m, .lo = if (v.lo <= 0 and v.hi >= 0) 0 else lo * lo, .hi = hi * hi }).mul(Iv.of(1));
}

/// RMSNorm of one row's fp32 values: the fp32 sum of squares, rsqrtf, then (v * r) * w rounded once to bf16.
fn normRow(s: Sums, x: []const Iv, w: []const u16, out: []Iv, scratch: []f64, sq: []Iv) void {
    for (sq, x) |*q, v| q.* = square(s, v);
    const total = s.sum(sq, rowDepth(.{ .rows = 1, .n = x.len, .k = 0 }), scratch);
    const r = total.div(Iv.of(@floatFromInt(x.len))).add(Iv.of(eps)).rsqrt();
    for (out, x, w) |*o, v, wv| o.* = v.mul(r).mul(Iv.of(ref.bf(wv))).bf();
}

/// Norm weights as the checkpoint's plain multipliers: mostly in [0.5, 1.5), one in 97 four times larger.
fn weights(a: std.mem.Allocator, rng: std.Random, n: usize) ![]u16 {
    const w = try a.alloc(u16, n);
    for (w, 0..) |*v, i| v.* = c.toBf16((0.5 + rng.float(f32)) * (if (i % 97 == 3) @as(f32, 4) else 1));
    return w;
}

/// bf16 values in [-2, 2), one in 64 ten times larger; stress: integers in [-lim, lim].
fn values(a: std.mem.Allocator, rng: std.Random, n: usize, fill: g.Fill, lim: i32) ![]u16 {
    const x = try a.alloc(u16, n);
    for (x, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress) ref.int(rng, -lim, lim) else (rng.float(f32) * 4 - 2) * (if (i % 64 == 7) @as(f32, 10) else 1));
    return x;
}

/// Rows x hidden x taps: an absorb's prompt rows (to 2,048), a draft's context rows, and f_rnn's one row.
pub fn mixSpec(drop: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = hidden, .k = taps }, .{ .rows = 1, .n = hidden, .k = taps }, .{ .rows = 7, .n = hidden, .k = taps }, .{ .rows = 2048, .n = hidden, .k = taps } };
    const name = if (drop) "inkling2 dspark control: the tap mix drops its last tap" else "inkling2 dspark tf_ink2_dsp_mix (alpha . taps in fp32, RMSNorm, rounded once)";
    return .{ .name = name, .accum = .f32, .depth = rowDepth, .out = .bf16, .served = served, .sums = .{} };
}

/// out[row] = RMSNorm(sum_t alpha[t] taps[row][t]) * w (vLLM's mix_rmsnorm).
pub const Mix = struct {
    kern: *const dk.Kernels,
    /// control: launched with one tap fewer (the head's input tap left out)
    drop: bool = false,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    alpha: []f32 = &.{},
    w: []u16 = &.{},
    in: [3]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Mix) c.Spec {
        return mixSpec(self.drop);
    }

    pub fn control(self: *const Mix) ?g.Gate {
        return if (self.drop) .shapes else null;
    }

    pub fn free(self: *Mix, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.alpha);
        a.free(self.w);
        self.x = &.{};
        self.alpha = &.{};
        self.w = &.{};
    }

    /// alpha a softmax row (positive, summing to 1); stress: small integers, so every mix is an exact integer.
    pub fn prepare(self: *Mix, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.n != hidden or shape.k != taps or shape.rows == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.x = try values(a, rng, shape.rows * taps * n, fill, 4);
        self.alpha = try a.alloc(f32, taps);
        var total: f32 = 0;
        for (self.alpha) |*v| {
            v.* = 0.2 + rng.float(f32);
            total += v.*;
        }
        const whole = [taps]f32{ 1, -1, 2, 1, -2, 1, 1, -1 };
        for (self.alpha, whole) |*v, x| v.* = if (fill == .stress) x else v.* / total;
        self.w = try weights(a, rng, n);
        if (fill == .nonfinite) {
            const at = [_][3]usize{ .{ 0, 3, n / 3 }, .{ 1, 0, n / 2 }, .{ 2, 7, n - 1 }, .{ 3, 1, 0 }, .{ 3, 2, n - 1 } };
            const put = [_]f32{ ref.nan, ref.inf, -ref.inf, ref.inf, -ref.inf };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.x[(p[0] * taps + p[1]) * n + p[2]] = c.toBf16(v);
            };
        }
        self.in = .{ try ref.input(rig, self.x), try ref.input(rig, self.alpha), try ref.input(rig, self.w) };
        self.out = try rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(self: *Mix, rig: *rig_.Rig) !void {
        try dk.mix(self.kern, rig.s, self.in[0], self.in[1], self.in[2], self.out, self.shape.rows, @as(usize, taps) - @intFromBool(self.drop), self.shape.n, eps);
    }

    pub fn count(self: *const Mix) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Mix, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const buf = try a.alloc(Iv, 2 * n);
        defer a.free(buf);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        var terms: [taps]Iv = undefined;
        for (0..self.shape.rows) |r| {
            for (buf[0..n], 0..) |*m, i| {
                for (&terms, 0..) |*t, k| t.* = times(s, Iv.of(ref.bf(self.x[(r * taps + k) * n + i])), Iv.of(self.alpha[k]));
                m.* = s.sum(&terms, taps, scratch);
            }
            normRow(s, buf[0..n], self.w, out[r * n ..][0..n], scratch, buf[n..]);
        }
    }
};

/// Which output an addnorm registration checks: the residual written back, or the next norm's rows.
pub const View = enum { resid, normed };

/// A draft block's rows x hidden x ranks' partials (0: the first layer's norm, 2: after o and down at TP=2).
pub fn addSpec(v: View, short: bool) c.Spec {
    const both = &[_]c.Shape{ .{ .rows = 8, .n = hidden, .k = 2 }, .{ .rows = 8, .n = hidden, .k = 0 } };
    const resid = &[_]c.Shape{.{ .rows = 8, .n = hidden, .k = 2 }};
    const name = if (short) "inkling2 dspark control: addnorm adds rank 0's partial alone" else if (v == .resid) "inkling2 dspark tf_ink2_dsp_addnorm, the residual written back" else "inkling2 dspark tf_ink2_dsp_addnorm, the next norm's rows";
    return .{ .name = name, .accum = .f32, .depth = rowDepth, .out = .bf16, .in_place = true, .served = if (v == .resid) resid else both, .sums = .{} };
}

/// resid (bf16) += the ranks' fp32 partials in rank order; out = RMSNorm of the unrounded sum * w.
pub const AddNorm = struct {
    kern: *const dk.Kernels,
    view: View,
    /// control: launched with one partial where two are due
    short: bool = false,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    parts: []f32 = &.{},
    w: []u16 = &.{},
    resid: u64 = 0,
    in: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const AddNorm) c.Spec {
        return addSpec(self.view, self.short);
    }

    pub fn control(self: *const AddNorm) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *AddNorm, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.parts);
        a.free(self.w);
        self.x = &.{};
        self.parts = &.{};
        self.w = &.{};
    }

    /// Partials in [-2, 2); stress: rank 0's integers near 2^16 cancelled by rank 1's (fp16 overflows, bf16 loses).
    pub fn prepare(self: *AddNorm, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != hidden or (shape.k != 0 and shape.k != 2)) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        self.x = try values(a, rng, rows * n, fill, 32);
        self.parts = try ref.f32s(a, rng, 2 * rows * n, -2, 2);
        if (fill == .stress) for (0..rows * n) |i| {
            const big = ref.int(rng, 40000, 120000);
            self.parts[i] = big;
            self.parts[rows * n + i] = ref.int(rng, -8, 8) - big;
        };
        self.w = try weights(a, rng, n);
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, n / 3 }, .{ 1, n / 2 }, .{ 2, n - 1 }, .{ 3, 0 }, .{ 3, n - 1 } };
            const put = [_]f32{ ref.nan, ref.inf, -ref.inf, ref.inf, -ref.inf };
            for (at, put) |p, v| if (p[0] < rows) {
                self.x[p[0] * n + p[1]] = c.toBf16(v);
            };
            self.parts[(rows - 1) * n + 5] = ref.nan; // the last row, rank 0
        }
        self.in = .{ try ref.input(rig, self.parts), try ref.input(rig, self.w) };
        if (self.view == .resid) {
            self.resid = try rig.output(rows * n * 2);
            self.out = try rig.scratch(rows * n * 2);
        } else {
            self.resid = try rig.scratch(rows * n * 2);
            self.out = try rig.output(rows * n * 2);
        }
        return true;
    }

    pub fn run(self: *AddNorm, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.resid, std.mem.sliceAsBytes(self.x)); // read and written in place: every run the same pass
        const parts = self.shape.k - @intFromBool(self.short and self.shape.k > 0);
        try dk.addNorm(self.kern, rig.s, self.resid, self.in[0], parts, self.shape.rows * self.shape.n, self.in[1], self.out, self.shape.rows, self.shape.n, eps);
    }

    pub fn count(self: *const AddNorm) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const AddNorm, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const rows = self.shape.rows;
        const k = self.shape.k;
        const buf = try a.alloc(Iv, 2 * n);
        defer a.free(buf);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        var terms: [2]Iv = undefined;
        for (0..rows) |r| {
            const row = buf[0..n];
            for (row, 0..) |*v, i| {
                for (terms[0..k], 0..) |*t, q| t.* = Iv.of(self.parts[(q * rows + r) * n + i]);
                const x = Iv.of(ref.bf(self.x[r * n + i]));
                v.* = if (k == 0) x else x.add(s.sum(terms[0..k], k, scratch));
            }
            if (self.view == .resid) {
                for (out[r * n ..][0..n], row) |*o, v| o.* = v.bf();
            } else normRow(s, row, self.w, out[r * n ..][0..n], scratch, buf[n..]);
        }
    }
};

/// The kernel's calls: a draft block's q, k, v (24 query, 4 key heads a TP=2 rank), or an absorb's context k, v.
pub const Form = enum { draft, absorb };

fn qHeads(f: Form) usize {
    return if (f == .draft) 24 else 0;
}

const kv_heads = 4;

fn width(f: Form) usize {
    return (qHeads(f) + 2 * kv_heads) * D;
}

/// The ring a call writes at its end position: a sliding layer's 4,096 rows, or the global layer's whole context.
fn ringOf(end: usize) usize {
    return if (end > 100_000) 1_050_624 else 4096;
}

/// A lane's two squares and their add, the warp's 5 shuffles, warp 0's 5 over the two partials.
fn headDepth(_: c.Shape) usize {
    return 12;
}

const dw = width(.draft);
const aw = width(.absorb);
const draft_served = [_]c.Shape{ .{ .rows = 8, .n = dw, .k = 3007 }, .{ .rows = 8, .n = dw, .k = 70_008 }, .{ .rows = 8, .n = dw, .k = 1_048_576 }, .{ .rows = 8, .n = dw, .k = 8 } };
const absorb_served = [_]c.Shape{ .{ .rows = 16, .n = aw, .k = 2016 }, .{ .rows = 2048, .n = aw, .k = 4096 }, .{ .rows = 2048, .n = aw, .k = 1_048_576 }, .{ .rows = 1, .n = aw, .k = 70_001 } };

/// Rows x the row's heads x the call's end position (pos0 + rows): early, deep in a ring, and at the 1M edge.
pub fn qkSpec(f: Form, late: bool) c.Spec {
    const served: []const c.Shape = switch (f) {
        .draft => &draft_served,
        .absorb => &absorb_served,
    };
    const name = if (late) "inkling2 dspark control: q/k prep rotates its rows one position late" else switch (f) {
        .draft => "inkling2 dspark tf_ink2_dsp_qk, a draft block's q, k, v",
        .absorb => "inkling2 dspark tf_ink2_dsp_qk, an absorb's context k, v",
    };
    return .{ .name = name, .accum = .f32, .depth = headDepth, .out = .bf16, .served = served, .sums = .{} };
}

/// Per (row, head): RMSNorm * w rounded to bf16, then neox RoPE at pos0 + row; key and value heads to the caches.
pub const Qk = struct {
    kern: *const dk.Kernels,
    form: Form,
    /// control: the device rotates every row one position late (its cache rows land where they are due)
    late: bool = false,
    shape: c.Shape = undefined,
    pos: usize = 0,
    x: []u16 = &.{},
    w: [2][]u16 = .{ &.{}, &.{} },
    dev: [3]u64 = @splat(0),
    outs: [3]u64 = @splat(0),

    pub const sums = {};

    pub fn spec(self: *const Qk) c.Spec {
        return qkSpec(self.form, self.late);
    }

    pub fn control(self: *const Qk) ?g.Gate {
        return if (self.late) .shapes else null;
    }

    pub fn free(self: *Qk, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
        for (&self.w) |*v| {
            a.free(v.*);
            v.* = &.{};
        }
    }

    /// Heads in [-2, 2) (one in 64 ten times larger); stress: integers in [-64, 64] (the squares pass 2^16).
    pub fn prepare(self: *Qk, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const wd = width(self.form);
        if (shape.n != wd or shape.rows == 0 or shape.k < shape.rows) return false;
        const ring = ringOf(shape.k);
        self.pos = shape.k - shape.rows;
        const shift: usize = @intFromBool(self.late);
        if ((self.pos + shift) % ring + shape.rows > ring) return false;
        self.free(a);
        self.shape = shape;
        const rows = shape.rows;
        self.x = try values(a, rng, rows * wd, fill, 64);
        if (fill == .nonfinite) {
            self.x[3] = c.toBf16(ref.nan); // row 0's first head: its every output NaN
            self.x[wd + D + 9] = c.toBf16(ref.inf);
            self.x[2 * wd + qHeads(self.form) * D + 70] = c.toBf16(-ref.inf); // row 2's first key head
            self.x[3 * wd + wd - 5] = c.toBf16(ref.inf); // row 3's last value head, copied as it is
        }
        for (&self.w) |*w| w.* = try weights(a, rng, D);
        self.dev = .{ try ref.input(rig, self.x), try ref.input(rig, self.w[0]), try ref.input(rig, self.w[1]) };
        const slot = (self.pos + shift) % ring; // the caches' base, so slot pos + row lands on the buffer's row
        const hq = qHeads(self.form);
        self.outs[0] = if (hq > 0) try rig.output(rows * hq * D * 2) else 0;
        self.outs[1] = try rig.output(rows * kv_heads * D * 2) - slot * kv_heads * D * 2;
        self.outs[2] = try rig.output(rows * kv_heads * D * 2) - slot * kv_heads * D * 2;
        return true;
    }

    pub fn run(self: *Qk, rig: *rig_.Rig) !void {
        const hq = qHeads(self.form);
        const d = self.dev;
        try dk.qk(self.kern, rig.s, .{ .x = d[0], .qw = if (hq > 0) d[1] else 0, .kw = d[2], .q = self.outs[0], .kc = self.outs[1], .vc = self.outs[2], .ld = @intCast(width(self.form)), .hq = @intCast(hq), .hk = kv_heads, .ring = @intCast(ringOf(self.shape.k)), .pos0 = @intCast(self.pos + @intFromBool(self.late)), .eps = eps, .theta = theta }, self.shape.rows);
    }

    pub fn count(self: *const Qk) usize {
        return self.shape.rows * self.shape.n;
    }

    /// One head's 128 outputs at row r: the norm (1 / sqrtf), then neox RoPE in fp32 at pos + r.
    fn head(self: *const Qk, s: Sums, r: usize, hd: usize, w: []const u16, out: []Iv, scratch: []f64) void {
        var x: [D]Iv = undefined;
        var sq: [D]Iv = undefined;
        var y: [D]Iv = undefined;
        for (&x, 0..) |*v, d| v.* = Iv.of(ref.bf(self.x[r * self.shape.n + hd * D + d]));
        for (&sq, x) |*q, v| q.* = square(s, v);
        const ms = s.sum(&sq, headDepth(self.shape), scratch).div(Iv.of(D)).add(Iv.of(eps));
        const root: Iv = .{ .m = @sqrt(ms.m), .lo = @sqrt(ms.lo) * (1 - ref.unit), .hi = @sqrt(ms.hi) * (1 + ref.unit) };
        const inv = Iv.of(1).div(root);
        for (&y, x, w) |*o, v, wv| o.* = v.mul(inv).mul(Iv.of(ref.bf(wv))).bf();
        const position: f64 = @floatFromInt(self.pos + r);
        for (0..D / 2) |i| {
            const e = @as(f64, @floatFromInt(2 * i)) / D;
            const am = position * std.math.pow(f64, theta, -e);
            // powf within 8 ulps (exactly 1 at i 0), the fp32 reciprocal and the product: half an ulp each
            const rel: f64 = if (i == 0) 0 else 9 * 0x1p-23 + 0x1p-50;
            const angle: Iv = .{ .m = am, .lo = am - am * rel, .hi = am + am * rel };
            const co = angle.trig(false);
            const si = angle.trig(true);
            out[i] = y[i].mul(co).sub(y[i + 64].mul(si)).bf();
            out[i + 64] = y[i + 64].mul(co).add(y[i].mul(si)).bf();
        }
    }

    pub fn solve(self: *const Qk, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const hq = qHeads(self.form);
        const rows = self.shape.rows;
        var scratch: [D]f64 = undefined;
        var o: usize = 0;
        for (0..rows) |r| for (0..hq) |hd| {
            self.head(s, r, hd, self.w[0], out[o..][0..D], &scratch);
            o += D;
        };
        for (0..rows) |r| for (0..kv_heads) |hd| {
            self.head(s, r, hq + hd, self.w[1], out[o..][0..D], &scratch);
            o += D;
        };
        for (0..rows) |r| for (0..kv_heads) |hd| for (0..D) |d| {
            out[o] = Iv.of(ref.bf(self.x[r * self.shape.n + (hq + kv_heads + hd) * D + d])); // the values as they are
            o += 1;
        };
        std.debug.assert(o == out.len);
    }
};
