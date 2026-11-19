//! Float64 models of Hy4-flash's hyper-connection kernels (ihc.metal): each fp32 step as a range, each bf16 point.
const std = @import("std");
const contract = @import("contract");
const h = @import("metal_hy4.zig");

const c = contract.spec;
const g = contract.gates;
pub const Range = h.fl.Range;
pub const bf = h.fl.bf;
const u = h.fl.u32_ulp;
const tiny = h.fl.tiny;

pub const streams = 4;
/// The seam's stream products a row (sum over columns of m_s m_u, s <= u), in the kernels' order.
pub const grams = 10;
/// float64's own rounding over a row's sums, added to every sum's range.
const slack = 0x1p-36;

/// The config's hyper-connection constants as the loaders pass them: rms_norm_eps, hc_eps and hc_magnitude in fp32.
pub const Params = struct { eps: f32, gate_eps: f32, magnitude: f32 };

/// How a model's sums run: float64 (.none), or gate 4's replay in an accumulator and order.
pub const Sum = struct {
    acc: c.Accum = .none,
    order: g.Order = .sequential,
    pub fn of(s: Sum, t: []const f64) f64 {
        return contract.rows.sumIn(s.acc, s.order, t);
    }
};

pub fn bv(bits: u16) f64 {
    return c.fromBf16(bits);
}

pub fn pt(v: f64) Range {
    return Range.point(v);
}

/// Each end through an fp32 step within k half-ulps and a flushed subnormal: the true ends bound every value between.
pub fn step(r: Range, k: f64) Range {
    return .{ .lo = r.lo - k * u * @abs(r.lo) - tiny, .hi = r.hi + k * u * @abs(r.hi) + tiny };
}

/// A sum of non-negative terms within gamma of its value, and float64's own rounding.
fn positive(v: f64, gam: f64) Range {
    return .{ .lo = @max(v * (1 - gam - slack), 0), .hi = v * (1 + gam + slack) };
}

/// rsqrt(x / n + eps) as the kernels take it: the division (2.5 ulps), the add, precise::rsqrt (2 ulps).
pub fn invRange(x: Range, n: f64, eps: f64) Range {
    const t = step(step((Range{ .lo = @max(x.lo, 0), .hi = @max(x.hi, 0) }).times(1 / n), 5).plus(pt(eps)), 1);
    return step(.{ .lo = 1 / @sqrt(t.hi), .hi = 1 / @sqrt(t.lo) }, 4);
}

pub fn inv(x: f64, n: f64, eps: f64) f64 {
    return 1 / @sqrt(x / n + eps);
}

/// A row's gate sums: its squares and up to 8 logits, float64 (or replayed) and the ranges fp32 sums allow.
pub const Sums = struct { ss: f64 = 0, dot: [8]f64 = @splat(0), ss_r: Range = .{ .lo = 0, .hi = 0 }, dot_r: [8]Range = @splat(.{ .lo = 0, .hi = 0 }) };

/// The gate sums of a row (x within lo..hi, each wide long) against proj [n_gates][wide]; ranges when `gam` is set.
pub fn sumsOf(x: []const f64, lo: []const f64, hi: []const f64, proj: []const f32, n_gates: usize, sum: Sum, gam: ?f64, terms: []f64) Sums {
    const wide = x.len;
    var s: Sums = .{};
    for (terms[0..wide], x) |*t, v| t.* = v * v;
    s.ss = sum.of(terms[0..wide]);
    for (0..n_gates) |j| {
        for (terms[0..wide], x, proj[j * wide ..][0..wide]) |*t, v, w| t.* = v * w;
        s.dot[j] = sum.of(terms[0..wide]);
    }
    const gm = gam orelse return s;
    var sl: f64 = 0;
    var sh: f64 = 0;
    for (lo, hi) |a, b| {
        sh += @max(a * a, b * b);
        sl += if (a <= 0 and b >= 0) 0 else @min(a * a, b * b);
    }
    s.ss_r = .{ .lo = @max(sl - (gm + slack) * sh, 0), .hi = sh * (1 + gm + slack) };
    for (0..n_gates) |j| {
        var dl: f64 = 0;
        var dh: f64 = 0;
        var mag: f64 = 0;
        for (lo, hi, proj[j * wide ..][0..wide]) |a, b, w32| {
            const w: f64 = w32;
            dl += @min(a * w, b * w);
            dh += @max(a * w, b * w);
            mag += @max(@abs(a * w), @abs(b * w));
        }
        s.dot_r[j] = .{ .lo = dl - (gm + slack) * mag, .hi = dh + (gm + slack) * mag };
    }
    return s;
}

/// A row's gates: sigmoid(dot rsqrt(ss / wide + eps) scale + bias), pre = g + hc_eps, post = hc_magnitude g + hc_eps.
pub const Gates = struct { pre: [4]f64 = @splat(0), post: [4]f64 = @splat(0), pre_r: [4]Range = undefined, post_r: [4]Range = undefined };

pub fn gatesOf(p: Params, s: Sums, wide: usize, n_gates: usize, bias: []const f32, scale: []const f32) Gates {
    var o: Gates = .{};
    const wf: f64 = @floatFromInt(wide);
    const iv = inv(s.ss, wf, p.eps);
    const iv_r = invRange(s.ss_r, wf, p.eps);
    const ge: f64 = p.gate_eps;
    const mg: f64 = p.magnitude;
    for (0..n_gates) |j| {
        const sc: f64 = scale[j / streams];
        const b: f64 = bias[j];
        const gate = 1 / (1 + @exp(-(s.dot[j] * iv * sc + b)));
        const z = step(step(step(s.dot_r[j].mul(iv_r), 1).times(sc), 1).plus(pt(b)), 1);
        const d = step(pt(1).plus(step(.{ .lo = @exp(-z.hi), .hi = @exp(-z.lo) }, 8)), 1);
        const gr = step(.{ .lo = 1 / d.hi, .hi = 1 / d.lo }, 5);
        if (j < streams) {
            o.pre[j] = gate + ge;
            o.pre_r[j] = step(gr.plus(pt(ge)), 1);
        } else {
            o.post[j - streams] = mg * gate + ge;
            o.post_r[j - streams] = step(step(gr.times(mg), 1).plus(pt(ge)), 1);
        }
    }
    return o;
}

/// The stream pairs of the 10 products, in the kernels' order.
pub fn pair(k: usize) [2]usize {
    var at: usize = 0;
    for (0..streams) |s| for (s..streams) |q| {
        if (at == k) return .{ s, q };
        at += 1;
    };
    unreachable;
}

/// The seam's stream products of a row of 4 W values (x within lo..hi); ranges when `gam` is set.
pub fn gramsOf(x: []const f64, lo: []const f64, hi: []const f64, width: usize, sum: Sum, gam: ?f64, terms: []f64, v: *[grams]f64, r: *[grams]Range) void {
    for (0..grams) |k| {
        const sq = pair(k);
        const xs = x[sq[0] * width ..][0..width];
        const xq = x[sq[1] * width ..][0..width];
        for (terms[0..width], xs, xq) |*t, a, b| t.* = a * b;
        v[k] = sum.of(terms[0..width]);
        const gm = gam orelse continue;
        var acc = pt(0);
        var mag: f64 = 0;
        for (0..width) |i| {
            const e = (Range{ .lo = lo[sq[0] * width + i], .hi = hi[sq[0] * width + i] }).mul(.{ .lo = lo[sq[1] * width + i], .hi = hi[sq[1] * width + i] });
            acc = acc.plus(e);
            mag += @max(@abs(e.lo), @abs(e.hi));
        }
        r[k] = acc.widen((gm + slack) * mag);
    }
}

/// The head's sum of squares of its collapsed row from the stream products: (1 or 2) pre_s pre_q, fma'd in order.
pub fn hsqOf(gt: Gates, v: [grams]f64) f64 {
    var acc: f64 = 0;
    for (0..grams) |k| {
        const sq = pair(k);
        acc += (if (sq[0] == sq[1]) @as(f64, 1) else 2) * gt.pre[sq[0]] * gt.pre[sq[1]] * v[k];
    }
    return acc;
}

pub fn hsqRange(gt: Gates, r: [grams]Range) Range {
    var acc = pt(0);
    for (0..grams) |k| {
        const sq = pair(k);
        const w = step(gt.pre_r[sq[0]].times(if (sq[0] == sq[1]) 1 else 2).mul(gt.pre_r[sq[1]]), 1);
        acc = step(acc.plus(w.mul(r[k])), 1);
    }
    return acc;
}

/// A normed value as the tail and head write it: bf16(bf16(x inv) gain).
pub fn normAt(x: f64, iv: f64, gain: f64) f64 {
    return bf(bf(x * iv) * gain);
}

pub fn normRange(x: Range, iv: Range, gain: f64) Range {
    return step(step(x.mul(iv), 1).bf16().times(gain), 1).bf16();
}

/// A launch's float64 values and their ranges for one output, row by row.
pub const Dest = struct {
    want: []f64,
    lo: []f64,
    hi: []f64,
    pub fn at(d: Dest, from: usize, n: usize) Dest {
        return .{ .want = d.want[from..][0..n], .lo = d.lo[from..][0..n], .hi = d.hi[from..][0..n] };
    }
    fn put(d: Dest, i: usize, want: f64, r: ?Range) void {
        d.want[i] = want;
        const x = r orelse return;
        d.lo[i] = x.lo;
        d.hi[i] = x.hi;
    }
};

/// What a head reads besides its hidden row: the collapse's weights, and the seam head's input-norm gain.
pub const Head = struct {
    p: Params,
    width: usize,
    n_gates: usize,
    proj: []const f32,
    bias: []const f32,
    scale: []const f32,
    /// The head's input-norm gain (null: the plain collapse, which writes no normed row).
    gain: ?[]const u16 = null,
    /// The longest rounding chain of the gate sums and of the stream products.
    depth: usize,
    gram_depth: usize = 0,
};

/// One row through a collapse: its gates, collapsed row and (the seam head) normed row; `exact` drops the sums' gamma.
pub fn headRow(hd: Head, sum: Sum, ranged: bool, exact: bool, x: []const f64, lo: []const f64, hi: []const f64, terms: []f64, col: ?Dest, norm: ?Dest, post: ?Dest) void {
    const W = hd.width;
    const gam: ?f64 = if (!ranged) null else if (exact) 0 else c.gamma(@floatFromInt(hd.depth), u);
    const s = sumsOf(x, lo, hi, hd.proj, hd.n_gates, sum, gam, terms);
    const gt = gatesOf(hd.p, s, streams * W, hd.n_gates, hd.bias, hd.scale);
    if (post) |d| for (0..streams) |q| d.put(q, gt.post[q], if (ranged) gt.post_r[q] else null);
    if (col == null and norm == null) return;
    var hiv: f64 = 0;
    var hiv_r: Range = pt(0);
    if (norm != null) {
        var gv: [grams]f64 = undefined;
        var gr: [grams]Range = undefined;
        const ggam: ?f64 = if (!ranged) null else if (exact) 0 else c.gamma(@floatFromInt(hd.gram_depth), u);
        gramsOf(x, lo, hi, W, sum, ggam, terms, &gv, &gr);
        const wf: f64 = @floatFromInt(W);
        hiv = inv(hsqOf(gt, gv), wf, hd.p.eps);
        if (ranged) hiv_r = invRange(hsqRange(gt, gr), wf, hd.p.eps);
    }
    for (0..W) |i| {
        var v: f64 = 0;
        var r = pt(0);
        for (0..streams) |q| {
            v += gt.pre[q] * x[q * W + i];
            if (ranged) r = step(r.plus(step(gt.pre_r[q].mul(.{ .lo = lo[q * W + i], .hi = hi[q * W + i] }), 1)), 1);
        }
        if (col) |d| d.put(i, bf(v), if (ranged) r.bf16() else null);
        if (norm) |d| {
            const gn = bv(hd.gain.?[i]);
            d.put(i, normAt(bf(v), hiv, gn), if (ranged) normRange(r.bf16(), hiv_r, gn) else null);
        }
    }
}

/// A tail's inputs: the branch (or two, added in bf16), norm gain, streams and post gates; `bare`: seamPartials.
pub const Tail = struct {
    p: Params,
    width: usize,
    bare: bool = false,
    b0: []const u16,
    b1: ?[]const u16 = null,
    gain: []const u16,
    res: []const u16,
    post: []const f32,
    depth: usize,
};

/// One row's mixed streams: m = bf16(residual + post_s bf16(bf16(v rsqrt(mean v^2 + eps)) gain)), into x (and lo, hi).
pub fn tailRow(t: Tail, sum: Sum, ranged: bool, exact: bool, terms: []f64, x: []f64, lo: []f64, hi: []f64) void {
    const W = t.width;
    if (t.bare) {
        for (x, t.res, 0..) |*o, b, i| {
            o.* = bv(b);
            lo[i] = o.*;
            hi[i] = o.*;
        }
        return;
    }
    for (terms[0..W], 0..) |*q, i| {
        const v = if (t.b1) |b1| bf(bv(t.b0[i]) + bv(b1[i])) else bv(t.b0[i]);
        q.* = v * v;
    }
    const ss = sum.of(terms[0..W]);
    const wf: f64 = @floatFromInt(W);
    const iv = inv(ss, wf, t.p.eps);
    const iv_r = if (ranged) invRange(positive(ss, if (exact) 0 else c.gamma(@floatFromInt(t.depth), u)), wf, t.p.eps) else pt(0);
    for (0..W) |i| {
        const v = if (t.b1) |b1| bf(bv(t.b0[i]) + bv(b1[i])) else bv(t.b0[i]);
        const gn = bv(t.gain[i]);
        const n = normAt(v, iv, gn);
        const nr = if (ranged) normRange(pt(v), iv_r, gn) else pt(0);
        for (0..streams) |q| {
            const k = q * W + i;
            const r = bv(t.res[k]);
            const ps: f64 = t.post[q];
            x[k] = bf(r + ps * n);
            if (!ranged) continue;
            const m = step(step(nr.times(ps), 1).plus(pt(r)), 1).bf16();
            lo[k] = m.lo;
            hi[k] = m.hi;
        }
    }
}

/// Gate 4's rows: integers, each stream's second half its first one column on (odd rows: 32 of stream 3 one less).
pub fn stressHidden(row: usize, s: usize, i: usize, width: usize) f32 {
    const half = width / 2;
    const at = if (i < half) i else (i - half + 1) % half;
    const v: usize = if (s == streams - 1) 40 + (at * 7 + row) % 11 else 2 + (at + s + row) % 4;
    const less: usize = @intFromBool(i >= half and row % 2 == 1 and s == streams - 1 and i - half < 32);
    return @floatFromInt(v - less);
}

/// Projections negated over each stream's second half the same way: even rows' logits are 0, odd rows' a few hundred.
pub fn stressProject(j: usize, s: usize, i: usize, width: usize) f32 {
    const half = width / 2;
    const at = if (i < half) i else (i - half + 1) % half;
    const w: f32 = @floatFromInt(1 + (at + 3 * j + s) % 7);
    return if (i < half) w else -w;
}
