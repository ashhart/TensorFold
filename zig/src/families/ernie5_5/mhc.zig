//! ERNIE 5.5's mHC (vLLM-ERNIE's mhc_pre, mix RMS 1 / (sqrt(mean) + eps); vLLM's mhc_post and hc_head).
const std = @import("std");

/// hc_mult streams of the hidden size, the (2 + streams) streams mixes, and a row's 4 streams side by side.
pub const streams = 4;
pub const width = 4096;
pub const mixes = (2 + streams) * streams;
pub const wide = streams * width;

/// rms_norm_eps (the mix RMS and the norm), hc_eps (pre's floor and Sinkhorn's) and hc_sinkhorn_iters.
pub const Params = struct { rms_eps: f32, hc_eps: f32, norm_eps: f32, iters: u32 };

/// rows rows of 4 streams xs apart, outputs os apart: 16-byte rows, as the kernel loads them.
pub const Shape = struct { rows: usize, xs: usize = wide, os: usize = width };

/// Rows that fit the grid, rows at least as wide as a row and 8-element aligned, at least one Sinkhorn round.
pub fn check(sh: Shape, p: Params) error{MhcShape}!void {
    if (sh.xs < wide or sh.os < width or sh.xs % 8 != 0 or sh.os % 8 != 0) return error.MhcShape;
    if (sh.rows > std.math.maxInt(i32) or p.iters == 0 or p.iters > std.math.maxInt(i32)) return error.MhcShape;
}

/// A row's outputs: post, comb, and the normed collapse before its bf16 rounding.
pub const Row = struct { post: [streams]f64, comb: [streams * streams]f64, out: [width]f64 };

pub fn sigmoid(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}

/// ERNIE's mix RMS of the squares' sum over a row's 4 streams.
pub fn mixRms(ss: f64, eps: f64) f64 {
    return 1 / (@sqrt(ss / wide) + eps);
}

/// The 24 mixes of row x (4 streams of bf16 values) with fn [24][wide], times mixRms; t: scratch, sums by `sum.of`.
pub fn mixRow(x: *const [wide]f64, fnw: []const f32, p: Params, sum: anytype, t: *[wide]f64, m: *[mixes]f64) void {
    for (t, x) |*o, v| o.* = v * v;
    const r = mixRms(sum.of(t), p.rms_eps);
    for (m, 0..) |*o, i| {
        for (t, x, fnw[i * wide ..][0..wide]) |*q, v, f| q.* = v * f;
        o.* = sum.of(t) * r;
    }
}

/// A row: post 2 sigmoid(m s1 + b), comb Sinkhorn of the rows' softmax, out the collapse's RMSNorm times w (unrounded).
pub fn pre(x: *const [wide]f64, fnw: []const f32, scale: *const [3]f32, base: *const [mixes]f32, w: *const [width]f64, p: Params, sum: anytype, t: *[wide]f64, r: *Row) void {
    var m: [mixes]f64 = undefined;
    mixRow(x, fnw, p, sum, t, &m);
    finish(&m, x, scale, base, w, p, sum, t, r);
}

/// The broadcast's rows: e rows es apart, 4 streams rs apart out, outputs os apart (16-byte rows).
pub const BcastShape = struct { rows: usize, es: usize = width, rs: usize = wide, os: usize = width };

pub fn checkBcast(sh: BcastShape, p: Params) error{MhcShape}!void {
    if (sh.es < width or sh.rs < wide or sh.os < width or sh.es % 8 != 0 or sh.rs % 8 != 0 or sh.os % 8 != 0) return error.MhcShape;
    if (sh.rows > std.math.maxInt(i32) or p.iters == 0 or p.iters > std.math.maxInt(i32)) return error.MhcShape;
}

/// The first layer's pre (vLLM-ERNIE's broadcast): pre of e's 4 copies x, mixes with fnb [24][width] and e's own RMS.
pub fn preBroadcast(e: *const [width]f64, fnb: []const f32, scale: *const [3]f32, base: *const [mixes]f32, w: *const [width]f64, p: Params, sum: anytype, x: *[wide]f64, t: *[wide]f64, r: *Row) void {
    for (0..streams) |s| @memcpy(x[s * width ..][0..width], e);
    const q = t[0..width];
    for (q, e) |*o, v| o.* = v * v;
    const rr = 1 / (@sqrt(sum.of(q) / width) + p.rms_eps);
    var m: [mixes]f64 = undefined;
    for (&m, 0..) |*o, i| {
        for (q, e, fnb[i * width ..][0..width]) |*a, v, f| a.* = v * f;
        o.* = sum.of(q) * rr;
    }
    finish(&m, x, scale, base, w, p, sum, t, r);
}

/// pre's outputs from its 24 mixes and the row's 4 streams x.
fn finish(m: *const [mixes]f64, x: *const [wide]f64, scale: *const [3]f32, base: *const [mixes]f32, w: *const [width]f64, p: Params, sum: anytype, t: *[wide]f64, r: *Row) void {
    var pw: [streams]f64 = undefined;
    for (0..streams) |k| {
        pw[k] = sigmoid(m[k] * scale[0] + base[k]) + p.hc_eps;
        r.post[k] = 2 * sigmoid(m[streams + k] * scale[1] + base[streams + k]);
    }
    sinkhorn(m[2 * streams ..][0 .. streams * streams], scale[2], base[2 * streams ..][0 .. streams * streams], p, &r.comb);
    const c = t[0..width];
    for (c, 0..) |*o, h| {
        var terms: [streams]f64 = undefined;
        for (&terms, pw, 0..) |*q, ps, s| q.* = ps * x[s * width + h];
        o.* = sum.of(&terms);
    }
    const sq = t[width .. 2 * width];
    for (sq, c) |*q, v| q.* = v * v;
    const rn = 1 / @sqrt(sum.of(sq) / width + p.norm_eps);
    for (&r.out, c, w) |*o, v, g| o.* = v * rn * g;
}

/// comb from its 16 mixes: each row's softmax plus eps, its columns' normalization, then iters - 1 rows' and columns'.
pub fn sinkhorn(m: *const [streams * streams]f64, s2: f32, base: *const [streams * streams]f32, p: Params, cm: *[streams * streams]f64) void {
    for (0..streams) |i| {
        var cl: [streams]f64 = undefined;
        var top = -std.math.inf(f64);
        for (&cl, 0..) |*v, j| {
            v.* = m[streams * i + j] * s2 + base[streams * i + j];
            top = @max(top, v.*);
        }
        var e: [streams]f64 = undefined;
        for (&e, cl) |*o, v| o.* = @exp(v - top);
        const tot = (e[0] + e[2]) + (e[1] + e[3]);
        for (e, 0..) |v, j| cm[streams * i + j] = v / tot + p.hc_eps;
    }
    normalize(cm, p.hc_eps, .col);
    for (1..p.iters) |_| {
        normalize(cm, p.hc_eps, .row);
        normalize(cm, p.hc_eps, .col);
    }
}

fn normalize(cm: *[streams * streams]f64, eps: f64, by: enum { row, col }) void {
    var tot: [streams]f64 = @splat(0);
    for (cm, 0..) |v, i| tot[if (by == .row) i / streams else i % streams] += v;
    for (cm, 0..) |*v, i| v.* /= tot[if (by == .row) i / streams else i % streams] + eps;
}

/// The post side's rows: x rows xs apart, 4 streams rs apart in and os apart out (16-byte rows, out apart from res).
pub const PostShape = struct { rows: usize, xs: usize = width, rs: usize = wide, os: usize = wide };

pub fn checkPost(sh: PostShape) error{MhcShape}!void {
    if (sh.xs < width or sh.rs < wide or sh.os < wide or sh.xs % 8 != 0 or sh.rs % 8 != 0 or sh.os % 8 != 0) return error.MhcShape;
    if (sh.rows > std.math.maxInt(i32)) return error.MhcShape;
}

/// vLLM's mhc_post: stream o = post[o] x + sum_i comb[i][o] res[i] (unrounded), its 5 terms summed by `sum.of`.
pub fn post(x: *const [width]f64, res: *const [wide]f64, pw: *const [streams]f32, comb: *const [streams * streams]f32, sum: anytype, out: *[wide]f64) void {
    for (0..streams) |o| for (0..width) |h| {
        var t: [streams + 1]f64 = undefined;
        t[0] = pw[o] * x[h];
        for (0..streams) |i| t[i + 1] = comb[i * streams + o] * res[i * width + h];
        out[o * width + h] = sum.of(&t);
    };
}

/// hc_head's rows: 4 streams xs apart in, one row os apart out (16-byte rows).
pub const HeadShape = struct { rows: usize, xs: usize = wide, os: usize = width };

pub fn checkHead(sh: HeadShape) error{MhcShape}!void {
    if (sh.xs < wide or sh.os < width or sh.xs % 8 != 0 or sh.os % 8 != 0 or sh.rows > std.math.maxInt(i32)) return error.MhcShape;
}

/// vLLM's hc_head: pre = sigmoid(m r scale + base) + hc_eps, r = 1 / sqrt(mean + eps), out = sum_s pre_s x_s.
pub fn head(x: *const [wide]f64, fnw: []const f32, scale: f32, base: *const [streams]f32, rms_eps: f32, hc_eps: f32, sum: anytype, t: *[wide]f64, out: *[width]f64) void {
    for (t, x) |*o, v| o.* = v * v;
    const r = 1 / @sqrt(sum.of(t) / wide + rms_eps);
    var pw: [streams]f64 = undefined;
    for (&pw, 0..) |*o, i| {
        for (t, x, fnw[i * wide ..][0..wide]) |*q, v, f| q.* = v * f;
        o.* = sigmoid(sum.of(t) * r * scale + base[i]) + hc_eps;
    }
    for (out, 0..) |*o, h| {
        var terms: [streams]f64 = undefined;
        for (&terms, pw, 0..) |*q, ps, s| q.* = ps * x[s * width + h];
        o.* = sum.of(&terms);
    }
}

/// Float64 sums in order.
pub const exact = struct {
    pub fn of(_: @This(), t: []const f64) f64 {
        var s: f64 = 0;
        for (t) |v| s += v;
        return s;
    }
}{};
