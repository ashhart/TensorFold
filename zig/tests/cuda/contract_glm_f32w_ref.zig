//! FP4-Local's float32 norm weight kernels in float64 (rmsnorm_4096_f32w, hc_finish_f32w): the inputs and references.
const std = @import("std");
const core = @import("core");

const contract = core.contract;
const c = contract.spec;
const Order = contract.gates.Order;
const Fill = contract.gates.Fill;
const sumIn = contract.rows.sumIn;

pub const d = 4096;
pub const s = 4;
pub const nb = 16; // glue.hc_blocks
pub const iters = 20;
pub const scale = [3]f32{ 0.7, 0.9, 1.1 };

/// What a call reads: x bf16 [rows][d] (norm) or [rows][s][d] (hc), the float32 weight, hc's partials and logits' base.
pub const In = struct {
    hc: bool,
    width: usize = d,
    rows: usize,
    x: []u16,
    gamma: []f32,
    part: []f32, // [rows][nb][32]: 24 dots, then the K block's sum of squares
    base: [24]f32,
    eps: f32,
    hc_eps: f32,
};

/// A row's sums: exact in float64, or as an accumulator in an order would add them (gate 4's simulation).
pub const Sum = struct {
    acc: ?c.Accum = null,
    order: Order = .sequential,

    pub fn of(m: Sum, t: []const f64) f64 {
        if (m.acc) |a| return sumIn(a, m.order, t);
        var v: f64 = 0;
        for (t) |x| v += x;
        return v;
    }
};

pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// fmaxf: a NaN operand gives the other one.
fn fmax(a: f64, b: f64) f64 {
    return if (std.math.isNan(a)) b else if (std.math.isNan(b)) a else @max(a, b);
}

fn sigmoid(v: f64) f64 {
    return 1 / (@exp(0 - v) + 1);
}

/// Inputs for a fill (stress: integer terms); `exact`: rows whose norm is x itself, so the 64-sums' reference is exact.
pub fn fill(a: std.mem.Allocator, hc: bool, rows: usize, mode: Fill, seed: u64, exact: bool) !In {
    return fillWidth(a, hc, d, rows, mode, seed, exact);
}

/// BF16 template replacements use the same fixture at their fixed 512, 1536 and 4096 widths.
pub fn fillWidth(a: std.mem.Allocator, hc: bool, width: usize, rows: usize, mode: Fill, seed: u64, exact: bool) !In {
    if (width != 512 and width != 1536 and width != d or hc and width != d) return error.UnsupportedWidth;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const x = try a.alloc(u16, rows * width * @as(usize, if (hc) s else 1));
    const gamma = try a.alloc(f32, width);
    const part = try a.alloc(f32, if (hc) rows * nb * 32 else 0);
    var in: In = .{ .hc = hc, .width = width, .rows = rows, .x = x, .gamma = gamma, .part = part, .base = undefined, .eps = 1e-6, .hc_eps = 1e-6 };
    const stress = mode == .stress;
    for (gamma) |*g| g.* = if (exact) 1 else 0.5 + rng.float(f32);
    for (&in.base) |*b| b.* = (rng.float(f32) * 2 - 1) * 0.5;
    for (part, 0..) |*p, i| p.* = if (i % 32 == 24) // stress logits under 30, where fp32's exp keeps every entry
        (if (stress) @floatFromInt(1024 + rng.uintLessThan(u32, 1024)) else 200 + rng.float(f32) * 200)
    else if (stress) @as(f32, @floatFromInt(rng.uintLessThan(u32, 5))) - 2 else rng.float(f32) * 2 - 1;
    if (exact or (hc and stress)) { // signed permutations of one row; eps 1 - mean square; hc's c = stream 0 / 2
        const row = try a.alloc(f64, width);
        defer a.free(row);
        for (row, 0..) |*v, i| v.* = if (stress)
            @as(f64, @floatFromInt(rng.uintLessThan(u32, 64))) - 31
        else
            bf((rng.float(f64) * 2 - 1) * (if (i % 64 == 7) @as(f64, 20) else 1));
        var ss: f64 = 0;
        for (row) |v| ss += v * v;
        if (exact) in.eps = @floatCast(1 - ss / @as(f64, @floatFromInt(width)));
        if (hc) { // pre 0.5: zero dots, base and hc_eps; squares that keep the logits' rms finite under that eps
            in.hc_eps = 0;
            for (in.base[0..4]) |*b| b.* = 0;
            const sq: f32 = @ceil((@abs(in.eps) + 1) * @as(f32, @floatFromInt(s * width)) / nb);
            for (0..rows) |r| for (0..nb) |b| {
                for (0..4) |k| part[(r * nb + b) * 32 + k] = 0;
                if (exact) part[(r * nb + b) * 32 + 24] = sq;
            };
        }
        const order = try a.alloc(usize, width);
        defer a.free(order);
        for (0..rows) |r| {
            for (order, 0..) |*o, i| o.* = i;
            rng.shuffle(usize, order);
            for (0..width) |i| {
                const v = row[order[i]] * @as(f64, if (rng.boolean()) -1 else 1);
                if (hc) {
                    x[(r * s) * width + i] = c.toBf16(@floatCast(2 * v));
                    for (1..s) |k| x[(r * s + k) * width + i] = 0;
                } else x[r * width + i] = c.toBf16(@floatCast(v));
            }
        }
    } else for (x, 0..) |*v, i| {
        v.* = c.toBf16(if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32) else (rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1));
    }
    if (mode == .nonfinite) plant(&in);
    return in;
}

/// Three BF16 roundings exceed the two-rounding bound when the float32 gamma is narrowed before the multiply.
pub fn narrowControl(in: *In) void {
    for (in.x, 0..) |*v, i| v.* = if (i % 2 == 0) 0x3e83 else 0x3f21;
    @memset(in.gamma, @as(f32, @bitCast(@as(u32, 0x3f077fff))));
}

/// NaN and both infinities in rows 0 to 3: x (the norms' outputs), and hc's dots for post (4, 5) and comb (8, 9).
fn plant(in: *In) void {
    const width = in.width;
    const nan = c.toBf16(std.math.nan(f32));
    const inf = c.toBf16(std.math.inf(f32));
    const ninf = c.toBf16(-std.math.inf(f32));
    const w: usize = if (in.hc) s * width else width;
    const at = [_][2]usize{ .{ 0, width / 3 }, .{ 1, width / 2 }, .{ 2, width - 1 }, .{ 3, 0 } };
    const put = [_]u16{ nan, inf, ninf, inf };
    for (at, put) |p, v| if (p[0] < in.rows) {
        in.x[p[0] * w + p[1]] = v;
    };
    if (in.rows > 3) in.x[3 * w + 1] = ninf;
    if (!in.hc) return;
    const dots = [_][3]usize{ .{ 0, 3, 4 }, .{ 1, 5, 5 }, .{ 2, 0, 8 }, .{ 3, 15, 9 } };
    const vals = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
    for (dots, vals) |p, v| if (p[0] < in.rows) {
        in.part[(p[0] * nb + p[1]) * 32 + p[2]] = v;
    };
}

fn xv(in: In, i: usize) f64 {
    return c.fromBf16(in.x[i]);
}

/// A row's hc logits: the 24 mixing sums times rinv of the squares' sum, and each one's |terms| times that rinv.
fn mixRow(in: In, r: usize, m: Sum, mixn: *[24]f64, mass: *[24]f64) void {
    var t: [nb]f64 = undefined;
    for (0..nb) |b| t[b] = in.part[(r * nb + b) * 32 + 24];
    const rinv = 1 / @sqrt(in.eps + m.of(&t) / (s * d));
    for (0..24) |k| {
        var abs: f64 = 0;
        for (0..nb) |b| {
            t[b] = in.part[(r * nb + b) * 32 + k];
            abs += @abs(t[b]);
        }
        mixn[k] = m.of(&t) * rinv;
        mass[k] = abs * rinv;
    }
}

/// rmsnorm_4096_f32w's outputs [rows][d] or hc_finish_f32w's normed rows, unrounded; `mag` each one's sums' terms.
pub fn normed(a: std.mem.Allocator, in: In, m: Sum, want: []f64, mag: []f64) !void {
    const width = in.width;
    const cv = try a.alloc(f64, width);
    defer a.free(cv);
    const cm = try a.alloc(f64, width); // the |terms| of each c: hc's collapse and its logits' sums
    defer a.free(cm);
    const sq = try a.alloc(f64, width);
    defer a.free(sq);
    for (0..in.rows) |r| {
        if (in.hc) {
            var mixn: [24]f64 = undefined;
            var mass: [24]f64 = undefined;
            mixRow(in, r, m, &mixn, &mass);
            var pre: [s]f64 = undefined;
            var dpre: [s]f64 = undefined; // pre's sensitivity to its logit's sum: sigmoid' <= 1/4, rinv's own error 1/2
            for (0..s) |k| {
                pre[k] = in.hc_eps + sigmoid(mixn[k] * scale[0] + in.base[k]);
                dpre[k] = 0.25 * scale[0] * 1.5 * mass[k];
            }
            for (cv, cm, 0..) |*o, *om, i| {
                var t: [s]f64 = undefined;
                om.* = 0;
                for (0..s) |k| {
                    const xk = xv(in, (r * s + k) * width + i);
                    t[k] = pre[k] * xk;
                    om.* += @abs(t[k]) + @abs(xk) * dpre[k];
                }
                o.* = m.of(&t);
            }
        } else for (cv, cm, 0..) |*o, *om, i| {
            o.* = xv(in, r * width + i);
            om.* = 0;
        }
        for (sq, cv) |*q, v| q.* = v * v;
        const rinv = 1 / @sqrt(in.eps + m.of(sq) / @as(f64, @floatFromInt(width)));
        for (want[r * width ..][0..width], mag[r * width ..][0..width], cv, cm, in.gamma) |*w, *g, v, om, gm| {
            w.* = gm * rinv * v;
            g.* = gm * rinv * om + @abs(w.*) / 2; // y ~ ss^-1/2
        }
    }
}

/// Each 64 outputs' sum, the outputs rounded as the kernel does, bf16(gamma bf16(rinv c)), from `fill`'s exact rows.
pub fn sums(a: std.mem.Allocator, in: In, m: Sum, want: []f64, mag: []f64) !void {
    const width = in.width;
    const row = try a.alloc(f64, width);
    defer a.free(row);
    for (0..in.rows) |r| {
        var sq: f64 = 0;
        for (row, 0..) |*v, i| {
            v.* = if (in.hc) bf(0.5 * xv(in, (r * s) * width + i)) else xv(in, r * width + i);
            sq += v.* * v.*;
        }
        const rinv = 1 / @sqrt(in.eps + sq / @as(f64, @floatFromInt(width)));
        for (row, in.gamma) |*o, gm| o.* = bf(gm * bf(rinv * o.*));
        for (0..width / 64) |g| {
            const t = row[g * 64 ..][0..64];
            var abs: f64 = 0;
            for (t) |v| abs += @abs(v);
            want[r * (width / 64) + g] = m.of(t);
            mag[r * (width / 64) + g] = abs;
        }
    }
}

/// hc_finish_f32w's post: 2 sigmoid of the logits 4 to 7; `mag` the logits' sums' terms through sigmoid' (1/4).
pub fn post(in: In, m: Sum, want: []f64, mag: []f64) void {
    for (0..in.rows) |r| {
        var mixn: [24]f64 = undefined;
        var mass: [24]f64 = undefined;
        mixRow(in, r, m, &mixn, &mass);
        for (0..4) |k| {
            want[r * 4 + k] = 2 * sigmoid(mixn[4 + k] * scale[1] + in.base[4 + k]);
            mag[r * 4 + k] = 0.5 * scale[1] * 1.5 * mass[4 + k] + @abs(want[r * 4 + k]);
        }
    }
}

/// hc_finish_f32w's comb: rows' softmax of the 4x4 logits plus hc_eps, then Sinkhorn's column and row normalizations.
pub fn comb(in: In, m: Sum, want: []f64, mag: []f64) void {
    for (0..in.rows) |r| {
        var mixn: [24]f64 = undefined;
        var mass: [24]f64 = undefined;
        mixRow(in, r, m, &mixn, &mass);
        var cm: [16]f64 = undefined;
        var most: f64 = 0;
        for (0..16) |i| most = @max(most, mass[8 + i]);
        for (0..4) |i| {
            var cl: [4]f64 = undefined;
            var top: f64 = std.math.nan(f64);
            for (0..4) |j| {
                cl[j] = mixn[8 + 4 * i + j] * scale[2] + in.base[8 + 4 * i + j];
                top = fmax(top, cl[j]);
            }
            var e: [4]f64 = undefined;
            for (&e, cl) |*o, v| o.* = @exp(v - top);
            const sum = (e[0] + e[1]) + (e[2] + e[3]);
            for (0..4) |j| cm[4 * i + j] = in.hc_eps + e[j] / sum;
        }
        normalize(&cm, in.hc_eps, .col);
        for (1..iters) |_| {
            normalize(&cm, in.hc_eps, .row);
            normalize(&cm, in.hc_eps, .col);
        }
        for (0..16) |i| {
            want[r * 16 + i] = cm[i];
            mag[r * 16 + i] = @abs(cm[i]) * 2 * scale[2] * 1.5 * most; // a logit's error moves each entry twice it
        }
    }
}

fn normalize(cm: *[16]f64, eps: f64, by: enum { row, col }) void {
    var tot: [4]f64 = @splat(0);
    for (0..16) |idx| tot[if (by == .row) idx / 4 else idx % 4] += cm[idx];
    for (0..16) |idx| cm[idx] /= eps + tot[if (by == .row) idx / 4 else idx % 4];
}
