//! glm_kda.cu's chain in float64 at any head count, end to end and the recurrence alone (glm-kda-ling).
const std = @import("std");

pub const dk = 128;

/// x rounded to the nearest bf16, ties to even (finite normal values).
pub fn bf16(x: f64) f64 {
    const u: u64 = @bitCast(x);
    const lsb = (u >> 45) & 1;
    return @bitCast((u + ((@as(u64, 1) << 44) - 1) + lsb) & ~((@as(u64, 1) << 45) - 1));
}

pub fn bf(u: u16) f64 {
    return @as(f32, @bitCast(@as(u32, u) << 16));
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

const p7 = std.math.pow(f64, 2, -7);
const p8 = std.math.pow(f64, 2, -8);
/// The recurrence's bound from the saves: fp32 sums of up to 144 terms a row, contracting rows, 2^4 to spare.
const p12 = std.math.pow(f64, 2, -12);

/// The call's shape: p rows hold q | k | v (3 H 128 channels) and beta's logits at b_off; a and g rows are H 128 wide.
pub const Layout = struct { heads: usize, p_stride: usize, b_off: usize, lower: f64, eps: f64 };

/// Raw rows and weights as the chain takes them: cs [3][3 H 128] oldest first, taps [3 H 128][4], state [H][v][k].
pub const Inputs = struct { p: []const u16, a: []const u16, g: []const u16, cs: []const u16, cw: []const f32, state: []const f32, a_log: []const f32, dt_bias: []const f32, norm: []const u16 };

/// prep's rows [rows][H][128]: q (scaled), k, v bf16, decay factors, and beta [rows][H].
pub const Saved = struct { q: []const f32, k: []const f32, v: []const u16, g: []const f32, b: []const f32 };

/// Outputs [rows][H 128] and the final state [H][128][128], each with a per-element bound.
pub const Want = struct { out: []f64, out_tol: []f64, state: []f64, state_tol: []f64 };

fn want(a: std.mem.Allocator, heads: usize, rows: usize) !Want {
    const n = heads * dk;
    return .{ .out = try a.alloc(f64, rows * n), .out_tol = try a.alloc(f64, rows * n), .state = try a.alloc(f64, n * dk), .state_tol = try a.alloc(f64, n * dk) };
}

/// A delta step: decay, u = beta (v - S k), S += k u, o = S q; mag bounds |S|; o_mag kda_ref's or (fp32) sum mag |q|.
fn step(s: []f64, mag: []f64, q: []const f64, k: []const f64, v: []const f64, g: []const f64, beta: f64, o: []f64, o_mag: []f64, fp32: bool) void {
    var kq: f64 = 0;
    for (q, k) |qq, kk| kq += @abs(qq * kk);
    for (0..dk) |j| {
        const row = s[j * dk ..][0..dk];
        const mrow = mag[j * dk ..][0..dk];
        var kv: f64 = 0;
        var km: f64 = 0;
        for (row, mrow, k, g) |*x, *m, kk, gg| {
            x.* *= gg;
            m.* *= gg;
            kv += x.* * kk;
            km += m.* * @abs(kk);
        }
        const u = beta * (v[j] - kv);
        const um = beta * (@abs(v[j]) + km);
        var acc: f64 = 0;
        var am: f64 = 0;
        var aq: f64 = 0;
        for (row, mrow, k, q) |*x, *m, kk, qq| {
            x.* += kk * u;
            m.* += @abs(kk) * um;
            acc += x.* * qq;
            am += m.* * @abs(qq);
            aq += @abs(x.* * qq);
        }
        o[j] = acc;
        o_mag[j] = if (fp32) am else aq + beta * @abs(v[j]) * kq;
    }
}

/// kda_ref.gatedNorm: o / rms * w * sigmoid(gate); tol: o's bound in, y's bound out (first order) + y's bf16 rounding.
fn gatedNorm(o: []const f64, gate: []const f64, w: []const f64, eps: f64, y: []f64, tol: []f64) void {
    var ss: f64 = 0;
    var cross: f64 = 0;
    for (o, tol) |x, t| {
        ss += x * x;
        cross += @abs(x) * t;
    }
    const nn: f64 = @floatFromInt(o.len);
    const rstd = 1 / @sqrt(ss / nn + eps);
    const moved = cross / nn / (ss / nn + eps);
    for (y, tol, o, gate, w) |*out, *t, x, z, ww| {
        out.* = x * rstd * ww * sigmoid(z);
        t.* = @abs(ww * sigmoid(z)) * rstd * (t.* + @abs(x) * moved) + p8 * @abs(out.*);
    }
}

fn l2norm(x: []f64, scale: f64) void {
    var ss: f64 = 0;
    for (x) |e| ss += e * e;
    const f = scale / @sqrt(ss + 1e-6);
    for (x) |*e| e.* *= f;
}

/// End to end: conv+SiLU, q/k L2 norms, safe gate, bf16 beta, delta rule, bf16(S q), gated norm; kda_ref's 2^-7 bounds.
pub fn chain(a: std.mem.Allocator, l: Layout, in: Inputs, rows: usize) !Want {
    const n = l.heads * dk;
    const cc = 3 * n;
    var w = try want(a, l.heads, rows);
    const mag = try a.alloc(f64, dk * dk);
    var qkv: [3][dk]f64 = undefined;
    var g: [dk]f64 = undefined;
    var o: [dk]f64 = undefined;
    var om: [dk]f64 = undefined;
    var gate: [dk]f64 = undefined;
    var nw: [dk]f64 = undefined;
    for (&nw, in.norm[0..dk]) |*d, x| d.* = bf(x);
    for (0..l.heads) |h| {
        const s = w.state[h * dk * dk ..][0 .. dk * dk];
        for (s, in.state[h * dk * dk ..][0 .. dk * dk]) |*d, x| d.* = x;
        for (mag, s) |*m, x| m.* = @abs(x);
        const rate = @exp(@as(f64, in.a_log[h]));
        for (0..rows) |row| {
            for (0..3) |part| for (0..dk) |i| {
                const c = part * n + h * dk + i;
                var acc: f64 = 0;
                for (0..4) |tap| {
                    const at = row + tap;
                    const x = if (at < 3) bf(in.cs[at * cc + c]) else bf(in.p[(at - 3) * l.p_stride + c]);
                    acc += @as(f64, in.cw[c * 4 + tap]) * x;
                }
                qkv[part][i] = bf16(acc * sigmoid(acc));
            };
            l2norm(&qkv[0], 1 / @sqrt(@as(f64, dk)));
            l2norm(&qkv[1], 1);
            for (&g, 0..) |*e, i| e.* = @exp(l.lower * sigmoid(rate * (bf(in.a[row * n + h * dk + i]) + in.dt_bias[h * dk + i])));
            const beta = bf16(sigmoid(bf(in.p[row * l.p_stride + l.b_off + h])));
            step(s, mag, &qkv[0], &qkv[1], &qkv[2], &g, beta, &o, &om, false);
            const tol = w.out_tol[row * n + h * dk ..][0..dk];
            for (&o, tol, om) |*e, *t, m| {
                t.* = p7 * m;
                e.* = bf16(e.*);
            }
            for (&gate, 0..) |*z, t| z.* = bf(in.g[row * n + h * dk + t]);
            gatedNorm(&o, &gate, &nw, l.eps, w.out[row * n + h * dk ..][0..dk], tol);
        }
        for (w.state_tol[h * dk * dk ..][0 .. dk * dk], mag) |*t, m| t.* = p7 * m + 1e-30;
    }
    return w;
}

/// The recurrence alone from prep's saves: S q (tol 2^-8 |y| + 2^-12 sum |mag q|), state (tol 2^-12 of magnitudes).
pub fn recurrence(a: std.mem.Allocator, heads: usize, state0: []const f32, sv: Saved, rows: usize) !Want {
    const n = heads * dk;
    var w = try want(a, heads, rows);
    const mag = try a.alloc(f64, dk * dk);
    var q: [dk]f64 = undefined;
    var k: [dk]f64 = undefined;
    var v: [dk]f64 = undefined;
    var g: [dk]f64 = undefined;
    var om: [dk]f64 = undefined;
    for (0..heads) |h| {
        const s = w.state[h * dk * dk ..][0 .. dk * dk];
        for (s, state0[h * dk * dk ..][0 .. dk * dk]) |*d, x| d.* = x;
        for (mag, s) |*m, x| m.* = @abs(x);
        for (0..rows) |row| {
            const at = row * n + h * dk;
            for (&q, &k, &v, &g, sv.q[at..][0..dk], sv.k[at..][0..dk], sv.v[at..][0..dk], sv.g[at..][0..dk]) |*qq, *kk, *vv, *gg, a0, a1, a2, a3| {
                qq.* = a0;
                kk.* = a1;
                vv.* = bf(a2);
                gg.* = a3;
            }
            const o = w.out[at..][0..dk];
            step(s, mag, &q, &k, &v, &g, sv.b[row * heads + h], o, &om, true);
            for (w.out_tol[at..][0..dk], o, om) |*t, e, m| t.* = p8 * @abs(e) + p12 * m + 1e-30;
        }
        for (w.state_tol[h * dk * dk ..][0 .. dk * dk], mag) |*t, m| t.* = p12 * m + 1e-30;
    }
    return w;
}

/// The worst |got - want| / tol.
pub fn fraction(got: []const f64, w: []const f64, tol: []const f64) f64 {
    var most: f64 = 0;
    for (got, w, tol) |x, e, t| {
        const q = @abs(x - e) / t;
        if (!(q <= most)) most = if (std.math.isNan(q)) std.math.inf(f64) else q;
    }
    return most;
}

/// Relative L2 distance of got from want.
pub fn relL2(got: []const f64, w: []const f64) f64 {
    var dd: f64 = 0;
    var ww: f64 = 0;
    for (got, w) |x, e| {
        dd += (x - e) * (x - e);
        ww += e * e;
    }
    return @sqrt(dd / ww);
}

test "bf16 rounding ties to even" {
    try std.testing.expectEqual(@as(f64, 1.0), bf16(1.0 + 1.0 / 256.0));
    try std.testing.expectEqual(@as(f64, 1.0 + 1.0 / 64.0), bf16(1.0 + 3.0 / 256.0));
}
