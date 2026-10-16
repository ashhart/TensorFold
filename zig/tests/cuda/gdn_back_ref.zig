//! Float64 reference of a Gated DeltaNet layer, forward (bf16 roundings when `round`) and input-gradient backward.
const std = @import("std");

pub const taps = 4;

pub const Dims = struct {
    rows: usize,
    nk: usize,
    nv: usize,
    dk: usize = 128,
    dv: usize = 128,

    pub fn conv(d: Dims) usize {
        return 2 * d.nk * d.dk + d.nv * d.dv;
    }

    pub fn proj(d: Dims) usize {
        return d.conv() + d.nv * d.dv + 2 * d.nv;
    }

    /// The first column of z, b and a in a projection row.
    pub fn zAt(d: Dims) usize {
        return d.conv();
    }

    pub fn bAt(d: Dims) usize {
        return d.conv() + d.nv * d.dv;
    }

    pub fn aAt(d: Dims) usize {
        return d.bAt() + d.nv;
    }
};

/// The layer's inputs: P [rows, proj], hist [3, conv] or null, conv_w, a_log, dt_bias, norm_w, init or null.
pub const Inputs = struct {
    p: []const f64,
    hist: ?[]const f64,
    conv_w: []const f64,
    a_log: []const f64,
    dt_bias: []const f64,
    norm_w: []const f64,
    init: ?[]const f64,
    eps: f64,
};

pub const Fwd = struct {
    act: []f64, // [rows, conv]
    q: []f64, // [rows, nk, dk]
    k: []f64,
    g: []f64, // [rows, nv]
    beta: []f64,
    y: []f64, // [rows, nv, dv]
    out: []f64, // [rows, nv, dv]
};

pub const Grads = struct {
    dy: []f64, // [rows, nv, dv]: the chain's output gradient
    dq: []f64, // [rows, nk, dk]: the normalized q's gradient, summed over the key head's value heads
    dk: []f64,
    dact: []f64, // [rows, conv]: q, k after their norms; v
    dg: []f64, // [rows, nv]
    dbeta: []f64,
    dproj: []f64, // [rows, proj]: qkv, z, b, a
};

pub fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7fc0;
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub fn bf16Value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn bfRound(x: f64) f64 {
    return bf16Value(bf16Bits(@floatCast(x)));
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

fn softplus(x: f64) f64 {
    return if (x > 20) x else std.math.log1p(@exp(x));
}

fn convPre(d: Dims, in: Inputs, r: usize, c: usize) f64 {
    const C = d.conv();
    var acc: f64 = 0;
    for (0..taps) |tap| {
        const at = @as(isize, @intCast(r)) - (taps - 1) + @as(isize, @intCast(tap));
        const x: f64 = if (at >= 0) in.p[@as(usize, @intCast(at)) * d.proj() + c] else if (in.hist) |h| h[@as(usize, @intCast(at + taps - 1)) * C + c] else 0;
        acc += in.conv_w[c * taps + tap] * x;
    }
    return acc;
}

fn gate(d: Dims, in: Inputs, r: usize, h: usize) f64 {
    const a = in.p[r * d.proj() + d.aAt() + h];
    return @exp(-@exp(in.a_log[h]) * softplus(a + in.dt_bias[h]));
}

/// q (or k) of row r, key head hk from act: x r (times dk^-0.5 for q), r = 1 / sqrt(|x|^2 + 1e-6).
fn l2(d: Dims, act: []const f64, r: usize, col: usize, scale: f64, out: []f64) void {
    var ss: f64 = 0;
    for (0..d.dk) |i| ss += act[r * d.conv() + col + i] * act[r * d.conv() + col + i];
    const inv = 1 / @sqrt(ss + 1e-6);
    for (0..d.dk) |i| out[i] = act[r * d.conv() + col + i] * inv * scale;
}

fn qk(d: Dims, act: []const f64, q: []f64, k: []f64) void {
    const sc = 1 / @sqrt(@as(f64, @floatFromInt(d.dk)));
    for (0..d.rows) |r| for (0..d.nk) |hk| {
        l2(d, act, r, hk * d.dk, sc, q[(r * d.nk + hk) * d.dk ..][0..d.dk]);
        l2(d, act, r, d.nk * d.dk + hk * d.dk, 1, k[(r * d.nk + hk) * d.dk ..][0..d.dk]);
    };
}

/// One head's bf16(bf16(y rms) w) in the GPU's own fp32 steps, so the rounded forward point is the GPU's bit for bit.
fn normedRounded(y: []const f64, w: []const f64, eps: f64, out: []f64) void {
    var lanes: [32]f32 = undefined;
    for (&lanes, 0..) |*l, lane| {
        var ss: f32 = 0;
        for (0..4) |i| {
            const v: f32 = @floatCast(y[lane * 4 + i]);
            ss = ss + v * v;
        }
        l.* = ss;
    }
    var o: usize = 16;
    while (o > 0) : (o >>= 1) {
        var next: [32]f32 = undefined;
        for (&next, 0..) |*x, lane| x.* = lanes[lane] + lanes[lane ^ o];
        lanes = next;
    }
    const rinv: f32 = 1.0 / @sqrt(lanes[0] / @as(f32, @floatFromInt(y.len)) + @as(f32, @floatCast(eps)));
    for (out, y, w) |*x, v, wv| {
        const n = bf16Value(bf16Bits(@as(f32, @floatCast(v)) * rinv));
        x.* = bf16Value(bf16Bits(n * @as(f32, @floatCast(wv))));
    }
}

/// yn of one head: the rounded forward's (normedRounded), else y rms(y) w in float64.
fn normed(y: []const f64, w: []const f64, eps: f64, round: bool, out: []f64) void {
    if (round) return normedRounded(y, w, eps, out);
    var ss: f64 = 0;
    for (y) |v| ss += v * v;
    const rinv = 1 / @sqrt(ss / @as(f64, @floatFromInt(y.len)) + eps);
    for (out, y, w) |*x, v, wv| x.* = v * rinv * wv;
}

pub fn forward(a: std.mem.Allocator, d: Dims, in: Inputs, round: bool) !Fwd {
    const C = d.conv();
    const P = d.proj();
    const act = try a.alloc(f64, d.rows * C);
    const beta = try a.alloc(f64, d.rows * d.nv);
    for (0..d.rows) |r| for (0..C) |c| {
        const pre = convPre(d, in, r, c);
        const v = pre * sigmoid(pre);
        act[r * C + c] = if (round) bfRound(v) else v;
    };
    for (0..d.rows) |r| for (0..d.nv) |h| {
        const b = sigmoid(in.p[r * P + d.bAt() + h]);
        beta[r * d.nv + h] = if (round) bfRound(b) else b;
    };
    return forwardFrom(a, d, in, act, beta, round);
}

/// The forward after the conv outputs act and the betas (the reference's own, or the GPU's).
pub fn forwardFrom(a: std.mem.Allocator, d: Dims, in: Inputs, act: []f64, betas: []f64, round: bool) !Fwd {
    const C = d.conv();
    const P = d.proj();
    const f: Fwd = .{
        .act = act,
        .q = try a.alloc(f64, d.rows * d.nk * d.dk),
        .k = try a.alloc(f64, d.rows * d.nk * d.dk),
        .g = try a.alloc(f64, d.rows * d.nv),
        .beta = betas,
        .y = try a.alloc(f64, d.rows * d.nv * d.dv),
        .out = try a.alloc(f64, d.rows * d.nv * d.dv),
    };
    qk(d, f.act, f.q, f.k);
    for (0..d.rows) |r| for (0..d.nv) |h| {
        f.g[r * d.nv + h] = gate(d, in, r, h);
    };
    const S = try a.alloc(f64, d.dv * d.dk);
    defer a.free(S);
    for (0..d.nv) |h| {
        const hk = h / (d.nv / d.nk);
        if (in.init) |s0| @memcpy(S, s0[h * d.dv * d.dk ..][0 .. d.dv * d.dk]) else @memset(S, 0);
        for (0..d.rows) |r| {
            const g = f.g[r * d.nv + h];
            const beta = f.beta[r * d.nv + h];
            const kr = f.k[(r * d.nk + hk) * d.dk ..][0..d.dk];
            const qr = f.q[(r * d.nk + hk) * d.dk ..][0..d.dk];
            for (0..d.dv) |j| {
                const row = S[j * d.dk ..][0..d.dk];
                var kv: f64 = 0;
                for (row, kr) |*s, kk| {
                    s.* *= g;
                    kv += s.* * kk;
                }
                const delta = beta * (f.act[r * C + 2 * d.nk * d.dk + h * d.dv + j] - kv);
                var o: f64 = 0;
                for (row, kr, qr) |*s, kk, qq| {
                    s.* += kk * delta;
                    o += s.* * qq;
                }
                f.y[(r * d.nv + h) * d.dv + j] = if (round) bfRound(o) else o;
            }
        }
    }
    const yn = try a.alloc(f64, d.dv);
    defer a.free(yn);
    for (0..d.rows) |r| for (0..d.nv) |h| {
        normed(f.y[(r * d.nv + h) * d.dv ..][0..d.dv], in.norm_w, in.eps, round, yn);
        for (0..d.dv) |j| {
            const z = in.p[r * P + d.zAt() + h * d.dv + j];
            const o = yn[j] * sigmoid(z);
            f.out[(r * d.nv + h) * d.dv + j] = if (round) bfRound(o) else o;
        }
    };
    return f;
}

/// The gradients at the forward point act, beta, y (the reference's or the GPU's) from dout; `round` rounds yn.
pub fn backward(a: std.mem.Allocator, d: Dims, in: Inputs, act: []const f64, beta: []const f64, y: []const f64, dout: []const f64, round: bool) !Grads {
    const C = d.conv();
    const P = d.proj();
    const dkn = d.dk;
    const gr: Grads = .{
        .dy = try a.alloc(f64, d.rows * d.nv * d.dv),
        .dq = try a.alloc(f64, d.rows * d.nk * dkn),
        .dk = try a.alloc(f64, d.rows * d.nk * dkn),
        .dact = try a.alloc(f64, d.rows * C),
        .dg = try a.alloc(f64, d.rows * d.nv),
        .dbeta = try a.alloc(f64, d.rows * d.nv),
        .dproj = try a.alloc(f64, d.rows * P),
    };
    @memset(gr.dq, 0);
    @memset(gr.dk, 0);
    const q = try a.alloc(f64, d.rows * d.nk * dkn);
    defer a.free(q);
    const k = try a.alloc(f64, d.rows * d.nk * dkn);
    defer a.free(k);
    qk(d, act, q, k);
    // the gated RMS norm
    for (0..d.rows) |r| for (0..d.nv) |h| {
        const yr = y[(r * d.nv + h) * d.dv ..][0..d.dv];
        var ss: f64 = 0;
        for (yr) |v| ss += v * v;
        const rinv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d.dv)) + in.eps);
        var dot: f64 = 0;
        const dn = try a.alloc(f64, d.dv);
        defer a.free(dn);
        const yn = try a.alloc(f64, d.dv);
        defer a.free(yn);
        normed(yr, in.norm_w, in.eps, round, yn);
        for (0..d.dv) |j| {
            const zc = r * P + d.zAt() + h * d.dv + j;
            const sg = sigmoid(in.p[zc]);
            const go = dout[(r * d.nv + h) * d.dv + j];
            gr.dproj[zc] = go * yn[j] * sg * (1 - sg);
            dn[j] = go * sg * in.norm_w[j];
            dot += yr[j] * dn[j];
        }
        for (0..d.dv) |j| gr.dy[(r * d.nv + h) * d.dv + j] = rinv * dn[j] - rinv * rinv * rinv * yr[j] * dot / @as(f64, @floatFromInt(d.dv));
    };
    // the chain, each value head with every state it passes through
    const size = d.dv * dkn;
    const states = try a.alloc(f64, (d.rows + 1) * size);
    defer a.free(states);
    const ds = try a.alloc(f64, size);
    defer a.free(ds);
    const dqh = try a.alloc(f64, dkn);
    defer a.free(dqh);
    const dkh = try a.alloc(f64, dkn);
    defer a.free(dkh);
    const sd = try a.alloc(f64, dkn);
    defer a.free(sd);
    for (0..d.nv) |h| {
        const hk = h / (d.nv / d.nk);
        const v0 = 2 * d.nk * dkn + h * d.dv;
        if (in.init) |s0| @memcpy(states[0..size], s0[h * size ..][0..size]) else @memset(states[0..size], 0);
        for (0..d.rows) |r| {
            const g = gate(d, in, r, h);
            const b = beta[r * d.nv + h];
            const kr = k[(r * d.nk + hk) * dkn ..][0..dkn];
            const cur = states[(r + 1) * size ..][0..size];
            @memcpy(cur, states[r * size ..][0..size]);
            for (0..d.dv) |j| {
                const row = cur[j * dkn ..][0..dkn];
                var kv: f64 = 0;
                for (row, kr) |*s, kk| {
                    s.* *= g;
                    kv += s.* * kk;
                }
                const delta = b * (act[r * C + v0 + j] - kv);
                for (row, kr) |*s, kk| s.* += kk * delta;
            }
        }
        @memset(ds, 0);
        var r = d.rows;
        while (r > 0) {
            r -= 1;
            const g = gate(d, in, r, h);
            const b = beta[r * d.nv + h];
            const kr = k[(r * d.nk + hk) * dkn ..][0..dkn];
            const qr = q[(r * d.nk + hk) * dkn ..][0..dkn];
            const prev = states[r * size ..][0..size];
            const cur = states[(r + 1) * size ..][0..size];
            @memset(dqh, 0);
            @memset(dkh, 0);
            var dg: f64 = 0;
            var db: f64 = 0;
            for (0..d.dv) |j| {
                const sp = prev[j * dkn ..][0..dkn];
                const sn = cur[j * dkn ..][0..dkn];
                const dsj = ds[j * dkn ..][0..dkn];
                var kv: f64 = 0;
                for (sd, sp, kr) |*x, p, kk| {
                    x.* = g * p;
                    kv += x.* * kk;
                }
                const err = act[r * C + v0 + j] - kv;
                const delta = b * err;
                const dj = gr.dy[(r * d.nv + h) * d.dv + j];
                var dd: f64 = 0;
                for (dsj, qr, kr, sn, dqh) |*x, qq, kk, s, *dq| {
                    x.* += dj * qq;
                    dq.* += dj * s;
                    dd += x.* * kk;
                }
                gr.dact[r * C + v0 + j] = b * dd;
                db += dd * err;
                for (dsj, kr, sd, sp, dkh) |*x, kk, s, p, *dk| {
                    dk.* += delta * x.* - b * dd * s;
                    const dsp = x.* - b * dd * kk;
                    dg += dsp * p;
                    x.* = g * dsp;
                }
            }
            for (0..dkn) |i| {
                gr.dq[(r * d.nk + hk) * dkn + i] += dqh[i];
                gr.dk[(r * d.nk + hk) * dkn + i] += dkh[i];
            }
            gr.dg[r * d.nv + h] = dg;
            gr.dbeta[r * d.nv + h] = db;
        }
    }
    // the L2 norms
    const sc = 1 / @sqrt(@as(f64, @floatFromInt(dkn)));
    for (0..d.rows) |r| for (0..d.nk) |hk| for (0..2) |which| {
        const col = which * d.nk * dkn + hk * dkn;
        const dqk = (if (which == 0) gr.dq else gr.dk)[(r * d.nk + hk) * dkn ..][0..dkn];
        const x = act[r * C + col ..][0..dkn];
        var ss: f64 = 0;
        var dot: f64 = 0;
        for (x, dqk) |xi, di| {
            ss += xi * xi;
            dot += xi * di;
        }
        const inv = 1 / @sqrt(ss + 1e-6);
        const s = if (which == 0) sc else 1;
        for (0..dkn) |i| gr.dact[r * C + col + i] = s * inv * (dqk[i] - inv * inv * x[i] * dot);
    };
    // the gates
    for (0..d.rows) |r| for (0..d.nv) |h| {
        const b = sigmoid(in.p[r * P + d.bAt() + h]);
        const x = in.p[r * P + d.aAt() + h] + in.dt_bias[h];
        const slope = if (x > 20) 1 else sigmoid(x);
        gr.dproj[r * P + d.bAt() + h] = gr.dbeta[r * d.nv + h] * b * (1 - b);
        gr.dproj[r * P + d.aAt() + h] = -gr.dg[r * d.nv + h] * gate(d, in, r, h) * @exp(in.a_log[h]) * slope;
    };
    // the conv and its SiLU
    for (0..d.rows) |r| for (0..C) |c| {
        var acc: f64 = 0;
        for (0..taps) |tap| {
            const at = r + taps - 1 - tap;
            if (at >= d.rows) continue;
            const pre = convPre(d, in, at, c);
            const s = sigmoid(pre);
            acc += in.conv_w[c * taps + tap] * gr.dact[at * C + c] * s * (1 + pre * (1 - s));
        }
        gr.dproj[r * P + c] = acc;
    };
    return gr;
}

/// sum(dout * out) of the unrounded forward: the loss whose P-gradient the backward gives, for finite differences.
pub fn loss(a: std.mem.Allocator, d: Dims, in: Inputs, dout: []const f64) !f64 {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const f = try forward(arena.allocator(), d, in, false);
    var s: f64 = 0;
    for (f.out, dout) |o, g| s += o * g;
    return s;
}

/// Central differences of `loss` at `count` coordinates of P against the backward: the worst relative difference.
pub fn finiteCheck(a: std.mem.Allocator, d: Dims, in: Inputs, dout: []const f64, rng: std.Random, count: usize) !struct { worst: f64, scale: f64 } {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const f = try forward(aa, d, in, false);
    const gr = try backward(aa, d, in, f.act, f.beta, f.y, dout, false);
    var top: f64 = 0;
    for (gr.dproj) |v| top = @max(top, @abs(v));
    const p = try aa.dupe(f64, in.p);
    var moved = in;
    moved.p = p;
    var worst: f64 = 0;
    const P = d.proj();
    const kinds = [_][2]usize{ .{ 0, d.nk * d.dk }, .{ d.nk * d.dk, 2 * d.nk * d.dk }, .{ 2 * d.nk * d.dk, d.conv() }, .{ d.zAt(), d.bAt() }, .{ d.bAt(), d.aAt() }, .{ d.aAt(), P } };
    for (0..count) |i| {
        const kind = kinds[i % kinds.len];
        const r = rng.uintLessThan(usize, d.rows);
        const c = kind[0] + rng.uintLessThan(usize, kind[1] - kind[0]);
        const at = r * P + c;
        const h = 1e-5 * @max(1, @abs(in.p[at]));
        p[at] = in.p[at] + h;
        const up = try loss(aa, d, moved, dout);
        p[at] = in.p[at] - h;
        const down = try loss(aa, d, moved, dout);
        p[at] = in.p[at];
        const fd = (up - down) / (2 * h);
        const e = @abs(fd - gr.dproj[at]) / @max(@abs(gr.dproj[at]), 1e-3 * top);
        worst = @max(worst, e);
    }
    return .{ .worst = worst, .scale = top };
}
