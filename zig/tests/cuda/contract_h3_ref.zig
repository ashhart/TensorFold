//! MiniMax H3's bf16 CUDA kernels in float64: each kernel's formula unrounded; its bf16 roundings are its spec's mids.
const std = @import("std");
const core = @import("core");

const contract = core.contract;
const c = contract.spec;
const Order = contract.gates.Order;
const sumIn = contract.rows.sumIn;

/// A value rounded to bf16 as the kernels round an input table (to nearest even); NaN and the infinities kept.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// What the references read: every input as float64 values (bf16 inputs already hold bf16 values) and the sizes.
pub const In = struct {
    a: []const f64 = &.{},
    b: []const f64 = &.{},
    d: []const f64 = &.{},
    e: []const f64 = &.{},
    f: []const f64 = &.{},
    g: []const f64 = &.{},
    ints: []const i32 = &.{},
    more: []const i32 = &.{},
    rows: usize,
    n: usize,
    k: usize,
    heads: usize = 1,
    slots: usize = 0,
    scale: f64 = 1,
    eps: f64 = 1e-5,
};

/// How the sums are taken: in float64 (the reference) or in an accumulator and order (gate 4's simulation).
pub const Sum = struct {
    acc: c.Accum = .none,
    order: Order = .sequential,

    pub fn of(s: Sum, t: []const f64) f64 {
        if (s.acc == .none) {
            var x: f64 = 0;
            for (t) |v| x += v;
            return x;
        }
        return sumIn(s.acc, s.order, t);
    }
};

fn absSum(t: []const f64) f64 {
    var x: f64 = 0;
    for (t) |v| x += @abs(v);
    return x;
}

/// The modulated norm of one value (h3.cu's `modulated`, which rounds the norm, 1 + scale and their product to bf16).
pub fn modulated(x: f64, inv: f64, w: f64, scale: f64, shift: f64) f64 {
    return x * inv * w * (1 + scale) + shift;
}

/// h3_norm: x += gate y, then the modulated norm (a x, b y, d w, e gate, f scale, g shift); stream out, then normed.
pub fn norm(in: In, s: Sum, scratch: []f64, want: []f64, mag: []f64) void {
    const C = in.n;
    const half = in.rows * C;
    for (0..in.rows) |r| {
        const x = scratch[0..C];
        for (x, 0..) |*v, i| v.* = in.a[r * C + i] + in.e[i] * in.b[r * C + i];
        const sq = scratch[C .. 2 * C];
        for (sq, x) |*q, v| q.* = v * v;
        const inv = 1 / @sqrt(s.of(sq) / @as(f64, @floatFromInt(C)) + in.eps);
        for (0..C) |i| {
            want[r * C + i] = x[i];
            mag[r * C + i] = @abs(in.a[r * C + i]) + @abs(in.e[i] * in.b[r * C + i]);
            want[half + r * C + i] = modulated(x[i], inv, in.d[i], in.f[i], in.g[i]);
            mag[half + r * C + i] = @abs(x[i] * inv * in.d[i] * (1 + in.f[i])) + @abs(in.g[i]);
        }
    }
}

/// h3_heads: q, k norms and split-half rotary on `k` channels (a q, b k, d v, e cos, f sin with k / 2 a row, g norms).
pub fn heads(in: In, s: Sum, scratch: []f64, want: []f64, mag: []f64) void {
    const H = in.heads;
    const half = in.k / 2;
    const block = H * in.slots * 128;
    for (0..in.rows) |r| for (0..H) |h| {
        const base = (r * H + h) * 128;
        const at = (h * in.slots + @as(usize, @intCast(in.ints[r]))) * 128;
        for ([_][]const f64{ in.a, in.b }, 0..) |x, which| {
            const sq = scratch[0..128];
            for (sq, 0..) |*q, i| q.* = x[base + i] * x[base + i];
            const inv = 1 / @sqrt(s.of(sq) / 128 + in.eps);
            const w = in.g[which * 128 ..][0..128];
            for (0..128) |i| {
                var y = x[base + i] * inv * w[i];
                var m = @abs(y);
                if (i < in.k) {
                    const low = i < half;
                    const p = if (low) i + half else i - half;
                    const t = if (low) i else i - half;
                    const sign: f64 = if (low) -1 else 1;
                    const pair = x[base + p] * inv * w[p];
                    y = y * bf(in.e[r * half + t]) + sign * pair * bf(in.f[r * half + t]);
                    m = 4 * @abs(x[base + i] * inv * w[i]) + 4 * @abs(pair);
                }
                want[which * block + at + i] = y;
                mag[which * block + at + i] = m;
            }
        }
        for (0..128) |i| {
            want[2 * block + at + i] = in.d[base + i];
            mag[2 * block + at + i] = @abs(in.d[base + i]);
        }
    };
}

/// h3_pool: each tile's mean q, k and v over its real rows. a, b, d: q, k, v in tile order; ints: the tile sizes.
pub fn pool(in: In, s: Sum, scratch: []f64, want: []f64, mag: []f64) void {
    const T = in.rows;
    const block = in.heads * T * 128;
    for ([_][]const f64{ in.a, in.b, in.d }, 0..) |x, which| for (0..in.heads) |h| for (0..T) |t| for (0..128) |i| {
        const size: usize = @intCast(in.ints[t]);
        const terms = scratch[0..size];
        for (terms, 0..) |*v, j| v.* = x[((h * in.slots) + t * 64 + j) * 128 + i];
        const o = which * block + (h * T + t) * 128 + i;
        want[o] = s.of(terms) / @as(f64, @floatFromInt(size));
        mag[o] = absSum(terms) / @as(f64, @floatFromInt(size));
    };
}

/// h3_tile_scores: S[h, tq, tk] = QP[h, tq] . KP[h, tk] / sqrt(128). a QP, b KP.
pub fn scores(in: In, s: Sum, scratch: []f64, want: []f64, mag: []f64) void {
    const T = in.rows;
    for (0..in.heads) |h| for (0..T) |q| for (0..T) |k| {
        const t = scratch[0..128];
        for (t, 0..) |*v, i| v.* = in.a[(h * T + q) * 128 + i] * in.b[(h * T + k) * 128 + i];
        const o = (h * T + q) * T + k;
        want[o] = s.of(t) * 0.08838834764831845;
        mag[o] = absSum(t) * 0.08838834764831845;
    };
}

/// h3_coarse: C[h, tq] = softmax over tiles of S[h, tq] times VP[h]. a S, b VP.
pub fn coarse(in: In, s: Sum, scratch: []f64, want: []f64, mag: []f64) void {
    const T = in.rows;
    for (0..in.heads) |h| for (0..T) |q| {
        const row = in.a[(h * T + q) * T ..][0..T];
        var top = -std.math.inf(f64);
        for (row) |v| top = @max(top, v);
        const e = scratch[0..T];
        for (e, row) |*w, v| w.* = @exp(v - top);
        const total = s.of(e);
        const t = scratch[T .. 2 * T];
        for (0..128) |i| {
            for (t, 0..) |*v, j| v.* = e[j] * in.b[(h * T + j) * 128 + i];
            const o = (h * T + q) * 128 + i;
            want[o] = s.of(t) / total;
            mag[o] = absSum(t) / total;
        }
    };
}

/// h3_attention: softmax(q k^T scale) v over each query tile's key tiles (ints), the online softmax's order kept.
pub fn attention(in: In, s: Sum, scratch: []f64, want: []f64, mag: []f64) void {
    const T = in.rows;
    const ksel = in.k / 64;
    const H = in.heads;
    var row: usize = 0;
    for (0..T) |tq| {
        const size: usize = @intCast(in.more[tq]);
        for (0..size) |r| {
            for (0..H) |h| attendRow(in, s, scratch, h, tq, r, ksel, want[(row * H + h) * 128 ..][0..128], mag[(row * H + h) * 128 ..][0..128]);
            row += 1;
        }
    }
}

fn attendRow(in: In, s: Sum, scratch: []f64, h: usize, tq: usize, r: usize, ksel: usize, want: []f64, mag: []f64) void {
    const q = in.a[((h * in.slots) + tq * 64 + r) * 128 ..][0..128];
    const list = in.ints[(h * in.rows + tq) * ksel ..][0..ksel];
    var top = -std.math.inf(f64);
    var total: f64 = 0;
    const acc = scratch[0..128];
    const accm = scratch[128..256];
    @memset(acc, 0);
    @memset(accm, 0);
    const t = scratch[256..384];
    const w = scratch[384..448];
    const terms = scratch[448 .. 448 + 64];
    for (list) |tile_i| {
        const tile: usize = @intCast(tile_i);
        const size: usize = @intCast(in.more[tile]);
        var next = top;
        var sc: [64]f64 = undefined;
        for (0..64) |j| {
            if (j >= size) {
                sc[j] = -std.math.inf(f64);
                continue;
            }
            const key = in.b[((h * in.slots) + tile * 64 + j) * 128 ..][0..128];
            for (t, q, key) |*v, a, b| v.* = a * b;
            sc[j] = s.of(t) * in.scale;
            next = @max(next, sc[j]);
        }
        const shrink = if (next == -std.math.inf(f64)) 1 else @exp(top - next);
        total *= shrink;
        for (acc, accm) |*o, *m| {
            o.* *= shrink;
            m.* *= shrink;
        }
        top = next;
        for (w, sc) |*x, v| x.* = if (next == -std.math.inf(f64)) 0 else @exp(v - next);
        total += s.of(w);
        for (0..128) |i| {
            for (terms, w, 0..) |*v, x, j| v.* = x * in.d[((h * in.slots) + tile * 64 + j) * 128 + i];
            acc[i] += s.of(terms);
            accm[i] += absSum(terms);
        }
    }
    for (want, mag, acc, accm) |*o, *m, a, b| {
        o.* = a / total;
        m.* = b / total;
    }
}

/// h3_swiglu: silu(gate) * value from the first projection's rows [gate | value]. a the rows.
pub fn swiglu(in: In, want: []f64, mag: []f64) void {
    const N = in.n;
    for (0..in.rows) |r| for (0..N) |n| {
        const g = in.a[r * 2 * N + n];
        const v = in.a[r * 2 * N + N + n];
        const silu = g / (1 + @exp(-g));
        want[r * N + n] = silu * v;
        mag[r * N + n] = @abs(silu * v);
    };
}

/// h3_gate_add: X += g y per row (a x, b y, e the gate row). h3_gate_mix: Y += C[tile] G (a y, b g, d C).
pub fn gated(in: In, mix: bool, want: []f64, mag: []f64) void {
    for (0..in.rows) |r| for (0..in.n) |i| {
        const o = r * in.n + i;
        const g = if (mix) in.d[((i / 128) * in.slots + @as(usize, @intCast(in.ints[r])) / 64) * 128 + i % 128] else in.e[i];
        const add = g * in.b[o];
        want[o] = in.a[o] + add;
        mag[o] = @abs(in.a[o]) + @abs(add);
    };
}
