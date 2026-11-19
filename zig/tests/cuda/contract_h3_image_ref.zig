//! MiniMax-H3-Image's CUDA kernels in float64 with their bf16 mid-roundings, before each output's store rounding.
const std = @import("std");
const core = @import("core");

const contract = core.contract;
const c = contract.spec;
const Order = contract.gates.Order;
const sumIn = contract.rows.sumIn;

/// fp32 to bf16 and back, as the kernels' __float2bfloat16 (round to nearest even); NaN and infinities kept.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

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

/// What the references read: every input as float64 values (bf16 inputs already hold bf16 values) and the sizes.
pub const In = struct {
    a: []const f64 = &.{}, // primary activation (lat, tok, x, q, acc, g, times...)
    b: []const f64 = &.{}, // second input or first weight (pw, ew, g, k, w, bias...)
    d: []const f64 = &.{}, // third (pb, eb, v, sn, s-table...)
    e: []const f64 = &.{}, // fourth (reg, cs, residual y...)
    f: []const f64 = &.{}, // fifth (sn, value rows...)
    g: []const f64 = &.{}, // sixth (layer-norm bias)
    ints: []const u32 = &.{}, // origins (pairs), ids, idx, pads
    rows: usize = 0,
    n: usize = 0,
    k: usize = 0,
    tiles: usize = 0,
    tokens: usize = 0,
    h: usize = 0,
    w: usize = 0,
    heads: usize = 1,
    kv_heads: usize = 1,
    head: usize = 128,
    q_row: usize = 0,
    k_row: usize = 0,
    eps: f64 = 1e-5,
    scale: f64 = 0.125,
    mode: u32 = 0,
    causal: bool = false,
    rotary: bool = false,
    stride: usize = 0, // heads' spacing in a row; 0 = head (tightly packed)
};

fn magOf(terms: []const f64) f64 {
    var m: f64 = 0;
    for (terms) |t| m += @abs(t);
    return m;
}

/// h3d_post_quant: per tile cell, bf16(pw . bf16(lat) + pb); a lat, ints origin, b pw, d pb; out tiles*cells*24.
pub fn postQuant(in: In, s: Sum, want: []f64, mag: []f64) void {
    const cells = in.h * in.w;
    const plane = in.rows * in.n;
    var terms: [24]f64 = undefined;
    for (0..in.tiles * cells) |y| {
        const tile = y / cells;
        const i = y % cells;
        const r = in.ints[2 * tile] + i / in.w;
        const col = in.ints[2 * tile + 1] + i % in.w;
        for (0..24) |x| {
            for (&terms, 0..) |*t, k| t.* = in.b[x * 24 + k] * bf(in.a[k * plane + r * in.n + col]);
            want[y * 24 + x] = s.of(&terms) + in.d[x];
            mag[y * 24 + x] = magOf(&terms);
        }
    }
}

/// h3d_embed: cells tokens from ew . tok + eb to bf16, 4 register rows and one zero; a tok, b ew, d eb, e reg.
pub fn embed(in: In, s: Sum, want: []f64, mag: []f64) void {
    const cells = in.h * in.w;
    var terms: [24]f64 = undefined;
    for (0..in.tiles * in.tokens) |y| {
        const tile = y / in.tokens;
        const t = y % in.tokens;
        const out = want[y * 1024 ..][0..1024];
        const om = mag[y * 1024 ..][0..1024];
        if (t < cells) {
            for (out, om, 0..) |*o, *m, j| {
                for (&terms, 0..) |*v, k| v.* = in.b[j * 24 + k] * in.a[(tile * cells + t) * 24 + k];
                o.* = s.of(&terms) + in.d[j];
                m.* = magOf(&terms);
            }
        } else if (t < cells + 4) {
            for (out, om, in.e[(t - cells) * 1024 ..][0..1024]) |*o, *m, v| {
                o.* = v;
                m.* = 0;
            }
        } else {
            @memset(out, 0);
            @memset(om, 0);
        }
    }
}

/// h3d_rms_norm: bf16(x * rsqrt(mean(x^2) + eps) * g), the squares summed in fp32; a x [m, 1024], b g [1024].
pub fn rmsNorm(in: In, s: Sum, want: []f64, mag: []f64) void {
    var terms: [1024]f64 = undefined;
    for (0..in.rows) |r| {
        const xr = in.a[r * 1024 ..][0..1024];
        for (&terms, xr) |*t, v| t.* = v * v;
        const inv = 1.0 / @sqrt(s.of(&terms) / 1024.0 + in.eps);
        for (0..1024) |i| {
            want[r * 1024 + i] = xr[i] * inv * in.b[i];
            mag[r * 1024 + i] = xr[i] * xr[i];
        }
    }
}

/// h3d_layer_norm: bf16((x - mean) * rstd * g + b), mean and rstd from fp32 sums; a x, b g, d b.
pub fn layerNorm(in: In, s: Sum, want: []f64, mag: []f64) void {
    var terms: [1024]f64 = undefined;
    for (0..in.rows) |r| {
        const xr = in.a[r * 1024 ..][0..1024];
        for (&terms, xr) |*t, v| t.* = v;
        const mean = s.of(&terms) / 1024.0;
        const abs = magOf(&terms);
        for (&terms, xr) |*t, v| t.* = (v - mean) * (v - mean);
        const rstd = 1.0 / @sqrt(s.of(&terms) / 1024.0 + in.eps);
        for (0..1024) |i| {
            const d = xr[i] - mean;
            want[r * 1024 + i] = d * rstd * in.b[i] + in.d[i];
            // the mean's sum moves every d by its error; the variance's moves rstd by half its relative error
            mag[r * 1024 + i] = @abs(in.b[i]) * rstd * (abs / 1024.0 + @abs(d));
        }
    }
}

/// h3d_finish: mode 0 bf16(acc + b); mode 1 bf16(y + bf16(bf16(acc + b) * s)); a acc, b bias, d scale, e y.
pub fn finish(in: In, want: []f64, mag: []f64) void {
    for (0..in.rows * in.n) |i| {
        const cix = i % in.n;
        const out = in.a[i] + in.b[cix];
        want[i] = if (in.mode == 0) out else in.e[i] + bf(bf(out) * in.d[cix]);
        mag[i] = 0;
    }
}

/// h3d_heads: q and k RMS-normed to bf16, their first 32 dims half-turned by cs/sn, v copied; a qkv, b cs, d sn.
pub fn heads(in: In, s: Sum, want: []f64, mag: []f64) void {
    const T = in.tokens;
    const block = in.tiles * 16 * T * 64; // elements of one of q, k, v
    var terms: [64]f64 = undefined;
    for (0..in.tiles) |tile| for (0..T) |t| for (0..16) |head| {
        const row = tile * T + t;
        const src = in.a[row * 3072 + head * 192 ..][0..192];
        const to = ((tile * 16 + head) * T + t) * 64;
        for (0..2) |part| {
            const x = src[part * 64 ..][0..64];
            for (&terms, x) |*q, v| q.* = v * v;
            const inv = 1.0 / @sqrt(s.of(&terms) / 64.0 + in.eps);
            var na: [64]f64 = undefined;
            for (&na, x) |*o, v| o.* = bf(v * inv);
            for (0..32) |lane| {
                const other = if (lane < 16) -na[lane + 16] else na[lane - 16];
                want[part * block + to + lane] = bf(na[lane] * in.b[t * 32 + lane]) + bf(other * in.d[t * 32 + lane]);
                want[part * block + to + 32 + lane] = x[32 + lane] * inv;
                mag[part * block + to + lane] = @abs(na[lane] * in.b[t * 32 + lane]) + @abs(other * in.d[t * 32 + lane]);
                mag[part * block + to + 32 + lane] = terms[32 + lane];
            }
        }
        for (0..64) |i| {
            want[2 * block + to + i] = src[128 + i];
            mag[2 * block + to + i] = 0;
        }
    };
}

/// h3d_attention: softmax(q k^T * scale) v per (tile, head, query); a q, b k, d v; NaN scores drop as warp_max.
pub fn attention(in: In, s: Sum, want: []f64, mag: []f64) void {
    const T = in.tokens;
    var sc: [288]f64 = undefined;
    var terms: [64]f64 = undefined;
    var vt: [288]f64 = undefined;
    for (0..in.tiles) |tile| for (0..16) |head| for (0..T) |query| {
        const base = ((tile * 16 + head) * T) * 64;
        const qr = in.a[base + query * 64 ..][0..64];
        var top: f64 = -std.math.inf(f64);
        for (0..T) |key| {
            const kr = in.b[base + key * 64 ..][0..64];
            for (&terms, qr, kr) |*t, q, kv| t.* = q * kv;
            sc[key] = s.of(&terms) * in.scale;
            if (!std.math.isNan(sc[key])) top = @max(top, sc[key]);
        }
        var total: f64 = 0;
        for (0..T) |key| {
            sc[key] = @exp(sc[key] - top);
            total += sc[key];
        }
        const at = ((tile * T + query) * 16 + head) * 64;
        for (0..64) |ch| {
            for (vt[0..T], 0..) |*t, key| t.* = sc[key] * in.d[base + key * 64 + ch];
            want[at + ch] = s.of(vt[0..T]) / total;
            mag[at + ch] = magOf(vt[0..T]);
        }
    };
}

/// h3d_swiglu: act [m, 4096] = bf16(bf16(silu(gate)) * value); a wide [m, 8192].
pub fn swiglu(in: In, want: []f64, mag: []f64) void {
    for (0..in.rows) |r| for (0..4096) |x| {
        const g = in.a[r * 8192 + x];
        const silu = bf(g / (1.0 + @exp(-g)));
        want[r * 4096 + x] = silu * in.a[r * 8192 + 4096 + x];
        mag[r * 4096 + x] = 0;
    };
}

/// h3p_norm: RMSNorm(x) by w, mode 0 bf16(w * bf16(x * r)) (Qwen), mode 1 bf16(x * r * w) (H3); a x, b w.
pub fn norm(in: In, s: Sum, want: []f64, mag: []f64) void {
    const width = in.n;
    var terms: [4096]f64 = undefined;
    for (0..in.rows) |r| {
        const xr = in.a[r * width ..][0..width];
        for (terms[0..width], xr) |*t, v| t.* = v * v;
        const inv = 1.0 / @sqrt(s.of(terms[0..width]) / @as(f64, @floatFromInt(width)) + in.eps);
        for (0..width) |i| {
            want[r * width + i] = if (in.mode == 0) in.b[i] * bf(xr[i] * inv) else xr[i] * inv * in.b[i];
            mag[r * width + i] = xr[i] * xr[i];
        }
    }
}

/// h3p_heads: q and k normed (mode as h3p_norm), turned by cos/sin when rotary; fused qkv's slices rewritten.
pub fn pheads(in: In, s: Sum, want: []f64, mag: []f64, fused: bool) void {
    const D = in.head;
    const stride = if (in.stride == 0) D else in.stride;
    var terms: [128]f64 = undefined;
    if (fused) @memcpy(want, in.a) else {
        @memcpy(want[0..in.a.len], in.a);
        @memcpy(want[in.a.len..], in.b);
    }
    @memset(mag, 0);
    for (0..in.rows) |r| for (0..in.heads + in.kv_heads) |hh| {
        const is_q = hh < in.heads;
        const at = if (fused)
            r * in.q_row + (if (is_q) @as(usize, 0) else D)
        else if (is_q)
            r * in.q_row + hh * stride
        else
            in.rows * in.q_row + r * in.k_row + (hh - in.heads) * stride;
        const src = if (is_q or fused) in.a[at..][0..D] else in.b[r * in.k_row + (hh - in.heads) * stride ..][0..D];
        const w = if (is_q) in.d else in.e;
        for (&terms, src) |*t, v| t.* = v * v;
        const inv = 1.0 / @sqrt(s.of(&terms) / @as(f64, @floatFromInt(D)) + in.eps);
        var v: [128]f64 = undefined;
        for (&v, src, w) |*o, xv, wv| o.* = if (in.mode == 0) wv * bf(xv * inv) else xv * inv * wv;
        var o = v;
        if (in.rotary) for (0..D) |ch| {
            const table_at = r * 64 + ch % 64;
            const other = if (ch < 64) -bf(v[ch + 64]) else bf(v[ch - 64]);
            o[ch] = bf(bf(v[ch]) * in.f[table_at]) + bf(other * in.g[table_at]);
        };
        for (0..D) |i| {
            want[at + i] = o[i];
            mag[at + i] = terms[i];
        }
    };
}

/// h3p_attention: softmax(q k^T scale) v, two-pass (the top score, then weight and value sums); a q, b k, d v.
pub fn pattention(in: In, s: Sum, want: []f64, mag: []f64) void {
    const D = in.head;
    const group = in.heads / in.kv_heads;
    var p: [1032]f64 = undefined;
    var vt: [1032]f64 = undefined;
    var terms: [128]f64 = undefined;
    for (0..in.rows) |r| for (0..in.heads) |h| {
        const qr = in.a[r * in.q_row + h * in.head ..][0..D];
        const kv = h / group;
        const end = if (in.causal) r + 1 else in.rows;
        var top: f64 = -std.math.inf(f64);
        for (0..end) |j| {
            const kr = in.b[j * in.k_row + kv * in.head ..][0..D];
            for (&terms, qr, kr) |*t, q, kv2| t.* = q * kv2;
            p[j] = s.of(&terms) * in.scale;
            if (!std.math.isNan(p[j])) top = @max(top, p[j]);
        }
        var l: f64 = 0;
        for (p[0..end]) |*e| {
            e.* = @exp(e.* - top);
            l += e.*;
        }
        for (0..D) |ch| {
            for (vt[0..end], 0..) |*t, j| t.* = p[j] * in.d[j * in.k_row + kv * in.head + ch];
            want[(r * in.heads + h) * D + ch] = s.of(vt[0..end]) / l;
            mag[(r * in.heads + h) * D + ch] = magOf(vt[0..end]);
        }
    };
}

/// h3p_glu: silu(g) * v, mode 0 bf16(bf16(g/e) v), mode 1 bf16(bf16(g * bf16(1/e)) v); a g, b v.
pub fn glu(in: In, want: []f64, mag: []f64) void {
    for (0..in.rows * in.n) |i| {
        const x = in.a[i];
        const e = 1.0 + @exp(-x);
        const sv = if (in.mode == 0) bf(x / e) else bf(x * bf(1.0 / e));
        want[i] = sv * in.b[i];
        mag[i] = 0;
    }
}

/// h3p_add: x = bf16(x + bf16(y)) in place; a x (bf16-valued), b y fp32.
pub fn add(in: In, want: []f64, mag: []f64) void {
    for (want, mag, in.a, in.b) |*o, *m, x, y| {
        o.* = x + bf(y);
        m.* = 0;
    }
}

/// h3p_scatter_add: rows idx[i] = bf16(x + y[i]); a x in place, b y [k*n], ints idx [k]; want is all of x.
pub fn scatterAdd(in: In, want: []f64, mag: []f64) void {
    @memcpy(want, in.a);
    @memset(mag, 0);
    for (0..in.k) |i| for (0..in.n) |ch| {
        const to = @as(usize, in.ints[i]) * in.n + ch;
        want[to] = want[to] + in.b[i * in.n + ch];
    };
}

/// h3p_embed: x rows = table rows ids[i]; a table [k*n], ints ids [rows].
pub fn gather(in: In, want: []f64, mag: []f64) void {
    for (0..in.rows) |i| {
        @memcpy(want[i * in.n ..][0..in.n], in.a[@as(usize, in.ints[i]) * in.n ..][0..in.n]);
        @memset(mag[i * in.n ..][0..in.n], 0);
    }
}

/// h3p_splice: x rows pads[i] = rows[i]; a x [rows*n] in place, b rows [k*n], ints pads [k].
pub fn splice(in: In, want: []f64, mag: []f64) void {
    @memcpy(want, in.a);
    @memset(mag, 0);
    for (0..in.k) |i| @memcpy(want[@as(usize, in.ints[i]) * in.n ..][0..in.n], in.b[i * in.n ..][0..in.n]);
}

/// y [rows, n] fp32 = x [rows, k] bf16 . w [n, k] bf16^T, each K sum sequential in fp32.
pub fn linear(in: In, s: Sum, want: []f64, mag: []f64) void {
    var row: [8192]f64 = undefined;
    for (0..in.rows) |r| for (0..in.n) |o| {
        for (row[0..in.k], in.a[r * in.k ..][0..in.k], in.b[o * in.k ..][0..in.k]) |*t, x, w| t.* = x * w;
        want[r * in.n + o] = s.of(row[0..in.k]);
        mag[r * in.n + o] = magOf(row[0..in.k]);
    };
}

/// h3p_round: y = bf16(x) elementwise; a x fp32.
pub fn round(in: In, want: []f64, mag: []f64) void {
    for (want, mag, in.a) |*o, *m, v| {
        o.* = bf(v);
        m.* = 0;
    }
}
