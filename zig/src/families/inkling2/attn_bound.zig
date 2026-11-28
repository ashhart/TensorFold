//! The float64 formula of one (row, head) and the page's bound on the kernels' order against it, evaluated in float64.
const std = @import("std");
const attn = @import("attn.zig");
const order = @import("attn_order.zig");

const u: f64 = 0x1p-24; // fp32 unit roundoff
const u_mma: f64 = 0x1p-23; // a tensor-core add: its adders may truncate
const u_bf: f64 = 0x1p-8; // bf16 unit roundoff
const most_live: usize = 1 << 20; // the domain's limit on the keys of a row
const huge: f64 = 0x1p120; // the domain's limit on every magnitude that enters a product or a sum
const gate_min: f64 = -87; // 1 + expf(-gate) stays below 2^126 and the sigmoid is a normal number

pub const Error = error{AttnDomain} || std.mem.Allocator.Error;

fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}

fn lambda(a: f64) f64 {
    return -@log(1 - a);
}

/// One head's float64 result per output index, the magnitude of its terms and the bound's exponent and floor term.
pub const Head = struct {
    want: []f64,
    mag: []f64,
    big: f64, // L
    floor: f64, // (nk + 1) (2 vmax + 1) 2^-130, the subnormal term before its factors
    sg: []f64,

    pub fn free(h: Head, a: std.mem.Allocator) void {
        a.free(h.want);
        a.free(h.mag);
        a.free(h.sg);
    }

    /// The bound on |y - want[d]| of the fp32 value before the output's rounding; +inf where exp(L) overflows.
    pub fn limit(h: Head, d: usize) f64 {
        const grow = @exp(h.big);
        if (!std.math.isFinite(grow)) return std.math.inf(f64);
        return (1 + 0x1p-20) * (h.mag[d] * std.math.expm1(h.big) + h.sg[d] * h.floor * grow) + 0x1p-149;
    }

    /// The bound on the bf16-rounded output: the fp32 bound, the output's own rounding, bf16's subnormal step.
    pub fn limitOut(h: Head, d: usize) f64 {
        const b = h.limit(d);
        return b + u_bf * (@abs(h.want[d]) + b) + 0x1p-134;
    }
};

fn finite(xs: []const f32) bool {
    for (xs) |x| if (!std.math.isFinite(x)) return false;
    return true;
}

/// Row `row`, head `h`: the formula's result and the bound's terms; error.AttnDomain outside the page's domain.
pub fn head(a: std.mem.Allocator, s: attn.Shape, in: attn.Inputs, row: usize, h: usize) Error!Head {
    const hd = s.hd;
    const t = s.pos0 + row;
    const lo = attn.first(s, t);
    const nk = t - lo + 1;
    if (nk > most_live) return error.AttnDomain;
    const kh = h / (s.heads / s.kv_heads);
    const hrow = row * s.heads + h;
    const q = in.q[hrow * hd ..][0..hd];
    const rel = in.rel[hrow * s.drel ..][0..s.drel];
    if (!finite(q) or !finite(rel) or !(@abs(@as(f64, in.sink[hrow])) < huge)) return error.AttnDomain;
    for (in.gate[hrow * hd ..][0..hd]) |g| if (!std.math.isFinite(g) or @as(f64, g) < gate_min) return error.AttnDomain;
    // the keys a block of this call loads: from the first row's window start to the last row's position
    const loaded_lo = attn.first(s, s.pos0);
    for (loaded_lo..s.pos0 + s.rows) |key| {
        const base = (key * s.kv_heads + kh) * hd;
        if (!finite(in.k[base..][0..hd]) or !finite(in.v[base..][0..hd])) return error.AttnDomain;
    }
    const sigma = std.math.pow(f64, @floatFromInt(hd), -0.5);
    const l2e = std.math.log2e;
    const mma: f64 = @floatFromInt((hd + 15) / 16);
    const g_dot = (1 + (std.math.pow(f64, 1 + 17 * u_mma, mma) - 1)) * (1 + gamma(6)) - 1;
    const g_bias = (1 + gamma(@floatFromInt(s.drel))) * (1 + u_bf) * (1 + gamma(3)) - 1;
    const xs = try a.alloc(f64, nk);
    defer a.free(xs);
    var xmax: f64 = -std.math.inf(f64);
    var xmin: f64 = std.math.inf(f64);
    var dmax: f64 = 0;
    for (xs, lo..) |*x, key| {
        const k = in.k[(key * s.kv_heads + kh) * hd ..][0..hd];
        var dot: f64 = 0;
        var abs: f64 = 0;
        for (q, k) |p, r| {
            dot += @as(f64, p) * r;
            abs += @abs(@as(f64, p) * r);
        }
        if (!(abs < huge)) return error.AttnDomain;
        var bias: f64 = 0;
        var babs: f64 = 0;
        if (t - key < s.rext) {
            for (0..s.drel) |j| {
                const w = @as(f64, in.rel[hrow * s.drel + j]) * in.proj[j * s.rext + t - key];
                if (!std.math.isFinite(w)) return error.AttnDomain;
                bias += w;
                babs += @abs(w);
            }
            if (!(babs < huge)) return error.AttnDomain;
        }
        x.* = (dot * sigma + bias) * l2e;
        xmax = @max(xmax, x.*);
        xmin = @min(xmin, x.*);
        dmax = @max(dmax, l2e * (sigma * abs * g_dot + babs * g_bias) + 0x1p-133);
    }
    const xz = @as(f64, in.sink[hrow]) * l2e;
    dmax = @max(dmax, gamma(2) * @abs(xz) + 0x1p-133);
    xmax = @max(xmax, xz);
    xmin = @min(xmin, xz);
    const eta = @log(2.0) * (dmax + u * ((xmax - xmin) + 2 * dmax));
    // keys a block holds and the seams a weight crosses: tiles and sub-chunks a step, chunks a row
    const global = s.window == 0;
    const tiles: f64 = if (global) order.sub / order.tile else order.sliding_chunk / order.tile;
    const subs: f64 = if (global) order.global_chunk / order.sub else 0;
    const chunks: f64 = @floatFromInt(if (global) t / order.global_chunk + 1 else (s.window - 1) / order.sliding_chunk + 2);
    const events = (tiles - 1) + subs + chunks + 1; // a rescale: exp2f and a product
    const common = eta + lambda(gamma(5)) + lambda(u_bf) + events * lambda(gamma(6));
    const num = common + 4 * tiles * lambda(17 * u_mma) + (subs + chunks) * lambda(u);
    const den = common + (16 * tiles + 2 + subs + chunks + 1) * lambda(u);
    const big = num + den + lambda(gamma(5)) + 4 * lambda(u);
    var z: f64 = std.math.exp2(xz - xmax);
    for (xs) |x| z += std.math.exp2(x - xmax);
    const want = try a.alloc(f64, hd);
    const mag = try a.alloc(f64, hd);
    const sg = try a.alloc(f64, hd);
    var vmax: f64 = 0;
    for (0..hd) |d| {
        var num_d: f64 = 0;
        var abs_d: f64 = 0;
        for (xs, lo..) |x, key| {
            const v: f64 = in.v[(key * s.kv_heads + kh) * hd + d];
            const p = std.math.exp2(x - xmax);
            num_d += p * v;
            abs_d += p * @abs(v);
            vmax = @max(vmax, @abs(v));
        }
        if (!(@as(f64, @floatFromInt(nk)) * vmax < huge)) return error.AttnDomain;
        sg[d] = 1 / (1 + @exp(-@as(f64, in.gate[hrow * hd + d])));
        want[d] = num_d / z * sg[d];
        mag[d] = abs_d / z * sg[d];
    }
    const floor = @as(f64, @floatFromInt(nk + 1)) * (2 * vmax + 1) * 0x1p-130;
    return .{ .want = want, .mag = mag, .big = big, .floor = floor, .sg = sg };
}
