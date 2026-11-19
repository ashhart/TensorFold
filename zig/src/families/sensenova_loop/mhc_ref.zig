//! IEEE-754 float64 reference and PyTorch BF16-rounded simulation for two-stream mHC (H <= 4096).
const std = @import("std");

pub const streams: usize = 2;
pub const max_hidden: usize = 4096;
pub const coeff_width: usize = 6; // { post0, post1, mix00, mix01, mix10, mix11 }

pub const Accum = enum { none, f32, f16, bf16 };
pub const Order = enum { sequential, pairwise, strided };

pub const Params = struct {
    hidden: usize,
    iterations: u32 = 20,
    pre_scale: f32 = 1.0,
    eps: f32 = 1e-6,
    shift_norm: bool = true,
    exit: bool = false,
    xs_group: usize = 0,
    xs_stride: usize = 0,
};

pub const PreIn = struct {
    r_in: []const u16,
    w: []const f32,
    b: []const f32,
    sub_w: []const u16,
};

pub const PreRefOut = struct {
    coeff: []f64,
    mag_coeff: []f64,
    norm: []f64,
    mag_norm: []f64,
    xs: ?[]f64 = null,
    mag_xs: ?[]f64 = null,
};

pub const PreSimOut = struct {
    coeff: []u16,
    norm: []u16,
    xs: ?[]f32 = null,
};

/// fp32 to bf16 bits, round to nearest even (NaN stays NaN).
pub fn toBf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn roundBf16(v: f32) f32 {
    return fromBf16(toBf16(v));
}

/// Exact IEEE-754 float64 value rounded to nearest BF16 (ties to even), returned as float64.
pub fn bf64(v: f64) f64 {
    if (!std.math.isFinite(v) or v == 0) return v;
    const e = @max(std.math.frexp(v).exponent, -125);
    const step = std.math.ldexp(@as(f64, 1), e - 8);
    const q = v / step;
    var t = @floor(q);
    const d = q - t;
    if (d > 0.5 or (d == 0.5 and @mod(t, 2) != 0)) t += 1;
    const out = std.math.copysign(t * step, v);
    return if (@abs(out) >= 0x1p128) std.math.copysign(std.math.inf(f64), v) else out;
}

fn bfVal(h: u16) f64 {
    return @as(f64, fromBf16(h));
}

fn roundIn(acc: Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => fromBf16(toBf16(@floatCast(v))),
        .none => v,
    };
}

fn chainSum(acc: Accum, t: []const f64, stride: usize, first: usize) f64 {
    var s: f64 = 0;
    var i = first;
    while (i < t.len) : (i += stride) s = roundIn(acc, s + roundIn(acc, t[i]));
    return s;
}

fn halvesSum(acc: Accum, t: []const f64) f64 {
    if (t.len == 0) return 0;
    if (t.len == 1) return roundIn(acc, t[0]);
    const mid = t.len / 2;
    return roundIn(acc, halvesSum(acc, t[0..mid]) + halvesSum(acc, t[mid..]));
}

pub fn sumIn(acc: Accum, order: Order, t: []const f64) f64 {
    if (acc == .none) {
        var s: f64 = 0;
        for (t) |v| s += v;
        return s;
    }
    return switch (order) {
        .sequential => chainSum(acc, t, 1, 0),
        .pairwise => halvesSum(acc, t),
        .strided => blk: {
            var lanes: [32]f64 = @splat(0);
            for (&lanes, 0..) |*l, i| l.* = chainSum(acc, t, 32, i);
            break :blk halvesSum(acc, &lanes);
        },
    };
}

fn lse2(a: f64, b: f64) f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return std.math.nan(f64);
    const mx = @max(a, b);
    if (std.math.isInf(mx) and mx < 0) return mx;
    return mx + @log(1.0 + @exp(-@abs(a - b)));
}

fn lse2F32(a: f32, b: f32) f32 {
    if (std.math.isNan(a) or std.math.isNan(b)) return std.math.nan(f32);
    const mx = @max(a, b);
    if (std.math.isInf(mx) and mx < 0) return mx;
    return mx + @log(1.0 + @exp(-@abs(a - b)));
}

fn sinkhorn2x2(z: [4]f64, iters: u32) [4]f64 {
    var l = z;
    for (0..iters) |_| {
        const r0 = lse2(l[0], l[1]);
        l[0] -= r0;
        l[1] -= r0;
        const r1 = lse2(l[2], l[3]);
        l[2] -= r1;
        l[3] -= r1;
        const c0 = lse2(l[0], l[2]);
        l[0] -= c0;
        l[2] -= c0;
        const c1 = lse2(l[1], l[3]);
        l[1] -= c1;
        l[3] -= c1;
    }
    return .{ @exp(l[0]), @exp(l[1]), @exp(l[2]), @exp(l[3]) };
}

fn sinkhorn2x2F32(z: [4]f32, iters: u32) [4]f32 {
    var l = z;
    for (0..iters) |_| {
        const r0 = lse2F32(l[0], l[1]);
        l[0] -= r0;
        l[1] -= r0;
        const r1 = lse2F32(l[2], l[3]);
        l[2] -= r1;
        l[3] -= r1;
        const c0 = lse2F32(l[0], l[2]);
        l[0] -= c0;
        l[2] -= c0;
        const c1 = lse2F32(l[1], l[3]);
        l[1] -= c1;
        l[3] -= c1;
    }
    return .{ @exp(l[0]), @exp(l[1]), @exp(l[2]), @exp(l[3]) };
}

/// Float64 reference for `sn_mhc_enter`: expands `h` (optionally `bf16(h + pending)`) to two identical streams.
pub fn enterRef(
    h_in: []const u16,
    pending: ?[]const u16,
    rows: usize,
    hidden: usize,
    r_out: []f64,
    r_mag: []f64,
    h_out: ?[]f64,
    h_mag: ?[]f64,
) void {
    for (0..rows) |r| {
        for (0..hidden) |d| {
            const hv = bfVal(h_in[r * hidden + d]);
            const v = if (pending) |pend| bf64(hv + bfVal(pend[r * hidden + d])) else hv;
            const mg = if (pending) |pend| @abs(hv) + @abs(bfVal(pend[r * hidden + d])) else 0;
            r_out[(r * 2 + 0) * hidden + d] = v;
            r_out[(r * 2 + 1) * hidden + d] = v;
            r_mag[(r * 2 + 0) * hidden + d] = mg;
            r_mag[(r * 2 + 1) * hidden + d] = mg;
            if (h_out) |ho| ho[r * hidden + d] = v;
            if (h_mag) |hm| hm[r * hidden + d] = mg;
        }
    }
}

/// PyTorch BF16 simulation for `sn_mhc_enter`.
pub fn enterSim(h_in: []const u16, pending: ?[]const u16, rows: usize, hidden: usize, r_out: []u16, h_out: ?[]u16) void {
    for (0..rows) |r| {
        for (0..hidden) |d| {
            const bits = if (pending) |pend|
                toBf16(fromBf16(h_in[r * hidden + d]) + fromBf16(pend[r * hidden + d]))
            else
                h_in[r * hidden + d];
            r_out[(r * 2 + 0) * hidden + d] = bits;
            r_out[(r * 2 + 1) * hidden + d] = bits;
            if (h_out) |ho| ho[r * hidden + d] = bits;
        }
    }
}

/// Float64 reference (and Gate 4 sum simulator) for `sn_mhc_pre_norm`.
pub fn preNormRef(a: std.mem.Allocator, in: PreIn, rows: usize, p: Params, acc: Accum, order: Order, out: PreRefOut) !void {
    const h = p.hidden;
    const two_h = 2 * h;
    const scratch = try a.alloc(f64, two_h * 2 + h);
    defer a.free(scratch);
    const rv = scratch[0..two_h];
    const tmp = scratch[two_h .. 2 * two_h];
    const br = scratch[2 * two_h ..];

    for (0..rows) |r| {
        const rx = in.r_in[r * two_h ..][0..two_h];
        for (rv, rx, tmp) |*o, bits, *sq| {
            const v = bfVal(bits);
            o.* = v;
            sq.* = v * v;
        }
        const ss = sumIn(acc, order, tmp);
        const inv_rms = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(two_h)) + @as(f64, p.eps));

        var z: [8]f64 = undefined;
        var dz: [8]f64 = undefined;
        for (0..8) |m| {
            const wm = in.w[m * two_h ..][0..two_h];
            var abs_dot: f64 = 0;
            for (0..two_h) |j| {
                const prod = rv[j] * @as(f64, wm[j]);
                tmp[j] = prod;
                abs_dot += @abs(prod);
            }
            const dot = sumIn(acc, order, tmp);
            z[m] = dot * inv_rms + @as(f64, in.b[m]);
            dz[m] = abs_dot * @abs(inv_rms) + 0.5 * @abs(dot * inv_rms);
        }

        const pre0 = 1.0 / (1.0 + @exp(-z[0]));
        const pre1 = 1.0 / (1.0 + @exp(-z[1]));
        const post0 = 2.0 / (1.0 + @exp(-z[2]));
        const post1 = 2.0 / (1.0 + @exp(-z[3]));
        const mix = sinkhorn2x2(.{ z[4], z[5], z[6], z[7] }, p.iterations);

        const co = out.coeff[r * coeff_width ..][0..coeff_width];
        const cm = out.mag_coeff[r * coeff_width ..][0..coeff_width];
        co.* = .{ post0, post1, mix[0], mix[1], mix[2], mix[3] };
        cm[0] = dz[2] + @abs(post0) * (@abs(z[2]) + 3.0);
        cm[1] = dz[3] + @abs(post1) * (@abs(z[3]) + 3.0);
        const dz_max = @max(@max(dz[4], dz[5]), @max(dz[6], dz[7]));
        const sk_scale: f64 = @as(f64, @floatFromInt(p.iterations)) * 4.0 + 6.0;
        for (0..4) |k| cm[2 + k] = dz_max * sk_scale + @abs(mix[k]) * (@abs(z[4 + k]) + sk_scale);

        const p0 = bf64(pre0);
        const p1 = bf64(pre1);
        for (0..h) |d| {
            const b_unscaled = bf64(p0 * rv[d] + p1 * rv[h + d]);
            const bv = if (p.pre_scale != 1.0) bf64(@as(f64, p.pre_scale) * b_unscaled) else b_unscaled;
            br[d] = bv;
            tmp[d] = bv * bv;
        }
        const ss_b = sumIn(acc, order, tmp[0..h]);
        const inv_b = 1.0 / @sqrt(ss_b / @as(f64, @floatFromInt(h)) + @as(f64, p.eps));
        for (0..h) |d| {
            const sw = bfVal(in.sub_w[d]);
            const w_eff = if (p.shift_norm) 1.0 + sw else sw;
            const yv = (br[d] * inv_b) * w_eff;
            out.norm[r * h + d] = yv;
            out.mag_norm[r * h + d] = @abs(yv) * 0.5;
        }
        if (p.xs_group > 0) {
            const xs_w = out.xs.?;
            const xs_m = out.mag_xs.?;
            const groups = h / p.xs_group;
            for (0..groups) |g| {
                const base = g * p.xs_group;
                var ab: f64 = 0;
                for (0..p.xs_group) |i| {
                    const oq = bf64(out.norm[r * h + base + i]);
                    tmp[i] = oq;
                    ab += @abs(oq);
                }
                xs_w[g * p.xs_stride + r] = sumIn(acc, order, tmp[0..p.xs_group]);
                xs_m[g * p.xs_stride + r] = ab;
            }
        }
    }
}

/// Exact PyTorch BF16-rounded FP32 simulation for `sn_mhc_pre_norm`.
pub fn preNormSim(a: std.mem.Allocator, in: PreIn, rows: usize, p: Params, out: PreSimOut) !void {
    const h = p.hidden;
    const two_h = 2 * h;
    const rv = try a.alloc(f32, two_h);
    defer a.free(rv);
    const br = try a.alloc(f32, h);
    defer a.free(br);

    for (0..rows) |r| {
        var ss: f32 = 0;
        for (0..two_h) |j| {
            const v = fromBf16(in.r_in[r * two_h + j]);
            rv[j] = v;
            ss += v * v;
        }
        const inv_rms = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(two_h)) + p.eps);

        var z: [8]f32 = undefined;
        for (0..8) |m| {
            const wm = in.w[m * two_h ..][0..two_h];
            var dot: f32 = 0;
            for (0..two_h) |j| dot += rv[j] * wm[j];
            z[m] = dot * inv_rms + in.b[m];
        }

        const pre0 = 1.0 / (1.0 + @exp(-z[0]));
        const pre1 = 1.0 / (1.0 + @exp(-z[1]));
        const post0 = 2.0 / (1.0 + @exp(-z[2]));
        const post1 = 2.0 / (1.0 + @exp(-z[3]));
        const mix = sinkhorn2x2F32(.{ z[4], z[5], z[6], z[7] }, p.iterations);

        const co = out.coeff[r * coeff_width ..][0..coeff_width];
        co.* = .{ toBf16(post0), toBf16(post1), toBf16(mix[0]), toBf16(mix[1]), toBf16(mix[2]), toBf16(mix[3]) };

        const p0 = roundBf16(pre0);
        const p1 = roundBf16(pre1);
        var ss_b: f32 = 0;
        for (0..h) |d| {
            var bv = roundBf16(p0 * rv[d] + p1 * rv[h + d]);
            if (p.pre_scale != 1.0) bv = roundBf16(p.pre_scale * bv);
            br[d] = bv;
            ss_b += bv * bv;
        }
        const inv_b = 1.0 / @sqrt(ss_b / @as(f32, @floatFromInt(h)) + p.eps);
        for (0..h) |d| {
            const sw = fromBf16(in.sub_w[d]);
            const w_eff = if (p.shift_norm) 1.0 + sw else sw;
            out.norm[r * h + d] = toBf16((br[d] * inv_b) * w_eff);
        }
        if (p.xs_group > 0) {
            const xs = out.xs.?;
            for (0..h / p.xs_group) |g| {
                const base = g * p.xs_group;
                var gs: f32 = 0;
                for (0..p.xs_group) |i| gs += fromBf16(out.norm[r * h + base + i]);
                xs[g * p.xs_stride + r] = gs;
            }
        }
    }
}

/// Float64 reference (and Gate 4 sum simulator) for `sn_mhc_post_merge` (exit or stream carry).
pub fn postMergeRef(
    r_in: []const u16,
    y_in: []const u16,
    coeff: []const u16,
    rows: usize,
    hidden: usize,
    exit: bool,
    acc: Accum,
    order: Order,
    want: []f64,
    mag: []f64,
) void {
    for (0..rows) |r| {
        const co = coeff[r * coeff_width ..][0..coeff_width];
        const post = [2]f64{ bfVal(co[0]), bfVal(co[1]) };
        const mix = [4]f64{ bfVal(co[2]), bfVal(co[3]), bfVal(co[4]), bfVal(co[5]) };
        for (0..hidden) |d| {
            const r0 = bfVal(r_in[(r * 2 + 0) * hidden + d]);
            const r1 = bfVal(r_in[(r * 2 + 1) * hidden + d]);
            const yv = bfVal(y_in[r * hidden + d]);
            var rn: [2]f64 = undefined;
            var rm: [2]f64 = undefined;
            for (0..2) |s| {
                const mt = [2]f64{ mix[s * 2 + 0] * r0, mix[s * 2 + 1] * r1 };
                const mixed = bf64(sumIn(acc, order, &mt));
                const upd = bf64(post[s] * yv);
                const add_t = [2]f64{ mixed, upd };
                rn[s] = sumIn(acc, order, &add_t);
                rm[s] = @abs(mt[0]) + @abs(mt[1]) + @abs(upd);
            }
            if (exit) {
                const b0 = bf64(rn[0]);
                const b1 = bf64(rn[1]);
                const et = [2]f64{ 0.5 * b0, 0.5 * b1 };
                want[r * hidden + d] = sumIn(acc, order, &et);
                mag[r * hidden + d] = 0.5 * (@abs(b0) + @abs(b1)) + 0.5 * (rm[0] + rm[1]);
            } else {
                want[(r * 2 + 0) * hidden + d] = rn[0];
                want[(r * 2 + 1) * hidden + d] = rn[1];
                mag[(r * 2 + 0) * hidden + d] = rm[0];
                mag[(r * 2 + 1) * hidden + d] = rm[1];
            }
        }
    }
}

/// Exact PyTorch BF16-rounded FP32 simulation for `sn_mhc_post_merge`.
pub fn postMergeSim(
    r_in: []const u16,
    y_in: []const u16,
    coeff: []const u16,
    rows: usize,
    hidden: usize,
    exit: bool,
    out: []u16,
) void {
    for (0..rows) |r| {
        const co = coeff[r * coeff_width ..][0..coeff_width];
        const post = [2]f32{ fromBf16(co[0]), fromBf16(co[1]) };
        const mix = [4]f32{ fromBf16(co[2]), fromBf16(co[3]), fromBf16(co[4]), fromBf16(co[5]) };
        for (0..hidden) |d| {
            const r0 = fromBf16(r_in[(r * 2 + 0) * hidden + d]);
            const r1 = fromBf16(r_in[(r * 2 + 1) * hidden + d]);
            const yv = fromBf16(y_in[r * hidden + d]);
            var rn: [2]u16 = undefined;
            for (0..2) |s| {
                const mixed = roundBf16(mix[s * 2 + 0] * r0 + mix[s * 2 + 1] * r1);
                const upd = roundBf16(post[s] * yv);
                rn[s] = toBf16(mixed + upd);
            }
            if (exit) {
                out[r * hidden + d] = toBf16(0.5 * (fromBf16(rn[0]) + fromBf16(rn[1])));
            } else {
                out[(r * 2 + 0) * hidden + d] = rn[0];
                out[(r * 2 + 1) * hidden + d] = rn[1];
            }
        }
    }
}

/// Float64 reference for fused `sn_mhc_post_and_pre_norm`, matching sequential `postMerge` -> `preNorm`.
pub fn postAndPreNormRef(
    a: std.mem.Allocator,
    r_in: []const u16,
    y_in: []const u16,
    coeff_in: []const u16,
    w_next: []const f32,
    b_next: []const f32,
    sub_w_next: []const u16,
    rows: usize,
    p: Params,
    acc: Accum,
    order: Order,
    want_r_next: []f64,
    mag_r_next: []f64,
    out: PreRefOut,
) !void {
    postMergeRef(r_in, y_in, coeff_in, rows, p.hidden, false, acc, order, want_r_next, mag_r_next);
    const r_bf = try a.alloc(u16, rows * 2 * p.hidden);
    defer a.free(r_bf);
    for (r_bf, want_r_next) |*dst, v| dst.* = toBf16(@floatCast(bf64(v)));
    try preNormRef(a, .{ .r_in = r_bf, .w = w_next, .b = b_next, .sub_w = sub_w_next }, rows, p, acc, order, out);
}

/// Exact PyTorch BF16 simulation for fused `sn_mhc_post_and_pre_norm`.
pub fn postAndPreNormSim(
    a: std.mem.Allocator,
    r_in: []const u16,
    y_in: []const u16,
    coeff_in: []const u16,
    w_next: []const f32,
    b_next: []const f32,
    sub_w_next: []const u16,
    rows: usize,
    p: Params,
    r_next_out: []u16,
    out: PreSimOut,
) !void {
    postMergeSim(r_in, y_in, coeff_in, rows, p.hidden, false, r_next_out);
    try preNormSim(a, .{ .r_in = r_next_out, .w = w_next, .b = b_next, .sub_w = sub_w_next }, rows, p, out);
}

test "mhc_ref: Sinkhorn produces doubly stochastic 2x2 matrix and preNorm/postMerge match" {
    const a = std.testing.allocator;
    const h: usize = 64;
    const rows: usize = 2;
    const r_in = try a.alloc(u16, rows * 2 * h);
    defer a.free(r_in);
    const w = try a.alloc(f32, 8 * 2 * h);
    defer a.free(w);
    const b = [_]f32{ -1.0986123, 1.0986123, 0.0, 0.0, 2.639057, 0.0, 0.0, 2.639057 };
    const sub_w = try a.alloc(u16, h);
    defer a.free(sub_w);
    const y_in = try a.alloc(u16, rows * h);
    defer a.free(y_in);

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (r_in) |*v| v.* = toBf16((rng.float(f32) * 2 - 1) * 3.0);
    for (w) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.02;
    for (sub_w) |*v| v.* = toBf16((rng.float(f32) * 2 - 1) * 0.1);
    for (y_in) |*v| v.* = toBf16((rng.float(f32) * 2 - 1) * 2.0);

    const p = Params{ .hidden = h, .iterations = 20, .pre_scale = 0.5, .shift_norm = true, .xs_group = 64, .xs_stride = rows };
    const wc = try a.alloc(f64, rows * coeff_width);
    defer a.free(wc);
    const mc = try a.alloc(f64, rows * coeff_width);
    defer a.free(mc);
    const wn = try a.alloc(f64, rows * h);
    defer a.free(wn);
    const mn = try a.alloc(f64, rows * h);
    defer a.free(mn);
    const wxs = try a.alloc(f64, rows);
    defer a.free(wxs);
    const mxs = try a.alloc(f64, rows);
    defer a.free(mxs);

    const pin = PreIn{ .r_in = r_in, .w = w, .b = &b, .sub_w = sub_w };
    try preNormRef(a, pin, rows, p, .none, .sequential, .{ .coeff = wc, .mag_coeff = mc, .norm = wn, .mag_norm = mn, .xs = wxs, .mag_xs = mxs });
    for (0..rows) |r| {
        const c = wc[r * coeff_width ..];
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), c[2] + c[3], 1e-3);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), c[4] + c[5], 1e-3);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), c[2] + c[4], 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), c[3] + c[5], 1e-12);
    }

    const sc = try a.alloc(u16, rows * coeff_width);
    defer a.free(sc);
    const sn = try a.alloc(u16, rows * h);
    defer a.free(sn);
    const sxs = try a.alloc(f32, rows);
    defer a.free(sxs);
    try preNormSim(a, pin, rows, p, .{ .coeff = sc, .norm = sn, .xs = sxs });
    for (wn, sn) |ref_v, sim_bits| {
        const sim_v: f64 = @as(f64, fromBf16(sim_bits));
        try std.testing.expect(@abs(ref_v - sim_v) <= @abs(ref_v) * 0x1p-6 + 0x1p-12);
    }

    const r_fused = try a.alloc(u16, rows * 2 * h);
    defer a.free(r_fused);
    const c_fused = try a.alloc(u16, rows * coeff_width);
    defer a.free(c_fused);
    const n_fused = try a.alloc(u16, rows * h);
    defer a.free(n_fused);
    const xs_fused = try a.alloc(f32, rows);
    defer a.free(xs_fused);
    try postAndPreNormSim(a, r_in, y_in, sc, w, &b, sub_w, rows, p, r_fused, .{ .coeff = c_fused, .norm = n_fused, .xs = xs_fused });

    const r_seq = try a.alloc(u16, rows * 2 * h);
    defer a.free(r_seq);
    const c_seq = try a.alloc(u16, rows * coeff_width);
    defer a.free(c_seq);
    const n_seq = try a.alloc(u16, rows * h);
    defer a.free(n_seq);
    const xs_seq = try a.alloc(f32, rows);
    defer a.free(xs_seq);
    postMergeSim(r_in, y_in, sc, rows, h, false, r_seq);
    try preNormSim(a, .{ .r_in = r_seq, .w = w, .b = &b, .sub_w = sub_w }, rows, p, .{ .coeff = c_seq, .norm = n_seq, .xs = xs_seq });
    try std.testing.expectEqualSlices(u16, r_seq, r_fused);
    try std.testing.expectEqualSlices(u16, c_seq, c_fused);
    try std.testing.expectEqualSlices(u16, n_seq, n_fused);
    try std.testing.expectEqualSlices(f32, xs_seq, xs_fused);
}
