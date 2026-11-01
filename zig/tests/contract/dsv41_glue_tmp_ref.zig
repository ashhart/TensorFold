//! Float64 oracles for DeepSeek V4.1's phase-1 glue beyond the rows family: the FP4 packings, pool, cand and hc mixes.
const std = @import("std");
const contract = @import("contract");
pub const base = @import("dsv41_glue_ref.zig");

const c = contract.spec;
const rows = contract.rows;

pub const e2m1 = base.e2m1;
pub const e4m3 = base.e4m3;
pub const nearest = base.nearest;
pub const ceilLog2 = base.ceilLog2;

/// Metal's fmax and fmin: a NaN operand is ignored, so a NaN never wins a maximum.
fn fmax(a: f32, b: f32) f32 {
    return if (std.math.isNan(a)) b else if (std.math.isNan(b)) a else @max(a, b);
}

fn fmin(a: f32, b: f32) f32 {
    return if (std.math.isNan(a)) b else if (std.math.isNan(b)) a else @min(a, b);
}

/// The kernel's tf_g_pow2: 2^e from the exponent bits, wrapping, so e = -128 gives -inf.
fn pow2i(e: i32) f32 {
    return @bitCast(@as(u32, @bitCast(e +% 127)) << 23);
}

/// The kernel's tf_g_log2c: ceil(log2(v)) of a positive float from its bits.
fn log2c(v: f32) i32 {
    const b: u32 = @bitCast(v);
    return @as(i32, @intCast((b >> 23) & 0xFF)) - 127 + @intFromBool((b & 0x7FFFFF) != 0);
}

/// The kernel's tf_g_e2m1: a magnitude of at most 6 as its nearest code, ties to the even code.
fn e2m1Code(m: f32) u8 {
    if (m <= 0.25) return 0;
    if (m < 0.75) return 1;
    if (m <= 1.25) return 2;
    if (m < 1.75) return 3;
    if (m <= 2.5) return 4;
    if (m < 3.5) return 5;
    if (m <= 5.0) return 6;
    return 7;
}

/// The kernel's tf_g_e4m3: a magnitude of at most 448 as its nearest code, ties to even.
fn e4m3Code(a: f32) u32 {
    const b: u32 = @bitCast(a);
    const r = b +% 0x7FFFF +% ((b >> 20) & 1);
    return if (a < 0.015625) @intFromFloat(@round(a * 512.0)) else (((r >> 23) - 120) << 3) | ((r >> 20) & 7);
}

/// The kernel's tf_g_e4m3v: the value of a 7-bit E4M3 code.
fn e4m3Val(cd: u32) f32 {
    return @floatCast(e4m3[cd]);
}

/// One 32-value block of fp4pack: sixteen code bytes (low nibble first) and the E8M0 scale byte.
pub fn packBlock(x: []const u16, codes: *[16]u8) u8 {
    var v: [32]f32 = undefined;
    var amax: f32 = 0;
    for (x, 0..) |h, j| {
        v[j] = c.fromBf16(h);
        amax = fmax(amax, @abs(v[j]));
    }
    const e = log2c(fmax(amax, 6 * pow2i(-126)) * @as(f32, 1.0 / 6.0));
    const s = pow2i(-e);
    for (0..16) |j| {
        var byte: u8 = 0;
        for (0..2) |k| {
            const q = e2m1Code(fmin(@abs(v[2 * j + k]) * s, 6)) | @as(u8, if (std.math.signbit(v[2 * j + k])) 8 else 0);
            byte |= q << @intCast(4 * k);
        }
        codes[j] = byte;
    }
    return @truncate(@as(u32, @bitCast(e + 127)));
}

/// One 16-value block of fp4rt: E2M1 over an E4M3 scale, back to bf16 bits.
pub fn rtBlock(x: []const u16, out: *[16]u16) void {
    var v: [16]f32 = undefined;
    var amax: f32 = 0;
    for (x, 0..) |h, j| {
        v[j] = c.fromBf16(h);
        amax = fmax(amax, @abs(v[j]));
    }
    const s = e4m3Val(e4m3Code(fmin(fmax(amax, 6 * pow2i(-9)) / 6, 448)));
    for (v, out) |a, *o| {
        const q: f32 = @as(f32, @floatCast(e2m1[e2m1Code(fmin(@abs(a / s), 6))])) * s;
        o.* = c.toBf16(if (std.math.signbit(a)) -q else q);
    }
}

/// A value and the term magnitude its bound is scaled by.
pub const Val = struct { v: f64, mag: f64 };

/// The pool's softmax over two slots, as the kernel's precise::exp and divides give it.
pub fn pool(k: [2]f64, s: [2]f64) Val {
    const m = @max(s[0], s[1]);
    const e0 = @exp(s[0] - m);
    const e1 = @exp(s[1] - m);
    const sum = e0 + e1;
    return .{ .v = (k[0] * e0 + k[1] * e1) / sum, .mag = (@abs(k[0] * e0) + @abs(k[1] * e1)) / sum };
}

/// A row's block maxima and its top-`topk` mask: the last visible block pinned, ties to the lower block.
pub fn cand(scores: []const f32, visible: i32, blocks: usize, topk: usize, bm: []f32, mask: []u8) void {
    const inf = std.math.inf(f32);
    const last: i64 = if (visible - 1 >= 0) @divTrunc(visible - 1, 8) else -1;
    for (0..blocks) |j| {
        var m: f32 = -inf;
        if (8 * j < scores.len) for (scores[8 * j .. @min(8 * j + 8, scores.len)]) |v| {
            if (!std.math.isNan(v)) m = @max(m, v);
        };
        bm[j] = if (@as(i64, @intCast(j)) == last) inf else m;
    }
    for (0..blocks) |j| {
        const sj = bm[j];
        var rank: usize = 0;
        for (0..blocks) |i| if (bm[i] > sj or (bm[i] == sj and i < j)) {
            rank += 1;
        };
        mask[j] = if (sj > -inf and rank < topk) 1 else 0;
    }
}

pub const MixIn = struct {
    x: []const u16, // the row's stream [dim]
    fnw: []const f32, // hc_fn [24][dim]
    scale: [3]f32,
    base: []const f32, // [24]
    iters: u32,
    eps: f64,
    hc_eps: f64,
    alpha: f64,
};

pub const MixOut = struct {
    pre: [4]f64,
    post: [4]f64,
    comb: [16]f64,
    z: [24]f64,
    mpre: [24]f64,
    mpost: [4]f64,
    mcomb: [16]f64,
};

/// hc_mixes then hc_split_sinkhorn of one row; `scratch` holds dim terms, each mag is its bound's scale.
pub fn mix(i: MixIn, acc: c.Accum, order: rows.Order, scratch: []f64) MixOut {
    const dim = i.x.len;
    const u: f64 = 0x1p-24;
    for (scratch[0..dim], i.x) |*o, h| {
        const v: f64 = c.fromBf16(h);
        o.* = v * v;
    }
    const inv = 1 / @sqrt(rows.sumIn(acc, order, scratch[0..dim]) / @as(f64, @floatFromInt(dim)) + i.eps);
    var o: MixOut = undefined;
    for (0..24) |m| {
        const f = i.fnw[m * dim ..][0..dim];
        var abs: f64 = 0;
        for (scratch[0..dim], f, i.x) |*t, w, h| {
            t.* = @as(f64, w) * c.fromBf16(h);
            abs += @abs(t.*);
        }
        const sc: f64 = @as(f64, i.scale[@min(m / 4, 2)]);
        o.z[m] = rows.sumIn(acc, order, scratch[0..dim]) * inv * sc + @as(f64, i.base[m]);
        o.mpre[m] = abs * @abs(inv) * sc + @abs(o.z[m]) * 2 * u;
    }
    for (0..4) |j| {
        o.pre[j] = 1 / (1 + @exp(-o.z[j])) + i.hc_eps;
        o.mpre[j] += (@abs(o.z[j]) + 4) * @abs(o.pre[j]) * u;
        o.post[j] = i.alpha / (1 + @exp(-o.z[4 + j]));
        o.mpost[j] = o.mpre[4 + j] + (@abs(o.z[4 + j]) + 3) * @abs(o.post[j]) * u;
    }
    sinkhorn(o.z[8..24].*, i.iters, i.hc_eps, &o.comb);
    const steps: f64 = @as(f64, @floatFromInt(i.iters)) * 6 + 8;
    for (0..16) |q| o.mcomb[q] = o.mpre[8 + q] * (@as(f64, @floatFromInt(i.iters)) * 4 + 6) + @abs(o.comb[q]) * (steps + @abs(o.z[8 + q])) * u;
    return o;
}

/// kernel.py's hc_split_sinkhorn order: a row softmax with +eps, a column pass, then iters - 1 (row, column) passes.
fn sinkhorn(z: [16]f64, iters: u32, eps: f64, out: *[16]f64) void {
    for (0..4) |j| {
        const mx = @max(@max(z[4 * j], z[4 * j + 1]), @max(z[4 * j + 2], z[4 * j + 3]));
        var s: f64 = 0;
        for (0..4) |k| {
            out[4 * j + k] = @exp(z[4 * j + k] - mx);
            s += out[4 * j + k];
        }
        for (0..4) |k| out[4 * j + k] = out[4 * j + k] / s + eps;
    }
    for (0..4) |k| {
        const s = ((out[k] + out[4 + k]) + out[8 + k]) + out[12 + k];
        for (0..4) |j| out[4 * j + k] = out[4 * j + k] / (s + eps);
    }
    for (1..iters) |_| {
        for (0..4) |j| {
            const s = ((out[4 * j] + out[4 * j + 1]) + out[4 * j + 2]) + out[4 * j + 3];
            for (0..4) |k| out[4 * j + k] = out[4 * j + k] / (s + eps);
        }
        for (0..4) |k| {
            const s = ((out[k] + out[4 + k]) + out[8 + k]) + out[12 + k];
            for (0..4) |j| out[4 * j + k] = out[4 * j + k] / (s + eps);
        }
    }
}

/// hc_pre of one element: the four streams through the row's pre mix, added in stream order, one round to bf16.
pub fn pre(p: []const f32, x: []const u16, dim: usize, q: usize, acc: c.Accum, order: rows.Order) Val {
    var t: [4]f64 = undefined;
    var abs: f64 = 0;
    for (0..4) |j| {
        t[j] = @as(f64, p[j]) * c.fromBf16(x[j * dim + q]);
        abs += @abs(t[j]);
    }
    return .{ .v = rows.sumIn(acc, order, &t), .mag = abs };
}

/// hc_post of one element: stream k sums comb[j][k] * res_j in j order, then post[k] * branch, one round to bf16.
pub fn post(comb: []const f32, postw: []const f32, res: []const u16, branch: []const u16, dim: usize, k: usize, q: usize, acc: c.Accum, order: rows.Order) Val {
    var t: [5]f64 = undefined;
    var abs: f64 = 0;
    for (0..4) |j| {
        t[j] = @as(f64, comb[4 * j + k]) * c.fromBf16(res[j * dim + q]);
        abs += @abs(t[j]);
    }
    t[4] = @as(f64, postw[k]) * c.fromBf16(branch[q]);
    abs += @abs(t[4]);
    return .{ .v = rows.sumIn(acc, order, &t), .mag = abs };
}
