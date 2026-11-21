//! Independent float64 formulas of Inkling 2's attention inputs, written from transformers' modeling_inkling2
//! (Inkling2ScaledL2Clip, Inkling2L2Clip, Inkling2UnweightedRMSNorm, and the log scaling in Inkling2Attention).
//! They read whole [rows][heads][dim] arrays at absolute positions: no ring, no slot and no chunk exists here.
//! Nothing is rounded to bf16: the probes' bounds carry the roundings (spec.mids counts the one before tau's product).
const std = @import("std");
const contract = @import("contract");
const g = contract.gates;
const c = contract.spec;

pub const qk_threshold: f64 = 22.627416997969522;
pub const rel_threshold: f64 = 8.0;

/// Log scaling of the full-attention layers: tau = 1 + alpha * log(max((pos + 1) / floor, 1)), `pos` absolute.
pub const Tau = struct { alpha: f64, floor: f64, pos0: usize };

pub fn tau(t: ?Tau, row: usize) f64 {
    const s = t orelse return 1;
    const pos: f64 = @floatFromInt(s.pos0 + row);
    return 1 + s.alpha * @log(@max((pos + 1) / s.floor, 1.0));
}

/// torch: (threshold / (norm + 1e-6)).clamp(max=1.0). clamp keeps a NaN, where Zig's @min would drop it.
pub fn clipFactor(sum_squares: f64, threshold: f64) f64 {
    const f = threshold / (@sqrt(sum_squares) + 1e-6);
    return if (std.math.isNan(f)) f else @min(f, 1.0);
}

/// 1 / sqrt(mean(v^2) + eps) from the sum of squares of `n` values (torch.rsqrt).
pub fn rsqrtMean(sum_squares: f64, n: usize, eps: f64) f64 {
    return 1 / @sqrt(sum_squares / @as(f64, @floatFromInt(n)) + eps);
}

fn magnitude(v: f64) f64 {
    return if (std.math.isFinite(v)) @abs(v) else 0;
}

/// Inkling2ScaledL2Clip then, with `t`, the context scaling: x [rows][heads][dim], scale [heads][dim].
/// out = (x * scale) * min(threshold / (|x * scale| + 1e-6), 1) * tau(pos0 + row). `mag` is |out|.
pub fn scaledClip(x: []const f64, scale: []const f64, rows: usize, heads: usize, dim: usize, threshold: f64, t: ?Tau, out: []f64, mag: []f64) void {
    for (0..rows) |r| for (0..heads) |h| {
        const at = (r * heads + h) * dim;
        var ss: f64 = 0;
        for (0..dim) |i| {
            const v = x[at + i] * scale[h * dim + i];
            ss += v * v;
        }
        const factor = clipFactor(ss, threshold) * tau(t, r);
        for (0..dim) |i| {
            out[at + i] = x[at + i] * scale[h * dim + i] * factor;
            mag[at + i] = magnitude(out[at + i]);
        }
    };
}

/// The same with the sum of squares replayed in `acc` (gate 4): one head's squares, summed in `order`.
pub fn scaledClipIn(acc: c.Accum, order: g.Order, x: []const f64, scale: []const f64, rows: usize, heads: usize, dim: usize, threshold: f64, t: ?Tau, out: []f64) void {
    var terms: [256]f64 = undefined;
    for (0..rows) |r| for (0..heads) |h| {
        const at = (r * heads + h) * dim;
        for (0..dim) |i| {
            const v = x[at + i] * scale[h * dim + i];
            terms[i] = v * v;
        }
        const factor = clipFactor(contract.rows.sumIn(acc, order, terms[0..dim]), threshold) * tau(t, r);
        for (0..dim) |i| out[at + i] = x[at + i] * scale[h * dim + i] * factor;
    };
}

/// Key of one position: Inkling2L2Clip (no scale) of k[heads][dim]; value: Inkling2UnweightedRMSNorm of v[heads][dim].
pub fn kvRow(k: []const f64, v: []const f64, heads: usize, dim: usize, threshold: f64, eps: f64, key: []f64, value: []f64, key_mag: []f64, value_mag: []f64) void {
    for (0..heads) |h| {
        const at = h * dim;
        var kss: f64 = 0;
        var vss: f64 = 0;
        for (0..dim) |i| {
            kss += k[at + i] * k[at + i];
            vss += v[at + i] * v[at + i];
        }
        const factor = clipFactor(kss, threshold);
        const inv = rsqrtMean(vss, dim, eps);
        for (0..dim) |i| {
            key[at + i] = k[at + i] * factor;
            value[at + i] = v[at + i] * inv;
            key_mag[at + i] = magnitude(key[at + i]);
            value_mag[at + i] = magnitude(value[at + i]);
        }
    }
}

pub fn kvRowIn(acc: c.Accum, order: g.Order, k: []const f64, v: []const f64, heads: usize, dim: usize, threshold: f64, eps: f64, key: []f64, value: []f64) void {
    var kt: [256]f64 = undefined;
    var vt: [256]f64 = undefined;
    for (0..heads) |h| {
        const at = h * dim;
        for (0..dim) |i| {
            kt[i] = k[at + i] * k[at + i];
            vt[i] = v[at + i] * v[at + i];
        }
        const factor = clipFactor(contract.rows.sumIn(acc, order, kt[0..dim]), threshold);
        const inv = rsqrtMean(contract.rows.sumIn(acc, order, vt[0..dim]), dim, eps);
        for (0..dim) |i| {
            key[at + i] = k[at + i] * factor;
            value[at + i] = v[at + i] * inv;
        }
    }
}

test "a short vector is unchanged and a long one lands on the threshold" {
    const x = [_]f64{ 3, 4 };
    const scale = [_]f64{ 1, 1 };
    var out: [2]f64 = undefined;
    var mag: [2]f64 = undefined;
    scaledClip(&x, &scale, 1, 1, 2, 10, null, &out, &mag);
    try std.testing.expectApproxEqAbs(@as(f64, 3), out[0], 1e-12);
    scaledClip(&x, &scale, 1, 1, 2, 2.5, null, &out, &mag);
    try std.testing.expectApproxEqAbs(2.5, @sqrt(out[0] * out[0] + out[1] * out[1]), 1e-5);
}

test "an infinity clips to zero and itself to NaN; a NaN poisons the head" {
    const x = [_]f64{ std.math.inf(f64), 1, 2, std.math.nan(f64) };
    const scale = [_]f64{ 1, 1 };
    var out: [4]f64 = undefined;
    var mag: [4]f64 = undefined;
    scaledClip(&x, &scale, 2, 1, 2, 10, null, &out, &mag);
    try std.testing.expect(std.math.isNan(out[0]));
    try std.testing.expectEqual(@as(f64, 0), out[1]);
    try std.testing.expect(std.math.isNan(out[2]) and std.math.isNan(out[3]));
}

test "tau is 1 below the floor" {
    const t = Tau{ .alpha = 0.1, .floor = 16, .pos0 = 0 };
    try std.testing.expectEqual(@as(f64, 1), tau(t, 14));
    try std.testing.expectApproxEqAbs(1 + 0.1 * @log(2.0), tau(t, 31), 1e-12);
    try std.testing.expectEqual(@as(f64, 1), tau(null, 99));
}
