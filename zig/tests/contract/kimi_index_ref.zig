//! Independent binary64 formulas for the explicitly assumed K3.1 indexer, with simulated reduced sums.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const rows = contract.rows;
pub const heads = 32;
pub const dim = 128;
pub const topk = 2048;

pub fn f(bits: u16) f64 {
    return c.fromBf16(bits);
}

pub fn relu(x: f64) f64 {
    return if (std.math.isNan(x)) x else @max(x, 0);
}

pub const Sum = struct {
    acc: c.Accum = .none,
    order: rows.Order = .sequential,
    pub fn of(s: Sum, terms: []const f64) f64 {
        return rows.sumIn(s.acc, s.order, terms);
    }
};

pub const Score = struct { value: f64, magnitude: f64 };

/// BF16 inputs, exact products and binary64 sums; no FP32 imitation serves as the numeric oracle.
pub fn score(q: []const u16, key_bits: []const u16, weights: []const u16, sum: Sum) Score {
    var hterms: [heads]f64 = undefined;
    var magnitude: f64 = 0;
    for (0..heads) |h| {
        var terms: [dim]f64 = undefined;
        var mag: f64 = 0;
        for (&terms, 0..) |*term, j| {
            term.* = f(q[h * dim + j]) * f(key_bits[j]);
            mag += @abs(term.*);
        }
        hterms[h] = f(weights[h]) * relu(sum.of(&terms)) / 64.0;
        magnitude += @abs(f(weights[h])) * mag / 64.0;
    }
    return .{ .value = sum.of(&hterms), .magnitude = magnitude };
}

/// Biased population LayerNorm without RoPE; the affine operation is performed in binary64.
pub fn key(x: []const u16, w: []const u16, bias: []const u16, sum: Sum, out: []f64, mag: []f64) void {
    var values: [dim]f64 = undefined;
    for (&values, x) |*v, bits| v.* = f(bits);
    const mean = sum.of(&values) / dim;
    var variance_terms: [dim]f64 = undefined;
    for (&variance_terms, values) |*v, xj| v.* = (xj - mean) * (xj - mean);
    const variance = sum.of(&variance_terms) / dim;
    const inv = 1.0 / @sqrt(variance + @as(f64, @as(f32, 1e-5)));
    for (out, mag, values, w, bias) |*o, *m, xj, weight, b| {
        o.* = (xj - mean) * inv * f(weight) + f(b);
        // Sensitivity terms for the mean, centered-square sum, inverse sqrt and affine arithmetic.
        m.* = @abs((xj - mean) * inv * f(weight)) + @abs(f(b)) + @abs(mean * inv * f(weight));
    }
}

/// Numeric descending, NaNs last, signed zero tied, lower positions first. Output order is chronological.
pub fn before(values: []const f32, a: u32, b: u32) bool {
    const x = values[a];
    const y = values[b];
    if (std.math.isNan(x)) return std.math.isNan(y) and a < b;
    if (std.math.isNan(y)) return true;
    return x > y or (x == y and a < b);
}

pub fn selected(a: std.mem.Allocator, values: []const f32, out: []u32) !void {
    if (values.len <= out.len) {
        for (out[0..values.len], 0..) |*v, i| v.* = @intCast(i);
        return;
    }
    const ids = try a.alloc(u32, values.len);
    defer a.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i);
    std.mem.sort(u32, ids, values, before);
    @memcpy(out, ids[0..out.len]);
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
}

/// Absorbed MLA: separate 512 latent and 64 rotary dot products, softmax, then the 512 latent values.
pub fn attend(q: []const u16, qlat: []const f32, cache: []const u16, list: []const u32, scale: f32, out: []f64, mag: []f64) void {
    var max_score = -std.math.inf(f64);
    var invalid = false;
    for (list) |pos| {
        const k = cache[@as(usize, pos) * 576 ..][0..576];
        const value = dot(q, qlat, k) * scale;
        invalid = invalid or !std.math.isFinite(value);
        max_score = @max(max_score, value);
    }
    if (invalid) {
        @memset(out, std.math.nan(f64));
        @memset(mag, std.math.nan(f64));
        return;
    }
    @memset(out, 0);
    @memset(mag, 0);
    var z: f64 = 0;
    for (list) |pos| {
        const k = cache[@as(usize, pos) * 576 ..][0..576];
        const p = @exp(dot(q, qlat, k) * scale - max_score);
        z += p;
        for (out, mag, k[0..512]) |*o, *m, value| {
            o.* += p * f(value);
            m.* += p * @abs(f(value));
        }
    }
    for (out, mag) |*o, *m| {
        o.* /= z;
        m.* /= z;
    }
}

fn dot(q: []const u16, qlat: []const f32, k: []const u16) f64 {
    var sum: f64 = 0;
    for (qlat, k[0..512]) |x, y| sum += @as(f64, x) * f(y);
    for (q[128..192], k[512..576]) |x, y| sum += f(x) * f(y);
    return sum;
}

test "Kimi independent binary64 index formula has the two scale factors and biased key norm" {
    var q: [4096]u16 = @splat(0);
    var key_bits: [128]u16 = @splat(0);
    var weights: [32]u16 = @splat(0);
    q[0] = c.toBf16(8);
    key_bits[0] = c.toBf16(1);
    weights[0] = c.toBf16(2);
    try std.testing.expectEqual(@as(f64, 0.25), score(&q, &key_bits, &weights, .{}).value);
    const norm: [128]u16 = @splat(c.toBf16(1));
    const bias: [128]u16 = @splat(c.toBf16(0.25));
    @memset(&key_bits, 0);
    var out: [128]f64 = undefined;
    var mag: [128]f64 = undefined;
    key(&key_bits, &norm, &bias, .{}, &out, &mag);
    for (out) |v| try std.testing.expectEqual(@as(f64, 0.25), v);
}

test "Kimi index independent comparison sort keeps causal ties and exceptional values" {
    var scores: [2051]f32 = @splat(0);
    scores[0] = std.math.nan(f32);
    scores[1] = -std.math.inf(f32);
    scores[2] = -0.0;
    scores[2049] = std.math.inf(f32);
    scores[2050] = std.math.nan(f32);
    var out: [2048]u32 = undefined;
    try selected(std.testing.allocator, &scores, &out);
    try std.testing.expectEqual(@as(u32, 2), out[0]);
    try std.testing.expectEqual(@as(u32, 2049), out[2047]);
}
