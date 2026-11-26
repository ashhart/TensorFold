//! ERNIE 5.5's indexer keys (vLLM-ERNIE's Ernie5_5Indexer): k's LayerNorm, then NeoX RoPE on its first 64 dims.
const std = @import("std");

/// index_head_dim, the roped dims (mla_qk_rope_head_dim) and k_norm's epsilon (LayerNorm(128, eps=1e-6) in its code).
pub const head_dim = 128;
pub const rope = 64;
pub const eps: f32 = 1e-6;

/// rows keys, k rows ks apart in and out rows os apart, row r at position pos0 + r (below 2^24: exact in fp32).
pub const Shape = struct { rows: usize, ks: usize = head_dim, os: usize = head_dim };

/// Rows that fit the grid and rows at least a key wide.
pub fn check(sh: Shape) error{KeyShape}!void {
    if (sh.ks < head_dim or sh.os < head_dim or sh.rows > std.math.maxInt(i32)) return error.KeyShape;
}

/// Pair i's angle as the kernel and vLLM form it: the fp32 product of inv[i] and the fp32 position.
pub fn angle(inv: f32, pos: i64) f64 {
    return inv * @as(f32, @floatFromInt(pos));
}

/// LayerNorm's mean and 1 / sqrt(var + eps), each sum as `sum.of` gives it.
pub const Norm = struct { mean: f64, rstd: f64 };

pub fn norm(x: *const [head_dim]f64, sum: anytype) Norm {
    const mean = sum.of(x) / head_dim;
    var sq: [head_dim]f64 = undefined;
    for (&sq, x) |*t, v| t.* = (v - mean) * (v - mean);
    return .{ .mean = mean, .rstd = 1 / @sqrt(sum.of(&sq) / head_dim + eps) };
}

/// Float64 sums in order.
pub const exact = struct {
    pub fn of(_: @This(), t: []const f64) f64 {
        var s: f64 = 0;
        for (t) |v| s += v;
        return s;
    }
}{};

/// One key: y = (x - mean) rstd w + b, then for i < 32, (y[i], y[i + 32]) turned by angle(inv[i], pos).
pub fn key(x: *const [head_dim]f64, w: *const [head_dim]f64, b: *const [head_dim]f64, inv: *const [rope / 2]f32, pos: i64, sum: anytype, out: *[head_dim]f64) void {
    const n = norm(x, sum);
    for (out, x, w, b) |*y, v, g, h| y.* = (v - n.mean) * n.rstd * g + h;
    for (inv, 0..) |f, i| {
        const a = angle(f, pos);
        const c = @cos(a);
        const s = @sin(a);
        const lo = out[i];
        const hi = out[i + rope / 2];
        out[i] = lo * c - hi * s;
        out[i + rope / 2] = hi * c + lo * s;
    }
}
