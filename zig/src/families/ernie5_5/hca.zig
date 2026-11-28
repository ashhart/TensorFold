//! ERNIE 5.5's HCA compressor (vLLM's DeepseekCompressor at ratio 128): a block's softmax pool, RMSNorm and RoPE.
const std = @import("std");

/// The HCA head (nope + rope), the tokens a compressed row pools, and the rope dims GPT-J's pairs turn.
pub const head_dim = 512;
pub const ratio = 128;
pub const nope = 448;
pub const rope = 64;

/// rows compressed rows; token rows (kv then score, fp32) ts apart, `ratio` of them a row; outputs os apart.
pub const Shape = struct { rows: usize, ts: usize = 2 * head_dim, os: usize = head_dim };

/// Rows that fit the grid, token rows at least kv and score wide, 8-byte aligned and outputs 4-byte aligned.
pub fn check(sh: Shape) error{HcaShape}!void {
    if (sh.ts < 2 * head_dim or sh.os < head_dim or sh.ts % 2 != 0 or sh.os % 2 != 0) return error.HcaShape;
    if (sh.rows > std.math.maxInt(i32) / ratio) return error.HcaShape;
}

/// Pair l's angle as the kernel and vLLM's cos_sin_cache form it: the fp32 product of freq[l] and the fp32 position.
pub fn angle(freq: f32, pos: i64) f64 {
    return freq * @as(f32, @floatFromInt(pos));
}

/// One compressed row from its tokens' kv, score [ratio][head_dim] and ape: the pools, the RMSNorm times w, turned.
pub fn compress(kv: []const f64, score: []const f64, ape: []const f32, w: *const [head_dim]f64, freq: *const [rope / 2]f32, pos: i64, eps: f32, sum: anytype, t: *[2 * ratio]f64, out: *[head_dim]f64) void {
    for (out, 0..) |*o, c| o.* = pool(kv, score, ape, c, sum, t);
    var sq: [head_dim]f64 = undefined;
    for (&sq, out) |*q, v| q.* = v * v;
    const r = 1 / @sqrt(sum.of(&sq) / head_dim + eps);
    for (out, w) |*o, g| o.* = o.* * r * g;
    turn(out, freq, pos);
}

/// Channel c's pool: sum_i kv_i e^(s_i - M) / sum_i e^(s_i - M), s_i = score_i + ape_i, M their max.
pub fn pool(kv: []const f64, score: []const f64, ape: []const f32, c: usize, sum: anytype, t: *[2 * ratio]f64) f64 {
    var top = -std.math.inf(f64);
    for (0..ratio) |i| top = @max(top, score[i * head_dim + c] + ape[i * head_dim + c]);
    const e = t[0..ratio];
    const p = t[ratio..];
    for (e, p, 0..) |*x, *y, i| {
        x.* = @exp(score[i * head_dim + c] + ape[i * head_dim + c] - top);
        y.* = kv[i * head_dim + c] * x.*;
    }
    return sum.of(p) / sum.of(e);
}

/// GPT-J's pairs (nope + 2l, nope + 2l + 1) turned by angle(freq[l], pos).
pub fn turn(out: *[head_dim]f64, freq: *const [rope / 2]f32, pos: i64) void {
    for (freq, 0..) |f, l| {
        const a = angle(f, pos);
        const c = @cos(a);
        const s = @sin(a);
        const x0 = out[nope + 2 * l];
        const x1 = out[nope + 2 * l + 1];
        out[nope + 2 * l] = x0 * c - x1 * s;
        out[nope + 2 * l + 1] = x0 * s + x1 * c;
    }
}

/// Float64 sums in order.
pub const exact = struct {
    pub fn of(_: @This(), t: []const f64) f64 {
        var s: f64 = 0;
        for (t) |v| s += v;
        return s;
    }
}{};
