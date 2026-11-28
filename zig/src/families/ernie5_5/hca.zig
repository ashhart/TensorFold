//! ERNIE 5.5's HCA compressor (vLLM's DeepseekCompressor at ratio 128): a block's softmax pool, RMSNorm and RoPE.
const std = @import("std");

/// The HCA head (nope + rope), the tokens a compressed row pools, and the rope dims GPT-J's pairs turn.
pub const head_dim = 512;
pub const ratio = 128;
pub const nope = 448;
pub const rope = 64;

/// rows compressed rows; token rows (kv then score, fp32) ts apart, `ratio` of them a row; outputs os apart.
pub const Shape = struct { rows: usize, ts: usize = 2 * head_dim, os: usize = head_dim };

/// Rows that fit the grid, token rows at least kv and score wide, 8-byte aligned and outputs 4-byte aligned, in i64.
pub fn check(sh: Shape) error{HcaShape}!void {
    if (sh.ts < 2 * head_dim or sh.os < head_dim or sh.ts % 2 != 0 or sh.os % 2 != 0) return error.HcaShape;
    if (sh.rows > std.math.maxInt(i32) / ratio) return error.HcaShape;
    // the kernel's furthest offsets, (128 rows - 1) ts + 1023 and (rows - 1) os + 511 floats, in its i64
    const most: u64 = std.math.maxInt(i64);
    const rows: u64 = @max(sh.rows, 1);
    if (sh.ts > (most - 2 * head_dim) / (rows * ratio) or sh.os > (most - head_dim) / rows) return error.HcaShape;
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

/// The attention's window: a query at position p sees tokens max(0, p - 127) .. p, its own included.
pub const window = 128;

/// The first token of p's window.
pub fn windowStart(p: u64) u64 {
    return p - @min(p, window - 1);
}

/// The compressed rows p sees: every block whose last token, 128 j + 127, is at most p.
pub fn compressedRows(p: u64) u64 {
    return (p + 1) / ratio;
}

/// The keys p attends: its window's rows, then its compressed rows.
pub fn keys(p: u64) u64 {
    return p + 1 - windowStart(p) + compressedRows(p);
}

/// Keys a chunk of the attention's tiles.
pub const chunk = 512;

/// rows query rows of `heads` heads, a row's keys listed `w` wide, nch chunks of them.
pub const Attn = struct { rows: usize, heads: usize, w: usize, nch: usize };

/// Grids that fit, chunks that cover a list's width, and the lists' and partials' offsets in the kernels' i64.
pub fn attnCheck(sh: Attn) error{HcaShape}!void {
    if (sh.heads == 0 or sh.w == 0 or sh.nch == 0 or sh.nch > 65535 or (sh.heads + 15) / 16 > 65535) return error.HcaShape;
    if (sh.nch < (sh.w + chunk - 1) / chunk or sh.rows > std.math.maxInt(i32)) return error.HcaShape;
    const most: u64 = std.math.maxInt(i64);
    const per: u64 = @as(u64, sh.heads) * sh.nch * head_dim;
    if (sh.rows > most / sh.w or sh.rows > most / per) return error.HcaShape;
}

/// One head: e^(s_j - M) k_j summed over e^(sink - M) + e^(s_j - M) summed, s_j = scale <q, k_j>, M their max.
pub fn attend(q: *const [head_dim]f64, kv: []const f64, sink: f64, scale: f64, sum: anytype, t: []f64, out: *[head_dim]f64) void {
    const n = kv.len / head_dim;
    // t: the scores [n], the sink's then the keys' weights [1 + n], then one sum's terms [max(n, 512)]
    const s = t[0..n];
    const e = t[n..][0 .. n + 1];
    const x = t[2 * n + 1 ..];
    for (s, 0..) |*v, j| {
        for (x[0..head_dim], q, kv[j * head_dim ..][0..head_dim]) |*y, a, b| y.* = a * b;
        v.* = scale * sum.of(x[0..head_dim]);
    }
    var top = sink;
    for (s) |v| top = @max(top, v);
    e[0] = @exp(sink - top);
    for (e[1..], s) |*w, v| w.* = @exp(v - top);
    const den = sum.of(e);
    for (out, 0..) |*o, d| {
        for (x[0..n], e[1..], 0..) |*y, w, j| y.* = w * kv[j * head_dim + d];
        o.* = sum.of(x[0..n]) / den;
    }
}

/// VHA's heads and rank (vLLM's VHAPostMix, VHA_RANK): U and V are bf16 [heads][rank].
pub const vha_heads = 64;
pub const vha_rank = 16;

/// One dim of a token's heads, vLLM's TP form: o + V (U^T o), each z_r = sum_h U_hr o_h first, then o_h added last.
pub fn vha(o: *const [vha_heads]f64, u: *const [vha_heads * vha_rank]f64, v: *const [vha_heads * vha_rank]f64, sum: anytype, t: *[vha_heads]f64, out: *[vha_heads]f64) void {
    var z: [vha_rank]f64 = undefined;
    for (&z, 0..) |*zr, r| {
        for (t, o, 0..) |*x, a, i| x.* = u[i * vha_rank + r] * a;
        zr.* = sum.of(t);
    }
    for (out, 0..) |*y, i| {
        for (t[0..vha_rank], &z, 0..) |*x, b, r| x.* = v[i * vha_rank + r] * b;
        t[vha_rank] = o[i];
        y.* = sum.of(t[0 .. vha_rank + 1]);
    }
}

/// rows tokens of d dims: a width in (0, 2^31) and a grid of 256-thread blocks that fits, which keeps offsets in i64.
pub fn vhaCheck(rows: usize, d: usize) error{HcaShape}!void {
    if (d == 0 or d > std.math.maxInt(i32)) return error.HcaShape;
    const cols = std.math.mul(u64, rows, d) catch return error.HcaShape;
    if (cols > @as(u64, std.math.maxInt(i32)) * 256) return error.HcaShape;
}

/// Float64 sums in order.
pub const exact = struct {
    pub fn of(_: @This(), t: []const f64) f64 {
        var s: f64 = 0;
        for (t) |v| s += v;
        return s;
    }
}{};
