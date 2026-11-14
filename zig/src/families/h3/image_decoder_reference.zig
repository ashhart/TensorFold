//! The image decoder on the host for the self-test: the Metal kernels' bf16 rounding points, float64 sums.
const std = @import("std");
const rounding = @import("bf16_round");
const host = @import("image_decoder.zig");
const vae = @import("picture_vae.zig");

const W = host.width;
const D = host.head_dim;
const F = host.ffn;
const eps = vae.eps;

/// The decoder's bf16 weights, laid out as `plan` says.
pub const Model = struct {
    w: []const u16,
    plan: host.Plan,

    fn head(m: Model, part: host.Head) []const u16 {
        return m.w[m.plan.head(part)..][0..host.count(host.head_parts[@backingInt(part)])];
    }

    fn block(m: Model, i: usize, part: host.Block) []const u16 {
        return m.w[m.plan.block(i, part)..][0..host.count(host.block_parts[@backingInt(part)])];
    }

    fn tail(m: Model, part: host.Tail) []const u16 {
        return m.w[m.plan.tail(part)..][0..host.count(host.tail_parts[@backingInt(part)])];
    }
};

fn val(b: u16) f32 {
    return rounding.value(b);
}

/// fp32 to bf16 and back, as a bf16 tensor op rounds its fp32 result.
fn r32(v: f32) f32 {
    return rounding.round32(v);
}

/// float64 to bf16 in one rounding.
fn r64(v: f64) f32 {
    return @floatCast(rounding.bf16(v));
}

fn dot(a: []const f32, b: []const f32) f64 {
    var acc: @Vector(8, f64) = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) {
        const x: @Vector(8, f32) = a[i..][0..8].*;
        const y: @Vector(8, f32) = b[i..][0..8].*;
        acc += @as(@Vector(8, f64), @floatCast(x)) * @as(@Vector(8, f64), @floatCast(y));
    }
    var s: f64 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += @as(f64, a[i]) * b[i];
    return s;
}

/// y [rows, n] = bf16(x w^T + b) for x [rows, k] bf16-exact and w [n, k], b [n] in bf16; outputs split over threads.
pub fn linear(x: []const f32, w: []const u16, b: []const u16, rows: usize, n: usize, k: usize, y: []f32) !void {
    std.debug.assert(x.len == rows * k and w.len == n * k and b.len == n and y.len == rows * n and k <= 8192);
    const Job = struct {
        fn run(xs: []const f32, ws: []const u16, bs: []const u16, r: usize, nn: usize, kk: usize, ys: []f32, from: usize, to: usize) void {
            var row: [8192]f32 = undefined;
            for (from..to) |j| {
                for (row[0..kk], ws[j * kk ..][0..kk]) |*o, v| o.* = val(v);
                for (0..r) |i| ys[i * nn + j] = r64(dot(xs[i * kk ..][0..kk], row[0..kk]) + val(bs[j]));
            }
        }
    };
    var threads: [16]std.Thread = undefined;
    const workers = @min(threads.len, std.Thread.getCpuCount() catch 1, n);
    var made: usize = 0;
    defer for (threads[0..made]) |t| t.join();
    for (0..workers) |t| {
        const from = n * t / workers;
        const to = n * (t + 1) / workers;
        threads[t] = try std.Thread.spawn(.{}, Job.run, .{ x, w, b, rows, n, k, y, from, to });
        made += 1;
    }
}

/// RMSNorm a row of 1,024: bf16((x * rsqrt(mean(x^2) + eps)) * g), the products in fp32.
fn rms(x: []const f32, g: []const u16, y: []f32) void {
    var s: f64 = 0;
    for (x) |v| s += @as(f64, v) * v;
    const ms: f32 = @floatCast(s / @as(f64, @floatFromInt(x.len)));
    const inv: f32 = @floatCast(1.0 / @sqrt(@as(f64, ms + eps)));
    for (y, x, g) |*o, v, gv| o.* = r32(v * inv * val(gv));
}

/// The first `cells` rows of a tile's sequence: its raw latent [24, h, w] through post_quant_conv and x_embedder.
pub fn embed(m: Model, latent: []const f32, h: usize, w: usize, x: []f32) !void {
    const cells = h * w;
    const n = try host.tokens(h, w);
    if (latent.len != host.channels * cells or x.len != n * W) return error.H3BadLatentLength;
    const pw = m.head(.post_w);
    const pb = m.head(.post_b);
    const ew = m.head(.embed_w);
    const eb = m.head(.embed_b);
    for (0..n) |t| {
        const out = x[t * W ..][0..W];
        switch (try host.token(t, cells)) {
            .cell => |c| {
                var in: [host.channels]f32 = undefined;
                var tok: [host.channels]f32 = undefined;
                for (&in, 0..) |*v, k| v.* = r32(latent[k * cells + c]);
                for (&tok, 0..) |*v, o| {
                    var s: f64 = 0;
                    for (in, 0..) |iv, k| s += @as(f64, val(pw[o * host.channels + k])) * iv;
                    v.* = r64(s + val(pb[o]));
                }
                for (out, 0..) |*v, j| {
                    var s: f64 = 0;
                    for (tok, 0..) |tv, k| s += @as(f64, val(ew[j * host.channels + k])) * tv;
                    v.* = r64(s + val(eb[j]));
                }
            },
            .register => |r| for (out, m.head(.registers)[r * W ..][0..W]) |*v, b| {
                v.* = val(b);
            },
            .zero => @memset(out, 0),
        }
    }
}

/// q and k of one head RMS-normed to bf16 and rotated over their first 32 dims, as the heads kernel does.
fn turn(src: []const f32, cos: []const u16, sin: []const u16, out: []f32) void {
    var s: f64 = 0;
    for (src) |v| s += @as(f64, v) * v;
    const ms: f32 = @floatCast(s / D);
    const inv: f32 = @floatCast(1.0 / @sqrt(@as(f64, ms + eps)));
    var n: [D]f32 = undefined;
    for (&n, src) |*o, v| o.* = r32(v * inv);
    const half = vae.rotated / 2;
    for (0..vae.rotated) |i| {
        const other = if (i < half) -n[i + half] else n[i - half];
        out[i] = r32(r32(n[i] * val(cos[i])) + r32(other * val(sin[i])));
    }
    @memcpy(out[vae.rotated..D], n[vae.rotated..D]);
}

/// Scratch for one tile's layer.
pub const Scratch = struct {
    xn: []f32,
    qkv: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    o: []f32,
    y: []f32,
    wide: []f32,
    act: []f32,

    pub fn init(a: std.mem.Allocator, n: usize) !Scratch {
        var s: Scratch = undefined;
        inline for (.{ .{ "xn", W }, .{ "qkv", 3 * W }, .{ "q", W }, .{ "k", W }, .{ "v", W }, .{ "o", W }, .{ "y", W }, .{ "wide", 2 * F }, .{ "act", F } }) |kv| {
            @field(s, kv[0]) = try a.alloc(f32, n * kv[1]);
        }
        return s;
    }
};

/// Block `layer` on a tile's sequence x [n, 1,024] in place, the rotary table [n, 32] of its tokens.
pub fn layer(m: Model, s: Scratch, index: usize, x: []f32, n: usize, cos: []const u16, sin: []const u16) !void {
    if (x.len != n * W or cos.len != n * vae.rotated or sin.len != cos.len) return error.H3BadLatentLength;
    for (0..n) |t| rms(x[t * W ..][0..W], m.block(index, .norm1), s.xn[t * W ..][0..W]);
    try linear(s.xn[0 .. n * W], m.block(index, .qkv_w), m.block(index, .qkv_b), n, 3 * W, W, s.qkv[0 .. n * 3 * W]);
    // per head: q and k normed and turned, v as it is; stored [head, token, 64]
    for (0..n) |t| for (0..host.heads) |hd| {
        const src = s.qkv[t * 3 * W + hd * 3 * D ..][0 .. 3 * D];
        const at = (hd * n + t) * D;
        const c = cos[t * vae.rotated ..][0..vae.rotated];
        const sn = sin[t * vae.rotated ..][0..vae.rotated];
        turn(src[0..D], c, sn, s.q[at..][0..D]);
        turn(src[D .. 2 * D], c, sn, s.k[at..][0..D]);
        @memcpy(s.v[at..][0..D], src[2 * D .. 3 * D]);
    };
    var p: [host.most_tokens]f64 = undefined;
    for (0..host.heads) |hd| for (0..n) |t| {
        const q = s.q[(hd * n + t) * D ..][0..D];
        var top: f64 = -std.math.inf(f64);
        for (0..n) |j| {
            p[j] = dot(q, s.k[(hd * n + j) * D ..][0..D]) * scale;
            top = @max(top, p[j]);
        }
        var total: f64 = 0;
        for (p[0..n]) |*e| {
            e.* = @exp(e.* - top);
            total += e.*;
        }
        for (0..D) |d| {
            var acc: f64 = 0;
            for (0..n) |j| acc += p[j] * s.v[(hd * n + j) * D + d];
            s.o[t * W + hd * D + d] = r64(acc / total);
        }
    };
    try linear(s.o[0 .. n * W], m.block(index, .out_w), m.block(index, .out_b), n, W, W, s.y[0 .. n * W]);
    residual(x, s.y[0 .. n * W], m.block(index, .scale1));
    for (0..n) |t| rms(x[t * W ..][0..W], m.block(index, .norm2), s.xn[t * W ..][0..W]);
    try linear(s.xn[0 .. n * W], m.block(index, .w1_w), m.block(index, .w1_b), n, 2 * F, W, s.wide[0 .. n * 2 * F]);
    for (0..n) |t| for (0..F) |j| {
        const g = s.wide[t * 2 * F + j];
        const silu = r32(g / (1.0 + @exp(-g)));
        s.act[t * F + j] = r32(silu * s.wide[t * 2 * F + F + j]);
    };
    try linear(s.act[0 .. n * F], m.block(index, .w2_w), m.block(index, .w2_b), n, W, F, s.y[0 .. n * W]);
    residual(x, s.y[0 .. n * W], m.block(index, .scale2));
}

const scale: f64 = host.scale;

/// x = bf16(x + bf16(y * s)) per channel, the LayerScale residual in bf16 elementwise.
fn residual(x: []f32, y: []const f32, s: []const u16) void {
    for (x, y, 0..) |*o, v, i| o.* = r32(o.* + r32(v * val(s[i % W])));
}

/// norm_out (fp32 LayerNorm, rounded to bf16 for proj_out) and proj_out on the first `cells` rows: out [cells, 768].
pub fn final(m: Model, s: Scratch, x: []const f32, cells: usize, out: []f32) !void {
    if (x.len < cells * W or out.len != cells * host.patch) return error.H3BadLatentLength;
    const g = m.tail(.norm_w);
    const b = m.tail(.norm_b);
    for (0..cells) |t| {
        const row = x[t * W ..][0..W];
        var sum: f64 = 0;
        for (row) |v| sum += v;
        const mean = sum / W;
        var sq: f64 = 0;
        for (row) |v| sq += (v - mean) * (v - mean);
        const rstd = 1.0 / @sqrt(sq / W + eps);
        for (s.xn[t * W ..][0..W], row, g, b) |*o, v, gv, bv| o.* = r64((v - mean) * rstd * val(gv) + val(bv));
    }
    try linear(s.xn[0 .. cells * W], m.tail(.proj_w), m.tail(.proj_b), cells, host.patch, W, out);
}

/// A tile's raw latent [24, h, w] at cell (row, col) of a latent [24, rows, cols].
pub fn crop(latent: []const f32, rows: usize, cols: usize, row: usize, col: usize, h: usize, w: usize, out: []f32) void {
    for (0..host.channels) |c| for (0..h) |y| {
        @memcpy(out[(c * h + y) * w ..][0..w], latent[(c * rows + row + y) * cols + col ..][0..w]);
    };
}
