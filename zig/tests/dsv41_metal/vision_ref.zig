//! The vision tower in float64 from the math alone: the ViT's stages and the aligner, over the release's bf16 weights.
const std = @import("std");
const host = @import("dsv41");

pub const Block = struct { n1: []f64, wqkv: []f64, bqkv: []f64, wo: []f64, bo: []f64, n2: []f64, w1: []f64, w2: []f64 };

pub const Model = struct {
    a: std.mem.Allocator,
    layers: u32,
    dim: u32,
    heads: u32,
    inter: u32,
    patch_values: u32,
    ratio: u32,
    out_dim: u32,
    theta: f64,
    eps: f64,
    pw: []f64,
    pb: []f64,
    blocks: []Block,
    norm: []f64,
    a1: []f64,
    ab1: []f64,
    a2: []f64,
    ab2: []f64,

    /// Every tensor of the pack as float64 (the bf16 values widened exactly).
    pub fn load(a: std.mem.Allocator, io: std.Io, pack: *const host.pack.Pack, cfg: anytype) !Model {
        var m: Model = .{ .a = a, .layers = cfg.layers, .dim = cfg.dim, .heads = cfg.heads, .inter = cfg.inter, .patch_values = cfg.patchValues(), .ratio = cfg.ratio, .out_dim = cfg.out_dim, .theta = cfg.theta, .eps = cfg.eps, .pw = undefined, .pb = undefined, .blocks = undefined, .norm = undefined, .a1 = undefined, .ab1 = undefined, .a2 = undefined, .ab2 = undefined };
        m.pw = try read(a, io, pack, "vision.patch_embed.proj.weight");
        m.pb = try read(a, io, pack, "vision.patch_embed.proj.bias");
        m.blocks = try a.alloc(Block, cfg.layers);
        var buf: [96]u8 = undefined;
        for (m.blocks, 0..) |*b, i| {
            const f = struct {
                fn r(al: std.mem.Allocator, io_: std.Io, p: *const host.pack.Pack, bb: []u8, k: usize, s: []const u8) ![]f64 {
                    return read(al, io_, p, try std.fmt.bufPrint(bb, "vision.blocks.{d}.{s}", .{ k, s }));
                }
            }.r;
            b.* = .{ .n1 = try f(a, io, pack, &buf, i, "norm1.weight"), .wqkv = try f(a, io, pack, &buf, i, "attn.wqkv.weight"), .bqkv = try f(a, io, pack, &buf, i, "attn.wqkv.bias"), .wo = try f(a, io, pack, &buf, i, "attn.wo.weight"), .bo = try f(a, io, pack, &buf, i, "attn.wo.bias"), .n2 = try f(a, io, pack, &buf, i, "norm2.weight"), .w1 = try f(a, io, pack, &buf, i, "mlp.w1.weight"), .w2 = try f(a, io, pack, &buf, i, "mlp.w2.weight") };
        }
        m.norm = try read(a, io, pack, "vision.norm.weight");
        m.a1 = try read(a, io, pack, "aligner.w1.weight");
        m.ab1 = try read(a, io, pack, "aligner.w1.bias");
        m.a2 = try read(a, io, pack, "aligner.w2.weight");
        m.ab2 = try read(a, io, pack, "aligner.w2.bias");
        return m;
    }

    fn read(a: std.mem.Allocator, io: std.Io, pack: *const host.pack.Pack, name: []const u8) ![]f64 {
        const info = try pack.need(name);
        const raw = try a.alloc(u8, @intCast(info.nbytes));
        defer a.free(raw);
        try pack.read(io, info, .{ .offset = 0, .len = info.nbytes }, raw);
        const out = try a.alloc(f64, raw.len / 2);
        for (out, 0..) |*o, i| o.* = @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16));
        return out;
    }

    pub fn deinit(m: *Model) void {
        for (m.blocks) |b| for ([_][]f64{ b.n1, b.wqkv, b.bqkv, b.wo, b.bo, b.n2, b.w1, b.w2 }) |s| m.a.free(s);
        m.a.free(m.blocks);
        for ([_][]f64{ m.pw, m.pb, m.norm, m.a1, m.ab1, m.a2, m.ab2 }) |s| m.a.free(s);
    }
};

/// y [rows][n] = x [rows][k] w^T + bias.
fn linear(y: []f64, x: []const f64, w: []const f64, bias: ?[]const f64, rows: usize, k: usize, n: usize) void {
    for (0..rows) |r| for (0..n) |c| {
        var s: f64 = if (bias) |b| b[c] else 0;
        for (0..k) |i| s += x[r * k + i] * w[c * k + i];
        y[r * n + c] = s;
    };
}

fn rms(y: []f64, x: []const f64, w: []const f64, rows: usize, d: usize, eps: f64) void {
    for (0..rows) |r| {
        var s: f64 = 0;
        for (x[r * d ..][0..d]) |v| s += v * v;
        const inv = 1.0 / @sqrt(s / @as(f64, @floatFromInt(d)) + eps);
        for (0..d) |i| y[r * d + i] = w[i] * x[r * d + i] * inv;
    }
}

/// erf by its series below |x| = 4 (about 1e-9 absolute there) and the complementary asymptotic series past it.
pub fn erf(x: f64) f64 {
    const ax = @abs(x);
    if (ax >= 6) return std.math.sign(x);
    if (ax < 4) {
        var term = ax;
        var sum = ax;
        var n: f64 = 0;
        while (@abs(term) > 1e-18 * @abs(sum)) {
            n += 1;
            term *= -ax * ax / n;
            sum += term / (2 * n + 1);
        }
        return std.math.sign(x) * sum * 2 / @sqrt(std.math.pi);
    }
    // erfc(x) = exp(-x^2) / (x sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4) - ...), 14 terms
    var t: f64 = 1;
    var s: f64 = 1;
    for (1..14) |i| {
        t *= -@as(f64, @floatFromInt(2 * i - 1)) / (2 * ax * ax);
        s += t;
    }
    return std.math.sign(x) * (1 - @exp(-ax * ax) / (ax * @sqrt(std.math.pi)) * s);
}

pub fn gelu(x: f64) f64 {
    return 0.5 * x * (1 + erf(x * 0.7071067811865476));
}

/// The residual stream after the patch embedding: [n][dim].
pub fn embed(a: std.mem.Allocator, m: *const Model, patches: []const f64, n: usize) ![]f64 {
    const x = try a.alloc(f64, n * m.dim);
    linear(x, patches, m.pw, m.pb, n, m.patch_values, m.dim);
    return x;
}

/// The residual stream after the patch embedding and the first `blocks` blocks: [n][dim].
pub fn stream(a: std.mem.Allocator, m: *const Model, patches: []const f64, gh: usize, gw: usize, blocks: u32) ![]f64 {
    const x = try embed(a, m, patches, gh * gw);
    try advance(a, m, x, gh, gw, 0, blocks);
    return x;
}

/// Blocks [from, to) over the stream `x` in place.
pub fn advance(a: std.mem.Allocator, m: *const Model, x: []f64, gh: usize, gw: usize, from: u32, to: u32) !void {
    const n = gh * gw;
    const D: usize = m.dim;
    const hd = D / m.heads;
    const xn = try a.alloc(f64, n * D);
    defer a.free(xn);
    const qkv = try a.alloc(f64, n * 3 * D);
    defer a.free(qkv);
    const o = try a.alloc(f64, n * D);
    defer a.free(o);
    const t = try a.alloc(f64, n * D);
    defer a.free(t);
    const g = try a.alloc(f64, n * 2 * m.inter);
    defer a.free(g);
    const h = try a.alloc(f64, n * m.inter);
    defer a.free(h);
    const scores = try a.alloc(f64, n);
    defer a.free(scores);
    for (m.blocks[from..to]) |b| {
        rms(xn, x, b.n1, n, D, m.eps);
        linear(qkv, xn, b.wqkv, b.bqkv, n, D, 3 * D);
        // 2D RoPE on q and k: the first quarter of the pairs by the row, the rest by the column
        const half = hd / 2;
        const quarter = half / 2;
        for (0..n) |p| for (0..m.heads) |head| for ([_]usize{ 0, 1 }) |part| for (0..half) |j| {
            const pos: f64 = @floatFromInt(if (j < quarter) p / gw else p % gw);
            const inv = 1.0 / std.math.pow(f64, m.theta, @as(f64, @floatFromInt(2 * (j % quarter))) / @as(f64, @floatFromInt(half)));
            const cs = @cos(pos * inv);
            const sn = @sin(pos * inv);
            const at = p * 3 * D + part * D + head * hd + j;
            const x1 = qkv[at];
            const x2 = qkv[at + half];
            qkv[at] = x1 * cs - x2 * sn;
            qkv[at + half] = x2 * cs + x1 * sn;
        };
        for (0..n) |p| for (0..m.heads) |head| {
            const q = qkv[p * 3 * D + head * hd ..][0..hd];
            var mx: f64 = -std.math.inf(f64);
            for (0..n) |j| {
                const k = qkv[j * 3 * D + D + head * hd ..][0..hd];
                var s: f64 = 0;
                for (q, k) |u, v| s += u * v;
                scores[j] = s / 8.0;
                mx = @max(mx, scores[j]);
            }
            var sum: f64 = 0;
            for (scores) |*s| {
                s.* = @exp(s.* - mx);
                sum += s.*;
            }
            for (0..hd) |d| {
                var acc: f64 = 0;
                for (0..n) |j| acc += scores[j] * qkv[j * 3 * D + 2 * D + head * hd + d];
                o[p * D + head * hd + d] = acc / sum;
            }
        };
        linear(t, o, b.wo, b.bo, n, D, D);
        for (x, t) |*xv, tv| xv.* += tv;
        rms(xn, x, b.n2, n, D, m.eps);
        linear(g, xn, b.w1, null, n, D, 2 * m.inter);
        for (0..n) |p| for (0..m.inter) |c| {
            const gt = g[p * 2 * m.inter + c];
            h[p * m.inter + c] = gt / (1 + @exp(-gt)) * g[p * 2 * m.inter + m.inter + c];
        };
        linear(t, h, b.w2, null, n, m.inter, D);
        for (x, t) |*xv, tv| xv.* += tv;
    }
}

/// The aligner's rows [lh * lw][out_dim] of a residual stream `x` after the last block: norm, unshuffle, w1, GELU, w2.
pub fn aligner(a: std.mem.Allocator, m: *const Model, x: []const f64, gh: usize, gw: usize) ![]f64 {
    const n = gh * gw;
    const D: usize = m.dim;
    const r: usize = m.ratio;
    const lh = (gh + r - 1) / r;
    const lw = (gw + r - 1) / r;
    const xn = try a.alloc(f64, n * D);
    defer a.free(xn);
    rms(xn, x, m.norm, n, D, m.eps);
    const kin = D * r * r;
    const u = try a.alloc(f64, lh * lw * kin);
    defer a.free(u);
    for (0..lh * lw) |l| for (0..D) |c| for (0..r * r) |k| {
        const py = (l / lw) * r + k / r;
        const px = (l % lw) * r + k % r;
        u[l * kin + c * r * r + k] = if (py < gh and px < gw) xn[(py * gw + px) * D + c] else 0;
    };
    const A: usize = m.out_dim;
    const t = try a.alloc(f64, lh * lw * A);
    defer a.free(t);
    linear(t, u, m.a1, m.ab1, lh * lw, kin, A);
    for (t) |*v| v.* = gelu(v.*);
    const out = try a.alloc(f64, lh * lw * A);
    linear(out, t, m.a2, m.ab2, lh * lw, A, A);
    return out;
}

test "erf: the series and the asymptotic branch meet, and known values hold" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.8427007929497149), erf(1), 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9999779095030014), erf(3), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, -0.9999999845827421), erf(-4), 1e-9);
    try std.testing.expectApproxEqAbs(erf(3.9999999), erf(4.0000001), 1e-8);
}
