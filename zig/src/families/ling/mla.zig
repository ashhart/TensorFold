//! Ling MLA mixer after the linears: unabsorbed python-style (ling_forward.py mla), plus RoPE helpers.
const std = @import("std");
const config = @import("config.zig");
const bf16_round = @import("bf16_round");

pub const lat: u32 = 512;
pub const rope_dim: u32 = 64;
pub const nope: u32 = 128;
pub const vd: u32 = 128;
pub const key: u32 = lat + rope_dim;
pub const qd: u32 = nope + rope_dim;
pub const nfreq: u32 = rope_dim / 2;
pub const theta_tiny: f64 = 6_000_000;
pub const eps_tiny: f32 = 1e-6;

pub const Shape = struct {
    heads: u32,
    eps: f32,
    theta: f64,

    pub fn of(c: *const config.Config) Shape {
        return .{ .heads = c.mla_heads, .eps = c.eps, .theta = c.rope_theta };
    }

    pub fn scale(s: Shape) f64 {
        _ = s;
        return 1.0 / @sqrt(@as(f64, @floatFromInt(qd)));
    }

    pub fn outWidth(s: Shape) u32 {
        return s.heads * vd;
    }
};

/// 1 / theta ** (arange(0, rope, 2) / rope): frequency i is 2i / 64.
pub fn invFreq(theta: f64, i: usize) f64 {
    return 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rope_dim)));
}

/// One 64-vector at absolute pos: even/odd pairs, then cat(a*c-b*s, b*c+a*s). out and z are rope_dim long.
pub fn rotate(out: []f64, z: []const f64, pos: u32, theta: f64) void {
    std.debug.assert(out.len == rope_dim and z.len == rope_dim);
    for (0..nfreq) |i| {
        const ang = @as(f64, @floatFromInt(pos)) * invFreq(theta, i);
        const c = @cos(ang);
        const s = @sin(ang);
        const a = z[2 * i];
        const b = z[2 * i + 1];
        out[i] = a * c - b * s;
        out[nfreq + i] = b * c + a * s;
    }
}

/// T rows of 64 at pos0 + t, the chunk/resume case. out and z are T * rope_dim.
pub fn rotateRows(out: []f64, z: []const f64, rows: u32, pos0: u32, theta: f64) void {
    std.debug.assert(out.len == rows * rope_dim and z.len == rows * rope_dim);
    for (0..rows) |t| {
        rotate(out[t * rope_dim ..][0..rope_dim], z[t * rope_dim ..][0..rope_dim], pos0 + @as(u32, @intCast(t)), theta);
    }
}

/// (T, H) to (T, H*128): each head gate repeated across the value columns.
pub fn expandGate(out: []u16, g: []const u16, rows: u32, heads: u32) void {
    std.debug.assert(g.len == rows * heads and out.len == rows * heads * vd);
    for (0..rows) |t| {
        for (0..heads) |h| {
            const v = g[t * heads + h];
            const lo = (t * heads + h) * vd;
            for (0..vd) |d| out[lo + d] = v;
        }
    }
}

fn r64(comptime round: bool, v: f64) f64 {
    if (!round) return v;
    return bf16_round.bf16(v);
}

fn sig(x: f64) f64 {
    if (x >= 0) {
        const e = @exp(-x);
        return 1.0 / (1.0 + e);
    }
    const e = @exp(x);
    return e / (1.0 + e);
}

fn mixAs(
    comptime round: bool,
    alloc: std.mem.Allocator,
    y: []f64,
    q: []const f32,
    kv: []const f32,
    w_rms: []const f32,
    kvb: []const f32,
    g: []const f32,
    s: Shape,
    rows: u32,
    pos0: u32,
) !void {
    const H = s.heads;
    const T = rows;
    std.debug.assert(y.len == T * H * vd);
    std.debug.assert(q.len == T * H * qd);
    std.debug.assert(kv.len == T * key);
    std.debug.assert(w_rms.len == lat);
    std.debug.assert(kvb.len == H * (nope + vd) * lat);
    std.debug.assert(g.len == T * H);

    const lat_t = try alloc.alloc(f64, T * lat);
    defer alloc.free(lat_t);
    const krope = try alloc.alloc(f64, T * rope_dim);
    defer alloc.free(krope);
    const qrope = try alloc.alloc(f64, T * H * rope_dim);
    defer alloc.free(qrope);
    const k_nope = try alloc.alloc(f64, T * H * nope);
    defer alloc.free(k_nope);
    const v = try alloc.alloc(f64, T * H * vd);
    defer alloc.free(v);

    const eps: f64 = s.eps;
    for (0..T) |t| {
        var ss: f64 = 0;
        for (0..lat) |i| {
            const x: f64 = kv[t * key + i];
            ss += x * x;
        }
        const rstd = 1.0 / @sqrt(ss / @as(f64, lat) + eps);
        for (0..lat) |i| {
            const yn = @as(f64, kv[t * key + i]) * rstd;
            lat_t[t * lat + i] = r64(round, r64(round, yn) * w_rms[i]);
        }
        var z: [rope_dim]f64 = undefined;
        var out: [rope_dim]f64 = undefined;
        for (0..rope_dim) |i| z[i] = kv[t * key + lat + i];
        rotate(&out, &z, pos0 + @as(u32, @intCast(t)), s.theta);
        for (0..rope_dim) |i| krope[t * rope_dim + i] = r64(round, out[i]);
        for (0..H) |h| {
            for (0..rope_dim) |i| z[i] = q[(t * H + h) * qd + nope + i];
            rotate(&out, &z, pos0 + @as(u32, @intCast(t)), s.theta);
            for (0..rope_dim) |i| qrope[(t * H + h) * rope_dim + i] = r64(round, out[i]);
        }
    }

    for (0..H) |h| {
        for (0..T) |t| {
            for (0..nope) |d| {
                var acc: f64 = 0;
                const w0 = ((h * (nope + vd) + d) * lat);
                for (0..lat) |c| acc += lat_t[t * lat + c] * kvb[w0 + c];
                k_nope[(t * H + h) * nope + d] = r64(round, acc);
            }
            for (0..vd) |d| {
                var acc: f64 = 0;
                const w0 = ((h * (nope + vd) + nope + d) * lat);
                for (0..lat) |c| acc += lat_t[t * lat + c] * kvb[w0 + c];
                v[(t * H + h) * vd + d] = r64(round, acc);
            }
        }
    }

    const sc = s.scale();
    const scores = try alloc.alloc(f64, H * T * T);
    defer alloc.free(scores);
    for (0..H) |h| {
        for (0..T) |t| {
            for (0..T) |k| {
                if (k > t) {
                    scores[(h * T + t) * T + k] = -std.math.inf(f64);
                    continue;
                }
                var p: f64 = 0;
                const qb = (t * H + h) * qd;
                const kb = (k * H + h) * nope;
                for (0..nope) |d| p += @as(f64, q[qb + d]) * k_nope[kb + d];
                const qr = (t * H + h) * rope_dim;
                for (0..rope_dim) |d| p += qrope[qr + d] * krope[k * rope_dim + d];
                scores[(h * T + t) * T + k] = p * sc;
            }
        }
    }

    for (0..H) |h| {
        for (0..T) |t| {
            var m: f64 = -std.math.inf(f64);
            const row = scores[(h * T + t) * T ..][0..T];
            for (0..t + 1) |k| m = @max(m, row[k]);
            var den: f64 = 0;
            for (0..t + 1) |k| {
                row[k] = @exp(row[k] - m);
                den += row[k];
            }
            for (0..t + 1) |k| row[k] /= den;
            var o: [vd]f64 = @splat(0);
            for (0..t + 1) |k| {
                const a = row[k];
                const vb = (k * H + h) * vd;
                for (0..vd) |d| o[d] += a * v[vb + d];
            }
            const gate = r64(round, sig(g[t * H + h]));
            for (0..vd) |d| y[(t * H + h) * vd + d] = r64(round, r64(round, o[d]) * gate);
        }
    }
}

/// Unabsorbed MLA in float64, r = identity. Causal over the given rows; pos0 is RoPE only.
pub fn mixF64(
    alloc: std.mem.Allocator,
    y: []f64,
    q: []const f32,
    kv: []const f32,
    w_rms: []const f32,
    kvb: []const f32,
    g: []const f32,
    s: Shape,
    rows: u32,
    pos0: u32,
) !void {
    try mixAs(false, alloc, y, q, kv, w_rms, kvb, g, s, rows, pos0);
}

/// Same mixer with r = bf16 at ling_forward.py's points (RMS, kv_b, rot, attn@v, gate).
pub fn mixBf16(
    alloc: std.mem.Allocator,
    y: []f64,
    q: []const f32,
    kv: []const f32,
    w_rms: []const f32,
    kvb: []const f32,
    g: []const f32,
    s: Shape,
    rows: u32,
    pos0: u32,
) !void {
    try mixAs(true, alloc, y, q, kv, w_rms, kvb, g, s, rows, pos0);
}

test "tiny MLA geometry is 16 heads, 512 latent, 64 rope, 128 nope/v, softmax 1/sqrt(192)" {
    const tiny = @embedFile("fixtures/tiny_config.json");
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, tiny, &why);
    const s = Shape.of(&c);
    try std.testing.expectEqual(@as(u32, 16), s.heads);
    try std.testing.expectEqual(@as(u32, 512), lat);
    try std.testing.expectEqual(@as(u32, 64), c.rope_dim);
    try std.testing.expectEqual(@as(u32, 128), c.nope);
    try std.testing.expectEqual(@as(u32, 128), c.v_dim);
    try std.testing.expectEqual(config.RopeStyle.gptj, c.rope_style);
    try std.testing.expectEqual(@as(f64, 6000000), s.theta);
    try std.testing.expectEqual(@as(f32, @floatCast(s.scale())), c.smScale());
}

test "expandGate repeats each head value across 128 columns" {
    const g = [_]u16{ 0x3f80, 0x4000, 0x4040, 0xc000 };
    var out: [2 * 2 * vd]u16 = undefined;
    expandGate(&out, &g, 2, 2);
    for (0..vd) |d| {
        try std.testing.expectEqual(@as(u16, 0x3f80), out[d]);
        try std.testing.expectEqual(@as(u16, 0x4000), out[vd + d]);
        try std.testing.expectEqual(@as(u16, 0x4040), out[2 * vd + d]);
        try std.testing.expectEqual(@as(u16, 0xc000), out[3 * vd + d]);
    }
}

test "mixF64 of zeros is zeros" {
    const a = std.testing.allocator;
    const s = Shape{ .heads = 1, .eps = eps_tiny, .theta = theta_tiny };
    var y: [vd]f64 = undefined;
    const q: [qd]f32 = @splat(0);
    const kv: [key]f32 = @splat(0);
    const w: [lat]f32 = @splat(1);
    const kvb: [1 * (nope + vd) * lat]f32 = @splat(0);
    const g = [_]f32{0};
    try mixF64(a, &y, &q, &kv, &w, &kvb, &g, s, 1, 0);
    for (y) |v| try std.testing.expectEqual(@as(f64, 0), v);
}
