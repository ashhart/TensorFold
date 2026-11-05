//! The tower reference's float64 operators: torch's Linear, LayerNorm, exact GELU, attention, each row-parallel.
const std = @import("std");

/// A stored bf16 value widened.
pub fn widen(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

/// x rounded to fp32, then to bf16 (nearest, ties to even), as a torch bf16 op stores its result.
pub fn bf16(x: f64) f64 {
    const f: f32 = @floatCast(x);
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return f;
    const r = u + 0x7fff + ((u >> 16) & 1);
    return @as(f32, @bitCast(r & 0xffff0000));
}

/// Rounding points: `exact` keeps every value in float64; `torch` rounds where a torch bf16 forward stores a tensor.
pub const Points = enum { exact, torch };

pub fn round(p: Points, x: f64) f64 {
    return if (p == .torch) bf16(x) else x;
}

pub fn roundAll(p: Points, xs: []f64) void {
    if (p == .torch) for (xs) |*x| {
        x.* = bf16(x.*);
    };
}

/// A bf16 tensor's values, widened.
pub fn values(gpa: std.mem.Allocator, bytes: []const u8) ![]f64 {
    const out = try gpa.alloc(f64, bytes.len / 2);
    for (out, 0..) |*o, i| o.* = widen(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
    return out;
}

fn threads() usize {
    return @min(std.Thread.getCpuCount() catch 1, 16);
}

/// `job(ctx, lo, hi)` over [0, n) in contiguous slices, one a thread.
pub fn parallel(n: usize, ctx: anytype, comptime job: fn (@TypeOf(ctx), usize, usize) void) void {
    const t = @min(threads(), @max(1, n / 4));
    if (t <= 1) return job(ctx, 0, n);
    var pool: [16]?std.Thread = @splat(null);
    const step = (n + t - 1) / t;
    for (0..t) |i| {
        const lo = i * step;
        const hi = @min(n, lo + step);
        if (lo >= hi) break;
        pool[i] = std.Thread.spawn(.{}, job, .{ ctx, lo, hi }) catch {
            job(ctx, lo, hi);
            continue;
        };
    }
    for (pool) |th| if (th) |h| h.join();
}

const LinearJob = struct { y: []f64, x: []const f64, w: []const f64, b: ?[]const f64, n: usize, k: usize };

fn linearRows(j: *const LinearJob, lo: usize, hi: usize) void {
    for (lo..hi) |r| {
        const xr = j.x[r * j.k ..][0..j.k];
        for (0..j.n) |o| {
            const wr = j.w[o * j.k ..][0..j.k];
            var s: f64 = if (j.b) |b| b[o] else 0;
            for (xr, wr) |a, c| s += a * c;
            j.y[r * j.n + o] = s;
        }
    }
}

/// y [rows, n] = x [rows, k] . w [n, k]^T + b.
pub fn linear(y: []f64, x: []const f64, w: []const f64, b: ?[]const f64, rows: usize, n: usize, k: usize) void {
    const j: LinearJob = .{ .y = y, .x = x, .w = w, .b = b, .n = n, .k = k };
    parallel(rows, &j, linearRows);
}

/// torch's LayerNorm over rows of `width`, eps 1e-5, in place.
pub fn layerNorm(x: []f64, w: []const f64, b: []const f64, width: usize) void {
    var r: usize = 0;
    while (r < x.len) : (r += width) {
        const row = x[r..][0..width];
        var mean: f64 = 0;
        for (row) |v| mean += v;
        mean /= @floatFromInt(width);
        var vr: f64 = 0;
        for (row) |v| vr += (v - mean) * (v - mean);
        vr /= @floatFromInt(width);
        const inv = 1.0 / @sqrt(vr + 1e-5);
        for (row, w, b) |*v, g, c| v.* = (v.* - mean) * inv * g + c;
    }
}

/// torch's exact GELU: x Phi(x).
pub fn gelu(x: f64) f64 {
    return 0.5 * x * (1.0 + erf(x / std.math.sqrt2));
}

/// erf by its Taylor series near 0 and its continued fraction from 2 (float64 to a few ulps).
pub fn erf(x: f64) f64 {
    const ax = @abs(x);
    if (ax < 2.0) {
        var term = ax;
        var sum = ax;
        var n: f64 = 0;
        while (n < 200) : (n += 1) {
            term *= -ax * ax / (n + 1);
            const add = term / (2 * n + 3);
            sum += add;
            if (@abs(add) < 1e-17 * @abs(sum)) break;
        }
        const r = 2.0 / @sqrt(std.math.pi) * sum;
        return if (x < 0) -r else r;
    }
    // erfc(x) = exp(-x^2) / (x sqrt(pi)) / (1 + 1/(2x^2) / (1 + 2/(2x^2) / ...)), Lentz's evaluation
    var f: f64 = ax;
    var n: f64 = 200;
    while (n >= 1) : (n -= 1) f = ax + (n / 2.0) / f;
    const erfc = @exp(-ax * ax) / @sqrt(std.math.pi) / f;
    return if (x < 0) erfc - 1.0 else 1.0 - erfc;
}

const AttnJob = struct { out: []f64, qkv: []const f64, start: usize, len: usize, heads: usize, dim: usize, scale: f64 };

fn attnRows(j: *const AttnJob, lo: usize, hi: usize) void {
    const d = j.heads * j.dim;
    var p: [2048]f64 = undefined;
    for (lo..hi) |i| {
        const r = j.start + i;
        for (0..j.heads) |h| {
            const q = j.qkv[r * 3 * d + h * j.dim ..][0..j.dim];
            var top: f64 = -std.math.inf(f64);
            for (0..j.len) |t| {
                const k = j.qkv[(j.start + t) * 3 * d + d + h * j.dim ..][0..j.dim];
                var s: f64 = 0;
                for (q, k) |a, b| s += a * b;
                p[t] = s * j.scale;
                top = @max(top, p[t]);
            }
            var den: f64 = 0;
            for (p[0..j.len]) |*v| {
                v.* = @exp(v.* - top);
                den += v.*;
            }
            const o = j.out[r * d + h * j.dim ..][0..j.dim];
            @memset(o, 0);
            for (0..j.len) |t| {
                const v = j.qkv[(j.start + t) * 3 * d + 2 * d + h * j.dim ..][0..j.dim];
                const wt = p[t] / den;
                for (o, v) |*a, b| a.* += wt * b;
            }
        }
    }
}

/// Bidirectional attention of rows [start, start + len) over themselves; qkv rows are [q | k | v], heads by head.
pub fn attention(out: []f64, qkv: []const f64, start: usize, len: usize, heads: usize, dim: usize) !void {
    if (len > 2048) return error.AttentionWindow;
    const j: AttnJob = .{ .out = out, .qkv = qkv, .start = start, .len = len, .heads = heads, .dim = dim, .scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dim))) };
    parallel(len, &j, attnRows);
}

test "bf16: ties to even, values kept" {
    try std.testing.expectEqual(@as(f64, 1.0), bf16(1.0 + 1.0 / 256.0));
    try std.testing.expectEqual(@as(f64, 1.0 + 1.0 / 64.0), bf16(1.0 + 3.0 / 256.0));
    try std.testing.expectEqual(@as(f64, -2.5), bf16(-2.5));
}

test "erf: against known values, odd" {
    const cases = [_][2]f64{ .{ 0.5, 0.5204998778130465 }, .{ 1.0, 0.8427007929497149 }, .{ 2.5, 0.9995930479825550 }, .{ 3.5, 0.9999992569016276 } };
    for (cases) |c| {
        try std.testing.expectApproxEqAbs(c[1], erf(c[0]), 1e-13);
        try std.testing.expectApproxEqAbs(-c[1], erf(-c[0]), 1e-13);
    }
}

test "linear and layerNorm: a hand-sized case" {
    var y: [2]f64 = undefined;
    linear(&y, &.{ 1, 2 }, &.{ 3, 4, 5, 6 }, &.{ 0.5, -1 }, 1, 2, 2);
    try std.testing.expectEqualSlices(f64, &.{ 11.5, 16 }, &y);
    var x = [_]f64{ 1, 3 };
    layerNorm(&x, &.{ 1, 1 }, &.{ 0, 0 }, 2);
    try std.testing.expectApproxEqAbs(@as(f64, -1), x[0], 1e-5);
}
