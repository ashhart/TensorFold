//! RMS norm rows in float64, the reference every backend's norm kernels are checked against (CUDA: cuda/norms.zig).
const std = @import("std");
const bf = @import("bf16_round.zig");

/// y = x / sqrt(mean(x^2) + eps) * w for one row (w null: no weight).
pub fn rms(x: []const f64, w: ?[]const f64, eps: f64, y: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (x, y, 0..) |v, *o, i| o.* = v * inv * (if (w) |ww| ww[i] else 1);
}

/// The residual step for one row: res += rms(x) * w1 (w1 null: res += x), then y = rms(res) * w2.
pub fn addRms(x: []const f64, res: []f64, w1: ?[]const f64, w2: []const f64, eps: f64, y: []f64) void {
    if (w1) |w| {
        rms(x, w, eps, y); // y as scratch: rms(x) * w1
        for (res, y) |*r, v| r.* += v;
    } else for (res, x) |*r, v| {
        r.* += v;
    }
    rms(res, w2, eps, y);
}

/// One once-rounded output's tolerance: half an ulp of its kind at |want| plus 2^-20 of |want| for fp32 arithmetic.
pub fn tolerance(want: f64, kind: Kind) f64 {
    const half_ulp: f64 = switch (kind) {
        .f32 => std.math.pow(f64, 2, -24),
        .f16 => std.math.pow(f64, 2, -11),
        .bf16 => std.math.pow(f64, 2, -8),
    };
    return @abs(want) * (half_ulp + std.math.pow(f64, 2, -20)) + 1e-30;
}

/// The residual step's bounds: res_i 2^-20 of |res0_i| + |add_i| (add: what it adds), y_i that error carried on.
pub fn addRmsBounds(res0: []const f64, add: []const f64, w2: []const f64, y: []const f64, res: []const f64, kind: Kind, res_tol: []f64, y_tol: []f64) void {
    var ss: f64 = 0;
    for (res) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(res.len)));
    for (res0, add, w2, y, res_tol, y_tol) |r0, a, w, yy, *rt, *yt| {
        rt.* = (@abs(r0) + @abs(a)) * std.math.pow(f64, 2, -20) + 1e-30;
        yt.* = tolerance(yy, kind) + rt.* * inv * @abs(w);
    }
}

/// y's sums over each 64 values.
pub fn groupSums(y: []const f64, xs: []f64) void {
    for (xs, 0..) |*o, j| {
        var t: f64 = 0;
        for (y[64 * j ..][0..64]) |v| t += v;
        o.* = t;
    }
}

/// A sum of 64 fp32 values' tolerance: 64 roundings of at most 2^-24 of the running |sum| <= sum |y|.
pub fn groupSumTolerance(y: []const f64) f64 {
    var t: f64 = 0;
    for (y) |v| t += @abs(v);
    return t * 64 * std.math.pow(f64, 2, -24) + 1e-30;
}

/// silu(z) * rms(y) * w for one head.
pub fn gatedRms(y: []const f64, z: []const f64, w: []const f64, eps: f64, out: []f64) void {
    rms(y, w, eps, out);
    for (out, z) |*o, zz| o.* *= zz / (1 + @exp(-zz));
}

/// A bf16 residual's tail (cuda/norms.zig's addRmsTail) at its rounding points; `round`: MLX's bf16(w * bf16(v * inv)).
pub const Tail = struct {
    round: bool,
    residual: ?[]const f64 = null,
    branch: ?[]const f64 = null,
    scalar: ?f64 = null,
    eps: f64 = 1e-6,
    /// The relative spread of an fp32 inverse RMS and the products by it: the outputs are ranges over it (0: points).
    spread: f64 = 0,
};

/// Sums of squares in float64, the tail's exact sums; a test may pass its own `squares` (another accumulator).
pub const Exact = struct {
    pub fn squares(_: Exact, v: []const f64) f64 {
        var s: f64 = 0;
        for (v) |x| s += x * x;
        return s;
    }
};

/// norm(v) w at one value: w bf16(v inv) rounded (`round`), else v inv w rounded once.
fn normed(v: f64, inv: f64, w: f64, round: bool) f64 {
    return bf.bf16(if (round) w * bf.bf16(v * inv) else v * inv * w);
}

/// a and b in order; NaN in either gives NaN for both.
fn order(a: f64, b: f64) [2]f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return @splat(std.math.nan(f64));
    return .{ @min(a, b), @max(a, b) };
}

/// One row's h = bf16(bf16(r + norm(x) wb) sc) and y = norm(h) w, each a range over inverse RMS values within spread.
pub fn tail(t: Tail, x: []const f64, w: []const f64, sums: anytype, h_lo: []f64, h_hi: []f64, y_lo: []f64, y_hi: []f64) void {
    const n: f64 = @floatFromInt(x.len);
    const inv1 = if (t.branch != null) 1 / @sqrt(sums.squares(x) / n + t.eps) else 0;
    for (x, h_lo, h_hi, 0..) |v, *lo, *hi, i| {
        var ends: [2]f64 = undefined;
        for (&ends, [_]f64{ 1 - t.spread, 1 + t.spread }) |*e, f| {
            var s = if (t.branch) |wb| normed(v, inv1 * f, wb[i], t.round) else v;
            if (t.residual) |r| s = r[i] + s;
            e.* = if (t.scalar) |sc| bf.bf16(bf.bf16(s) * sc) else bf.bf16(s);
        }
        lo.*, hi.* = order(ends[0], ends[1]);
    }
    for (h_lo, h_hi, y_lo, y_hi) |lo, hi, *small, *big| { // the squares' least and largest terms, as scratch
        const a, const b = order(@abs(lo), @abs(hi));
        small.* = if (lo <= 0 and hi >= 0) 0 else a;
        big.* = b;
    }
    const inv_lo = (1 - t.spread) / @sqrt(sums.squares(y_hi) / n + t.eps);
    const inv_hi = (1 + t.spread) / @sqrt(sums.squares(y_lo) / n + t.eps);
    for (h_lo, h_hi, w, y_lo, y_hi) |lo, hi, ww, *ylo, *yhi| {
        var most = [2]f64{ std.math.inf(f64), -std.math.inf(f64) };
        for ([_]f64{ lo * inv_lo, lo * inv_hi, hi * inv_lo, hi * inv_hi }) |p| {
            const o = normed(p, 1, ww, t.round); // monotone in p: the corners bound it
            if (std.math.isNan(o)) most = @splat(std.math.nan(f64)) else most = .{ @min(most[0], o), @max(most[1], o) };
        }
        ylo.*, yhi.* = most;
    }
}

/// The element kinds the kernels read and write, numbered as they are (hip/ops/common.hpp: 0 fp32, 1 fp16, 2 bf16).
pub const Kind = enum(u8) { f32 = 0, f16 = 1, bf16 = 2 };

test "a row's norm and the residual step against hand values" {
    var y: [4]f64 = undefined;
    rms(&.{ 1, -1, 1, -1 }, &.{ 2, 2, 2, 2 }, 0, &y);
    try std.testing.expectEqualSlices(f64, &.{ 2, -2, 2, -2 }, &y);
    rms(&.{ 3, 4 }, null, 0, y[0..2]);
    try std.testing.expectApproxEqRel(@as(f64, 3.0 / @sqrt(12.5)), y[0], 1e-15);
    var res = [_]f64{ 1, 1, 1, 1 };
    addRms(&.{ 1, -1, 1, -1 }, &res, null, &.{ 1, 1, 1, 1 }, 0, &y);
    try std.testing.expectEqualSlices(f64, &.{ 2, 0, 2, 0 }, &res);
    try std.testing.expectApproxEqRel(@as(f64, @sqrt(2.0)), y[0], 1e-15);
    res = .{ 0, 0, 0, 0 };
    addRms(&.{ 2, -2, 2, -2 }, &res, &.{ 3, 3, 3, 3 }, &.{ 1, 1, 1, 1 }, 0, &y); // res = rms(x) * 3 = +-3
    try std.testing.expectEqualSlices(f64, &.{ 3, -3, 3, -3 }, &res);
    try std.testing.expect(tolerance(1, .bf16) > tolerance(1, .f32));
}

test "a tail's rounding points: none set passes x, the scalar rounds the sum first, MLX's rows differ from one rounding" {
    const x = [_]f64{ 1, -1, 1, -1 };
    const w = [_]f64{ 1, 1, 1, 1 };
    var h: [2][4]f64 = undefined;
    var y: [2][4]f64 = undefined;
    tail(.{ .round = true, .eps = 0 }, &x, &w, Exact{}, &h[0], &h[1], &y[0], &y[1]);
    try std.testing.expectEqualSlices(f64, &x, &h[0]);
    try std.testing.expectEqualSlices(f64, &x, &y[0]);
    const r = [_]f64{ 1, 1, 1, 1 };
    const tiny = [_]f64{ 0x1p-8, 0x1p-8, 0x1p-8, 0x1p-8 };
    tail(.{ .round = false, .residual = &r, .scalar = 1 + 0x1p-7, .eps = 0 }, &tiny, &w, Exact{}, &h[0], &h[1], &y[0], &y[1]);
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-7), h[0][0]); // bf16(1 + 2^-8) = 1 (a tie to even), then 1 + 2^-7
    var prng = std.Random.DefaultPrng.init(3);
    var v: [64]f64 = undefined;
    var wb: [64]f64 = undefined;
    for (&v, &wb) |*a, *b| {
        a.* = bf.bf16(prng.random().float(f64) * 4 - 2);
        b.* = bf.bf16(0.5 + prng.random().float(f64));
    }
    var rows: [4][64]f64 = undefined;
    var ys: [2][64]f64 = undefined;
    tail(.{ .round = true, .branch = &wb }, &v, &wb, Exact{}, &rows[0], &rows[1], &ys[0], &ys[1]);
    tail(.{ .round = false, .branch = &wb }, &v, &wb, Exact{}, &rows[2], &rows[3], &ys[0], &ys[1]);
    try std.testing.expectEqualSlices(f64, &rows[0], &rows[1]); // spread 0: points
    try std.testing.expect(!std.mem.eql(f64, &rows[0], &rows[2]));
    tail(.{ .round = true, .branch = &wb, .spread = 0x1p-12 }, &v, &wb, Exact{}, &rows[0], &rows[1], &ys[0], &ys[1]);
    for (rows[0], rows[1], ys[0], ys[1]) |a, b, c, d| try std.testing.expect(a <= b and c <= d);
}
