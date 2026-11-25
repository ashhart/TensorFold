//! tf_idol_norm and tf_idol_norm32 under the contract: LayerNorm32 at idol-sdk's fp16 points, fp32 sums in gate 4.
const std = @import("std");
const cuda = @import("cuda");
const idol = @import("idol_cuda");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const Range = r.Range;
const Sum = r.Sum;

const why = "an output follows the sums as (x - mean) / sqrt(var): an fp16 or bf16 sum's error is many output ulps";

/// A thread's share of the row in turn, the warp's 5 shuffles, then the 8 warps' partials in order.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

const served = [_]c.Shape{ .{ .rows = r.joint, .n = r.hidden, .k = r.hidden }, .{ .rows = r.latent, .n = r.hidden, .k = r.hidden }, .{ .rows = r.cond, .n = r.hidden, .k = r.hidden } };

fn rowSpec(name: []const u8, out: c.Elem) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = 2 * (r.div_ulps + r.sqrt_ulps) + 3,
        .mids = 2,
        .out = out,
        .served = &served,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = why },
    };
}

fn rowOk(s: c.Shape) bool {
    return s.rows > 0 and s.k == r.hidden and s.n == r.hidden;
}

/// One row's mean and 1 / sqrt(var + eps) as float64 and in range, each sum formed as `sum` says.
const Stats = struct {
    mean: f64,
    mr: Range,
    inv: f64,
    ir: Range,

    fn of(a: std.mem.Allocator, x: []const f64, eps: f64, sum: Sum) !Stats {
        const k: f64 = @floatFromInt(x.len);
        const dp = (x.len + 255) / 256 + 5 + 8;
        const m = sum.total(x, x, x, dp);
        const mean = m.v / k;
        const mr = (Range{ .lo = m.r.lo / k, .hi = m.r.hi / k }).fp32(r.div_ulps);
        const buf = try a.alloc(f64, 3 * x.len);
        defer a.free(buf);
        const sq = buf[0..x.len];
        const lo = buf[x.len .. 2 * x.len];
        const hi = buf[2 * x.len ..];
        for (x, sq, lo, hi) |v, *q, *l, *u| {
            q.* = (v - mean) * (v - mean);
            const qr = Range.of(v).sub(mr).fp32(0.5).square().fp32(0.5);
            l.* = qr.lo;
            u.* = qr.hi;
        }
        const s = sum.total(sq, lo, hi, dp);
        const vr = (Range{ .lo = s.r.lo / k, .hi = s.r.hi / k }).fp32(r.div_ulps).add(Range.of(eps)).fp32(0.5);
        const root = vr.up(sqrt).fp32(r.sqrt_ulps);
        return .{ .mean = mean, .mr = mr, .inv = 1 / @sqrt(s.v / k + eps), .ir = Range.of(1).div(root).fp32(r.div_ulps) };
    }

    /// (x - mean) inv: its float64 value and range.
    fn at(st: Stats, x: f64) struct { v: f64, r: Range } {
        return .{ .v = (x - st.mean) * st.inv, .r = Range.of(x).sub(st.mr).fp32(0.5).mul(st.ir).fp32(0.5) };
    }
};

fn sqrt(x: f64) f64 {
    return @sqrt(@max(x, 0));
}

/// Gate 4's rows: integers whose second half mirrors the first, so the mean is 0 and every fp32 sum is exact.
fn rowsFill(rng: std.Random, fill: c.Shape, mode: r.g.Fill, x: []f64) void {
    if (mode == .stress) {
        const k = fill.k;
        for (0..fill.rows) |row| {
            const v = x[row * k ..][0..k];
            r.integers(rng, v[0 .. k / 2], 8);
            for (v[k / 2 ..], v[0 .. k / 2]) |*o, a| o.* = -a;
        }
    } else r.uniform(rng, x, -3, 3);
    if (mode == .nonfinite) {
        const k = fill.k;
        r.plant(x, &.{ k / 3, k + k / 2, 2 * k + k - 1 });
    }
}

/// tf_idol_norm: fp16(y w + b) (variant 1, affine) or fp16(fp16(fp16(y) s1) + sh) (variant 2, modulated).
pub const Norm = struct {
    pub const kinds = [_]r.Kind{ .f16, .f32, .f32, .f16, .f16 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = null;
    pub const accepts = rowOk;

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return rowSpec(name, .f16);
    }

    pub fn sizes(s: c.Shape) [5]usize {
        return .{ s.rows * s.k, s.k, s.k, s.k, s.k };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.k;
    }

    pub fn fill(rng: std.Random, mode: r.g.Fill, s: c.Shape, _: u32, v: *[5][]f64) void {
        rowsFill(rng, s, mode, v[0]);
        r.uniform(rng, v[1], 0.25, 2);
        r.uniform(rng, v[2], -1, 1);
        // the modulation's 1 + scale, and its shift
        r.uniform(rng, v[3], 0.5, 1.5);
        r.uniform(rng, v[4], -1, 1);
        if (mode == .nonfinite) for (v[1..]) |col| r.plant(col, &.{7});
    }

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [5]u64, o: u64, shape: c.Shape, variant: u32) !void {
        try idol.norm(k, s, dev[0], dev[1], dev[2], dev[3], dev[4], o, shape.rows, @as(f32, 1e-6), if (variant == 2) .modulated else .affine);
    }

    pub fn ref(a: std.mem.Allocator, shape: c.Shape, variant: u32, v: [5][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        const k = shape.k;
        for (0..shape.rows) |row| {
            const x = v[0][row * k ..][0..k];
            const st = try Stats.of(a, x, r.f(1e-6), sum);
            for (x, 0..) |xv, i| {
                const y = st.at(xv);
                const at = row * k + i;
                if (variant == 2) {
                    const s1 = v[3][i];
                    const sh = v[4][i];
                    e.put(want, mag, at, r.h(r.h(r.h(y.v) * s1) + sh), y.r.half().mul(Range.of(s1)).fp32(0.5).half().add(Range.of(sh)).fp32(0.5).half());
                } else {
                    const w = v[1][i];
                    const b = v[2][i];
                    e.put(want, mag, at, r.h(y.v * w + b), y.r.mul(Range.of(w)).fp32(0.5).add(Range.of(b)).fp32(0.5).half());
                }
            }
        }
    }
};

/// tf_idol_norm32: y w + b (variant 1, RePo's norm) or y (1 + w) + b (variant 2, the output's), in fp32.
pub const Norm32 = struct {
    pub const kinds = [_]r.Kind{ .f16, .f32, .f32 };
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = null;
    pub const accepts = rowOk;

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return rowSpec(name, .f32);
    }

    pub fn sizes(s: c.Shape) [3]usize {
        return .{ s.rows * s.k, s.k, s.k };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.k;
    }

    pub fn fill(rng: std.Random, mode: r.g.Fill, s: c.Shape, _: u32, v: *[3][]f64) void {
        rowsFill(rng, s, mode, v[0]);
        r.uniform(rng, v[1], -1, 2);
        r.uniform(rng, v[2], -1, 1);
        if (mode == .nonfinite) for (v[1..]) |col| r.plant(col, &.{9});
    }

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, variant: u32) !void {
        try idol.norm32(k, s, dev[0], dev[1], dev[2], o, shape.rows, @as(f32, 1e-5), if (variant == 2) .modulated else .affine);
    }

    pub fn ref(a: std.mem.Allocator, shape: c.Shape, variant: u32, v: [3][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        const k = shape.k;
        for (0..shape.rows) |row| {
            const x = v[0][row * k ..][0..k];
            const st = try Stats.of(a, x, r.f(1e-5), sum);
            for (x, 0..) |xv, i| {
                const y = st.at(xv);
                const w = v[1][i];
                const b = v[2][i];
                const at = row * k + i;
                if (variant == 2) {
                    const one = Range.of(1).add(Range.of(w)).fp32(0.5);
                    e.put(want, mag, at, y.v * (1 + w) + b, y.r.mul(one).fp32(0.5).add(Range.of(b)).fp32(0.5));
                } else e.put(want, mag, at, y.v * w + b, y.r.mul(Range.of(w)).fp32(0.5).add(Range.of(b)).fp32(0.5));
            }
        }
    }
};

test "a LayerNorm row's statistics lie in their ranges, and gate 4's mirrored integers sum to a zero mean" {
    const a = std.testing.allocator;
    var x: [1024]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    rowsFill(prng.random(), .{ .rows = 1, .n = 1024, .k = 1024 }, .stress, &x);
    const st = try Stats.of(a, &x, r.f(1e-6), .{ .range = true });
    try std.testing.expectEqual(@as(f64, 0), st.mean);
    try std.testing.expect(st.ir.lo <= st.inv and st.inv <= st.ir.hi);
    rowsFill(prng.random(), .{ .rows = 1, .n = 1024, .k = 1024 }, .random, &x);
    const rs = try Stats.of(a, &x, r.f(1e-6), .{ .range = false });
    try std.testing.expect(rs.mr.lo <= rs.mean and rs.mean <= rs.mr.hi and rs.ir.lo <= rs.inv and rs.inv <= rs.ir.hi);
}
