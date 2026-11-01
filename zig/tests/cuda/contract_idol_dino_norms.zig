//! tf_idol_dino_norm and _final under the contract: nn.LayerNorm to bf16, then the fp32 F.layer_norm; fp32 sums.
const std = @import("std");
const cuda = @import("cuda");
const dino = @import("idol_dino_cuda");
const r = @import("contract_idol_ref.zig");
const ops = @import("contract_idol_dino_ops.zig");

const c = r.c;
const Range = r.Range;
const Sum = r.Sum;
const W = ops.W;
const bf = ops.bf;
const bfr = ops.bfr;
const eps: f64 = @as(f32, 1e-5);

const why = "an output follows the sums as (x - mean) / sqrt(var): an fp16 or bf16 sum's error is many output ulps";

/// A thread's share of the row in turn, the warp's 5 shuffles, then the 8 warps' partials in order.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

const served = [_]c.Shape{.{ .rows = ops.rows, .n = W, .k = W }};

fn rowSpec(name: []const u8, out: c.Elem, stages: f64) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = stages * (2 * (r.div_ulps + r.sqrt_ulps) + 4),
        .mids = 1,
        .out = out,
        .served = &served,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = why },
    };
}

fn rowOk(s: c.Shape) bool {
    return s.rows > 0 and s.k == W and s.n == W;
}

/// A row's mean and 1 / sqrt(var + eps), value and range, from values `x` each anywhere in [lo, hi], summed as `sum`.
const Stats = struct {
    mean: f64,
    mr: Range,
    inv: f64,
    ir: Range,

    fn of(a: std.mem.Allocator, x: []const f64, lo: []const f64, hi: []const f64, sum: Sum) !Stats {
        const k: f64 = @floatFromInt(x.len);
        const dp = depth(.{ .rows = 1, .n = x.len, .k = x.len });
        const m = sum.total(x, lo, hi, dp);
        const mean = m.v / k;
        const mr = (Range{ .lo = m.r.lo / k, .hi = m.r.hi / k }).fp32(r.div_ulps);
        const buf = try a.alloc(f64, 3 * x.len);
        defer a.free(buf);
        const sq = buf[0..x.len];
        const ql = buf[x.len .. 2 * x.len];
        const qh = buf[2 * x.len ..];
        for (x, lo, hi, sq, ql, qh) |v, l, u, *q, *qlo, *qhi| {
            q.* = (v - mean) * (v - mean);
            const qr = (Range{ .lo = l, .hi = u }).sub(mr).fp32(0.5).square().fp32(0.5);
            qlo.* = qr.lo;
            qhi.* = qr.hi;
        }
        const s = sum.total(sq, ql, qh, dp);
        const vr = (Range{ .lo = s.r.lo / k, .hi = s.r.hi / k }).fp32(r.div_ulps).add(Range.of(eps)).fp32(0.5);
        const root = vr.up(sqrt).fp32(r.sqrt_ulps);
        return .{ .mean = mean, .mr = mr, .inv = 1 / @sqrt(s.v / k + eps), .ir = Range.of(1).div(root).fp32(r.div_ulps) };
    }

    /// (x - mean) inv for x in `xr`: its float64 value and range.
    fn at(st: Stats, x: f64, xr: Range) struct { v: f64, r: Range } {
        return .{ .v = (x - st.mean) * st.inv, .r = xr.sub(st.mr).fp32(0.5).mul(st.ir).fp32(0.5) };
    }
};

fn sqrt(x: f64) f64 {
    return @sqrt(@max(x, 0));
}

/// Gate 4's rows: integers whose second half mirrors the first, so the mean is 0 and every fp32 sum is exact.
fn mirrored(rng: std.Random, n_rows: usize, x: []f64, bound: u32) void {
    for (0..n_rows) |row| {
        const v = x[row * W ..][0..W];
        r.integers(rng, v[0 .. W / 2], bound);
        for (v[W / 2 ..], v[0 .. W / 2]) |*o, a| o.* = -a;
    }
}

fn rowsFill(rng: std.Random, s: c.Shape, mode: r.g.Fill, x: []f64) void {
    if (mode == .stress) mirrored(rng, s.rows, x, 8) else r.uniform(rng, x, -3, 3);
    if (mode == .nonfinite) r.plant(x, &.{ W / 3, W + W / 2, 2 * W + W - 1 });
}

/// One row's first stage: bf16((x - mean) inv w + b), each output's value and range.
fn first(a: std.mem.Allocator, x: []const f64, w: []const f64, b: []const f64, sum: Sum, out: []f64, lo: []f64, hi: []f64) !void {
    const st = try Stats.of(a, x, x, x, sum);
    for (x, w, b, out, lo, hi) |xv, wv, bv, *o, *l, *u| {
        const y = st.at(xv, Range.of(xv));
        const yr = bfr(y.r.mul(Range.of(wv)).fp32(0.5).add(Range.of(bv)).fp32(0.5));
        o.* = bf(y.v * wv + bv);
        l.* = yr.lo;
        u.* = yr.hi;
    }
}

/// tf_idol_dino_norm: nn.LayerNorm of bf16 rows, bf16(y w + b) with bf16 w and b.
pub const Norm = struct {
    pub const Kernels = dino.Kernels;
    pub const kinds = [_]r.Kind{ .bf16, .bf16, .bf16 };
    pub const out: r.Kind = .bf16;
    pub const in_place: ?usize = null;
    pub const accepts = rowOk;

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return rowSpec(name, .bf16, 1);
    }

    pub fn sizes(s: c.Shape) [3]usize {
        return .{ s.rows * W, W, W };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * W;
    }

    pub fn fill(rng: std.Random, mode: r.g.Fill, s: c.Shape, _: u32, v: *[3][]f64) void {
        rowsFill(rng, s, mode, v[0]);
        r.uniform(rng, v[1], 0.25, 2);
        r.uniform(rng, v[2], -1, 1);
        if (mode == .nonfinite) for (v[1..]) |col| r.plant(col, &.{7});
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dino.norm(k, s, dev[0], dev[1], dev[2], o, shape.rows);
    }

    pub fn ref(a: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        var o: [W]f64 = undefined;
        var lo: [W]f64 = undefined;
        var hi: [W]f64 = undefined;
        for (0..shape.rows) |row| {
            try first(a, v[0][row * W ..][0..W], v[1], v[2], sum, &o, &lo, &hi);
            for (0..W) |i| e.put(want, mag, row * W + i, o[i], .{ .lo = lo[i], .hi = hi[i] });
        }
    }
};

/// tf_idol_dino_final: h = the last norm (as tf_idol_dino_norm), y = F.layer_norm(h) in fp32 with no affine.
pub const Final = FinalOf(false);
/// tf_idol_dino_final's h, last_hidden_state (y in scratch).
pub const FinalHidden = FinalOf(true);

/// tf_idol_dino_final with one output checked, the other in scratch.
fn FinalOf(comptime hidden: bool) type {
    return struct {
        pub const Kernels = dino.Kernels;
        pub const kinds = [_]r.Kind{ .bf16, .bf16, .bf16 };
        pub const out: r.Kind = if (hidden) .bf16 else .f32;
        pub const in_place: ?usize = null;
        pub const accepts = rowOk;

        pub fn spec(name: []const u8, _: u32) c.Spec {
            return if (hidden) rowSpec(name, .bf16, 1) else rowSpec(name, .f32, 2);
        }

        pub fn sizes(s: c.Shape) [3]usize {
            return .{ s.rows * W, W, W };
        }

        pub fn count(s: c.Shape) usize {
            return s.rows * W;
        }

        pub fn scratch(s: c.Shape) usize {
            return s.rows * W * @as(usize, if (hidden) 4 else 2);
        }

        pub fn fill(rng: std.Random, mode: r.g.Fill, s: c.Shape, _: u32, v: *[3][]f64) void {
            rowsFill(rng, s, mode, v[0]);
            r.uniform(rng, v[1], 0.25, 2);
            r.uniform(rng, v[2], -1, 1);
            if (mode == .stress) {
                // h = bf16(0 + b) = b exactly: the second stage's sums see mirrored integers too
                @memset(v[1], 0);
                mirrored(rng, 1, v[2], 8);
            }
            // a non-finite w or b reaches every row's F.layer_norm: y's rows keep their finite ones
            if (mode == .nonfinite and hidden) for (v[1..]) |col| r.plant(col, &.{7});
        }

        pub fn launchWith(k: *const Kernels, s: cuda.Stream, dev: [3]u64, o: u64, extra: u64, shape: c.Shape, _: u32) !void {
            if (hidden) {
                try dino.final(k, s, dev[0], dev[1], dev[2], o, extra, shape.rows);
            } else try dino.final(k, s, dev[0], dev[1], dev[2], extra, o, shape.rows);
        }

        pub fn ref(a: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
            var h: [W]f64 = undefined;
            var lo: [W]f64 = undefined;
            var hi: [W]f64 = undefined;
            for (0..shape.rows) |row| {
                try first(a, v[0][row * W ..][0..W], v[1], v[2], sum, &h, &lo, &hi);
                if (hidden) {
                    for (0..W) |i| e.put(want, mag, row * W + i, h[i], .{ .lo = lo[i], .hi = hi[i] });
                    continue;
                }
                const st = try Stats.of(a, &h, &lo, &hi, sum);
                for (0..W) |i| {
                    const y = st.at(h[i], .{ .lo = lo[i], .hi = hi[i] });
                    e.put(want, mag, row * W + i, y.v, y.r);
                }
            }
        }
    };
}

test "a LayerNorm row's statistics lie in their ranges, gate 4's mirrored integers sum to a zero mean" {
    const a = std.testing.allocator;
    var x: [W]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    mirrored(prng.random(), 1, &x, 8);
    const st = try Stats.of(a, &x, &x, &x, .{ .range = true });
    try std.testing.expectEqual(@as(f64, 0), st.mean);
    try std.testing.expect(st.ir.lo <= st.inv and st.inv <= st.ir.hi);
    r.uniform(prng.random(), &x, -3, 3);
    const rs = try Stats.of(a, &x, &x, &x, .{ .range = false });
    try std.testing.expect(rs.mr.lo <= rs.mean and rs.mean <= rs.mr.hi and rs.ir.lo <= rs.inv and rs.inv <= rs.ir.hi);
}
