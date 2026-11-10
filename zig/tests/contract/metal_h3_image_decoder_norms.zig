//! h3d_rms_norm and h3d_layer_norm under the contract: fp32 statistics, one bf16 rounding, fp32 sums proven in gate 4.
const std = @import("std");
const mtl = @import("metal");
const parts = @import("metal_h3_image_decoder_parts.zig");

const c = parts.c;
const Range = parts.Range;
const Ends = parts.Ends;
const Sum = parts.Sum;
const bf = parts.bf;
const value = parts.value;
const W = parts.W;
const Wf: f64 = W;

/// A thread's 4 values in turn, the simdgroup's 5-level tree, then the 8 simdgroups' parts in turn.
fn depth(_: c.Shape) usize {
    return 4 + 5 + 8;
}

/// The variance's chain: each term's subtraction and square, then the same sum.
fn varianceDepth(s: c.Shape) usize {
    return depth(s) + 3;
}

/// Rows of 1,024: one tile of 261 tokens, the batch of 16 tiles, one small tile of 65.
const served = [_]c.Shape{ .{ .rows = 261, .n = W, .k = W }, .{ .rows = 4176, .n = W, .k = W }, .{ .rows = 65, .n = W, .k = W } };

/// Gate 1's plants: (row, column) of each of `parts.plants`; the last two in one row.
fn plant(x: []u16, rows: usize) void {
    const at = [_][2]usize{ .{ 0, W / 3 }, .{ 1, W / 2 }, .{ 2, W - 1 }, .{ 3, 0 }, .{ 3, W - 1 } };
    for (at, parts.plants) |p, b| if (p[0] < rows) {
        x[p[0] * W + p[1]] = b;
    };
}

/// A norm launch as the launcher makes it: one row a threadgroup of 256.
fn launchRows(t: anytype, e: mtl.ComputeEncoder, buffers: usize) void {
    for (0..buffers) |i| t.bind(e, i, i);
    e.setValue(parts.Shape{ .m = @intCast(t.shape.rows) }, buffers);
    e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
}

// --- h3d_rms_norm: y = bf16((x * rsqrt(mean(x^2) + eps)) * g), all in fp32 ------------------------------------------

pub const RmsNorm = struct {
    x: []u16 = &.{},
    gw: []u16 = &.{},

    /// Rows: rows; n and k: the row's 1,024 values.
    pub fn spec(name: []const u8, _: RmsNorm) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 2 * (parts.div_ulps + parts.rsqrt_ulps) + 3, .out = .bf16, .served = &served };
    }

    pub fn prepare(k: *RmsNorm, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        if (s.rows == 0 or s.n != W or s.k != W) return false;
        k.x = try t.host().alloc(u16, s.rows * W);
        k.gw = try t.host().alloc(u16, W);
        // gate 4: integers, squares to 1,024; a row's sum passes fp16's range and stays exact in fp32
        if (t.fill == .stress) parts.integers(rng, k.x, -32, 32) else parts.uniform(rng, k.x, -3, 3);
        if (t.fill == .nonfinite) plant(k.x, s.rows);
        parts.uniform(rng, k.gw, 0.5, 1.5);
        try t.input(k.x);
        try t.input(k.gw);
        try t.output(s.rows * W * 2);
        return true;
    }

    /// A block's norm1 and norm2.
    pub fn launch(_: *RmsNorm, t: anytype, e: mtl.ComputeEncoder) void {
        launchRows(t, e, 3);
    }

    pub fn count(_: *RmsNorm, s: c.Shape) usize {
        return s.rows * W;
    }

    const Job = struct {
        k: *const RmsNorm,
        sum: Sum,
        ends: *Ends,
        want: []f64,
        mag: []f64,

        fn rows(j: Job, from: usize, to: usize) void {
            var x: [W]f64 = undefined;
            var terms: [W]f64 = undefined;
            for (from..to) |r| {
                for (&x, &terms, j.k.x[r * W ..][0..W]) |*v, *q, b| {
                    v.* = value(b);
                    q.* = v.* * v.*;
                }
                const ss = j.sum.of(&terms, depth(served[0]));
                const inv = parts.invNorm(ss.r, Wf);
                const inv64 = 1 / @sqrt(ss.v / Wf + parts.eps);
                for (x, j.k.gw, 0..) |v, gb, i| {
                    const gv = value(gb);
                    const y = Range.of(v).mul(inv).fp32(0.5).mul(Range.of(gv)).fp32(0.5).bf16();
                    j.ends.put(j.want, j.mag, r * W + i, bf(v * inv64 * gv), y);
                }
            }
        }
    };

    pub fn outputs(k: *RmsNorm, t: anytype, sum: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        parts.parallel(t.shape.rows, Job{ .k = k, .sum = sum, .ends = ends, .want = want, .mag = mag }, Job.rows);
    }
};

// --- h3d_layer_norm: norm_out, LayerNorm in fp32 with weight and bias, then bf16 ------------------------------------

pub const LayerNorm = struct {
    x: []u16 = &.{},
    gw: []u16 = &.{},
    bw: []u16 = &.{},

    /// Rows: rows; n and k: the row's 1,024 values.
    pub fn spec(name: []const u8, _: LayerNorm) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = varianceDepth, .ops = 2 * (parts.div_ulps + parts.rsqrt_ulps) + 5, .out = .bf16, .served = &served };
    }

    pub fn prepare(k: *LayerNorm, t: anytype, rng: std.Random) !bool {
        const s = t.shape;
        if (s.rows == 0 or s.n != W or s.k != W) return false;
        k.x = try t.host().alloc(u16, s.rows * W);
        k.gw = try t.host().alloc(u16, W);
        k.bw = try t.host().alloc(u16, W);
        if (t.fill == .stress) {
            // gate 4: an integer mean near 100, deviations cancelling in pairs; every step exact in fp32
            for (0..s.rows) |r| {
                const mean = rng.intRangeAtMost(i32, 80, 120) * @as(i32, if (rng.boolean()) 1 else -1);
                for (0..W / 2) |i| {
                    const d = rng.intRangeAtMost(i32, -64, 64);
                    k.x[r * W + i] = parts.integer(mean + d);
                    k.x[r * W + W / 2 + i] = parts.integer(mean - d);
                }
            }
        } else parts.uniform(rng, k.x, -2, 4);
        if (t.fill == .nonfinite) plant(k.x, s.rows);
        parts.uniform(rng, k.gw, 0.5, 1.5);
        parts.uniform(rng, k.bw, -0.5, 0.5);
        try t.input(k.x);
        try t.input(k.gw);
        try t.input(k.bw);
        try t.output(s.rows * W * 2);
        return true;
    }

    /// The pass's final stage, norm_out.
    pub fn launch(_: *LayerNorm, t: anytype, e: mtl.ComputeEncoder) void {
        launchRows(t, e, 4);
    }

    pub fn count(_: *LayerNorm, s: c.Shape) usize {
        return s.rows * W;
    }

    const Job = struct {
        k: *const LayerNorm,
        sum: Sum,
        ends: *Ends,
        want: []f64,
        mag: []f64,

        fn rows(j: Job, from: usize, to: usize) void {
            var x: [W]f64 = undefined;
            var terms: [W]f64 = undefined;
            for (from..to) |r| {
                for (&x, j.k.x[r * W ..][0..W]) |*v, b| v.* = value(b);
                const total = j.sum.of(&x, depth(served[0]));
                const mean = total.v / Wf;
                const mr = total.r.div(Wf).fp32(parts.div_ulps);
                for (&terms, x) |*q, v| q.* = (v - mean) * (v - mean);
                const sq = j.sum.of(&terms, varianceDepth(served[0]));
                // the kernel's mean may sit off float64's by `off`: its squares then add W off^2 at most
                const off = @max(@abs(mr.lo - mean), @abs(mr.hi - mean));
                var vr = sq.r;
                if (std.math.isFinite(off)) vr.hi += Wf * off * off;
                const rr = parts.invNorm(vr, Wf);
                const rstd = 1 / @sqrt(sq.v / Wf + parts.eps);
                for (x, j.k.gw, j.k.bw, 0..) |v, gb, bb, i| {
                    const gv = value(gb);
                    const bv = value(bb);
                    const d = Range.of(v).sub(mr).fp32(0.5);
                    const y = d.mul(rr).fp32(0.5).mul(Range.of(gv)).fp32(0.5).add(Range.of(bv)).fp32(0.5).bf16();
                    j.ends.put(j.want, j.mag, r * W + i, bf((v - mean) * rstd * gv + bv), y);
                }
            }
        }
    };

    pub fn outputs(k: *LayerNorm, t: anytype, sum: Sum, ends: *Ends, want: []f64, mag: []f64) !void {
        parts.parallel(t.shape.rows, Job{ .k = k, .sum = sum, .ends = ends, .want = want, .mag = mag }, Job.rows);
    }
};
