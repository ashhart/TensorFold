//! Idol's DINOv3 epilogues under the contract: the bias, the RoPE heads, LayerScale with the residual, SwiGLU.
const std = @import("std");
const cuda = @import("cuda");
const dino = @import("idol_dino_cuda");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const Range = r.Range;
const Sum = r.Sum;
const Fill = r.g.Fill;
const rounding = @import("core").contract.bf16_round;

pub const W = 1280;
pub const F = 5120;
pub const D = 64;
/// The stream's rows (the class token, four registers, 64 x 64 patches) and the patches alone.
pub const rows = 4101;
pub const patches = 4096;
pub const prefix = 5;

/// A value rounded once to bf16, ties to even.
pub fn bf(v: f64) f64 {
    return rounding.bf16(v);
}

/// A range's ends each rounded to bf16.
pub fn bfr(x: Range) Range {
    return .{ .lo = bf(x.lo), .hi = bf(x.hi) };
}

/// bf16(a b) for bf16 a and b: the product is exact in fp32, so one rounding.
fn prod(a: f64, b: f64) Range {
    return bfr(Range.of(a * b));
}

fn none(_: c.Shape) usize {
    return 0;
}

const why = "no sums: each output is a few fp32 steps and its bf16 roundings, every one inside its interval";

fn plain(name: []const u8, served: []const c.Shape, ops: f64, mids: f64) c.Spec {
    return .{ .name = name, .accum = .none, .depth = none, .ops = ops, .mids = mids, .out = .bf16, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .why = why } };
}

fn wide(n_rows: usize, n: usize) c.Shape {
    return .{ .rows = n_rows, .n = n, .k = n };
}

/// tf_idol_dino_bias: y = bf16(acc + b), the patch embedding's fp32 sums and its bf16 bias.
pub const Bias = struct {
    pub const Kernels = dino.Kernels;
    pub const kinds = [_]r.Kind{ .f32, .bf16 };
    pub const out: r.Kind = .bf16;
    pub const in_place: ?usize = null;
    const served = [_]c.Shape{wide(patches, W)};

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n > 0 and s.n == s.k;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return plain(name, &served, 1, 0);
    }

    pub fn sizes(s: c.Shape) [2]usize {
        return .{ s.rows * s.n, s.n };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.n;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[2][]f64) void {
        r.uniform(rng, v[0], -6, 6);
        r.uniform(rng, v[1], -1, 1);
        if (mode == .nonfinite) for (v, 0..) |col, i| r.plant(col, &.{ 11 + 7 * i, 2048 + i });
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [2]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dino.bias(k, s, dev[0], dev[1], o, shape.rows, shape.n);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [2][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows * shape.n) |i| {
            const b = v[1][i % shape.n];
            e.put(want, mag, i, bf(v[0][i] + b), bfr(Range.of(v[0][i]).add(Range.of(b)).fp32(0.5)));
        }
    }
};

/// tf_idol_dino_scale_add: x = bf16(x + bf16(g bf16(acc + b))), in place over x (input 0): attn.proj's and fc2's.
pub const ScaleAdd = struct {
    pub const Kernels = dino.Kernels;
    pub const kinds = [_]r.Kind{ .bf16, .f32, .bf16, .bf16 };
    pub const out: r.Kind = .bf16;
    pub const in_place: ?usize = 0;
    const served = [_]c.Shape{wide(rows, W)};

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n == W and s.k == W;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return plain(name, &served, 2, 2);
    }

    pub fn sizes(s: c.Shape) [4]usize {
        return .{ s.rows * W, s.rows * W, W, W };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * W;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[4][]f64) void {
        r.uniform(rng, v[0], -8, 8);
        r.uniform(rng, v[1], -4, 4);
        r.uniform(rng, v[2], -1, 1);
        r.uniform(rng, v[3], -2, 2);
        if (mode == .nonfinite) for (v, 0..) |col, i| r.plant(col, &.{ 3 + 5 * i, 1000 + i });
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [4]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dino.scaleAdd(k, s, o, dev[1], dev[2], dev[3], shape.rows);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [4][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows * W) |i| {
            const x = v[0][i];
            const b = v[2][i % W];
            const g = v[3][i % W];
            const y = bf(v[1][i] + b);
            const yr = bfr(Range.of(v[1][i]).add(Range.of(b)).fp32(0.5));
            const gy = bfr(Range.of(g).mul(yr));
            e.put(want, mag, i, bf(x + bf(g * y)), bfr(gy.add(Range.of(x)).fp32(0.5)));
        }
    }
};

/// silu of one bf16 value g as the kernel forms it, g / (1 + expf(-g)) in fp32: its range.
fn siluAt(g: f64) Range {
    const den = Range.of(@exp(-g)).fp32(r.fn_ulps).add(Range.of(1)).fp32(0.5);
    return Range.of(g).div(den).fp32(r.div_ulps);
}

/// silu over g's range: both ends, and its minimum (-0.27846 at -1.27846) when the range holds it.
fn silu(gr: Range) Range {
    const a = siluAt(gr.lo);
    const b = siluAt(gr.hi);
    var out: Range = .{ .lo = @min(a.lo, b.lo), .hi = @max(a.hi, b.hi) };
    if (gr.lo <= -1.2784 and gr.hi >= -1.2785) out.lo = @min(out.lo, -0.27847);
    return out;
}

/// tf_idol_dino_swiglu: h = bf16(bf16(silu(g)) u), g and u each bf16(sum + bias) from fc1's sums [gate | up].
pub const Swiglu = struct {
    pub const Kernels = dino.Kernels;
    pub const kinds = [_]r.Kind{ .f32, .bf16 };
    pub const out: r.Kind = .bf16;
    pub const in_place: ?usize = null;
    const served = [_]c.Shape{wide(rows, F)};

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n == F and s.k == F;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return plain(name, &served, r.fn_ulps + r.div_ulps + 3, 3);
    }

    pub fn sizes(s: c.Shape) [2]usize {
        return .{ s.rows * 2 * F, 2 * F };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * F;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[2][]f64) void {
        r.uniform(rng, v[0], -8, 8);
        r.uniform(rng, v[1], -1, 1);
        if (mode == .nonfinite) {
            // a gate and an up sum of row 0, an up sum of row 1, a gate bias
            r.plant(v[0], &.{ 13, F + 17, 2 * F + F + 5 });
            v[1][29] = r.nan;
        }
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [2]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dino.swiglu(k, s, dev[0], dev[1], o, shape.rows);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [2][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows) |row| for (0..F) |j| {
            const sg = v[0][row * 2 * F + j];
            const su = v[0][row * 2 * F + F + j];
            const g = bf(sg + v[1][j]);
            const u = bf(su + v[1][F + j]);
            const gr = bfr(Range.of(sg).add(Range.of(v[1][j])).fp32(0.5));
            const ur = bfr(Range.of(su).add(Range.of(v[1][F + j])).fp32(0.5));
            const s = bf(g / (1 + @exp(-g)));
            e.put(want, mag, row * F + j, bf(s * u), bfr(bfr(silu(gr)).mul(ur)));
        };
    }
};

/// tf_idol_dino_heads: packed [rows, q | k | v] from qkv's fp32 sums, q and k turned past the prefix by bf16 cos, sin.
pub const Heads = struct {
    pub const Kernels = dino.Kernels;
    pub const kinds = [_]r.Kind{ .f32, .bf16, .bf16 };
    pub const out: r.Kind = .bf16;
    pub const in_place: ?usize = null;
    const served = [_]c.Shape{.{ .rows = rows, .n = 3 * W, .k = 1 }};

    pub fn accepts(s: c.Shape) bool {
        return s.rows > prefix and s.n == 3 * W;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return plain(name, &served, 1, 2);
    }

    pub fn sizes(s: c.Shape) [3]usize {
        return .{ s.rows * 3 * W, (s.rows - prefix) * D, (s.rows - prefix) * D };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * 3 * W;
    }

    pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, _: u32, v: *[3][]f64) void {
        r.uniform(rng, v[0], -4, 4);
        r.uniform(rng, v[1], -1, 1);
        r.uniform(rng, v[2], -1, 1);
        if (mode == .nonfinite) {
            // a q in the prefix and two past it, a v; a cos and a sin of the patches
            r.plant(v[0], &.{ 97, (6 % s.rows) * 3 * W + 300, (7 % s.rows) * 3 * W + W + 33, 2 * W + 11 });
            v[1][(1 % (s.rows - prefix)) * D + 5] = r.inf;
            v[2][(2 % (s.rows - prefix)) * D + 40] = r.nan;
        }
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dino.heads(k, s, dev[0], dev[1], dev[2], o, shape.rows);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows) |row| {
            const x = v[0][row * 3 * W ..][0 .. 3 * W];
            for (0..3 * W) |i| {
                const at = row * 3 * W + i;
                // the linear's rounding is one RN of an fp32 sum: exact to model
                const own = bf(x[i]);
                if (i >= 2 * W or row < prefix) {
                    e.put(want, mag, at, own, Range.of(own));
                    continue;
                }
                const j = i % D;
                // rotate_half: -x[j + 32] for j < 32, x[j - 32] past it
                const other = if (j < 32) -bf(x[i + 32]) else bf(x[i - 32]);
                const t = (row - prefix) * D + j;
                const cs = v[1][t];
                const sn = v[2][t];
                e.put(want, mag, at, bf(bf(own * cs) + bf(other * sn)), bfr(prod(own, cs).add(prod(other, sn)).fp32(0.5)));
            }
        }
    }
};

test "bf16 ranges round both ends to nearest even; silu's range holds its minimum" {
    const x = bfr(.{ .lo = 1 + 0x1p-8, .hi = 1 + 3 * 0x1p-8 });
    try std.testing.expectEqual(@as(f64, 1), x.lo);
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-6), x.hi);
    const m = silu(.{ .lo = -1.29, .hi = -1.27 });
    try std.testing.expect(m.lo <= -0.2784645 and m.hi > -0.2785 and m.hi < -0.2784);
    try std.testing.expect(silu(.{ .lo = -1.3, .hi = -1.29 }).lo > -0.27847);
}
