//! Idol's epilogues under the contract: gated residual, linear biases (GELU), condition sum, fp32 bias and the embed.
const std = @import("std");
const cuda = @import("cuda");
const idol = @import("idol_cuda");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const Range = r.Range;
const Sum = r.Sum;
const Fill = r.g.Fill;

fn none(_: c.Shape) usize {
    return 0;
}

const why = "no sums: each output is a few fp32 steps and its fp16 rounding, every one inside its interval";

/// A kernel with no sums: its outputs are held to their pointwise intervals alone.
fn plain(name: []const u8, out: c.Elem, served: []const c.Shape) c.Spec {
    return .{ .name = name, .accum = .none, .depth = none, .ops = 4, .mids = 2, .out = out, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .why = why } };
}

fn wide(rows: usize, n: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = n };
}

fn any(s: c.Shape) bool {
    return s.rows > 0 and s.n > 0 and s.n == s.k;
}

/// The 1024-wide residual stream's rows at every refiner.
const stream = [_]c.Shape{ wide(r.joint, r.hidden), wide(r.latent, r.hidden), wide(r.cond, r.hidden) };

/// tf_idol_gate_add: x = fp16(x + fp16(g y)), in place over x (input 0).
pub const GateAdd = struct {
    pub const kinds = [_]r.Kind{ .f16, .f16, .f16 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = 0;
    pub const accepts = any;

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return plain(name, .f16, &stream);
    }

    pub fn sizes(s: c.Shape) [3]usize {
        return .{ s.rows * s.n, s.rows * s.n, s.n };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.n;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[3][]f64) void {
        r.uniform(rng, v[0], -8, 8);
        r.uniform(rng, v[1], -4, 4);
        r.uniform(rng, v[2], -2, 2);
        if (mode == .nonfinite) for (v, 0..) |col, i| r.plant(col, &.{ 3 + 5 * i, 1000 + i });
    }

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, _: u32) !void {
        try idol.gateAdd(k, s, o, dev[1], dev[2], shape.rows, shape.n);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows * shape.n) |i| {
            const x = v[0][i];
            const y = v[1][i];
            const g = v[2][i % shape.n];
            const gy = Range.of(g).mul(Range.of(y)).fp32(0.5).half();
            e.put(want, mag, i, r.h(x + r.h(g * y)), gy.add(Range.of(x)).fp32(0.5).half());
        }
    }
};

/// The block linears' widths: attention's 3072 and 1024, the MLP's 4096 (GELU) and 1024, at every refiner.
const linears = [_]c.Shape{ wide(r.joint, 3 * r.hidden), wide(r.joint, r.ffn), wide(r.joint, r.hidden), wide(r.latent, r.ffn), wide(r.cond, r.ffn) };

/// torch's tanh GELU of an fp16 value as the kernel forms it in fp32 (tanhf within fn_ulps): value and range.
pub fn gelu(x: f64) struct { v: f64, r: Range } {
    const c1 = r.f(0.7978845608028654);
    const c2 = r.f(0.044715);
    const v = 0.5 * x * (1 + std.math.tanh(c1 * (x + c2 * x * x * x)));
    const xr = Range.of(x);
    const t = Range.of(c2).mul(xr).fp32(0.5).mul(xr).fp32(0.5).mul(xr).fp32(0.5);
    const inner = Range.of(c1).mul(xr.add(t).fp32(0.5)).fp32(0.5);
    const th = inner.up(tanh).fp32(r.fn_ulps);
    return .{ .v = v, .r = Range.of(0.5).mul(xr).mul(Range.of(1).add(th).fp32(0.5)).fp32(0.5) };
}

fn tanh(x: f64) f64 {
    return std.math.tanh(x);
}

/// tf_idol_bias: y = fp16(acc + b) from fp32 sums and an fp16 bias; variant 1 then takes fp16(GELU(y)).
pub const Bias = struct {
    pub const kinds = [_]r.Kind{ .f32, .f16 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = null;
    pub const accepts = any;

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return plain(name, .f16, &linears);
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

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [2]u64, o: u64, shape: c.Shape, variant: u32) !void {
        try idol.bias(k, s, dev[0], dev[1], o, shape.rows, shape.n, variant == 1);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, variant: u32, v: [2][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows * shape.n) |i| {
            const y = r.h(v[0][i] + v[1][i % shape.n]);
            const yr = Range.of(v[0][i]).add(Range.of(v[1][i % shape.n])).fp32(0.5).half();
            if (variant != 1) {
                e.put(want, mag, i, y, yr);
                continue;
            }
            // GELU's ranges at both fp16 ends, and its minimum (-0.16997 at -0.7518) when the range holds it
            const lo = gelu(yr.lo);
            const hi = gelu(yr.hi);
            const g = gelu(y);
            var gr: Range = .{ .lo = @min(@min(lo.r.lo, hi.r.lo), g.r.lo), .hi = @max(@max(lo.r.hi, hi.r.hi), g.r.hi) };
            if (yr.lo <= -0.751 and yr.hi >= -0.753) gr.lo = @min(gr.lo, -0.1701);
            e.put(want, mag, i, r.h(g.v), gr.half());
        }
    }
};

/// tf_idol_cond: y = fp16((a + ba) + (c + bc)), the two condition projections' fp32 sums and biases.
pub const Cond = struct {
    pub const kinds = [_]r.Kind{ .f32, .f32, .f32, .f32 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = null;
    pub const accepts = any;
    const served = [_]c.Shape{wide(r.cond, r.hidden)};

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return plain(name, .f16, &served);
    }

    pub fn sizes(s: c.Shape) [4]usize {
        return .{ s.rows * s.n, s.n, s.rows * s.n, s.n };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.n;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[4][]f64) void {
        for (v) |col| r.uniform(rng, col, -4, 4);
        if (mode == .nonfinite) for (v, 0..) |col, i| r.plant(col, &.{ 17 + 3 * i, 3000 + i });
    }

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [4]u64, o: u64, shape: c.Shape, _: u32) !void {
        try idol.cond(k, s, dev[0], dev[1], dev[2], dev[3], o, shape.rows, shape.n);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [4][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows * shape.n) |i| {
            const j = i % shape.n;
            const a = Range.of(v[0][i]).add(Range.of(v[1][j])).fp32(0.5);
            const b = Range.of(v[2][i]).add(Range.of(v[3][j])).fp32(0.5);
            e.put(want, mag, i, r.h((v[0][i] + v[1][j]) + (v[2][i] + v[3][j])), a.add(b).fp32(0.5).half());
        }
    }
};

/// tf_idol_bias32: y += b in fp32, in place over y (input 0): RePo's 256 gate | content sums, the output's 16.
pub const Bias32 = struct {
    pub const kinds = [_]r.Kind{ .f32, .f32 };
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = 0;
    pub const accepts = any;
    const served = [_]c.Shape{ wide(r.joint, 256), wide(r.latent, 256), wide(r.cond, 256), wide(r.latent, 16) };

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return plain(name, .f32, &served);
    }

    pub fn sizes(s: c.Shape) [2]usize {
        return .{ s.rows * s.n, s.n };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.n;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[2][]f64) void {
        r.uniform(rng, v[0], -10, 10);
        r.uniform(rng, v[1], -1, 1);
        if (mode == .nonfinite) for (v, 0..) |col, i| r.plant(col, &.{ 5 + 9 * i, 777 + i });
    }

    pub fn launch(k: *const idol.Kernels, s: cuda.Stream, dev: [2]u64, o: u64, shape: c.Shape, _: u32) !void {
        try idol.bias32(k, s, o, dev[1], shape.rows, shape.n);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [2][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows * shape.n) |i| {
            const b = v[1][i % shape.n];
            e.put(want, mag, i, r.f(v[0][i] + b), Range.of(v[0][i]).add(Range.of(b)).fp32(0.5));
        }
    }
};
