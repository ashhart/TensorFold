//! MLX affine-quantized projections for the contract: synthetic weights and rows, float64 reference, simulated sums.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");

/// The formula a kernel computes: MLX's per-group s*sum(q x) + b*sum(x), or each weight s*q + b first.
pub const Form = enum { factored, dequantized };

/// Host inputs of y = x W^T: x [rows, k] bf16, codes [n, k], scales and biases [n, k / group] bf16.
pub const Affine = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    bits: u8,
    group: usize,
    form: Form = .factored,
    /// What the kernel adds to each code before its sums (affine_mm's 4-bit codes enter as q + 128): the sums' size.
    off: f64 = 0,
    x: []u16,
    codes: []u8,
    scales: []u16,
    biases: []u16,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, bits: u8, group: usize) !Affine {
        if (shape.k % group != 0) return error.GroupSplitsRow;
        const x = try a.alloc(u16, shape.rows * shape.k);
        errdefer a.free(x);
        const codes = try a.alloc(u8, shape.n * shape.k);
        errdefer a.free(codes);
        const scales = try a.alloc(u16, shape.n * (shape.k / group));
        errdefer a.free(scales);
        return .{ .a = a, .shape = shape, .bits = bits, .group = group, .x = x, .codes = codes, .scales = scales, .biases = try a.alloc(u16, scales.len) };
    }

    pub fn deinit(f: Affine) void {
        f.a.free(f.x);
        f.a.free(f.codes);
        f.a.free(f.scales);
        f.a.free(f.biases);
    }

    pub fn outputs(f: Affine) usize {
        return f.shape.rows * f.shape.n;
    }

    pub fn fill(f: Affine, mode: rows.Fill, seed: u64) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const top: u32 = @as(u32, 1) << @intCast(f.bits);
        for (f.codes) |*q| q.* = @intCast(rng.uintLessThan(u32, top));
        for (f.scales) |*s| s.* = c.toBf16(rng.float(f32) * 0.02 + 0.001);
        for (f.biases) |*b| b.* = c.toBf16((rng.float(f32) - 0.66) * 0.15);
        for (f.x, 0..) |*v, i| v.* = if (i % 29 == 0) 0 else c.toBf16(rng.float(f32) * 4 - 2);
        switch (mode) {
            .random => {},
            .nonfinite => {
                const k = f.shape.k;
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |p, v| if (p[0] < f.shape.rows) {
                    f.x[p[0] * k + p[1]] = c.toBf16(v);
                };
            },
            .stress => try f.stress(rng),
        }
    }

    /// Gate 4's inputs: s 1, b -2^(bits-1); the second half mirrors the first, w negated and 8 inputs nudged by one.
    fn stress(f: Affine, rng: std.Random) !void {
        const k = f.shape.k;
        const h = k / 2;
        const half: u64 = @as(u64, 1) << @intCast(f.bits - 1);
        const top: u64 = 2 * half + @as(u64, @intFromFloat(f.off));
        const xmax: u64 = @min(255, ((1 << 24) - 1) / (top * k)); // every sum of |(q + off) x| under 2^24, x + 1 a bf16
        if (xmax == 0) return error.StressPastFp32;
        @memset(f.scales, c.toBf16(1));
        @memset(f.biases, c.toBf16(-@as(f32, @floatFromInt(half))));
        for (0..f.shape.n) |j| for (0..h) |i| {
            const q = half + 1 + rng.uintLessThan(u64, half - 1);
            f.codes[j * k + i] = @intCast(q);
            f.codes[j * k + h + i] = @intCast(2 * half - q);
        };
        for (0..f.shape.rows) |r| for (0..h) |i| {
            const v = 1 + rng.uintLessThan(u64, xmax);
            f.x[r * k + i] = c.toBf16(@floatFromInt(v));
            f.x[r * k + h + i] = c.toBf16(@floatFromInt(v + @intFromBool(i < 8)));
        };
        if (k % 2 == 1) for (0..f.shape.rows) |r| {
            f.x[r * k + k - 1] = 0;
        };
    }

    /// The codes as MLX packs them: a little-endian bit stream, k * bits / 32 words a row.
    pub fn words(f: Affine, out: []u32) void {
        @memset(out, 0);
        for (f.codes, 0..) |q, i| for (0..f.bits) |b| if (q >> @intCast(b) & 1 != 0) {
            const at = i * f.bits + b;
            out[at / 32] |= @as(u32, 1) << @intCast(at % 32);
        };
    }

    fn xv(f: Affine, r: usize, i: usize) f64 {
        return c.fromBf16(f.x[r * f.shape.k + i]);
    }

    fn sb(f: Affine, j: usize, g: usize) [2]f64 {
        const at = j * (f.shape.k / f.group) + g;
        return .{ c.fromBf16(f.scales[at]), c.fromBf16(f.biases[at]) };
    }

    /// Output (r, j) in float64 by the kernel's form, and the magnitude of the terms its sums add.
    fn one(f: Affine, r: usize, j: usize) [2]f64 {
        const k = f.shape.k;
        var y: f64 = 0;
        var mag: f64 = 0;
        for (0..k / f.group) |g| {
            const p = f.sb(j, g);
            var qx: f64 = 0;
            var xs: f64 = 0;
            var qa: f64 = 0;
            var xa: f64 = 0;
            for (g * f.group..(g + 1) * f.group) |i| {
                const x = f.xv(r, i);
                const q: f64 = @floatFromInt(f.codes[j * k + i]);
                switch (f.form) {
                    .factored => {
                        qx += q * x;
                        xs += x;
                    },
                    .dequantized => qx += (p[0] * q + p[1]) * x,
                }
                qa += @abs((q + f.off) * x);
                xa += @abs(x);
            }
            y += if (f.form == .factored) p[0] * qx + p[1] * xs else qx;
            mag += @abs(p[0]) * qa + @abs(p[1] - f.off * p[0]) * xa;
        }
        return .{ y, mag };
    }

    pub fn reference(f: Affine, want: []f64, mag: []f64) !void {
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            const o = f.one(r, j);
            want[r * f.shape.n + j] = o[0];
            mag[r * f.shape.n + j] = o[1];
        };
    }

    /// On gate 4's inputs (one scale and bias): each output as an accumulator leaves it, closest of three structures.
    pub fn simulate(f: Affine, acc: c.Accum, order: rows.Order, out: []f64) !void {
        const k = f.shape.k;
        const t = try f.a.alloc(f64, 2 * k);
        defer f.a.free(t);
        const gs = try f.a.alloc(f64, k / f.group);
        defer f.a.free(gs);
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            const want = f.one(r, j)[0];
            const p = f.sb(j, 0);
            for (0..k) |i| {
                const q: f64 = @floatFromInt(f.codes[j * k + i]);
                t[i] = (q + f.off) * f.xv(r, i);
                t[k + i] = f.xv(r, i);
            }
            const whole = rows.sumIn(acc, order, t[0..k]) * p[0] + rows.sumIn(acc, order, t[k..]) * (p[1] - f.off * p[0]);
            for (gs, 0..) |*y, g| {
                const pg = f.sb(j, g);
                y.* = rows.sumIn(acc, .sequential, t[g * f.group ..][0..f.group]) * pg[0] + rows.sumIn(acc, .sequential, t[k + g * f.group ..][0..f.group]) * (pg[1] - f.off * pg[0]);
            }
            const grouped = rows.sumIn(acc, order, gs);
            for (0..k) |i| t[i] = (p[0] * @as(f64, @floatFromInt(f.codes[j * k + i])) + p[1]) * f.xv(r, i);
            const direct = rows.sumIn(acc, order, t[0..k]);
            var best = whole;
            for ([_]f64{ grouped, direct }) |v| if (!std.math.isFinite(best) or @abs(v - want) < @abs(best - want)) {
                best = v;
            };
            out[r * f.shape.n + j] = best;
        };
    }
};

test "the stress weights: exact fp32 sums, every fp16 and bf16 structure far off" {
    const a = std.testing.allocator;
    for ([_]u8{ 2, 4, 8, 4 }, 0..) |bits, i| {
        var f = try Affine.init(a, .{ .rows = 2, .n = 8, .k = 4096 }, bits, 64);
        defer f.deinit();
        if (i == 3) f.off = 128;
        try f.fill(.stress, bits);
        var want: [16]f64 = undefined;
        var mag: [16]f64 = undefined;
        try f.reference(&want, &mag);
        var got: [16]f64 = undefined;
        for ([_]rows.Order{ .sequential, .pairwise, .strided }) |o| {
            try f.simulate(.f32, o, &got);
            try std.testing.expectEqualSlices(f64, &want, &got);
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                try f.simulate(acc, o, &got);
                var most: f64 = 0;
                for (got, want) |g, w| most = @max(most, if (std.math.isFinite(g)) @abs(g - w) / (0x1p-8 * @abs(w) + 0x1p-133) else std.math.inf(f64));
                try std.testing.expect(most >= 100);
            }
        }
    }
}

test "MLX's bit stream: 3-bit codes cross word boundaries" {
    const a = std.testing.allocator;
    const f = try Affine.init(a, .{ .rows = 1, .n = 1, .k = 32 }, 3, 32);
    defer f.deinit();
    for (f.codes, 0..) |*q, i| q.* = @intCast(i % 8);
    var w: [3]u32 = undefined;
    f.words(&w);
    try std.testing.expectEqual(@as(u32, 0b10_001_000) | (0b011 << 9) | (0b100 << 12), w[0] & 0x7fff);
    try std.testing.expectEqual(@as(u32, 2), (w[0] >> 30) | ((w[1] & 1) << 2)); // code 10 = 2 at bits 30-32
}
