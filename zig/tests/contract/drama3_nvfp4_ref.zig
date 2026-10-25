//! Independent packed-format FP64 projections and signed ranges at Drama's BF16 SwiGLU nodes.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const rows = contract.rows;
const bf = contract.bf16_round.bf16;
pub const Interval = @import("gemma4_math.zig").Interval;

pub const Variant = enum { up, down };
pub const slots = 8;
pub const experts = 9;

pub fn depth(s: c.Shape) usize {
    return 16 + (s.k + 511) / 512 + 5;
}

pub fn fp4(code: u8) f64 {
    const magnitude: f64 = switch (code % 8) {
        0 => 0,
        1 => 0.5,
        2 => 1,
        3 => 1.5,
        4 => 2,
        5 => 3,
        6 => 4,
        7 => 6,
        else => unreachable,
    };
    return if (code >= 8) -magnitude else magnitude;
}

pub fn scale(code: u8) f64 {
    const bits = code % 128;
    if (bits == 127) return std.math.nan(f64);
    const exponent = bits / 8;
    const mantissa = bits % 8;
    const value = if (exponent == 0)
        @as(f64, @floatFromInt(mantissa)) / 512
    else
        std.math.ldexp(@as(f64, @floatFromInt(8 + @as(u16, mantissa))), @as(i32, exponent) - 10);
    return if (code >= 128) -value else value;
}

pub const Dot = struct { value: f64, magnitude: f64 };
pub const Weights = struct {
    a: std.mem.Allocator,
    codes: []u8,
    scales: []u8,
    global: f32,
    k: usize,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, fill: rows.Fill, seed: u64, projection: usize) !Weights {
        const codes = try a.alloc(u8, shape.n * shape.k / 2);
        errdefer a.free(codes);
        const scales = try a.alloc(u8, shape.n * shape.k / 16);
        var rng_state = std.Random.DefaultPrng.init(seed);
        const rng = rng_state.random();
        for (codes) |*v| v.* = rng.int(u8);
        for (scales, 0..) |*v, i| v.* = switch (i % 131) {
            0 => 0,
            1 => 0x80,
            else => @intCast((i +% seed) % 127),
        };
        if (fill == .stress) {
            @memset(scales, 0x38);
            for (0..shape.n) |n| for (0..shape.k / 2) |j| {
                const constant = projection < 2 and n % 2 != projection;
                const lo: u8 = if (constant) (if (j == 0) 2 else 0) else if (2 * j < shape.k / 2) 2 else 10;
                const hi: u8 = if (constant) 0 else if (2 * j + 1 < shape.k / 2) 2 else 10;
                codes[n * (shape.k / 2) + j] = lo | (hi << 4);
            };
        }
        return .{
            .a = a,
            .codes = codes,
            .scales = scales,
            .k = shape.k,
            .global = if (fill == .stress) 1 else @as(f32, @floatFromInt(1 + seed % 7)) / 4096,
        };
    }

    pub fn deinit(w: Weights) void {
        w.a.free(w.codes);
        w.a.free(w.scales);
    }

    fn code(w: Weights, n: usize, k: usize) u8 {
        const byte = w.codes[n * (w.k / 2) + k / 2];
        return if (k % 2 == 0) byte % 16 else byte / 16;
    }

    pub fn dot(w: Weights, n: usize, x: []const u16) Dot {
        var sum: f64 = 0;
        var magnitude: f64 = 0;
        for (0..w.k / 16) |block| {
            var subtotal: f64 = 0;
            var absolute: f64 = 0;
            for (0..16) |j| {
                const k = block * 16 + j;
                const term = fp4(w.code(n, k)) * @as(f64, c.fromBf16(x[k]));
                subtotal += term;
                absolute += @abs(term);
            }
            const s = scale(w.scales[n * (w.k / 16) + block]);
            sum += subtotal * s;
            magnitude += absolute * @abs(s);
        }
        return .{ .value = sum * w.global, .magnitude = magnitude * @abs(w.global) };
    }

    pub fn simulate(w: Weights, n: usize, x: []const u16, acc: c.Accum, order: rows.Order) f64 {
        var blocks: [128]f64 = undefined;
        for (blocks[0 .. w.k / 16], 0..) |*out, block| {
            var terms: [16]f64 = undefined;
            for (&terms, 0..) |*term, j| {
                const k = block * 16 + j;
                term.* = fp4(w.code(n, k)) * @as(f64, c.fromBf16(x[k]));
            }
            out.* = rows.sumIn(acc, order, &terms) * scale(w.scales[n * (w.k / 16) + block]);
        }
        return rows.sumIn(acc, order, blocks[0 .. w.k / 16]) * w.global;
    }
};

pub fn fillX(x: []u16, k: usize, variant: Variant, fill: rows.Fill, seed: u64) void {
    var state = std.Random.DefaultPrng.init(seed);
    const rng = state.random();
    for (x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) / 32);
    const count = x.len / k;
    if (fill == .stress) {
        // Integer partials stay below 2^24; a unit residue is lost inside a block in reduced precision.
        for (0..count) |r| {
            for (0..k / 2) |i| {
                const v: f32 = @floatFromInt(256 * (1 + rng.uintLessThan(u32, 16)));
                x[r * k + i] = c.toBf16(v);
                x[r * k + k / 2 + i] = c.toBf16(v);
            }
            x[r * k] = c.toBf16(1);
            x[r * k + k / 2] = 0;
        }
    } else if (fill == .nonfinite) {
        for (0..count) |r| {
            const token = if (variant == .up) r else r / slots;
            switch (token) {
                0 => x[r * k + k / 3] = c.toBf16(std.math.nan(f32)),
                1 => x[r * k + k / 2] = c.toBf16(std.math.inf(f32)),
                2 => x[r * k + k - 1] = c.toBf16(-std.math.inf(f32)),
                3 => {
                    x[r * k] = c.toBf16(std.math.inf(f32));
                    x[r * k + k - 1] = c.toBf16(-std.math.inf(f32));
                },
                else => {},
            }
        }
    }
}

pub fn swiglu(gate: f64, up: f64) f64 {
    const g = bf(gate);
    return bf(g / (1 + @exp(-g)) * bf(up));
}

fn fp32(x: Interval) Interval {
    const error_bound = c.unit(.f32) * @max(@abs(x.lo), @abs(x.hi)) + 0x1p-149;
    return .{ .lo = x.lo - error_bound, .hi = x.hi + error_bound };
}

pub fn projectionRange(d: Dot, shape: c.Shape, exact: bool) Interval {
    const reduction = if (exact) 0 else c.gamma(@floatFromInt(depth(shape)), c.unit(.f32)) * d.magnitude;
    // The projection global is a separate FP32 multiply after the SIMD reduction.
    return fp32(Interval.around(d.value, reduction));
}

pub fn upRange(gate: Interval, up: Interval) Interval {
    const g = Interval{ .lo = bf(gate.lo), .hi = bf(gate.hi) };
    const u = Interval{ .lo = bf(up.lo), .hi = bf(up.hi) };
    // precise::exp's three-ulp FP32 envelope; the following add, divide and multiply each round once.
    const exp_error = 6 * c.unit(.f32);
    const ex = Interval{ .lo = @exp(-g.hi) * (1 - exp_error), .hi = @exp(-g.lo) * (1 + exp_error) };
    const denom = fp32(ex.add(Interval.point(1)));
    const inverse = Interval{ .lo = 1 / denom.hi, .hi = 1 / denom.lo };
    return fp32(g.mul(inverse)).mul(u).rounded();
}

pub const shared_gate_terms = 65;

pub fn sharedGateStressTerm(i: usize) f32 {
    return if (i < 16) 32768 else if (i < 48) 1 else if (i < 64) -32768 else if (i == 64) -31 else 0;
}

pub fn sharedGateOutput(dot: f64) f64 {
    return bf(2 / (1 + @exp(-dot)));
}

test "shared gate stress remains discriminating after sigmoid and BF16 output" {
    var terms: [2048]f64 = @splat(0);
    for (terms[0..shared_gate_terms], 0..) |*v, i| v.* = sharedGateStressTerm(i);
    const want = sharedGateOutput(1);
    try std.testing.expectEqual(@as(f64, 1.4609375), want);
    for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
        try std.testing.expectEqual(@as(f64, 1), rows.sumIn(.f32, order, &terms));
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            const got = sharedGateOutput(rows.sumIn(acc, order, &terms));
            // Exact FP32 sums yield one stable BF16 value, so the stress interval has zero width.
            try std.testing.expect(!std.math.isFinite(got) or got != want);
        }
    }
}

test "NVFP4 independent decoding includes signs, subnormal scales, signed zero and maximum finite scale" {
    try std.testing.expectEqual(@as(f64, 0.5), fp4(1));
    try std.testing.expectEqual(@as(f64, -6), fp4(15));
    try std.testing.expect(std.math.signbit(fp4(8)));
    try std.testing.expectEqual(@as(f64, 0x1p-9), scale(1));
    try std.testing.expectEqual(@as(f64, 1), scale(0x38));
    try std.testing.expectEqual(@as(f64, 448), scale(126));
    try std.testing.expectEqual(@as(f64, 0), scale(0x80));
    try std.testing.expect(std.math.signbit(scale(0x80)));
}

test "block16 integer cancellation discriminates both reduced accumulators in all three orders" {
    const a = std.testing.allocator;
    for ([_]usize{ 512, 2048 }) |k| {
        const shape = c.Shape{ .rows = 1, .n = 2, .k = k };
        const x = try a.alloc(u16, k);
        defer a.free(x);
        fillX(x, k, .up, .stress, 17);
        for (0..3) |projection| {
            const w = try Weights.init(a, shape, .stress, 23, projection);
            defer w.deinit();
            const n = if (projection < 2) projection else 0;
            const dot = w.dot(n, x);
            try std.testing.expectEqual(@as(f64, 1), dot.value);
            try std.testing.expect(dot.magnitude < 0x1p24);
            for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
                try std.testing.expectEqual(dot.value, w.simulate(n, x, .f32, order));
                for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                    const got = w.simulate(n, x, acc, order);
                    const want_up = if (projection == 0) swiglu(dot.value, 1) else swiglu(1, dot.value);
                    const got_up = if (projection == 0) swiglu(got, 1) else swiglu(1, got);
                    const miss = if (projection < 2) @abs(got_up - want_up) else @abs(got - dot.value);
                    const bound = if (projection < 2) c.unit(.bf16) * @abs(want_up) else c.unit(.f32);
                    try std.testing.expect(!std.math.isFinite(got) or miss >= 100 * bound);
                }
            }
        }
    }
}
