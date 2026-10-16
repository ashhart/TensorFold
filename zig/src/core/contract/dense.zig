//! Dot products against dense weights for the contract: x [rows, k] bf16 times W [n, k] as the kernel dequantizes it.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");

/// Host inputs: x as bf16 bits, W as the float64 values the kernel multiplies by; outputs [slices][rows][n].
pub const Dense = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    slices: usize = 1,
    x: []u16,
    w: []f64,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, slices: usize) !Dense {
        if (slices == 0 or shape.k % slices != 0) return error.SliceSplitsRow;
        const x = try a.alloc(u16, shape.rows * shape.k);
        errdefer a.free(x);
        return .{ .a = a, .shape = shape, .slices = slices, .x = x, .w = try a.alloc(f64, shape.n * shape.k) };
    }

    pub fn deinit(d: Dense) void {
        d.a.free(d.x);
        d.a.free(d.w);
    }

    pub fn outputs(d: Dense) usize {
        return d.slices * d.shape.rows * d.shape.n;
    }

    /// x for a gate: values in [-2, 2), NaN and infinities in rows 0-3, or gate 4's mirrored multiples of `step`.
    pub fn fillX(d: Dense, mode: rows.Fill, seed: u64, step: f32) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = d.shape.k;
        for (d.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        switch (mode) {
            .random => {},
            .nonfinite => {
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |p, v| if (p[0] < d.shape.rows) {
                    d.x[p[0] * k + p[1]] = c.toBf16(v);
                };
            },
            .stress => for (0..d.shape.rows) |r| for (0..k / 2) |i| {
                const v = step * @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 32)));
                d.x[r * k + i] = c.toBf16(v);
                d.x[r * k + k / 2 + i] = c.toBf16(v + if (i < 8) step else 0);
            },
        }
    }

    fn slice(d: Dense) usize {
        return d.shape.k / d.slices;
    }

    fn terms(d: Dense, s: usize, r: usize, j: usize, t: []f64) void {
        const k = d.shape.k;
        const per = d.slice();
        for (t, s * per..) |*o, i| o.* = c.fromBf16(d.x[r * k + i]) * d.w[j * k + i];
    }

    pub fn reference(d: Dense, want: []f64, mag: []f64) !void {
        const t = try d.a.alloc(f64, d.slice());
        defer d.a.free(t);
        for (0..d.slices) |s| for (0..d.shape.rows) |r| for (0..d.shape.n) |j| {
            d.terms(s, r, j, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            const o = (s * d.shape.rows + r) * d.shape.n + j;
            want[o] = sum;
            mag[o] = abs;
        };
    }

    pub fn simulate(d: Dense, acc: c.Accum, order: rows.Order, out: []f64) !void {
        const t = try d.a.alloc(f64, d.slice());
        defer d.a.free(t);
        for (0..d.slices) |s| for (0..d.shape.rows) |r| for (0..d.shape.n) |j| {
            d.terms(s, r, j, t);
            out[(s * d.shape.rows + r) * d.shape.n + j] = rows.sumIn(acc, order, t);
        };
    }
};

test "dense stress rows: integer terms, exact in fp32, far off in fp16 and bf16" {
    const a = std.testing.allocator;
    const d = try Dense.init(a, .{ .rows = 2, .n = 4, .k = 4096 }, 1);
    defer d.deinit();
    d.fillX(.stress, 5, 2);
    for (0..4) |j| for (0..4096) |i| {
        const w: f64 = @floatFromInt(1 + (i * 7 + j) % 6);
        d.w[j * 4096 + i] = if (i < 2048) w else -@as(f64, @floatFromInt(1 + ((i - 2048) * 7 + j) % 6));
    };
    var want: [8]f64 = undefined;
    var mag: [8]f64 = undefined;
    try d.reference(&want, &mag);
    var got: [8]f64 = undefined;
    for ([_]rows.Order{ .sequential, .pairwise, .strided }) |o| {
        try d.simulate(.f32, o, &got);
        try std.testing.expectEqualSlices(f64, &want, &got);
        try d.simulate(.bf16, o, &got);
        var most: f64 = 0;
        for (got, want) |g, w| most = @max(most, @abs(g - w) / (0x1p-8 * @abs(w) + 0x1p-133));
        try std.testing.expect(most >= 100);
    }
}
