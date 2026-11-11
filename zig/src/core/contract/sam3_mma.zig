const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");

pub const m = 16;
pub const n = 32;
pub const k = 16;
pub const Range = enum { normal, subnormal, extreme, half_left, half_right, half_both };

pub const Panel = struct {
    count: usize,
    ta: bool,
    tb: bool,
    x: [m * k]f32 = undefined,
    y: [k * n]f32 = undefined,

    fn xi(self: *const Panel, row: usize, col: usize) usize {
        return if (self.ta) col * m + row else row * k + col;
    }

    fn yi(self: *const Panel, row: usize, col: usize) usize {
        return if (self.tb) col * k + row else row * n + col;
    }

    pub fn fill(self: *Panel, mode: rows.Fill, seed: u64, range: Range) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (&self.x) |*v| v.* = random.float(f32) * 4 - 2;
        for (&self.y) |*v| v.* = random.float(f32) * 4 - 2;
        switch (mode) {
            .random => if (range == .subnormal or range == .extreme) {
                @memset(&self.x, 0);
                @memset(&self.y, 0);
                if (range == .subnormal) {
                    for (0..m) |r| self.x[self.xi(r, 0)] = @bitCast(@as(u32, @intCast(r + 1)));
                    for (0..n) |j| self.y[self.yi(0, j)] = std.math.ldexp(@as(f32, 1), @intCast(110 + j % 17));
                } else {
                    for (0..m) |r| self.x[self.xi(r, 0)] = 0x1p-70;
                    for (0..n) |j| self.y[self.yi(0, j)] = 0x1p-70;
                }
            } else {
                if (range == .half_left or range == .half_both) for (&self.x) |*v| {
                    v.* = @as(f16, @floatCast(v.*));
                };
                if (range == .half_right or range == .half_both) for (&self.y) |*v| {
                    v.* = @as(f16, @floatCast(v.*));
                };
            },
            .nonfinite => {
                const at = [_][2]usize{ .{ 0, 3 }, .{ 1, 5 }, .{ 2, 7 }, .{ 3, 1 }, .{ 3, 9 } };
                const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, values) |p, v| self.x[self.xi(p[0], p[1])] = v;
            },
            .stress => {
                for (0..m) |r| for (0..k / 2) |i| {
                    const value: f32 = @floatFromInt(1024 + random.uintLessThan(u32, 1024));
                    self.x[self.xi(r, i)] = value;
                    self.x[self.xi(r, i + k / 2)] = value + 1;
                };
                for (0..n) |j| for (0..k / 2) |i| {
                    const value: f32 = @floatFromInt(128 + random.uintLessThan(u32, 128));
                    self.y[self.yi(i, j)] = value;
                    self.y[self.yi(i + k / 2, j)] = -value;
                };
            },
        }
    }

    fn terms(self: *const Panel, r: usize, j: usize) [k]f64 {
        var out: [k]f64 = undefined;
        for (&out, 0..) |*term, i| term.* = @as(f64, self.x[self.xi(r, i)]) * self.y[self.yi(i, j)];
        return out;
    }

    pub fn reference(self: *const Panel, want: []f64, mag: []f64) void {
        for (0..self.count) |r| for (0..n) |j| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (self.terms(r, j)) |term| {
                sum += term;
                magnitude += @abs(term);
            }
            want[r * n + j] = sum;
            mag[r * n + j] = magnitude;
        };
    }

    pub fn simulate(self: *const Panel, accum: c.Accum, order: rows.Order, out: []f64) void {
        for (0..self.count) |r| for (0..n) |j| {
            const values = self.terms(r, j);
            out[r * n + j] = rows.sumIn(accum, order, &values);
        };
    }
};

test "SAM FP32 MMA stress stays exact and rejects reduced sums in every layout and order" {
    var want: [4 * n]f64 = undefined;
    var magnitude: [4 * n]f64 = undefined;
    var got: [4 * n]f64 = undefined;
    for ([_]bool{ false, true }) |ta| for ([_]bool{ false, true }) |tb| {
        var panel = Panel{ .count = 4, .ta = ta, .tb = tb };
        panel.fill(.stress, 19, .normal);
        panel.reference(&want, &magnitude);
        for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
            panel.simulate(.f32, order, &got);
            try std.testing.expectEqualSlices(f64, &want, &got);
            for ([_]c.Accum{ .f16, .bf16 }) |accum| {
                panel.simulate(accum, order, &got);
                var worst: f64 = 0;
                for (got, want) |x, y| worst = @max(worst, if (!std.math.isFinite(x)) std.math.inf(f64) else @abs(x - y) / (0x1p-24 * @abs(y)));
                try std.testing.expect(worst >= 100);
            }
        }
    };
}
