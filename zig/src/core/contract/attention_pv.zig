//! Independent scalar products for the shared attention P x V, including the legacy operand cut.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");

pub const Mode = enum { cut, fp32, cut_control };
pub const shapes = [_]c.Shape{ .{ .rows = 4, .n = 32, .k = 32 }, .{ .rows = 17, .n = 33, .k = 65 }, .{ .rows = 4, .n = 32, .k = 4096 } };
pub fn name(m: Mode) []const u8 {
    return switch (m) {
        .cut => "metal attention PV cut",
        .fp32 => "metal attention PV fp32",
        .cut_control => "control: metal attention PV cut against fp32",
    };
}
fn depth(s: c.Shape) usize {
    return s.k;
}
pub fn spec(m: Mode) c.Spec {
    return .{ .name = name(m), .accum = .f32, .depth = depth, .out = .f32, .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{ .size = 16 } }, .served = &shapes };
}

/// The observed M5 relaxed operand conversion, independent of the device helper.
pub fn cut(p: f32) f32 {
    return @bitCast(@as(u32, @bitCast(p)) & 0xffffe000);
}

pub const Fixture = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    p: []f32,
    v: []u16,

    pub fn init(a: std.mem.Allocator, s: c.Shape) !Fixture {
        const p = try a.alloc(f32, s.rows * s.k);
        errdefer a.free(p);
        return .{ .a = a, .shape = s, .p = p, .v = try a.alloc(u16, s.k * s.n) };
    }
    pub fn deinit(f: Fixture) void {
        f.a.free(f.p);
        f.a.free(f.v);
    }

    pub fn fill(f: Fixture, mode: rows.Fill, seed: u64) !void {
        const s = f.shape;
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        for (f.p) |*p| p.* = random.float(f32);
        for (f.v) |*v| v.* = c.toBf16(random.float(f32) * 2 - 1);
        if (mode == .stress) {
            const d = try rows.Rows.init(f.a, .dot, .{ .rows = s.rows, .n = 1, .k = s.k });
            defer d.deinit();
            d.fill(.stress, seed);
            for (f.p, d.x) |*p, x| p.* = c.fromBf16(x);
            for (0..s.k) |k| @memset(f.v[k * s.n ..][0..s.n], d.w[k]);
        } else if (mode == .nonfinite) {
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (values, 0..) |v, r| if (r < s.rows) {
                f.p[r * s.k] = v;
            };
            if (s.rows > 3) {
                f.p[3 * s.k] = 0;
                f.v[0] = c.toBf16(std.math.inf(f32));
            }
        } else if (s.k >= 2) {
            // Cancellation exposes a cut probability even when a long random row would hide it.
            @memset(f.p[0..s.k], 0);
            f.p[0] = 1;
            f.p[1] = 0.999267578125;
            for (0..s.n) |j| {
                f.v[j] = c.toBf16(1);
                f.v[s.n + j] = c.toBf16(-1);
            }
        }
    }
    fn terms(f: Fixture, m: Mode, r: usize, j: usize, out: []f64) void {
        for (out, 0..) |*t, k| {
            const p = f.p[r * f.shape.k + k];
            t.* = @as(f64, if (m == .cut) cut(p) else p) * c.fromBf16(f.v[k * f.shape.n + j]);
        }
    }
    pub fn reference(f: Fixture, m: Mode, want: []f64, mag: []f64) !void {
        const t = try f.a.alloc(f64, f.shape.k);
        defer f.a.free(t);
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            f.terms(m, r, j, t);
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (t) |v| {
                sum += v;
                magnitude += @abs(v);
            }
            want[r * f.shape.n + j] = sum;
            mag[r * f.shape.n + j] = magnitude;
        };
    }
    pub fn simulate(f: Fixture, m: Mode, acc: c.Accum, order: rows.Order, out: []f64) !void {
        const t = try f.a.alloc(f64, f.shape.k);
        defer f.a.free(t);
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            f.terms(m, r, j, t);
            out[r * f.shape.n + j] = rows.sumIn(acc, order, t);
        };
    }
};

test "attention PV: the cut mode fails the full probability's derived bound" {
    const f = try Fixture.init(std.testing.allocator, shapes[0]);
    defer f.deinit();
    try f.fill(.random, 1772);
    var want: [128]f64 = undefined;
    var mag: [128]f64 = undefined;
    var legacy: [128]f64 = undefined;
    try f.reference(.fp32, &want, &mag);
    try f.simulate(.cut, .f32, .sequential, &legacy);
    try std.testing.expectEqual(@as(f64, 3.0 / 4096.0), want[0]);
    try std.testing.expectEqual(@as(f64, 1.0 / 1024.0), legacy[0]);
    try std.testing.expect(@abs(legacy[0] - want[0]) > c.bound(spec(.fp32), shapes[0], want[0], mag[0]));
}

test "attention PV: stress requires fp32 sums in all three independent orders" {
    const f = try Fixture.init(std.testing.allocator, shapes[2]);
    defer f.deinit();
    try f.fill(.stress, 1772);
    var want: [128]f64 = undefined;
    var mag: [128]f64 = undefined;
    var got: [128]f64 = undefined;
    try f.reference(.fp32, &want, &mag);
    for (std.enums.values(rows.Order)) |order| {
        try f.simulate(.fp32, .f32, order, &got);
        try std.testing.expectEqualSlices(f64, &want, &got);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try f.simulate(.fp32, acc, order, &got);
            var most: f64 = 0;
            for (got, want) |g, v| most = @max(most, if (std.math.isFinite(g)) @abs(g - v) / c.boundExact(spec(.fp32), v) else std.math.inf(f64));
            try std.testing.expect(most >= 100);
        }
    }
}

test "attention PV: every M5 FMA supplier maps to the requested matrix coordinate" {
    for (0..32) |lane| for (0..16) |k| {
        const row = ((lane & 16) >> 2) | ((lane >> 1) & 3);
        const col = (lane & 8) + ((lane & 1) << 2);
        const pl = (lane & 22) | ((k >> 2) & 1) | ((k >> 3) << 3);
        const vl = ((k & 3) << 1) | ((k & 4) << 2) | (lane & 9);
        try std.testing.expectEqual(row, ((pl & 16) >> 2) | ((pl >> 1) & 3));
        try std.testing.expectEqual(k, (pl & 8) + ((pl & 1) << 2) + (k & 3));
        try std.testing.expectEqual(k, ((vl & 16) >> 2) | ((vl >> 1) & 3) | ((k >> 3) * 8));
        try std.testing.expectEqual(col, (vl & 8) + ((vl & 1) << 2));
    };
}
