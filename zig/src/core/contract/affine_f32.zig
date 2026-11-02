//! Independent affine values with native FP32 metadata, reusing only the logical code/input fixture and stress proof.
const std = @import("std");
const c = @import("contract.zig");
const rows = @import("rows.zig");
const Affine = @import("affine.zig").Affine;

pub const Fixture = struct {
    f: Affine,
    scales: []f32,
    biases: []f32,

    pub fn init(a: std.mem.Allocator, shape: c.Shape, bits: u8, group: usize) !Fixture {
        const f = try Affine.init(a, shape, bits, group);
        errdefer f.deinit();
        const scales = try a.alloc(f32, f.scales.len);
        errdefer a.free(scales);
        return .{ .f = f, .scales = scales, .biases = try a.alloc(f32, f.biases.len) };
    }
    pub fn deinit(p: Fixture) void {
        p.f.a.free(p.scales);
        p.f.a.free(p.biases);
        p.f.deinit();
    }
    pub fn fill(p: Fixture, mode: rows.Fill, seed: u64) !void {
        try p.f.fill(mode, seed);
        for (p.scales, p.biases, p.f.scales, p.f.biases) |*s, *b, hs, hb| {
            s.* = c.fromBf16(hs);
            b.* = c.fromBf16(hb);
            if (mode != .stress) {
                s.* += 0x1p-18;
                b.* -= 0x1p-19;
            }
        }
        if (mode == .random) p.witness();
    }
    pub fn witness(p: Fixture) void {
        @memset(p.f.x, c.toBf16(1));
        @memset(p.f.codes[0..p.f.shape.k], 1);
        @memset(p.scales[0 .. p.f.shape.k / p.f.group], 1.001953125);
        @memset(p.biases[0 .. p.f.shape.k / p.f.group], -1);
    }
    pub fn one(p: Fixture, row: usize, col: usize) [2]f64 {
        @setFloatMode(.strict);
        const f = p.f;
        var value: f64 = 0;
        var mag: f64 = 0;
        for (0..f.shape.k / f.group) |group| {
            var qx: f64 = 0;
            var xs: f64 = 0;
            var qm: f64 = 0;
            var xm: f64 = 0;
            for (group * f.group..(group + 1) * f.group) |i| {
                const x: f64 = c.fromBf16(f.x[row * f.shape.k + i]);
                const q: f64 = @floatFromInt(f.codes[col * f.shape.k + i]);
                qx += q * x;
                xs += x;
                qm += @abs(q * x);
                xm += @abs(x);
            }
            const at = col * (f.shape.k / f.group) + group;
            value += @as(f64, p.scales[at]) * qx + @as(f64, p.biases[at]) * xs;
            mag += @abs(@as(f64, p.scales[at])) * qm + @abs(@as(f64, p.biases[at])) * xm;
        }
        return .{ value, mag };
    }
    pub fn simulate(p: Fixture, acc: c.Accum, order: rows.Order, out: []f64) !void {
        for (p.scales, p.biases, p.f.scales, p.f.biases) |s, b, hs, hb| {
            if (s != c.fromBf16(hs) or b != c.fromBf16(hb)) return error.StressMetadataChanged;
        }
        return p.f.simulate(acc, order, out);
    }
};

test "all four native FP32 forms retain metadata that BF16 loses in a small projection" {
    for ([_][2]usize{ .{ 4, 32 }, .{ 5, 32 }, .{ 6, 16 }, .{ 8, 32 } }) |format| {
        var p = try Fixture.init(std.testing.allocator, .{ .rows = 3, .n = 8, .k = 128 }, @intCast(format[0]), format[1]);
        defer p.deinit();
        try p.fill(.random, 442);
        for (0..3) |row| try std.testing.expectEqual(@as(f64, 0.25), p.one(row, 0)[0]);
        try std.testing.expectEqual(@as(f32, 1), c.fromBf16(c.toBf16(p.scales[0])));
        try std.testing.expectEqual(@as(f32, -1), c.fromBf16(c.toBf16(p.biases[0])));
    }
}
