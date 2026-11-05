//! Float64 oracles and input fills for DeepSeek V4.1's glue kernels: the act-quant round trips, from the math alone.
const std = @import("std");
const contract = @import("contract");

const c = contract.spec;

/// The 127 finite E4M3 magnitudes by code: subnormals j * 2^-9, then (1 + m / 8) * 2^(e - 7) up to 448.
pub const e4m3 = blk: {
    var v: [127]f64 = undefined;
    for (&v, 0..) |*x, code| {
        const e: i32 = @intCast(code >> 3);
        const m: f64 = @floatFromInt(code & 7);
        x.* = if (e == 0) m * 0x1p-9 else (1 + m / 8) * @as(f64, @bitCast(@as(u64, @intCast(e - 7 + 1023)) << 52));
    }
    break :blk v;
};

pub const e2m1 = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// The value of `table` nearest to `a` (at least 0); a tie goes to the even code, as round to nearest even does.
pub fn nearest(table: []const f64, a: f64) f64 {
    var best: usize = 0;
    for (table, 0..) |t, i| {
        const d = @abs(a - t);
        const bd = @abs(a - table[best]);
        if (d < bd or (d == bd and i % 2 == 0 and best % 2 == 1)) best = i;
    }
    return table[best];
}

/// ceil(log2(x)) for a positive finite x, by frexp.
pub fn ceilLog2(x: f64) i32 {
    const f = std.math.frexp(x);
    return if (f.significand == 0.5) f.exponent - 1 else f.exponent;
}

pub const Kind = enum { fp8, fp4_e8m0, fp4_e4m3 };

pub fn block(kind: Kind) usize {
    return if (kind == .fp4_e4m3) 16 else 32;
}

/// One block's round trip as bf16 bits; a block holding a NaN or infinity is all NaN. x holds decoded bf16 values.
pub fn roundTrip(kind: Kind, x: []const f32, out: []u16) void {
    var amax: f32 = 0;
    for (x) |v| {
        if (!std.math.isFinite(v)) return @memset(out, 0x7fc0);
        amax = @max(amax, @abs(v));
    }
    const inv448: f32 = 1.0 / 448.0;
    const inv6: f32 = 1.0 / 6.0;
    for (x, out) |v, *o| {
        const mag: f64 = @abs(v);
        const q: f64 = switch (kind) {
            .fp8 => blk: {
                const s = std.math.pow(f64, 2, @floatFromInt(ceilLog2(@max(amax, 1e-4) * inv448)));
                break :blk nearest(&e4m3, @min(mag / s, 448)) * s;
            },
            .fp4_e8m0 => blk: {
                const s = std.math.pow(f64, 2, @floatFromInt(ceilLog2(@max(amax, 0x1.8p-124) * inv6)));
                break :blk nearest(&e2m1, @min(mag / s, 6)) * s;
            },
            .fp4_e4m3 => blk: {
                const s: f32 = @floatCast(nearest(&e4m3, @min(@max(amax, 0.01171875) / 6.0, 448)));
                const a: f32 = @min(@as(f32, @floatCast(mag)) / s, 6);
                break :blk @as(f32, @floatCast(nearest(&e2m1, a))) * s;
            },
        };
        const f: f32 = @floatCast(q);
        o.* = c.toBf16(if (std.math.signbit(v)) -f else f);
    }
}

fn pow2(e: i32) f32 {
    return std.math.pow(f32, 2, @floatFromInt(e));
}

/// Every block of x bf16 [rows][n]: its amax element sets the scale boundary, with ties, zeros, tiny values and spikes.
pub fn fillQuant(kind: Kind, x: []u16, blk: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const table: []const f64 = if (kind == .fp8) &e4m3 else &e2m1;
    const top: f32 = if (kind == .fp8) 448 else 6;
    var b: usize = 0;
    while (b * blk < x.len) : (b += 1) {
        const v = x[b * blk ..][0..blk];
        const e: i32 = rng.intRangeAtMost(i32, -8, 8);
        const mode = rng.uintLessThan(u32, 8);
        for (v, 0..) |*o, i| {
            const f: f32 = switch (mode) {
                0, 1 => (rng.float(f32) * 2 - 1) * pow2(rng.intRangeAtMost(i32, -10, 10)),
                2 => 0.0,
                3 => (rng.float(f32) * 2 - 1) * 1e-6,
                4 => if (i == 0) 1e30 else (rng.float(f32) - 0.5) * 1e-3,
                5 => (rng.float(f32) * 2 - 1) * top * pow2(e),
                else => blk2: {
                    const j = rng.uintLessThan(usize, table.len - 1);
                    const mid: f32 = @floatCast((table[j] + table[j + 1]) / 2);
                    break :blk2 if (i == 0) top * pow2(e) else if (rng.boolean()) -mid * pow2(e) else mid * pow2(e);
                },
            };
            o.* = c.toBf16(if (mode == 2 and i % 7 == 3) -f else f);
        }
        if (kind == .fp4_e4m3) for (v) |*o| if (mode >= 6) {
            o.* = c.toBf16(c.fromBf16(o.*) * 0.37);
        };
    }
}

/// The five planted values of the rows family (a NaN, both infinities, two at one row's ends) into bf16 rows [rows][k].
pub fn plant(x: []u16, rows: usize, k: usize, col0: usize, width: usize) usize {
    const at = [_][2]usize{ .{ 0, width / 3 }, .{ 1, width / 2 }, .{ 2, width - 1 }, .{ 3, 0 }, .{ 3, width - 1 } };
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
    var n: usize = 0;
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * k + col0 + p[1]] = c.toBf16(v);
        n += 1;
    };
    return n;
}

test "the oracle's tables: E4M3 reaches 448, ties go to the even code, E2M1 rounds as the format does" {
    try std.testing.expectEqual(@as(f64, 448), e4m3[126]);
    try std.testing.expectEqual(@as(f64, 0x1p-9), e4m3[1]);
    try std.testing.expectEqual(@as(f64, 0x1p-6), e4m3[8]);
    try std.testing.expectEqual(@as(f64, 1.0), nearest(&e2m1, 0.75));
    try std.testing.expectEqual(@as(f64, 1.0), nearest(&e2m1, 1.25));
    try std.testing.expectEqual(@as(f64, 2.0), nearest(&e2m1, 1.75));
    try std.testing.expectEqual(@as(f64, 4.0), nearest(&e2m1, 5.0));
    try std.testing.expectEqual(@as(f64, 0.0), nearest(&e2m1, 0.25));
    try std.testing.expectEqual(@as(i32, 3), ceilLog2(5.0));
    try std.testing.expectEqual(@as(i32, 2), ceilLog2(4.0));
    var out: [32]u16 = undefined;
    var x: [32]f32 = @splat(0);
    x[0] = 448;
    roundTrip(.fp8, &x, &out);
    try std.testing.expectEqual(@as(f32, 448), c.fromBf16(out[0]));
}
