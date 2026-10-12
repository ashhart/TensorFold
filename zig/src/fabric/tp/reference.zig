//! Host references for the collectives: reductions in rank order, bf16 / f16 added in fp32 and rounded once.
const std = @import("std");
const collective = @import("collective.zig");
const DType = collective.DType;
const Op = collective.Op;

/// bf16 bits to fp32 (exact).
pub fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// fp32 to bf16 bits, round to nearest even; NaN stays a quiet NaN.
pub fn f32ToBf16(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return @intCast((u >> 16) | 0x40);
    const round = 0x7FFF + ((u >> 16) & 1);
    return @intCast((u +% round) >> 16);
}

fn combine(comptime T: type, a: T, b: T, op: Op) T {
    return switch (op) {
        .sum => if (@typeInfo(T) == .int) a +% b else a + b,
        .max => @max(a, b),
        .min => @min(a, b),
    };
}

fn reduceTyped(comptime T: type, acc: []u8, src: []const u8, op: Op) void {
    const n = acc.len / @sizeOf(T);
    for (0..n) |i| {
        const a = std.mem.readInt(@Int(.unsigned, @bitSizeOf(T)), acc[i * @sizeOf(T) ..][0..@sizeOf(T)], .little);
        const b = std.mem.readInt(@Int(.unsigned, @bitSizeOf(T)), src[i * @sizeOf(T) ..][0..@sizeOf(T)], .little);
        const r: T = combine(T, @bitCast(a), @bitCast(b), op);
        std.mem.writeInt(@Int(.unsigned, @bitSizeOf(T)), acc[i * @sizeOf(T) ..][0..@sizeOf(T)], @bitCast(r), .little);
    }
}

fn reduceHalf(acc: []u8, src: []const u8, op: Op, bf: bool) void {
    const n = acc.len / 2;
    for (0..n) |i| {
        const a = std.mem.readInt(u16, acc[2 * i ..][0..2], .little);
        const b = std.mem.readInt(u16, src[2 * i ..][0..2], .little);
        const fa: f32 = if (bf) bf16ToF32(a) else @floatCast(@as(f16, @bitCast(a)));
        const fb: f32 = if (bf) bf16ToF32(b) else @floatCast(@as(f16, @bitCast(b)));
        const r = combine(f32, fa, fb, op);
        const out: u16 = if (bf) f32ToBf16(r) else @bitCast(@as(f16, @floatCast(r)));
        std.mem.writeInt(u16, acc[2 * i ..][0..2], out, .little);
    }
}

/// acc <- acc (op) src, element-wise; `acc` holds the lower ranks' result, `src` the next rank's input.
pub fn accumulate(acc: []u8, src: []const u8, t: DType, op: Op) void {
    std.debug.assert(acc.len == src.len and acc.len % t.size() == 0);
    switch (t) {
        .u8 => reduceTyped(u8, acc, src, op),
        .i32 => reduceTyped(i32, acc, src, op),
        .i64 => reduceTyped(i64, acc, src, op),
        .f32 => reduceTyped(f32, acc, src, op),
        .f64 => reduceTyped(f64, acc, src, op),
        .bf16 => reduceHalf(acc, src, op, true),
        .f16 => reduceHalf(acc, src, op, false),
    }
}

/// out <- inputs[0] (op) inputs[1] (op) ..., in rank order.
pub fn reduce(out: []u8, inputs: []const []const u8, t: DType, op: Op) void {
    @memcpy(out, inputs[0]);
    for (inputs[1..]) |in| accumulate(out, in, t, op);
}

/// Deterministic test data for `rank`'s input of `bytes`: finite values of type `t` that differ per rank and position.
pub fn fill(out: []u8, t: DType, rank: u32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed ^ (@as(u64, rank) *% 0x9E3779B97F4A7C15));
    const r = rng.random();
    const n = out.len / t.size();
    for (0..n) |i| {
        const x = r.float(f32) * 8.0 - 4.0;
        switch (t) {
            .u8 => out[i] = r.int(u8),
            .i32 => std.mem.writeInt(i32, out[4 * i ..][0..4], r.int(i32), .little),
            .i64 => std.mem.writeInt(i64, out[8 * i ..][0..8], r.int(i64), .little),
            .f32 => std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(x), .little),
            .f64 => std.mem.writeInt(u64, out[8 * i ..][0..8], @bitCast(@as(f64, x) * 1.000000123), .little),
            .bf16 => std.mem.writeInt(u16, out[2 * i ..][0..2], f32ToBf16(x), .little),
            .f16 => std.mem.writeInt(u16, out[2 * i ..][0..2], @bitCast(@as(f16, @floatCast(x))), .little),
        }
    }
}

test "bf16 rounding is nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    // 1 + 2^-8 is exactly halfway between bf16 1.0 and 1.0078125: rounds to the even one (1.0)
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.00390625));
    try std.testing.expectEqual(@as(u16, 0x3F82), f32ToBf16(1.01171875));
    try std.testing.expect(std.math.isNan(bf16ToF32(f32ToBf16(std.math.nan(f32)))));
}

test "rank-order reduction of bf16 adds in fp32 and rounds once" {
    var a: [4]u8 = undefined;
    var b: [4]u8 = undefined;
    std.mem.writeInt(u16, a[0..2], f32ToBf16(1.0), .little);
    std.mem.writeInt(u16, b[0..2], f32ToBf16(0.00390625), .little);
    std.mem.writeInt(u16, a[2..4], f32ToBf16(-2.5), .little);
    std.mem.writeInt(u16, b[2..4], f32ToBf16(3.0), .little);
    var out: [4]u8 = undefined;
    reduce(&out, &.{ &a, &b }, .bf16, .sum);
    try std.testing.expectEqual(f32ToBf16(1.0), std.mem.readInt(u16, out[0..2], .little));
    try std.testing.expectEqual(f32ToBf16(0.5), std.mem.readInt(u16, out[2..4], .little));
    reduce(&out, &.{ &a, &b }, .bf16, .max);
    try std.testing.expectEqual(f32ToBf16(3.0), std.mem.readInt(u16, out[2..4], .little));
}

test "integer sums wrap" {
    var a: [4]u8 = undefined;
    var b: [4]u8 = undefined;
    std.mem.writeInt(i32, &a, std.math.maxInt(i32), .little);
    std.mem.writeInt(i32, &b, 1, .little);
    var out: [4]u8 = undefined;
    reduce(&out, &.{ &a, &b }, .i32, .sum);
    try std.testing.expectEqual(@as(i32, std.math.minInt(i32)), std.mem.readInt(i32, &out, .little));
}
