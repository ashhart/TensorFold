//! CPU mirror of the Metal two-word circular decoder, tested against the deliberately scalar format oracle.
const std = @import("std");
const fmt = @import("exl3_format.zig");

fn word(tile: []const i16, i: usize) u32 {
    return @as(u32, @as(u16, @bitCast(tile[2 * i]))) | (@as(u32, @as(u16, @bitCast(tile[2 * i + 1]))) << 16);
}
pub fn state(width: fmt.Width, tile: []const i16, p: u32) u16 {
    const ring: u32 = 128 * @as(u32, width.halves);
    const end = ((p + 1) * width.halves - (if (width.isHalf()) (p + 1) & 1 else @as(u32, 0))) / 2;
    const start = (end + ring - 16) % ring;
    const wi = start / 32;
    const off = start % 32;
    const lo = word(tile, wi);
    if (off <= 16) return @truncate(lo >> @as(u5, @intCast(16 - off)));
    const hi = word(tile, (wi + 1) % (ring / 32));
    return @truncate((lo << @as(u5, @intCast(off - 16))) | (hi >> @as(u5, @intCast(48 - off))));
}

/// FP32 arithmetic used by Metal is exact here, followed by exactly one FP16 rounding.
pub fn value(codebook: fmt.Codebook, s: u16) f16 {
    var x: u32 = @as(u32, s) *% switch (codebook) {
        .mul1 => fmt.Mul1Mul,
        .mcg => fmt.McgMul,
        .inst3 => fmt.Inst3Mul,
    };
    if (codebook == .mul1) {
        const h: f32 = @floatFromInt(1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24));
        const scale: f16 = @bitCast(fmt.Mul1ScaleBits);
        const bias: f16 = @bitCast(fmt.Mul1BiasBits);
        return @floatCast(@mulAdd(f32, h, @floatCast(scale), @floatCast(bias)));
    }
    if (codebook == .inst3) x +%= fmt.Inst3Add;
    x = (x & fmt.Mask) ^ fmt.Flip;
    const lo: f16 = @bitCast(@as(u16, @truncate(x)));
    const hi: f16 = @bitCast(@as(u16, @truncate(x >> 16)));
    return @floatCast(@as(f32, @floatCast(lo)) + @as(f32, @floatCast(hi)));
}

/// The row/column inverse of the CUDA fragment permutation lets the MMA decoder place each stream value.
pub fn placement(p: u32) fmt.Placement {
    const l = p / 8;
    const j = p % 8;
    return .{ .row = @intCast(2 * (l % 4) + (j & 1) + 8 * ((j >> 1) & 1)), .col = @intCast(l / 4 + 8 * (j >> 2)) };
}

/// Same seven FP32 butterfly stages as the SIMDgroup transform, normalized only at the end.
pub fn hadamard(block: *[128]f32) void {
    var d: usize = 1;
    while (d < 128) : (d *= 2) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * d) {
            for (0..d) |j| {
                const a = block[base + j];
                const b = block[base + j + d];
                block[base + j] = a + b;
                block[base + j + d] = a - b;
            }
        }
    }
    for (block) |*x| x.* *= 0.08838834764831845;
}

test "two-word circular extraction agrees at every stream boundary and supported half width" {
    var prng = std.Random.DefaultPrng.init(0x45584c3353544154);
    const random = prng.random();
    var tile: [128]i16 = undefined;
    for ([_]u16{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 }) |halves| {
        const width = fmt.Width{ .halves = halves };
        for (0..12) |pattern| {
            for (tile[0..width.tileWords()], 0..) |*v, j| v.* = switch (pattern) {
                0 => 0,
                1 => -1,
                2 => @bitCast(@as(u16, if (j % 2 == 0) 0x8001 else 0x1234)),
                else => random.int(i16),
            };
            var expected: [256]u32 = undefined;
            fmt.tileStates(width, tile[0..width.tileWords()], &expected);
            for (expected, 0..) |want, p| try std.testing.expectEqual(@as(u16, @intCast(want)), state(width, tile[0..width.tileWords()], @intCast(p)));
        }
    }
}

test "Metal FP32 codebook arithmetic preserves FP16 bits for all 65536 states and all codebooks" {
    for ([_]fmt.Codebook{ .inst3, .mcg, .mul1 }) |codebook| {
        for (0..65536) |s| {
            const got: u16 = @bitCast(value(codebook, @intCast(s)));
            const want: u16 = @bitCast(fmt.codebookValue(codebook, @intCast(s)));
            try std.testing.expectEqual(want, got);
        }
    }
}

test "fragment placement is a bijection agreeing with the scalar oracle" {
    var expected: [256]fmt.Placement = undefined;
    fmt.tilePlacement(&expected);
    var visited: [256]bool = @splat(false);
    for (expected, 0..) |want, p| {
        const got = placement(@intCast(p));
        try std.testing.expectEqual(want, got);
        const at = @as(usize, got.row) * 16 + got.col;
        try std.testing.expect(!visited[at]);
        visited[at] = true;
    }
}

test "fixed butterfly transform preserves block signs and is its own inverse" {
    var x: [128]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 64)) / 32;
    const before = x;
    hadamard(&x);
    hadamard(&x);
    for (x, before) |got, want| try std.testing.expectApproxEqAbs(want, got, 0.000002);
    var impulse: [128]f32 = @splat(0);
    impulse[67] = 1;
    hadamard(&impulse);
    for (impulse, 0..) |got, i| {
        const want: f32 = if (@popCount(i & 67) & 1 == 0) 0.08838834764831845 else -0.08838834764831845;
        try std.testing.expectEqual(want, got);
    }
}
