//! Lane helpers shared by the exact GGUF block kernels; each is a plain load, shuffle or widening with no rounding.
const std = @import("std");

pub inline fn readU16(p: [*]const u8) u16 {
    return std.mem.readInt(u16, p[0..2], .little);
}

pub inline fn readF16(p: [*]const u8) f32 {
    return @floatCast(@as(f16, @bitCast(readU16(p))));
}

pub inline fn loadF32x8(p: [*]const f32) @Vector(8, f32) {
    return @as(*align(1) const [8]f32, @ptrCast(p)).*;
}

pub inline fn lowHalf(v: @Vector(16, u8)) @Vector(8, u8) {
    return @shuffle(u8, v, undefined, [8]i32{ 0, 1, 2, 3, 4, 5, 6, 7 });
}

pub inline fn highHalf(v: @Vector(16, u8)) @Vector(8, u8) {
    return @shuffle(u8, v, undefined, [8]i32{ 8, 9, 10, 11, 12, 13, 14, 15 });
}

pub inline fn u8x8ToF32x8(v: @Vector(8, u8)) @Vector(8, f32) {
    return @floatFromInt(@as(@Vector(8, u32), v));
}

pub inline fn lowHalfI8(v: @Vector(16, i8)) @Vector(8, i8) {
    return @shuffle(i8, v, undefined, [8]i32{ 0, 1, 2, 3, 4, 5, 6, 7 });
}

pub inline fn highHalfI8(v: @Vector(16, i8)) @Vector(8, i8) {
    return @shuffle(i8, v, undefined, [8]i32{ 8, 9, 10, 11, 12, 13, 14, 15 });
}

pub inline fn i8x8ToF32x8(v: @Vector(8, i8)) @Vector(8, f32) {
    return @floatFromInt(@as(@Vector(8, i32), v));
}

pub inline fn expandBits8(byte: u8) @Vector(8, u8) {
    const bit_mask: @Vector(8, u8) = .{ 1, 2, 4, 8, 16, 32, 64, 128 };
    const bits = @as(@Vector(8, u8), @splat(byte)) & bit_mask;
    return @select(u8, bits != @as(@Vector(8, u8), @splat(0)), @as(@Vector(8, u8), @splat(16)), @as(@Vector(8, u8), @splat(0)));
}
