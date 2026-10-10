//! Host-side element conversions the Python loader does with torch: `.float()` and bf16 rounding.

const std = @import("std");
const st = @import("../safetensors.zig");

const Tensor = st.Tensor;
const DType = st.DType;

pub const Error = error{ UnexpectedTensor, UnsupportedQuantization } || std.mem.Allocator.Error;

/// One element as fp32 (bf16 and fp16 widen exactly).
pub fn load(dtype: DType, bytes: []const u8, i: usize) f32 {
    return switch (dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)))),
        else => unreachable,
    };
}

pub fn isFloat(dtype: DType) bool {
    return switch (dtype) {
        .f16, .bf16, .f32, .f64 => true,
        else => false,
    };
}

/// The group-table dtypes the kernels read as stored.
pub fn tableDtype(dtype: DType) bool {
    return dtype == .f32 or dtype == .bf16 or dtype == .f16;
}

/// torch's fp32 -> bf16: round to nearest even, NaN stays a NaN.
pub fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// Writes `src` as little-endian fp32 into `dst` (4 bytes an element).
pub fn widenInto(dst: []u8, src: Tensor) Error!void {
    const n = src.numel();
    if (dst.len != n * 4) return error.UnexpectedTensor;
    switch (src.dtype) {
        .f32, .bf16, .f16 => for (0..n) |i| std.mem.writeInt(u32, dst[4 * i ..][0..4], @bitCast(load(src.dtype, src.bytes, i)), .little),
        .f64 => for (0..n) |i| {
            const v: f64 = @bitCast(std.mem.readInt(u64, src.bytes[8 * i ..][0..8], .little));
            std.mem.writeInt(u32, dst[4 * i ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little);
        },
        else => return error.UnexpectedTensor,
    }
}

/// `.float().contiguous()` of a floating tensor into memory of `a` (an fp32 tensor is shared, not copied).
pub fn float32(a: std.mem.Allocator, src: Tensor) Error!Tensor {
    if (src.dtype == .f32) return src;
    const out = try a.alloc(u8, src.numel() * 4);
    try widenInto(out, src);
    return .{ .dtype = .f32, .rank = src.rank, .shape = src.shape, .bytes = out };
}

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16(@bitCast(@as(u32, 0x3F807FFF))));
    try std.testing.expectEqual(@as(u16, 0x3F82), bf16(@bitCast(@as(u32, 0x3F818000))));
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16(@bitCast(@as(u32, 0x3F808000))));
    try std.testing.expect(std.math.isNan(@as(f32, @bitCast(@as(u32, bf16(std.math.nan(f32))) << 16))));
}
