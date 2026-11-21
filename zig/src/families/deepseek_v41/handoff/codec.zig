//! Exact scalar decoders of CUDA cache records (e4m3 codes + UE8M0 scale per 64 dims, RoPE bf16) into bf16 rows.
const std = @import("std");
const g = @import("geometry.zig");

pub const Error = error{ BadKvRow, BadIndexRow };

pub fn bf16ToF32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// Round to nearest even; NaN stays a quiet NaN.
pub fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// The value of an OCP e4m3fn code: bias 7, subnormals m * 2^-9, no infinity, 0x7f / 0xff are NaN.
pub fn e4m3(code: u8) f32 {
    const e: u32 = (code >> 3) & 15;
    const m: u32 = code & 7;
    if (e == 15 and m == 7) return std.math.nan(f32);
    const mag: f32 = if (e == 0) @as(f32, @floatFromInt(m)) * 0x1p-9 else std.math.ldexp(@as(f32, @floatFromInt(8 + m)), @as(i32, @intCast(e)) - 10);
    return if (code & 0x80 != 0) -mag else mag;
}

/// 2^(byte - 127); null for the UE8M0 NaN byte 255.
pub fn ue8m0(byte: u8) ?f64 {
    if (byte == 255) return null;
    return std.math.ldexp(@as(f64, 1), @as(i32, byte) - 127);
}

fn scaled(code: u8, scale: f64) !u16 {
    const v = e4m3(code);
    if (std.math.isNan(v)) return error.BadKvRow;
    return f32ToBf16(@floatCast(@as(f64, v) * scale));
}

/// One comp / ring row (576 value bytes, 8 scale bytes) as 512 bf16 values.
pub fn decodeComp(values: *const [g.cuda_values_bytes]u8, scales: *const [g.cuda_scale_bytes]u8, out: *[512]u16) Error!void {
    for (0..g.cuda_tiles) |t| {
        const s = ue8m0(scales[t]) orelse return error.BadKvRow;
        for (t * g.cuda_tile..(t + 1) * g.cuda_tile) |d| out[d] = try scaled(values[d], s);
    }
    for (0..64) |j| {
        const v = std.mem.readInt(u16, values[g.cuda_nope + 2 * j ..][0..2], .little);
        if (!finiteBf16(v)) return error.BadKvRow;
        out[g.cuda_nope + j] = v;
    }
}

pub fn finiteBf16(x: u16) bool {
    return x & 0x7f80 != 0x7f80;
}

/// A pool record (values then scales, 584 bytes) as 512 bf16 values.
pub fn decodeRecord(rec: *const [g.cuda_record_bytes]u8, out: *[512]u16) Error!void {
    return decodeComp(rec[0..g.cuda_values_bytes], rec[g.cuda_values_bytes..][0..g.cuda_scale_bytes], out);
}

/// One fp8 index row (132 bytes) as 128 bf16 values; a scale that is not finite is refused.
pub fn decodeIndexFp8(row: *const [g.cuda_index_fp8_bytes]u8, out: *[128]u16) Error!void {
    const scale: f32 = @bitCast(std.mem.readInt(u32, row[128..][0..4], .little));
    if (!std.math.isFinite(scale)) return error.BadIndexRow;
    for (0..128) |d| {
        const v = e4m3(row[d]);
        if (std.math.isNan(v)) return error.BadIndexRow;
        out[d] = f32ToBf16(@floatCast(@as(f64, v) * scale));
    }
}

/// A bf16 index row (256 bytes, little-endian) as 128 bf16 values.
pub fn decodeIndexBf16(row: *const [g.cuda_index_bf16_bytes]u8, out: *[128]u16) void {
    for (out, 0..) |*o, d| o.* = std.mem.readInt(u16, row[2 * d ..][0..2], .little);
}

pub fn storeRow(comptime n: usize, row: *const [n]u16, out: []u8) void {
    for (row, 0..) |v, i| std.mem.writeInt(u16, out[2 * i ..][0..2], v, .little);
}

pub fn loadRow(comptime n: usize, bytes: []const u8) [n]u16 {
    var r: [n]u16 = undefined;
    for (&r, 0..) |*v, i| v.* = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
    return r;
}
