//! Scalar twins of the Metal engines' cache packers and dequantisers, matching each kernel's float operations.
const std = @import("std");
const c = @import("codec.zig");

fn rne(x: f32) f32 {
    const f = @floor(x);
    const d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// The E4M3 code of x (magnitude saturated at 448), as `dsv41_e4code`.
pub fn e4code(x: f32) u8 {
    const a = @min(@abs(x), 448.0);
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    if (a == 0) return sign;
    const e: i32 = @max(-6, @as(i32, @intCast((@as(u32, @bitCast(a)) >> 23) & 255)) - 127);
    const q: i32 = @intFromFloat(rne(std.math.ldexp(a, 3 - e)));
    const code: i32 = if (e == -6 and q < 8) q else (e + 7) * 8 + q - 8;
    return @as(u8, @intCast(@min(code, 126))) | sign;
}

/// The E2M1 code of x, ties to the even code, as `tf_c_e2m1` and `dsv41_e2code` (identical tables).
pub fn e2code(x: f32) u8 {
    const a = @min(@abs(x), 6.0);
    const pick: u8 = if (a <= 0.25) 0 else if (a < 0.75) 1 else if (a <= 1.25) 2 else if (a < 1.75) 3 else if (a <= 2.5) 4 else if (a < 3.5) 5 else if (a <= 5.0) 6 else 7;
    return pick | (if (std.math.signbit(x)) @as(u8, 8) else 0);
}

pub fn e2value(code: u8) f32 {
    const v = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const m = v[code & 7];
    return if (code & 8 != 0) -m else m;
}

/// ceil(log2(z)) of a positive normal float, from its bits.
fn log2Ceil(z: f32) i32 {
    const b: u32 = @bitCast(z);
    return @as(i32, @intCast((b >> 23) & 255)) - 127 + @intFromBool(b & 0x7fffff != 0);
}

fn amaxOf(x: []const u16) f32 {
    var a: f32 = 0;
    for (x) |v| a = @max(a, @abs(c.bf16ToF32(v)));
    return a;
}

const min_e8: f32 = 6.0 * 0x1p-126;

/// main's index-key packer: 32-value E2M1 blocks, E8M0 scale; a NaN/inf block packs zero codes under byte 255.
pub fn releaseIndexKey(x: *const [128]u16, out: *[68]u8) void {
    const inv6: f32 = 1.0 / 6.0;
    for (0..4) |g| {
        const blk = x[32 * g ..][0..32];
        var bad = false;
        for (blk) |v| bad = bad or !c.finiteBf16(v);
        if (bad) {
            @memset(out[16 * g ..][0..16], 0);
            out[64 + g] = 255;
            continue;
        }
        const e = log2Ceil(@max(amaxOf(blk), min_e8) * inv6);
        const inv = @as(f32, @bitCast(@as(u32, @intCast(-e + 127)) << 23));
        for (0..16) |j| {
            const lo = c.bf16ToF32(blk[2 * j]);
            const hi = c.bf16ToF32(blk[2 * j + 1]);
            out[16 * g + j] = e2code(std.math.copysign(@abs(lo) * inv, lo)) | (e2code(std.math.copysign(@abs(hi) * inv, hi)) << 4);
        }
        out[64 + g] = @intCast(e + 127);
    }
}

/// kind 0, `window_fp8`: 32-value E4M3 blocks, UE8M0 scale 2^ceil(log2(max(amax, 1e-4) / 448)); 512 codes, 16 scales.
pub fn window(x: *const [512]u16, out: *[528]u8) void {
    for (0..16) |g| {
        const blk = x[32 * g ..][0..32];
        const e = log2Ceil(@max(amaxOf(blk), 1e-4) / 448.0);
        const scale = std.math.ldexp(@as(f32, 1), e);
        for (blk, 0..) |v, j| out[32 * g + j] = e4code(c.bf16ToF32(v) / scale);
        out[512 + g] = @intCast(e + 127);
    }
}

/// kind 1, `main_fp4`: 16-value blocks, E4M3 scale of max(amax, 6 2^-9) / 6, E2M1 nibbles; 256 code bytes, 32 scales.
pub fn mainRow(x: *const [512]u16, out: *[288]u8) void {
    for (0..32) |g| {
        const blk = x[16 * g ..][0..16];
        const sc = e4code(@max(amaxOf(blk), 6.0 * 0x1p-9) / 6.0);
        const scale = c.e4m3(sc);
        for (0..8) |j| {
            const lo = e2code(c.bf16ToF32(blk[2 * j]) / scale);
            const hi = e2code(c.bf16ToF32(blk[2 * j + 1]) / scale);
            out[8 * g + j] = lo | (hi << 4);
        }
        out[256 + g] = sc;
    }
}

/// kind 2, `index_fp4`: 32-value blocks, UE8M0 scale 2^ceil(log2(max(amax, 6 2^-126) / 6)); 64 codes, 4 scales.
pub fn index(x: *const [128]u16, out: *[68]u8) void {
    for (0..4) |g| {
        const blk = x[32 * g ..][0..32];
        const e = log2Ceil(@max(amaxOf(blk), min_e8) / 6.0);
        const scale = std.math.ldexp(@as(f32, 1), e);
        for (0..16) |j| {
            const lo = e2code(c.bf16ToF32(blk[2 * j]) / scale);
            const hi = e2code(c.bf16ToF32(blk[2 * j + 1]) / scale);
            out[16 * g + j] = lo | (hi << 4);
        }
        out[64 + g] = @intCast(e + 127);
    }
}

/// `dsv41_unpack` kind 0: bf16 of e4m3(code) * 2^(scale - 127).
pub fn unpackWindow(row: *const [528]u8, out: *[512]u16) void {
    for (0..512) |d| {
        const s = std.math.ldexp(@as(f32, 1), @as(i32, row[512 + d / 32]) - 127);
        out[d] = c.f32ToBf16(c.e4m3(row[d]) * s);
    }
}

/// `dsv41_unpack` kind 1.
pub fn unpackMain(row: *const [288]u8, out: *[512]u16) void {
    for (0..512) |d| {
        const nib = if (d & 1 != 0) row[d / 2] >> 4 else row[d / 2] & 15;
        out[d] = c.f32ToBf16(e2value(nib) * c.e4m3(row[256 + d / 16]));
    }
}

/// `dsv41_unpack` kind 2 (also main's index key): E2M1 nibble times 2^(scale - 127).
pub fn unpackIndex(row: *const [68]u8, out: *[128]u16) void {
    for (0..128) |d| {
        const nib = if (d & 1 != 0) row[d / 2] >> 4 else row[d / 2] & 15;
        out[d] = c.f32ToBf16(e2value(nib) * std.math.ldexp(@as(f32, 1), @as(i32, row[64 + d / 32]) - 127));
    }
}
