//! exl3-kv host codes: bit for bit what the int8 and int4 KV kernels write, and the float64 views of them.

const std = @import("std");
const rig = @import("../../flashnext_rig.zig");
const bf = rig.bf;
const bf16 = rig.bf16;

const head_dim = 256;
const group = 32;
const ng = head_dim / group;

const inv_sqrt32: f32 = 0.17677669529663688;

/// The normalized Sylvester Hadamard of 32 values in fp32, the kernel's butterfly order.
pub fn wht32(x: *[group]f32) void {
    var m: usize = 1;
    while (m < group) : (m <<= 1) {
        var y: [group]f32 = undefined;
        for (0..group) |i| y[i] = if (i & m != 0) x[i ^ m] - x[i] else x[i] + x[i ^ m];
        x.* = y;
    }
    for (x) |*v| v.* *= inv_sqrt32;
}

/// One group of 32 values (already rotated if wanted) into its codes and fp16 scale.
pub fn quant8(x: *const [group]f32, codes: *[group]i8) f16 {
    var peak: f32 = 0;
    for (x) |v| peak = @max(peak, @abs(v));
    const sh: f16 = @floatCast(@max(peak, 6.103515625e-05));
    const s: f32 = @floatCast(sh);
    for (x, codes) |v, *c| {
        const q = std.math.clamp(@floor(v / s * 128.0) + 128.0, 0.0, 255.0);
        c.* = @intCast(@as(i32, @intFromFloat(q)) - 128);
    }
    return sh;
}

pub fn dequant8(c: i8, s: f16) f32 {
    return (@as(f32, @floatFromInt(c)) + 0.5) * (@as(f32, @floatCast(s)) * 0.0078125);
}

/// int4 (host only, for the quality comparison): exl3_cache.zig quantize4's grid, the scale rounded to fp16 first.
pub fn roundTrip4(x: *const [group]f32, out: *[group]f32) void {
    var peak: f32 = 0;
    for (x) |v| peak = @max(peak, @abs(v));
    const s: f32 = @floatCast(@as(f16, @floatCast(@max(peak, 6.103515625e-05))));
    for (x, out) |v, *o| {
        const q = std.math.clamp(@floor(v / s * 8.0) + 8.0, 0.0, 15.0);
        o.* = (q - 7.5) * s * 0.125;
    }
}

/// A row of 256 bf16 into codes and scales (rot: each group rotated first).
pub fn packRow(x: []const u16, rot: bool, codes: []i8, scales: []f16) void {
    for (0..ng) |gi| {
        var g: [group]f32 = undefined;
        for (0..group) |j| g[j] = @floatCast(bf(x[gi * group + j]));
        if (rot) wht32(&g);
        scales[gi] = quant8(&g, codes[gi * group ..][0..group]);
    }
}

/// The dequantised row as the chunk kernel's bf16 tile holds it (still in the rotated domain when rot).
pub fn tileRow(codes: []const i8, scales: []const f16, out: []u16) void {
    for (0..head_dim) |d| out[d] = bf16(dequant8(codes[d], scales[d / group]));
}

/// The dequantised row back in the model's domain, in float64 (the oracle's view of what the cache holds).
pub fn valueRow(codes: []const i8, scales: []const f16, rot: bool, out: []f64) void {
    for (0..ng) |gi| {
        var g: [group]f64 = undefined;
        for (0..group) |j| g[j] = dequant8(codes[gi * group + j], scales[gi]);
        if (rot) {
            // H is symmetric and orthonormal: its own inverse
            var y: [group]f64 = undefined;
            for (0..group) |i| {
                var acc: f64 = 0;
                for (0..group) |j| acc += (if (@popCount(i & j) % 2 == 0) @as(f64, 1) else -1) * g[j];
                y[i] = acc / @sqrt(32.0);
            }
            g = y;
        }
        @memcpy(out[gi * group ..][0..group], &g);
    }
}
