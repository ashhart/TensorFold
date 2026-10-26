// SPDX-License-Identifier: Apache-2.0

//! Host registration for the EXL3 XPU kernels. The device gates are a separate run.

const std = @import("std");

/// One kernel's launch, taken from the OpenCL C source. The chip check is the device run.
pub const Launch = struct {
    name: []const u8,
    subgroup: u32,
    workgroup: [3]u32,
    local_bytes: u32,
};

/// Sub-group 16. rotate_out uses 8 sub-groups and (8-1)*128 fp32 local values.
pub const launches = [_]Launch{
    .{ .name = "exl3_rotate_in", .subgroup = 16, .workgroup = .{ 16, 1, 1 }, .local_bytes = 0 },
    .{ .name = "exl3_matvec", .subgroup = 16, .workgroup = .{ 16, 1, 1 }, .local_bytes = 0 },
    .{ .name = "exl3_rotate_out", .subgroup = 16, .workgroup = .{ 16, 8, 1 }, .local_bytes = 7 * 128 * 4 },
    .{ .name = "exl3_gemm", .subgroup = 16, .workgroup = .{ 16, 1, 1 }, .local_bytes = 0 },
    .{ .name = "exl3_moe_gate_up", .subgroup = 16, .workgroup = .{ 16, 1, 1 }, .local_bytes = 0 },
    .{ .name = "exl3_moe_down", .subgroup = 16, .workgroup = .{ 16, 1, 1 }, .local_bytes = 0 },
};

/// fp64 form of exl3_rotate_in for one 128 block. The device must match this, then the bf16 store.
pub fn rotateIn(x: []const f64, suh: []const f64, out: []f64) void {
    std.debug.assert(x.len == 128 and suh.len == 128 and out.len == 128);
    for (0..128) |i| out[i] = x[i] * suh[i];
    var d: usize = 1;
    while (d < 128) : (d *= 2) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * d) {
            for (0..d) |j| {
                const a = out[base + j];
                const b = out[base + j + d];
                out[base + j] = a + b;
                out[base + j + d] = a - b;
            }
        }
    }
    const scale = 1.0 / @sqrt(128.0);
    for (out) |*v| v.* *= scale;
}

/// The kernel store: a NaN keeps a NaN code. Same bits as core/bf16_round.zig bits.
fn storeBits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn bfValue(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// The old store: integer round with no NaN branch. A NaN can leave the NaN codes.
fn storeWithoutNanBranch(v: f32) u16 {
    var u: u32 = @bitCast(v);
    u +%= 0x7fff +% ((u >> 16) & 1);
    return @truncate(u >> 16);
}

test "launch records match the source and stay inside a 1024-thread work-group" {
    try std.testing.expectEqual(@as(usize, 6), launches.len);
    for (launches) |k| {
        try std.testing.expectEqual(@as(u32, 16), k.subgroup);
        const threads = k.workgroup[0] * k.workgroup[1] * k.workgroup[2];
        try std.testing.expect(threads <= 1024);
        try std.testing.expect(k.local_bytes <= 64 * 1024);
    }
    try std.testing.expectEqual(@as(u32, 3584), launches[2].local_bytes);
}

test "fp64 rotate of zeros stays zero" {
    var z: [128]f64 = @splat(0);
    var out: [128]f64 = undefined;
    rotateIn(&z, &z, &out);
    for (out) |v| try std.testing.expectEqual(@as(f64, 0), v);
}

test "planted controls: the two-half path and the old bf16 store both fail" {
    const bits: u32 = 0x3f800801;
    const f: f32 = @bitCast(bits);
    const hi: f16 = @floatCast(f);
    const lo: f16 = @floatCast(f - @as(f32, @floatCast(hi)));
    const back: f32 = @as(f32, hi) + @as(f32, lo);
    try std.testing.expect(bits != @as(u32, @bitCast(back)));
    const nan: f32 = @bitCast(@as(u32, 0x7fffffff));
    try std.testing.expect(std.math.isNan(nan));
    try std.testing.expect(!std.math.isNan(bfValue(storeWithoutNanBranch(nan))));
    try std.testing.expect(std.math.isNan(bfValue(storeBits(nan))));
}
