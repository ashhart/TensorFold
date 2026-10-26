// SPDX-License-Identifier: Apache-2.0

//! Host registration for the Sage XPU kernels. The device gates are a separate run.

const std = @import("std");

/// Group size X from Dev.kernel (Y and Z stay 1). The chip check is the device run.
pub const Launch = struct { name: []const u8, group: u32 };

/// The served path. Prompt tiles and gdn_one are not on this list.
pub const launches = [_]Launch{
    .{ .name = "attn_prep", .group = 256 },
    .{ .name = "attn_decode_span", .group = 16 },
    .{ .name = "attn_merge_span", .group = 256 },
    .{ .name = "attn_gate", .group = 256 },
    .{ .name = "gdn_pre", .group = 256 },
    .{ .name = "gdn_step", .group = 16 },
    .{ .name = "gated_norm", .group = 256 },
    .{ .name = "sage_router", .group = 16 },
    .{ .name = "sage_down_combine", .group = 64 },
};

fn storeBits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn bfValue(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn storeWithoutNanBranch(v: f32) u16 {
    var u: u32 = @bitCast(v);
    u +%= 0x7fff +% ((u >> 16) & 1);
    return @truncate(u >> 16);
}

test "served group sizes are multiples of 16 and at most 1024" {
    for (launches) |k| {
        try std.testing.expect(k.group > 0 and k.group <= 1024 and k.group % 16 == 0);
    }
}

test "planted control: the old bf16 store turns a NaN into a finite value" {
    const nan: f32 = @bitCast(@as(u32, 0x7fffffff));
    try std.testing.expect(std.math.isNan(nan));
    try std.testing.expect(!std.math.isNan(bfValue(storeWithoutNanBranch(nan))));
    try std.testing.expect(std.math.isNan(bfValue(storeBits(nan))));
}
