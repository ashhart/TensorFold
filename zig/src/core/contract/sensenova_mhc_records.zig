//! SenseNova-Loop two-stream mHC Metal launches: 256 threads reserved, under 17 KiB threadgroup memory.
const std = @import("std");
const limits = @import("limits.zig");
const M = limits.MetalLaunch;

pub const pre_norm_tg_memory: u32 = 16704;

pub const records = [_]M{
    .{ .name = "sensenova mhc enter sn_mhc_enter", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sensenova mhc pre_norm sn_mhc_pre_norm", .threads = 256, .tg_memory = pre_norm_tg_memory, .reserved = 256 },
    .{ .name = "sensenova mhc post_merge sn_mhc_post_merge", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sensenova mhc post_and_pre_norm sn_mhc_post_and_pre_norm", .threads = 256, .tg_memory = pre_norm_tg_memory, .reserved = 256 },
};

test "SenseNova mHC launches fit every Apple GPU family" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var problems: usize = 0;
    for (records) |l| for (limits.apple) |f| {
        problems += try limits.metal(l, f, &out, a);
    };
    try std.testing.expectEqual(@as(usize, 0), problems);
}
