//! New four-row-tile launches; runtime reflection sizes threads, and offline compile receipts remain pending.
const limits = @import("limits.zig");
pub const records = [_]limits.MetalLaunch{
    .{ .name = "simd_qmm tf_sq_mma_8_1_4", .threads = 256, .tg_memory = 8192, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_8_2_4", .threads = 256, .tg_memory = 16384, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_8_4_4", .threads = 256, .tg_memory = 32768, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_16_1_4", .threads = 512, .tg_memory = 16384, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_16_2_4", .threads = 512, .tg_memory = 32768, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_32_1_4", .threads = 512, .tg_memory = 32768, .adapts = true },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q4_8_1_4", .threads = 256, .tg_memory = 8192, .adapts = true },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q4_8_2_4", .threads = 256, .tg_memory = 16384, .adapts = true },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q4_8_4_4", .threads = 256, .tg_memory = 32768, .adapts = true },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q4_16_1_4", .threads = 512, .tg_memory = 16384, .adapts = true },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q4_16_2_4", .threads = 512, .tg_memory = 32768, .adapts = true },
    .{ .name = "simd_qmm_bits tf_sqw_mma_q4_32_1_4", .threads = 512, .tg_memory = 32768, .adapts = true },
};
test "simd32 records cover the new launches with reflection rather than invented compile limits" {
    const std = @import("std");
    for (records) |r| {
        var found: usize = 0;
        for (@import("registry.zig").all_metal) |active| if (std.mem.eql(u8, active.name, r.name)) {
            try std.testing.expectEqualDeep(r, active);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expect(r.adapts and r.m1 == 0 and r.m2 == 0);
        try std.testing.expect(r.tg_memory <= 32768);
        try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &.{r}, true));
    }
}
