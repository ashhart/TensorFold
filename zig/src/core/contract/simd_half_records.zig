//! simd_qmm's half-operand tiles (tf_sq_mmh) and their prepass (tf_sq_fits): M1 and M2 take the 8-row tile at 512.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");
pub const records = [_]limits.MetalLaunch{
    .{ .name = "simd_qmm tf_sq_mmh_8_4_1", .threads = 256, .tg_memory = 8192, .m1 = 512, .m2 = 512, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mmh_8_4_2", .threads = 256, .tg_memory = 16384, .m1 = 576, .m2 = 576, .adapts = true },
    .{ .name = "simd_qmm tf_sq_fits", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
test "the half-operand tiles and their prepass are in the live and offline collector and fit M1 and M2" {
    for (records) |record| {
        var found: usize = 0;
        for (registry.all_metal) |active| if (std.mem.eql(u8, active.name, record.name)) {
            try std.testing.expectEqualDeep(record, active);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkMetal(std.testing.allocator, &.{record}, true));
    }
}
