//! simd_qmm's chunk-pair tiles (tf_sq_mmp): M1 and M2 compile them for 224 threads, so those chips keep tf_sq_mma.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");
pub const records = [_]limits.MetalLaunch{
    .{ .name = "simd_qmm tf_sq_mmp_16_4_2_2", .threads = 256, .tg_memory = 16384, .m1 = 224, .m2 = 224, .adapts = true },
};
test "the chunk-pair tiles are in the live and offline collector and fit M1 and M2" {
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
