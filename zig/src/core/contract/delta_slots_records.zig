//! DeltaNet's register-slot pipelines: declared geometry and the M1 and M2 thread limits of Xcode's offline compiler.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");
pub const counts = [_]usize{ 1, 2, 4, 8, 16 };
pub const records = [_]limits.MetalLaunch{
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_1", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_2", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_4", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_8", .threads = 128, .tg_memory = 0, .m1 = 768, .m2 = 768 },
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_16", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
test "all five DeltaNet register-slot pipelines are in the live and offline collector and fit M1 and M2" {
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

/// Column launches reserve their SIMD group in source; offline compile receipts remain device work.
pub const columns = [_]limits.MetalLaunch{
    .{ .name = "core deltanet tf_delta_gdn_columns_1", .threads = 32, .tg_memory = 2048, .reserved = 32 },
    .{ .name = "core deltanet tf_delta_gdn_columns_2", .threads = 32, .tg_memory = 4096, .reserved = 32 },
    .{ .name = "core deltanet tf_delta_gdn_columns_4", .threads = 32, .tg_memory = 8192, .reserved = 32 },
};

test "q27 stack: all column replacements are active exactly once" {
    for (columns) |record| {
        var count: usize = 0;
        for (registry.all_metal) |active| if (std.mem.eql(u8, record.name, active.name)) {
            try std.testing.expectEqualDeep(record, active);
            count += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), count);
    }
}
