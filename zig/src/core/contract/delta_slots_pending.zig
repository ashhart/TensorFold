//! DeltaNet's register-slot variants have declared geometry; pipeline reflection remains required.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");
pub const counts = [_]usize{ 1, 2, 4, 8, 16 };
pub const records = [_]limits.MetalLaunch{
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_1", .threads = 128, .tg_memory = 0 },
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_2", .threads = 128, .tg_memory = 0 },
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_4", .threads = 128, .tg_memory = 0 },
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_8", .threads = 128, .tg_memory = 0 },
    .{ .name = "core deltanet tf_delta_gdn_tree_slots_16", .threads = 128, .tg_memory = 0 },
};
test "all five DeltaNet register-slot pipelines participate in the live and offline collector" {
    for (records) |record| {
        var found: usize = 0;
        for (registry.all_metal) |active| if (std.mem.eql(u8, active.name, record.name)) {
            try std.testing.expectEqualDeep(record, active);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 2), try registry.checkMetal(std.testing.allocator, &.{record}, true));
    }
}
