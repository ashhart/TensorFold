//! The pair seam's four launches (core/pair_rows.metal) with the M1/M2 limits the offline compile measures.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");

pub const records = [_]limits.MetalLaunch{
    .{ .name = "metal pair stage pair_stage", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal pair gather pair_gather", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal pair argmax pair_argmax", .threads = 1024, .tg_memory = 384, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal pair pick pair_pick", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "the pair seam's four launches are in the live and offline collector and fit every family" {
    for (records) |record| {
        var found: usize = 0;
        for (registry.all_metal) |active| if (std.mem.eql(u8, active.name, record.name)) {
            try std.testing.expectEqualDeep(record, active);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkMetal(std.testing.allocator, &.{record}, true));
        // A zero here would be a launch no M1/M2 compiler was asked about.
        try std.testing.expect(record.m1 > 0 and record.m2 > 0);
    }
}
