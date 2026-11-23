//! Inkling 2 Mini's BF16 dense tensor-unit launches: tensor ops, so no M1/M2 compile and 256 reserved threads.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 dense_units tf_ink2_dense_units_16", .threads = 256, .tg_memory = 8192, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 dense_units tf_ink2_dense_units_32", .threads = 256, .tg_memory = 8192, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 dense_units tf_ink2_dense_units_64", .threads = 256, .tg_memory = 8192, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 dense_units tf_ink2_dense6_units_16", .threads = 256, .tg_memory = 8192, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 dense_units tf_ink2_dense6_units_32", .threads = 256, .tg_memory = 8192, .reserved = 256, .offline = false },
    .{ .name = "metal inkling2 dense_units tf_ink2_dense6_units_64", .threads = 256, .tg_memory = 8192, .reserved = 256, .offline = false },
};

test "Inkling2 dense_units launches reserve their threads and carry no M1/M2 compile receipts" {
    for (records) |record| {
        try std.testing.expectEqual(@as(u32, 0), record.m1);
        try std.testing.expectEqual(@as(u32, 0), record.m2);
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(!record.offline);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 dense_units "));
    }
}
