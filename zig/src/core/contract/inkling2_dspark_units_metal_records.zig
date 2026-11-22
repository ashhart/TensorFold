//! Mini DSpark's tensor-unit attention launches (zig/tests/contract/metal_inkling2_dspark_units.zig).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 dspark_units tf_ink2_dspark_attn_units", .threads = 64, .tg_memory = 0, .reserved = 64, .offline = false },
    .{ .name = "metal inkling2 dspark_units tf_ink2_dspark_attn_merge", .threads = 128, .tg_memory = 32, .reserved = 128, .offline = false },
};

test "the tensor-unit attention's launches reserve their threads and carry no M1/M2 compile receipts" {
    for (records) |record| {
        try std.testing.expectEqual(@as(u32, 0), record.m1);
        try std.testing.expectEqual(@as(u32, 0), record.m2);
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(!record.offline);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 dspark_units "));
    }
}
