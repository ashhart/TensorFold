//! Inkling 2 Mini M3 Metal launches; every thread count is reserved by its kernel, so no M1/M2 receipt is needed.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 experts tf_ink2_nvfp4_gu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 experts tf_ink2_nvfp4_down", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 experts tf_ink2_mxfp8_gu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 experts tf_ink2_mxfp8_down", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 experts tf_ink2_nvfp4_quantize", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 experts tf_ink2_mxfp8_quantize", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 experts tf_ink2_combine", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

test "Inkling2 expert launches carry no fabricated M1/M2 compile receipts" {
    for (records) |record| {
        try std.testing.expectEqual(@as(u32, 0), record.m1);
        try std.testing.expectEqual(@as(u32, 0), record.m2);
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 experts "));
    }
}
