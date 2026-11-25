//! Inkling 2 Mini's simdgroup-matrix launches: every thread count is reserved by its kernel, so no M1/M2 receipt.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 mma tf_ink2_mma_group", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "metal inkling2 mma tf_ink2_nvfp4_gu_mma", .threads = 256, .tg_memory = 24576, .reserved = 256 },
    .{ .name = "metal inkling2 mma tf_ink2_nvfp4_down_mma", .threads = 256, .tg_memory = 24576, .reserved = 256 },
    .{ .name = "metal inkling2 mma tf_ink2_mxfp8_gu_mma", .threads = 256, .tg_memory = 24576, .reserved = 256 },
    .{ .name = "metal inkling2 mma tf_ink2_mxfp8_down_mma", .threads = 256, .tg_memory = 24576, .reserved = 256 },
    .{ .name = "metal inkling2 mma tf_ink2_dense_mma", .threads = 128, .tg_memory = 4096, .reserved = 128 },
    .{ .name = "metal inkling2 mma tf_inkling2_loop_gates_mma", .threads = 256, .tg_memory = 5696, .reserved = 256 },
};

test "Inkling2 simdgroup-matrix launches reserve their threads and carry no M1/M2 compile receipts" {
    for (records) |record| {
        try std.testing.expectEqual(@as(u32, 0), record.m1);
        try std.testing.expectEqual(@as(u32, 0), record.m2);
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 mma "));
    }
}
