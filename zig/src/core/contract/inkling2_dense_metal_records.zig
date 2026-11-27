//! Inkling 2 Mini register-tiled bf16 projection launch: threads reserved by the kernel, so no M1/M2 receipt.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 dense tf_ink2_dense_tiled", .threads = 256, .tg_memory = 8192, .reserved = 256 },
};

test "Inkling2 dense tiled launch reserves its threads and carries no M1/M2 compile receipt" {
    for (records) |record| {
        try std.testing.expectEqual(@as(u32, 0), record.m1);
        try std.testing.expectEqual(@as(u32, 0), record.m2);
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 dense "));
    }
}
