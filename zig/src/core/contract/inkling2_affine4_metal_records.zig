//! Inkling 2 Mini MLX 4-bit affine g64 Metal launches (zig/tests/contract/metal_inkling2_affine4.zig).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal inkling2 affine4 embed_norm tf_ink2_embed_norm_q4", .threads = 256, .tg_memory = 48, .reserved = 256 },
    .{ .name = "metal inkling2 affine4 dense tf_ink2_dense_q4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 affine4 logits tf_ink2_logits_q4", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

test "Inkling2 affine4 launches reserve their threads and carry no M1/M2 compile receipts" {
    for (records) |record| {
        try std.testing.expectEqual(@as(u32, 0), record.m1);
        try std.testing.expectEqual(@as(u32, 0), record.m2);
        try std.testing.expectEqual(record.threads, record.reserved);
        try std.testing.expect(std.mem.startsWith(u8, record.name, "metal inkling2 affine4 "));
    }
}
