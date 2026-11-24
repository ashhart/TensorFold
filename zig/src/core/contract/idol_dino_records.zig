//! Idol's DINOv3 encoder launches (idol_dino.metal after gemm_nax.metal), each source-reserved.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "idol dino dn_linear_0", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol dino dn_linear_1", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol dino dn_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "idol dino dn_final", .threads = 256, .tg_memory = 5264, .reserved = 256 },
    .{ .name = "idol dino dn_heads", .threads = 128, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol dino dn_scale_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol dino dn_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol dino dn_attention", .threads = 128, .tg_memory = 0, .reserved = 128 },
};

test "the Idol DINOv3 launches fit every Apple family, one record a pipeline" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
    try std.testing.expectEqual(@as(usize, 8), records.len);
}
