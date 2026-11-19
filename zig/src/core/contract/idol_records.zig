//! Idol's denoiser launches (idol.metal and idol_attention.metal after gemm_nax.metal), each source-reserved.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "idol id_linear_00", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_linear_01", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_linear_10", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_linear_11", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_linear_gelu_00", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_linear_gelu_01", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_linear_gelu_10", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_linear_gelu_11", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_split_0", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "idol id_split_1", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "idol id_gemm32", .threads = 128, .tg_memory = 19456, .reserved = 128 },
    .{ .name = "idol id_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "idol id_norm32", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "idol id_norm_split", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "idol id_repo", .threads = 64, .tg_memory = 704, .reserved = 64 },
    .{ .name = "idol id_qk", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_gate_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_cond", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_bias32", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol id_attention_00", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "idol id_attention_01", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "idol id_attention_10", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "idol id_attention_11", .threads = 128, .tg_memory = 0, .reserved = 128 },
};

test "the Idol launches fit every Apple family, one record a pipeline" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
    try std.testing.expectEqual(@as(usize, 24), records.len);
}
