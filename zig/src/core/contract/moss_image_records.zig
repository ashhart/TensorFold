//! MOSS-Image's transformer launches: its own kernels, and the prefill split-K GEMM it reads fp32 sums from.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "moss image gemm custom_kernel_tf_gemm_splitk_nax_bf16_n_t_n_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "moss image gemm custom_kernel_tf_gemm_splitk_nax_bf16_n_t_t_t_64_64_256_2_2_bfloat16_t_bfloat16_t_int32_t_float", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "moss image mi_norm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "moss image mi_qk", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "moss image mi_situ", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "moss image mi_gate_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "moss image mi_bias", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "moss image mi_attention_00", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "moss image mi_attention_01", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "moss image mi_attention_10", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "moss image mi_attention_11", .threads = 128, .tg_memory = 0, .reserved = 128 },
};

test "the MOSS-Image launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
