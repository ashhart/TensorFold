//! DeepSeek V4.1 opt-in FP64-certified BF16 dense FP8 Metal launches (M5 Ultra Metal 4.1 only, every count reserved).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

const mma_records = [_]M{
    .{ .name = "dsv41 fp8 cert norm tf_dsv41_fp8_cert_norm_w", .threads = 128, .tg_memory = 144, .reserved = 128, .offline = false },
    .{ .name = "dsv41 fp8 cert prep tf_dsv41_fp8_cert_prep_x", .threads = 128, .tg_memory = 80, .reserved = 128, .offline = false },
    .{ .name = "dsv41 fp8 cert zero tf_dsv41_fp8_cert_zero_cnt", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "dsv41 fp8 cert gemm tf_dsv41_fp8_cert_gemm", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "dsv41 fp8 cert repair tf_dsv41_fp8_cert_repair_bf16", .threads = 128, .tg_memory = 5632, .reserved = 128, .offline = false },
    .{ .name = "dsv41 fp8 cert prep small tf_dsv41_fp8_cert_prep_x_small", .threads = 256, .tg_memory = 160, .reserved = 256, .offline = false },
    .{ .name = "dsv41 fp8 cert small tf_dsv41_fp8_cert_small_gemm", .threads = 128, .tg_memory = 9232, .reserved = 128, .offline = false },
    .{ .name = "dsv41 fp8 cert fast bf16 tf_dsv41_fp8_fast_nax_bf16", .threads = 128, .tg_memory = 12288, .reserved = 128, .offline = false },
    .{ .name = "dsv41 fp8 cert fast f32 tf_dsv41_fp8_fast_nax_f32", .threads = 128, .tg_memory = 12288, .reserved = 128, .offline = false },
    .{ .name = "dsv41 fp8 cert f32 tf_dsv41_fp8_cert_f32", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
};

const tile_names = [_][]const u8{ "r8_c8_f32", "r8_c8_bf16", "r16_c8_f32", "r16_c8_bf16", "r8_c16_f32", "r8_c16_bf16", "r16_c16_f32", "r16_c16_bf16" };
const row_names = [_][]const u8{ "wa_r16_bf16", "wb_r16_bf16", "wc_r16_bf16", "wd_r16_bf16", "we_r16_bf16", "wf_r16_bf16" };

pub const records = blk: {
    var out: [mma_records.len + tile_names.len + row_names.len]M = undefined;
    for (mma_records, 0..) |r, i| out[i] = r;
    for (tile_names, 0..) |n, i| out[mma_records.len + i] = .{ .name = "dsv41 fp8 cert tile tf_dsv41_fp8_tile_" ++ n, .threads = 512, .tg_memory = 18432, .reserved = 512, .offline = false };
    for (row_names, 0..) |n, i| out[mma_records.len + tile_names.len + i] = .{ .name = "dsv41 fp8 cert rows tf_dsv41_fp8_" ++ n, .threads = 1024, .tg_memory = 8192, .reserved = 1024, .offline = false };
    break :blk out;
};

test "the DeepSeek V4.1 FP8 certified launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
