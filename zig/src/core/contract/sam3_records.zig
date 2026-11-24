const M = @import("limits.zig").MetalLaunch;

pub const metal = blk: {
    var out: [10]M = undefined;
    for ([_][]const u8{ "nn", "nt", "tn", "tt" }, 0..) |layout, i| {
        out[i] = .{ .name = "sam3 fp32 mma tf_sam3_mma_" ++ layout, .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false };
        out[i + 4] = .{ .name = "sam3 fp32 mma control tf_sam3_cut_" ++ layout, .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false };
    }
    out[8] = .{ .name = "sam3 fp32 mma helper tf_sam3_fma", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false };
    out[9] = .{ .name = "sam3 fp32 mma helper control tf_sam3_fma_flush", .threads = 32, .tg_memory = 0, .reserved = 32, .offline = false };
    break :blk out;
} ++ blk: {
    var out: [16]M = undefined;
    for ([_][]const u8{ "00", "01", "10", "11" }, 0..) |tag, i| {
        out[i] = .{ .name = "sam3 fp32 gemm qwv_gemm_f32_nt_" ++ tag, .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false };
        out[i + 4] = .{ .name = "sam3 fp32 gemm qwv_gemm_f32_exact_nt_" ++ tag, .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false };
        out[i + 8] = .{ .name = "sam3 fp32 gemm qwv_gemm_f32_compact_nt_" ++ tag, .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false };
        out[i + 12] = .{ .name = "sam3 fp32 gemm qwv_gemm_f32_exact_compact_nt_" ++ tag, .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false };
    }
    break :blk out;
} ++ [_]M{
    .{ .name = "sam3 sparse mask sam3_mask_first_conv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 sparse mask control sam3_mask_first_conv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 resize sam3_resample", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 resize control sam3_resample", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 resize nonfinite-control sam3_resample", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 compensated normalization sam_norm_compensated", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 compensated normalization control sam_norm_compensated", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 compensated normalization wide sam_norm_compensated_wide", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 compensated normalization wide sam_norm_wide_clear", .threads = 32, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 compensated normalization planes sam_norm_compensated_planes", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 compensated normalization planes sam_norm_planes_clear", .threads = 8, .tg_memory = 0, .reserved = 8 },
    .{ .name = "sam3 compensated normalization pair sam_norm_compensated_pair", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 compensated normalization pair sam_norm_pair_clear", .threads = 8, .tg_memory = 0, .reserved = 8 },
    .{ .name = "sam3 device projection planes sam_dense_planes_clear", .threads = 16, .tg_memory = 0, .reserved = 16, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_pack", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_plan", .threads = 1, .tg_memory = 0, .reserved = 1, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_restore", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_half_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_finish", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_gelu", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_residual", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_tiled", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_compensated", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes sam_dense_planes_compensated_residual", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection planes control sam_dense_planes", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_prepare_weights", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_clear", .threads = 8, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_pack", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_plan", .threads = 1, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_restore", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_gelu_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_plain", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_gelu_pair", .threads = 128, .tg_memory = 16384, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_residual", .threads = 128, .tg_memory = 16384, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair sam_dense_pair_tiled", .threads = 128, .tg_memory = 16384, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection pair control sam_dense_pair_plain", .threads = 256, .tg_memory = 32768, .reserved = 256, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device64", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device_rescale", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 device sam_attention_device_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 device sam_attention_device_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 device sam_attention_device32", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 device sam_attention_device32_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 device sam_attention_device32_rescale", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 split sam_attention_device_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 split sam_attention_device_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 split sam_attention_split32", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 split sam_attention_split32_merge", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 split sam_attention_device32_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 split sam_attention_device32_rescale", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 split control sam_attention_split32", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 split sam_attention_device_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 split sam_attention_device_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 split sam_attention_split64", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 split sam_attention_split32_merge", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 split sam_attention_device_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 split sam_attention_device_rescale", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 split control sam_attention_split64", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact half weight sam_dense_device_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact half weight sam_dense_half_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact half weight sam_dense_device_plan", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact half weight sam_dense_half_weight", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact half weight sam_dense_half_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact half weight control sam_dense_half_weight", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device_plan", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection GELU sam_dense_device_gelu", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection GELU sam_dense_device_gelu_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection GELU sam_dense_device_gelu_linear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 activation im_sam_gelu", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_expansion_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_device_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_expansion_pack", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_device_plan", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_expansion", .threads = 64, .tg_memory = 0, .reserved = 64, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_expansion_gelu", .threads = 64, .tg_memory = 0, .reserved = 64, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_expansion_drop_residual", .threads = 64, .tg_memory = 0, .reserved = 64, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_device_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection exact expansion sam_dense_device_gelu_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device_compensated", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device_nn", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device_nn_compensated", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection sam_dense_device_nn_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection control sam_dense_device", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide sam_dense_wide_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide sam_dense_wide_plain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide sam_dense_wide_compensated", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide sam_dense_wide_gelu", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide control sam_dense_wide_compensated", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide cached sam_dense_wide_operands", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide cached sam_dense_wide_seed", .threads = 32, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide cached sam_dense_wide_rows", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "sam3 device projection wide cached control sam_dense_wide_rows", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "sam3 device projection wide ready sam_dense_wide_plan", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide ready sam_dense_wide_gelu_flag", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide ready sam_dense_wide_plain_residual", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide ready sam_dense_wide_plain_tiled", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide ready sam_dense_wide_compensated_residual", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide ready sam_dense_wide_residual_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 device projection wide ready sam_dense_wide_tiled_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 small compensated projection tf_sam3_dense_small", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 small compensated projection baseline tf_sam3_dense_small", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 small compensated projection control tf_sam3_dense_small_drop_tail", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 deconvolution im_sam_deconv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 deconvolution baseline im_sam_deconv_baseline", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 deconvolution matrix sam3_deconv_pack", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "sam3 deconvolution matrix sam3_deconv_scatter", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "sam3 attention im_attention_f32_full256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention control im_attention_f32_full256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention two-pass im_attention_scores256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention two-pass im_attention_values256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention two-pass control im_attention_values256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D256 split sam_attention_split_scores", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D256 split sam_attention_split_values", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D256 split sam_attention_split_merge", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D256 split control sam_attention_split_values", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
} ++ @import("sam3_layout_records.zig").metal ++ @import("sam3_arithmetic_records.zig").metal ++
    @import("sam3_rotary_records.zig").metal ++ @import("sam3_rows_records.zig").metal ++
    @import("sam3_flash_records.zig").metal ++ @import("sam3_trunk_records.zig").metal ++ tensor_ops ++ core_attention;

/// The tensor library's FP32 attention at head dims 32 and 64: 128 threads, reserved in the source.
const core_attention = blk: {
    const functions = [_][]const u8{ "im_attention_f32_full32", "im_attention_f32_full64", "im_attention_f32_causal64" };
    var out: [2 * functions.len]M = undefined;
    for (functions, 0..) |f, i| {
        out[2 * i] = .{ .name = "sam3 core attention " ++ f, .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false };
        out[2 * i + 1] = .{ .name = "sam3 core attention control " ++ f, .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false };
    }
    break :blk out;
};

/// The tensor library's generic ops: groups of 256 (im_norm reserves them), heads at most 128; no M1/M2 build.
const tensor_ops = blk: {
    const functions = [_][]const u8{ "im_cast", "im_point", "im_norm", "im_heads", "im_gelu" };
    var out: [2 * functions.len]M = undefined;
    for (functions, 0..) |f, i| {
        const threads: u32 = if (i == 3) 128 else 256;
        const tg: u32 = if (i == 2) 144 else 0;
        const reserved: u32 = if (i == 2) 256 else 0;
        out[2 * i] = .{ .name = "sam3 tensor ops " ++ f, .threads = threads, .tg_memory = tg, .reserved = reserved, .offline = false };
        out[2 * i + 1] = .{ .name = "sam3 tensor ops control " ++ f, .threads = threads, .tg_memory = tg, .reserved = reserved, .offline = false };
    }
    break :blk out;
};
