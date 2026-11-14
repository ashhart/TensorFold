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
    .{ .name = "sam3 sparse mask im_sam_mask_downsample", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 sparse mask control im_sam_mask_downsample", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 resize im_sam_resize", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 resize control im_sam_resize", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 resize nonfinite-control im_sam_resize", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 compensated normalization sam_norm_compensated", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 compensated normalization control sam_norm_compensated", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 attention D64 device sam_attention_device_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device_center", .threads = 128, .tg_memory = 2560, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device64", .threads = 128, .tg_memory = 17152, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device_fallback", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D64 device sam_attention_device_rescale", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 device sam_attention_device_clear", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 device sam_attention_device_domain", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D32 device sam_attention_device_center", .threads = 128, .tg_memory = 2560, .reserved = 128, .offline = false },
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
    .{ .name = "sam3 small compensated projection tf_sam3_dense_small", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 small compensated projection baseline tf_sam3_dense_small", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 small compensated projection control tf_sam3_dense_small_drop_tail", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "sam3 deconvolution im_sam_deconv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 deconvolution baseline im_sam_deconv_baseline", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 deconvolution matrix im_sam_deconv_weight", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "sam3 deconvolution matrix im_sam_deconv_shuffle", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "sam3 attention im_attention_f32_full256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention control im_attention_f32_full256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention two-pass im_attention_scores256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention two-pass im_attention_values256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention two-pass control im_attention_values256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D256 split sam_attention_split_scores", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D256 split sam_attention_split_values", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D256 split sam_attention_split_merge", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention D256 split control sam_attention_split_values", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
};
