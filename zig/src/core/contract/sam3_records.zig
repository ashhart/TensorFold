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
    var out: [8]M = undefined;
    for ([_][]const u8{ "00", "01", "10", "11" }, 0..) |tag, i| {
        out[i] = .{ .name = "sam3 fp32 gemm qwv_gemm_f32_nt_" ++ tag, .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false };
        out[i + 4] = .{ .name = "sam3 fp32 gemm qwv_gemm_f32_exact_nt_" ++ tag, .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false };
    }
    break :blk out;
} ++ [_]M{
    .{ .name = "sam3 deconvolution im_sam_deconv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 deconvolution baseline im_sam_deconv_baseline", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "sam3 attention im_attention_f32_full256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "sam3 attention control im_attention_f32_full256", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
};
