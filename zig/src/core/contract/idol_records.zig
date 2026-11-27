//! Idol's launches, each source-reserved: the denoiser's, the decoders', the VAE's and BiRefNet's.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = denoiser ++ decoder ++ vae ++ birefnet;

const denoiser = [_]M{
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

const decoder = [_]M{
    .{ .name = "idol decoder dec_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol decoder dec_half", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol decoder dec_kv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol decoder dec_heads", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol decoder dec_attention_few", .threads = 256, .tg_memory = 2352, .reserved = 256 },
    .{ .name = "idol decoder dec_attention_00", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "idol decoder dec_attention_01", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "idol decoder dec_attention_10", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "idol decoder dec_attention_11", .threads = 128, .tg_memory = 0, .reserved = 128 },
};

const vae = [_]M{
    .{ .name = "idol vae iv_conv_in", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol vae iv_conv", .threads = 128, .tg_memory = 26624, .reserved = 128 },
    .{ .name = "idol vae iv_wino", .threads = 512, .tg_memory = 32768, .reserved = 512 },
    .{ .name = "idol vae iv_norm", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol vae iv_group_part", .threads = 512, .tg_memory = 2048, .reserved = 512 },
    .{ .name = "idol vae iv_group_finish", .threads = 1024, .tg_memory = 0, .reserved = 1024 },
    .{ .name = "idol vae iv_attention", .threads = 128, .tg_memory = 5168, .reserved = 128 },
    .{ .name = "idol vae iv_tokens", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

const birefnet = [_]M{
    .{ .name = "idol birefnet br_conv", .threads = 128, .tg_memory = 26624, .reserved = 128 },
    .{ .name = "idol birefnet br_norm", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "idol birefnet br_window", .threads = 192, .tg_memory = 28416, .reserved = 192 },
    .{ .name = "idol birefnet br_resize", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol birefnet br_tiles", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol birefnet br_conv3", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol birefnet br_deform", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol birefnet br_pool", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "idol birefnet br_gate", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "idol birefnet br_head", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

test "the Idol launches fit every Apple family, one record a pipeline" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
    try std.testing.expectEqual(@as(usize, 24 + 9 + 8 + 10), records.len);
}
