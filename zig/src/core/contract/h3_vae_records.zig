//! MiniMax-H3 video and image VAE launches: the decode ViT's and the encoder conv's kernels, each source-reserved.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const decode = [_]M{
    .{ .name = "h3 vae matmul h3_vae_matmul", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "h3 vae rms norm h3_vae_rmsnorm", .threads = 256, .tg_memory = 1024, .reserved = 256 },
    .{ .name = "h3 vae SwiGLU h3_vae_swiglu", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "h3 vae layer norm h3_vae_layernorm", .threads = 256, .tg_memory = 1024, .reserved = 256 },
    .{ .name = "h3 vae post quant h3_vae_postquant", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "h3 vae unpatch h3_vae_unpatch", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "h3 vae v extract h3_vae_vextract", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "h3 vae attention h3_vae_attention_il", .threads = 1, .tg_memory = 0, .reserved = 1 },
};
pub const encode = [_]M{
    .{ .name = "h3 vae encoder conv h3_vae_enc_conv", .threads = 1, .tg_memory = 0, .reserved = 1 },
};
pub const records = decode ++ encode;

test "the MiniMax-H3 VAE launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}