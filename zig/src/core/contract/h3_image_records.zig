//! MiniMax-H3-Image's launches: its edit encoder's, prompt path's and image decoder's kernels, each source-reserved.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const encoder = [_]M{
    .{ .name = "minimax image encoder hie_conv", .threads = 256, .tg_memory = 16384, .reserved = 256 },
    .{ .name = "minimax image encoder hie_group_part", .threads = 1024, .tg_memory = 0, .reserved = 1024 },
    .{ .name = "minimax image encoder hie_group_finish", .threads = 1024, .tg_memory = 0, .reserved = 1024 },
    .{ .name = "minimax image encoder hie_norm_silu", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
pub const prompt = [_]M{
    .{ .name = "minimax image prompt hp_gemm", .threads = 128, .tg_memory = 10240, .reserved = 128 },
    .{ .name = "minimax image prompt hp_norm", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "minimax image prompt hp_heads", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "minimax image prompt hp_attention", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "minimax image prompt hp_glu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "minimax image prompt hp_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "minimax image prompt hp_scatter_add", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
pub const decoder = [_]M{
    .{ .name = "minimax image decoder h3d_post_quant", .threads = 192, .tg_memory = 0, .reserved = 192 },
    .{ .name = "minimax image decoder h3d_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "minimax image decoder h3d_rms_norm", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "minimax image decoder h3d_layer_norm", .threads = 256, .tg_memory = 64, .reserved = 256 },
    .{ .name = "minimax image decoder h3d_gemm", .threads = 128, .tg_memory = 18432, .reserved = 128 },
    .{ .name = "minimax image decoder h3d_heads", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "minimax image decoder h3d_attention", .threads = 256, .tg_memory = 11264, .reserved = 256 },
    .{ .name = "minimax image decoder h3d_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
pub const records = encoder ++ prompt ++ decoder;

test "the MiniMax-H3-Image launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
