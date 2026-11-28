//! Small DSpark Metal launches (zig/tests/contract/metal_inkling2_small_dspark.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 small_dspark init_rows tf_ink2_small_dspark_init_rows", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 small_dspark fuse_taps tf_ink2_small_dspark_fuse_taps", .threads = 256, .tg_memory = 64, .reserved = 256 },
    .{ .name = "metal inkling2 small_dspark block_qkv tf_ink2_small_dspark_block_qkv", .threads = 128, .tg_memory = 16, .reserved = 128 },
    .{ .name = "metal inkling2 small_dspark attention tf_ink2_small_dspark_attention", .threads = 384, .tg_memory = 13824, .reserved = 384 },
    .{ .name = "metal inkling2 small_dspark residual_norm tf_ink2_small_dspark_residual_norm", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "metal inkling2 small_dspark kappa tf_ink2_small_dspark_kappa", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "metal inkling2 small_dspark base tf_ink2_small_dspark_base", .threads = 256, .tg_memory = 5120, .reserved = 256 },
};
