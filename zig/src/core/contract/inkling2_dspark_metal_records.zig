//! Mini DSpark Metal launches (zig/tests/contract/metal_inkling2_dspark.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 dspark init_rows tf_ink2_dspark_init_rows", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark fuse_taps tf_ink2_dspark_fuse_taps", .threads = 256, .tg_memory = 48, .reserved = 256 },
    .{ .name = "metal inkling2 dspark block_qkv tf_ink2_dspark_block_qkv", .threads = 128, .tg_memory = 16, .reserved = 128 },
    .{ .name = "metal inkling2 dspark context_kv tf_ink2_dspark_context_kv", .threads = 128, .tg_memory = 16, .reserved = 128 },
    .{ .name = "metal inkling2 dspark attention tf_ink2_dspark_attention", .threads = 768, .tg_memory = 20736, .reserved = 768, .offline = false },
    .{ .name = "metal inkling2 dspark residual_norm tf_ink2_dspark_residual_norm", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "metal inkling2 dspark swiglu tf_ink2_dspark_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 dspark kappa tf_ink2_dspark_kappa", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "metal inkling2 dspark recurrent tf_ink2_dspark_recurrent", .threads = 256, .tg_memory = 1024, .reserved = 256 },
    .{ .name = "metal inkling2 dspark base tf_ink2_dspark_base", .threads = 256, .tg_memory = 5120, .reserved = 256 },
    .{ .name = "metal inkling2 dspark logits tf_ink2_dspark_logits", .threads = 256, .tg_memory = 4224, .reserved = 256 },
    .{ .name = "metal inkling2 dspark logits_q4 tf_ink2_dspark_logits_q4", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
