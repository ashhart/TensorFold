//! Inkling 2 Mini runtime helper launches (zig/tests/contract/metal_inkling2_runtime.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 runtime copy_bf16 tf_ink2_copy_bf16", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 runtime copy_f32 tf_ink2_copy_f32", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 runtime widen tf_ink2_widen", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 runtime attention_partial tf_ink2_attention_partial", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 runtime mlp_partial tf_ink2_mlp_partial", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 runtime capture_tap tf_ink2_capture_tap", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 runtime stage_conv tf_ink2_stage_conv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 runtime commit_histories tf_ink2_commit_histories", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
