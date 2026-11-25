//! Inkling 2 Small Metal launches (zig/tests/contract/metal_inkling2_small.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 small embed_norm tf_ink2_small_embed_norm", .threads = 256, .tg_memory = 48, .reserved = 256 },
    .{ .name = "metal inkling2 small ar tf_ink2_small_ar", .threads = 256, .tg_memory = 656, .reserved = 256 },
    .{ .name = "metal inkling2 small conv_dynamic tf_ink2_small_conv_dynamic", .threads = 256, .tg_memory = 128, .reserved = 256 },
    .{ .name = "metal inkling2 small router tf_ink2_small_router", .threads = 256, .tg_memory = 4688, .reserved = 256 },
    .{ .name = "metal inkling2 small logits tf_ink2_small_logits", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 small attention_partial tf_ink2_small_attention_partial", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "metal inkling2 small capture_tap tf_ink2_small_capture_tap", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
