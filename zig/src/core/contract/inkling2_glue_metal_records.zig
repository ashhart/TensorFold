//! Inkling 2 Mini glue launches (zig/tests/contract/metal_inkling2_glue.zig).
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal inkling2 glue embed_norm tf_inkling2_embed_norm", .threads = 256, .tg_memory = 48, .reserved = 256 },
    .{ .name = "metal inkling2 glue ar tf_inkling2_ar", .threads = 256, .tg_memory = 368, .reserved = 256 },
    .{ .name = "metal inkling2 glue conv_static tf_inkling2_conv_static", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 glue conv_dynamic tf_inkling2_conv_dynamic", .threads = 256, .tg_memory = 128, .reserved = 256 },
    .{ .name = "metal inkling2 glue router tf_inkling2_router", .threads = 256, .tg_memory = 576, .reserved = 256 },
    .{ .name = "metal inkling2 glue swiglu tf_inkling2_swiglu", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 glue logits tf_inkling2_logits", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 glue expert_pick tf_inkling2_expert_pick", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 glue expert_combine tf_inkling2_expert_combine", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 glue route_scores tf_inkling2_route_scores", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "metal inkling2 glue route_select tf_inkling2_route_select", .threads = 32, .tg_memory = 0, .reserved = 32 },
};
