//! MiniMax-M3.1-Flash's nax launches (metal_minimax_nax.zig); the grouped experts reserve 64 threads, as GLM's rows do.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "minimax_m3 nax experts q3 mm_experts_grouped_q3", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "minimax_m3 nax experts q4 mm_experts_grouped_q4", .threads = 64, .tg_memory = 0, .reserved = 64 },
    // tensor ops before the M5 too (nax.h's simdgroup fragments still multiply through mpp): no offline build
    .{ .name = "minimax_m3 nax attention mm_prompt_attention", .threads = 64, .tg_memory = 9472, .reserved = 256, .offline = false },
    .{ .name = "minimax_m3 nax index mm_index_scores_nax", .threads = 64, .tg_memory = 256, .reserved = 256, .offline = false },
};
