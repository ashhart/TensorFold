//! MiniMax-M3.1-Flash's draft launches (metal_minimax_draft*.zig): dspark.metal in both builds, and dflash2.metal.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "minimax_m3 draft target mm_tap", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft target 8 q heads mm_draft_vec", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dspark 16 q heads mm_draft_vec", .threads = 512, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dspark mm_norm16", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dspark mm_draft_prep", .threads = 32, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dspark mm_markov_step", .threads = 256, .tg_memory = 1088, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dspark mm_markov_pick", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dflash2 df_norm", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dflash2 df_norm16", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dflash2 df_prep", .threads = 32, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dflash2 df_conv", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dflash2 df_silu", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dflash2 df_topk", .threads = 1024, .tg_memory = 16384, .m1 = 1024, .m2 = 1024 },
    .{ .name = "minimax_m3 draft dflash2 df_walk", .threads = 256, .tg_memory = 1040, .m1 = 1024, .m2 = 1024 },
};
