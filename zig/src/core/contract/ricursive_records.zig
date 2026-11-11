//! Ripple-65B's Metal launches (metal_ricursive*.zig): its glue, prompt attention, the draw and DFlash2's kernels.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "ricursive glue rc_embed", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // part_y[32] and part_r[32] floats; the source reserves its 256 threads
    .{ .name = "ricursive glue rc_add_norm", .threads = 256, .tg_memory = 256, .reserved = 256 },
    .{ .name = "ricursive glue rc_prep", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive glue rc_gate", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive glue rc_silu_clamp", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // a simdgroup for each of a key head's 10 query heads
    .{ .name = "ricursive glue rc_decode_vec", .threads = 320, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // tensor ops before the M5 too (nax.h's simdgroup fragments still multiply through mpp): no offline build
    .{ .name = "ricursive attention rc_prompt_attention", .threads = 64, .tg_memory = 9472, .reserved = 256, .offline = false },
    .{ .name = "ricursive fp4 rc_fp4_matvec_1", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive fp4 rc_fp4_matvec_2", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive fp4 rc_fp4_matvec_4", .threads = 256, .tg_memory = 0, .m1 = 768, .m2 = 768 },
    .{ .name = "ricursive fp4 rc_fp4_matvec_8", .threads = 256, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "ricursive fp4 rc_fp4_matvec_16", .threads = 256, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "ricursive fp4 rc_fp4_matvec_32", .threads = 256, .tg_memory = 0, .m1 = 448, .m2 = 448 },
    .{ .name = "ricursive fp4 rc_fp4_dequant", .threads = 256, .tg_memory = 0, .m1 = 832, .m2 = 832 },
    // gemm_nax's tiles: tensor ops, no offline build; the source reserves its 256 threads
    .{ .name = "ricursive fp4 gemm rc_fp4_gemm_00", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "ricursive fp4 gemm rc_fp4_gemm_01", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "ricursive fp4 gemm rc_fp4_gemm_10", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "ricursive fp4 gemm rc_fp4_gemm_11", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false },
    .{ .name = "ricursive sample tf_sample_full", .threads = 1024, .tg_memory = 9760, .reserved = 1024, .m1 = 1024, .m2 = 1024 },
    // dflash2.metal: the target's tap and the DFlash2 drafter's kernels
    .{ .name = "ricursive dflash2 rc_tap", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive dflash2 rc_df_norm", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive dflash2 rc_df_norm16", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive dflash2 rc_df_embed", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive dflash2 rc_df_prep", .threads = 32, .tg_memory = 512, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive dflash2 rc_df_conv", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive dflash2 rc_df_silu", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // a simdgroup for each of a key head's 10 query heads
    .{ .name = "ricursive dflash2 rc_df_vec", .threads = 320, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive dflash2 rc_df_topk", .threads = 1024, .tg_memory = 16384, .m1 = 1024, .m2 = 1024 },
    .{ .name = "ricursive dflash2 rc_df_walk", .threads = 256, .tg_memory = 1040, .m1 = 1024, .m2 = 1024 },
};
