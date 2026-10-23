//! Flash Next vision and rotary pipelines with the M1 and M2 thread limits their offline compiles report.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "flashnext vision fz_mask_rows", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flashnext vision fn_attn_prep_mrope", .threads = 256, .tg_memory = (256 + 8) * 4, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flashnext rotary target custom_kernel_q4_attn_prep_63a42a6cd8e0434a_bfloat16_t_int32_t_float_float_float_floatc_floatc_bfloat16_t_bfloat16_t_bfloat16_t", .threads = 256, .tg_memory = (256 + 8) * 4, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flashnext rotary mtp custom_kernel_q4_attn_prep_63a42a6cd8e0434a_bfloat16_t_int32_t_float_float_float_floatc_floatc_bfloat16_t_bfloat16_t_bfloat16_t", .threads = 256, .tg_memory = (256 + 8) * 4, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flashnext rotary pooled custom_kernel_q4_idx_pool_6d826f5ebecec197_bfloat16_t_int32_tc_float_floatc_floatc_bfloat16_t", .threads = 128, .tg_memory = (128 + 4) * 4, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flashnext rotary absolute fz_idx_pool_abs", .threads = 128, .tg_memory = (128 + 4) * 4, .m1 = 1024, .m2 = 1024 },
    .{ .name = "flashnext rotary positioned fz_idx_pool_pos", .threads = 128, .tg_memory = (128 + 4) * 4, .m1 = 1024, .m2 = 1024 },
};
