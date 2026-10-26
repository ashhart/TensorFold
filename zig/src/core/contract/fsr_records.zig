//! FSR variants reserve their existing group widths in source; no compiler measurements are fabricated.
const M = @import("limits.zig").MetalLaunch;
pub const route_memory = 2 * 320 * @sizeOf(i32);
pub const gather_memory = 64 * (64 + 16 / @sizeOf(u16)) * @sizeOf(u16);
pub const metal = [_]M{
    .{ .name = "flashnext FSR recorded custom_kernel_qa_expert_gateup_fd30a642315017a8_bfloat16_t_float_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_uint32_t_float", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "flashnext FSR recorded custom_kernel_qa_expert_down_y_346a3e7c6008e1ac_bfloat16_t_uint32_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_tc_bfloat16_t", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "flashnext FSR fz_xgu", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "flashnext FSR fz_xdown", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "flashnext FSR fz_route", .threads = 512, .tg_memory = route_memory, .reserved = 512 },
    .{ .name = "flashnext FSR tf_gather_qmm6_nax_32", .threads = 128, .tg_memory = gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_qmm6_nax_64", .threads = 128, .tg_memory = gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_qmm6_nax_128", .threads = 128, .tg_memory = gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_gu6_nax_32", .threads = 128, .tg_memory = 2 * gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_gu6_nax_64", .threads = 128, .tg_memory = 2 * gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_qmmw_nax_32", .threads = 128, .tg_memory = gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_qmmw_nax_64", .threads = 128, .tg_memory = gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_qmmw_nax_128", .threads = 128, .tg_memory = gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_guw_nax_32", .threads = 128, .tg_memory = 2 * gather_memory, .reserved = 128 },
    .{ .name = "flashnext FSR tf_gather_guw_nax_64", .threads = 128, .tg_memory = 2 * gather_memory, .reserved = 128 },
};
