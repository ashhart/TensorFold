//! The indexed expert kernels keep the recorded kernels' 64-thread groups, reserved in their source.
const M = @import("limits.zig").MetalLaunch;
pub const metal = [_]M{
    .{ .name = "flashnext index recorded custom_kernel_qa_expert_gateup_fd30a642315017a8_bfloat16_t_float_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_uint32_t_float", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "flashnext index recorded custom_kernel_qa_expert_down_y_346a3e7c6008e1ac_bfloat16_t_uint32_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_tc_bfloat16_t", .threads = 64, .tg_memory = 0, .reserved = 64 },
};
