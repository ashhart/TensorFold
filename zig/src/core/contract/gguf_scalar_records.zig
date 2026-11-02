//! F32 GGUF gain specializations and direct-coefficient GDN dispatches; measured device limits remain a gate.
const M = @import("limits.zig").MetalLaunch;
pub const metal = [_]M{
    .{ .name = "GGUF scalar input custom_kernel_q27_norm_input_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_float", .threads = 320, .tg_memory = 40 },
    .{ .name = "GGUF scalar residual custom_kernel_q27_norm_xs_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float", .threads = 320, .tg_memory = 40 },
    .{ .name = "GGUF scalar head q27_head_norm", .threads = 64, .tg_memory = 132 },
    .{ .name = "GGUF scalar prompt q27_prompt_decay", .threads = 64, .tg_memory = 0 },
    .{ .name = "GGUF scalar pre tf_delta_gdn_pre", .threads = 32, .tg_memory = 0 },
    .{ .name = "GGUF scalar step tf_delta_gdn_pre_step", .threads = 128, .tg_memory = 0 },
};
