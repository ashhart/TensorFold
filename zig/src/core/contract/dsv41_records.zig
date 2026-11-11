//! DeepSeek V4.1 gate GEMMs: the f16 and bf16 f32-out NAX forms the router gate launches.
const limits = @import("limits.zig");
pub const metal = [_]limits.MetalLaunch{
    .{ .name = "dsv41 gate gemm f16 custom_kernel_tf_gemm_nax_f16_f32out_n_t_n_t_n_64_128_256_2_4_float16_t_float16_t_int32_t_float", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "dsv41 gate gemm bf16 custom_kernel_tf_gemm_nax_bf16_f32out_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_float", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
