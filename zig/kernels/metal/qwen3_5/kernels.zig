//! Qwen3.5 kernel templates. The family prepends a checkpoint's geometry as constants when it compiles them.
pub const Kernel = struct { key: []const u8, function: [:0]const u8, source: []const u8 };
pub const qmm = Kernel{ .key = "qmm", .function = "qwen35_qmm", .source = @embedFile("qmm.metal") };
pub const all = [_]Kernel{
    .{ .key = "norm", .function = "custom_kernel_qwen35_norm_bfloat16_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t_bfloat16_t", .source = @embedFile("norm.metal") },
    .{ .key = "norm_nores", .function = "custom_kernel_qwen35_norm_nores_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("norm_nores.metal") },
    .{ .key = "gdn_pre", .function = "custom_kernel_qwen35_gdn_pre_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_bfloat16_t_float_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_float_bfloat16_t_bfloat16_t", .source = @embedFile("gdn_pre.metal") },
    .{ .key = "gdn_chain", .function = "custom_kernel_qwen35_gdn_chain__bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_float_bfloat16_t_float_int32_tc_bfloat16_t_float", .source = @embedFile("gdn_chain.metal") },
    .{ .key = "gdn_post", .function = "custom_kernel_qwen35_gdn_post_bfloat16_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("gdn_post.metal") },
    .{ .key = "mlp_act", .function = "custom_kernel_qwen35_mlp_act_bfloat16_t_bfloat16_t_bfloat16_t", .source = @embedFile("mlp_act.metal") },
    .{ .key = "attn_partial", .function = "custom_kernel_qwen35_attn_partial_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_float_float_float", .source = @embedFile("attn_partial.metal") },
    .{ .key = "attn_merge", .function = "custom_kernel_qwen35_attn_merge_float_float_float_int32_t_bfloat16_t", .source = @embedFile("attn_merge.metal") },
    .{ .key = "state_copy", .function = "qwen35_state_copy", .source = @embedFile("state_copy.metal") },
};
