//! MiniMax H3's Metal kernels: three sources joined and compiled at load, after the prefill kernels' gemm_nax.
pub const source = @embedFile("h3_rows.metal") ++ @embedFile("h3_routing.metal") ++ @embedFile("h3_attention.metal");
/// MLX's bf16 GEMM, x w^T with N a multiple of 128 and M a multiple of 64 (whole) or not (ragged).
pub const gemm_whole = "custom_kernel_tf_gemm_nax_bf16_n_t_t_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";
pub const gemm_ragged = "custom_kernel_tf_gemm_nax_bf16_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t";
pub const names = [_][:0]const u8{ "h3_rows_in", "h3_norm", "h3_gate_add", "h3_gate_mix", "h3_swiglu", "h3_final_norm", "h3_rows_out", "h3_heads", "h3_pool", "h3_tile_scores", "h3_topk", "h3_coarse", "h3_attention", "h3_modulation", gemm_whole, gemm_ragged };
