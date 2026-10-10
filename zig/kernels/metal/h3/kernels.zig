//! MiniMax H3's Metal kernels: three sources joined and compiled at load.
pub const source = @embedFile("h3_products.metal") ++ @embedFile("h3_rows.metal") ++ @embedFile("h3_attention.metal");
pub const names = [_][:0]const u8{ "h3_quant_weight_t", "h3_rows_in", "h3_norm_q8", "h3_gate_add", "h3_heads_q8", "h3_tile_scales", "h3_pool", "h3_tile_scores", "h3_topk", "h3_coarse", "h3_attention_tiles", "h3_value_sums", "h3_attention_w8", "h3_gate_mix", "h3_mix_quant", "h3_i8_in", "h3_i8_out", "h3_final_norm", "h3_rows_out", "h3p_quant_rows", "h3p_i8_linear_wide", "h3p_i8_swiglu", "h3_i8_swiglu" };
