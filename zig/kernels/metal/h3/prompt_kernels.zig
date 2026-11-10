//! MiniMax-H3-Image's prompt path kernels (any Apple GPU), compiled at load.
pub const source = @embedFile("h3_prompt.metal");
pub const names = [_][:0]const u8{ "hp_gemm", "hp_norm", "hp_heads", "hp_attention", "hp_glu", "hp_add", "hp_scatter_add" };
