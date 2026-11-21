//! Shader texts; the prompt's fragment helpers are inline so both chip paths use the same resolver.

pub const attention = @embedFile("attention.metal");
pub const prompt_attention = @embedFile("../nax.h") ++ @embedFile("prompt_attention.metal");
pub const prompt_attention_pre_m5 = "#define TF_SIMD_FRAGS 1\n" ++ prompt_attention;
