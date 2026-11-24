//! Embedded Metal sources for GLM-5.3-Flash TFQ3 (2D-AffineCoset/half4) decode and prefill.
pub const header = "#include \"../nax.h\"\nusing namespace metal;\n";
pub const common = @embedFile("common.metal");
pub const moe_mv = @embedFile("moe_mv.metal");
pub const moe_decode = @embedFile("moe_decode.metal");
pub const dense_decode = @embedFile("dense_decode.metal");
pub const prefill = @embedFile("prefill.metal");
pub const kda_q8 = @embedFile("kda_q8.metal");
