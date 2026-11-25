//! Embedded FlashNext TFQ3 Metal shader sources.
pub const common = @embedFile("common.metal");
pub const moe_mv = @embedFile("moe_mv.metal");
pub const moe_decode = @embedFile("moe_decode.metal");
pub const prefill = @embedFile("prefill.metal");
