//! Inkling 2 Mini's TFQ3/TFQ4 (2/225/3/4/5-bit) routed MoE & dense Metal source.
pub const tfq3 = @embedFile("tfq3.metal") ++ "\n" ++ @embedFile("tfq4_moe.metal");
