//! One compilation unit for the packed expert kernels.
pub const experts = @embedFile("experts_common.metal") ++ "\n" ++ @embedFile("experts_quant.metal") ++ "\n" ++ @embedFile("experts_dot.metal");
