//! One compilation unit for the packed expert kernels.
pub const experts = @embedFile("experts_common.metal") ++ "\n" ++ @embedFile("experts_dot.metal") ++ "\n" ++ @embedFile("../nax.h") ++ "\n" ++ @embedFile("experts_grouped_nax.metal");
