//! DeepSeek-V4.1's Metal kernel sources, embedded in the executable (families/deepseek_v41/metal).

/// DeepSeek V4.1's release FP8 dense prompt rows, decoded exactly into fp32 MMA groups.
pub const fp8_mma = @embedFile("fp8_mma.metal");
/// DeepSeek V4.1 release FP8 dense projections: 32x32 blocks and split-invariant row sums.
pub const fp8_rows = @embedFile("fp8_rows.metal");
/// DeepSeek V4.1's copy of Kimi's expert plan, chains and combine: the E8M0 scale over its full range.
pub const experts_kimi = @embedFile("experts_kimi.metal");
/// DeepSeek V4.1's routed-expert gate/up (clamped SwiGLU, the reference's activation rounding) over the copy.
pub const experts = @embedFile("experts.metal");
pub const xmma = @embedFile("experts_mma.metal");
