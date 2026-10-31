//! DeepSeek-V4.1's Metal kernel sources, embedded in the executable (families/deepseek_v41/metal).

/// DeepSeek V4.1's release FP8 dense prompt rows, decoded exactly into fp32 MMA groups.
pub const fp8_mma = @embedFile("fp8_mma.metal");
/// DeepSeek V4.1 release FP8 dense projections: 32x32 blocks and split-invariant row sums.
pub const fp8_rows = @embedFile("fp8_rows.metal");
