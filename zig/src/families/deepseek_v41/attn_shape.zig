//! Launch shape shared by the Metal checkpoint check and the CSA2 prompt encoder.

/// Heads per simdgroup in the Metal CSA2 prompt kernel.
pub const heads_per_group = 2;

pub fn check(heads: u32) !void {
    if (heads == 0 or heads % heads_per_group != 0) return error.UnsupportedAttentionHeadCount;
}
