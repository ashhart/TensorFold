//! Shader texts; the prompt's fragment helpers are inline so both chip paths use the same resolver.
const guard = @import("../threadgroup_source.zig");

pub const attention = guard.normalize(@embedFile("attention.metal"));
pub const prompt_attention = guard.normalize(@embedFile("../nax.h") ++ @embedFile("prompt_attention.metal"));
pub const prompt_attention_pre_m5 = guard.normalize("#define TF_SIMD_FRAGS 1\n" ++ prompt_attention);
