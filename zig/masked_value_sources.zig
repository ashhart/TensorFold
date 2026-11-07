//! Test-only access to existing source files; production kernels remain owned by their families.
pub const kimi_common = @embedFile("kernels/metal/kimi/kimi_common.h");
pub const kimi_mla = @embedFile("kernels/metal/kimi/mla.metal");
const sources = @import("kernel_sources");
pub const lfm3_attention = if (@hasDecl(sources, "lfm3_attention")) sources.lfm3_attention else "";
