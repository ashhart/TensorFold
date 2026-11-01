//! DeepSeek-V4.1's Metal kernel sources, embedded in the executable (families/deepseek_v41/metal).

/// DeepSeek V4.1's release FP8 dense prompt rows, decoded exactly into fp32 MMA groups.
pub const fp8_mma = @embedFile("fp8_mma.metal");
/// DeepSeek V4.1 release FP8 dense projections: 32x32 blocks and split-invariant row sums.
pub const fp8_rows = @embedFile("fp8_rows.metal");
/// DeepSeek V4.1 CSA2 attention: window, selected compressed rows and the sink over one shared latent row.
pub const csa2_attn = @embedFile("csa2_attn.metal");
/// DeepSeek V4.1's copy of Kimi's expert plan, chains and combine: the E8M0 scale over its full range.
pub const experts_kimi = @embedFile("experts_kimi.metal");
/// DeepSeek V4.1's routed-expert gate/up (clamped SwiGLU, the reference's activation rounding) over the copy.
pub const experts = @embedFile("experts.metal");
pub const xmma = @embedFile("experts_mma.metal");
pub const glue = @embedFile("glue.metal");
/// DeepSeek-V4.1 router, indexer, compressor and head matvecs: lane chains over K, one butterfly, any row count.
pub const mv = @embedFile("mv.metal");
/// DeepSeek-V4.1's phase-1 glue (deepseek_v41/glue_tmp.metal): the reference bits of the registered glue kernels.
pub const glue_tmp = @embedFile("glue_tmp.metal");
