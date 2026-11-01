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
/// DeepSeek-V4.1 sparse-attention indexer top-512 over fp32 score rows.
pub const topk512 = @embedFile("topk512.metal");
pub const glue = @embedFile("glue.metal");
/// DeepSeek V4.1 decode launches that split or fuse the phase-1 glue at the same bits (deepseek_v41/fused.metal).
pub const fused = @embedFile("fused.metal");
/// DeepSeek V4.1 decode kernels that fold a dependent glue launch into its neighbour (deepseek_v41/fold.metal).
pub const fold = @embedFile("fold.metal");
pub const compress = @embedFile("compress.metal");
/// DeepSeek-V4.1 router, indexer, compressor and head matvecs: lane chains over K, one butterfly, any row count.
pub const mv = @embedFile("mv.metal");
/// DeepSeek V4.1's vision tower: epilogues, RoPE, attention, SwiGLU, the aligner's unshuffle and the stream merge.
pub const vision = @embedFile("vision.metal");
/// DeepSeek-V4.1 candidate blocks: block maxima, the newest block pinned, an exact top-2048 as a byte mask.
pub const cand = @embedFile("cand.metal");
/// DeepSeek V4.1's indexer scores: FP4 query and key dots in one stated order, causal and block masks.
pub const index = @embedFile("index.metal");
/// DeepSeek V4.1 Engram: FP8 row dequantisation, F16 wkv, the gate and the residual add.
pub const engram = @embedFile("engram.metal");
/// DeepSeek-V4.1's MoE route: the top-k of sqrt(softplus(logit)) + bias, per row (deepseek_v41/route.metal).
pub const route = @embedFile("route.metal");
/// DeepSeek-V4.1's hyper-connection boundary (deepseek_v41/hc.metal): the mixes, the collapse, the expand.
pub const hc = @embedFile("hc.metal");
/// DeepSeek-V4.1's phase-1 glue (deepseek_v41/glue_tmp.metal): the reference bits of the registered glue kernels.
pub const glue_tmp = @embedFile("glue_tmp.metal");
